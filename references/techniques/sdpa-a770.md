# A770 Scaled Dot-Product Attention (SDPA): Decode (`Q=1`) and Prefill

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.1–2026.2, driver `32.0.101.8860` / `32.0.101.9150`. Prefill (`Q > 1`, Domain II XMX) and Decode (`Q = 1`, Domain I DRAM stream-through) are separate microarchitectural regimes -- never substitute one campaign's design for the other.

## 1. Decode Attention (`Q = 1`, Domain I: DRAM Stream-Through) `[MEASURED]`

Decode attention (`B*H` heads, `Q=1`, `S` KV tokens, `D in {64, 128, 256}`, MHA/GQA/MQA) moves the entire KV cache once per token (`XMX = 0%`). Target the `403.1 GB/s` DRAM read roof.

### 1.1 Production `bf16` Decode SDPA (`SDPA_ORDER=1 + SDPA_REDUCE=1`)

Measured on `BH=128, S=2048, D=128` (`134.2 MB` KV, `errors=0`, supports GQA `1:1..12:1` and `D=64/128/256`):

- **16-slice split-reduction per head (`WG=512`, `1` head per WG)**: A lone 32-lane sub-group per head leaves `87.5%` of A770 idle (`12.5%` occupancy, `557 us`). Splitting the KV reduction across `16` sub-groups within the **same** head raises occupancy to full saturation without widening cross-head barriers (`557 -> 403 us`). Never put multiple heads inside one work-group (widens the barrier domain and regresses).
- **Mandatory interleaved key-block ownership (`SDPA_ORDER=1`)**: Giving each of the 16 slices a contiguous `128`-key window creates `2,048` strided streams (`328.5 GB/s`, `403–409 us`). **Interleaving key blocks across the 16 slices** (same bytes, same math) makes concurrent reads adjacent in DRAM and unlocks **`+16% to +22%` bandwidth across `S in {512, 2048, 8192}` and `D in {64, 128}`**:
  - Pinned SYCL event median: **`343.3–343.6 us` (`390.7 GB/s` = `96.9%` of `403.1 GB/s` roof)**.
  - Native L0 Regular Command List (`10x`-batched): **`347.8 us` (`386.0 GB/s` = `95.7%` roof)**; `1`-shot replay: `437.1–439.0 us` (includes WDDM submit tax).
- **SLM reduction (`SDPA_REDUCE=1`) -- critical correctness fix `[CORRECTNESS]`**: Under register pressure, IGC narrows `sycl::reduce_over_group` from SIMD32 to SIMD16/8 without emitting a cross-pass SLM merge, silently halving the `128`-dim dot product! Stage per-lane shares in **separate per-slice SLM slots** (sharing one staging buffer across slices corrupts `~7,200/16,384` outputs), run one work-group barrier, and sum as `float4`. Do **not** pin `reqd_sub_group_size(32)` (`3,705 us`, `6.6x` slower vs SLM reduce).
- **Never hoist next KV block across the barrier**: Pipelining KV loads across the reduction barrier spills registers (`11/5` to `55/30` scratch ops) and slows execution **`2.8x–3.2x` (`556 -> 1567–1763 us`)** vs zero-spill baseline `[MEASURED]`.

### 1.2 Paged KV and `f32` Decode Rules `[DISPATCH]` `[MEASURED]`

- **Paged KV (`64`-token physical pages + page table)**: Adds only **`+0.5% to +6%`** device overhead (`+4–88 us` at `KV=2048`, `+87–177 us` at `KV=32768`) vs contiguous chunk layout while supporting dynamic KV allocation.
- **Short KV crossover (`KV <= 512`, `f32`)**: Standard 3-kernel naive path can beat fused online decode at `KV=512` (`492–655 us` vs `715–863 us`) because `S`/`P` fit L3 and parallelism is higher; fused online / chunk / paged paths take over for `KV >= 2048` (`2x–3x` faster vs naive).
- **Negative path**: Never split along `D` (`split-d` VLEN 4/8/16 duplicates `QK^T` reduction per d-slice and runs **`~15x` slower** at `21.1 ms` vs `1.4 ms` `[MEASURED]`).

## 2. Prefill / Dense Multi-Row SDPA (`Q > 1`, Domain II: ESIMD XMX) `[MEASURED]`

### 2.1 Hard DG2 Porting Constraints `[ARCH]` `[BUG]`

Xe2/PVC flash-attention kernels fail on A770 unless adapted to four DG2 gates:

1. **Keep ESIMD `WG = 32`**: `WG = 512` triggers Windows `VIDEO_TDR` (LiveKernelEvent 141) even on a trivial kernel; `WG = 64` non-fused attention produces wrong results on single tiles `[BUG]`.
2. **`fp32` accumulators only**: DG2 VISA rejects `dpas` with `hf` destination (`dst:f` required).
3. **Zero compiler spill gate**: Any non-zero register spill with `RPT > 4` triggers `UR_RESULT_ERROR_DEVICE_LOST` on launch; check `spillMemSize == 0` and disassembly `send.dc0 == 0` before running.
4. **No `load_2d` / `named_barrier`**: Use standard `block_load` / `slm_block_load` (`<= 256 B`) and standard `esimd::barrier()`.

### 2.2 Production Prefill Design & Dispatch (`fp16` / `bf16`, `D = 128`)

- **Fused single-kernel path** (`qLen <= 1024` AND `qTiles * kvTiles <= 128`): `WG=32`, `RPT=4` (`4` Q rows/thread), `BN=64`, on-the-fly VNNI K/V packing into SLM in one launch (avoids multi-submit host overhead).
- **Non-fused packed path** (larger `qLen` / `kvLen`): Prepack `Q` (DPAS A-operand layout, `256 B`/slice) and `K/V` (VNNI layout) once, then run `WG=32, RPT=8` (100% XMX row utilization, zero spill) with `BN=64` (or `BN=128` using `K 32 KB + V 32 KB = 64 KB` SLM to drop the `QK -> SxV` barrier).
- **High-leverage micro-optimizations (cut attn instructions `404G -> 194G`, overtaking PyTorch SDPA by `10%–25%` across `L = 512..16384`)**:
  - Use ESIMD hardware `exp2` (`native_exp2`) instead of scalar `sycl::exp2` (SP ops `130G -> 41G`).
  - Keep scores and per-thread softmax/accumulators strictly in **registers** (storing per-thread state in shared SLM causes 32-lane races); keep SLM for shared K/V tiles only (`direct global packedV` regresses `105 -> 172 ms`).
  - Pass original `kv_len` and mask padding rows by row index (eliminates `kvZero` flag arrays).
  - Do not zero-fill unused DPAS `A` rows `4..7` when `RPT=4` (`48 -> 40 ms`).
  - For tiny `f32` prefill (`seq <= 256`), 3-kernel naive fits L3 and beats flash-style tiling (`1.82 vs 2.01 ms`).

### 2.3 Numerically Stable Online-Softmax Rescale `[CORRECTNESS]`

Never compute `exp2((m_old - m_new) * log2e)` when `m_new < m_old` -- positive exponents overflow fp32 to `inf`/`NaN` on large score spans. Always clamp rescale factors `<= 1.0`:
- If `m_tile > m_old`: scale previous `acc` and `l` by `exp2((m_old - m_tile) * log2e)`.
- If `m_tile < m_old`: scale current tile `p` and `l_tile` by `exp2((m_tile - m_old) * log2e)`.

Copy-ready core: [Stable online-softmax rescale core](../api/code-snippets.md#8-stable-online-softmax-rescale-core).

### 2.4 Integration Pitfalls `[CORRECTNESS]`

- **Use `if constexpr`, never `#if FUSED`** inside templated kernels (`#if` evaluates to `0` at preprocessing time and takes null-pointer branches).
- **SLM offsets are byte offsets**: Missing `* 4` on `uint32_t` V-pack SLM offsets silently corrupts V tiles.
- **LRU buffer cache keys**: Include `kv_heads` in packed K/V buffer cache keys for GQA shapes, and bound LRU capacity (a single `seq ~ 20,000` shape consumes `~0.9 GB`).
