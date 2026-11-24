# Weight-Only Quantized GEMM and Decode GEMV (`INT4` / `NF4`) on A770

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`. Operator: `y = x * W` (`W` packed 4-bit `[N, K/2]`, low nibble first along `K`, per-group scales). Distinct from `GEMV-N1` (`A * x`).

## 1. Production ESIMD Online-Dequant Fast Path (`oneAPI 2026.2`) `[MEASURED]`

Earlier oneAPI 2026.1 attempts stalled at `1.7–5.7 ms` (forcing impractical host pre-dequantization). On **oneAPI 2026.2**, zero-buffer on-device dequantization achieves high throughput by aligning `BK` to `group_size`, coalescing `16 B` reads, and using SLM LUTs:

### 1.1 `4-Bit` (`NF4` / `INT4Sym`) Online-Dequant ESIMD `dpasw` GEMM (`512x4096x4096, gs=64`)

- **Geometry (`BM=64, BN=64, BK=64, WG=32 (8x4)`)**: Setting **`BK == group_size == 64`** guarantees exactly one scale per K-tile, eliminating inner-loop integer division and branching.
- **Coalesced `4x16 B` OWord reads + SLM LUT**: Load packed weights via `esimd::block_load<uint32_t, 4>` (`16 B`, 8 nibbles/`uint32_t`), stage the 16-entry `float` NF4 codebook at the tail of SLM (total SLM `32,832 B <= 48 KB` -> `2 WGs/Xe-core`), look up via SIMD16 `esimd::slm_gather<float, 16>`, multiply by the tile scale, convert to `bf16`, and write 16 B-aligned VNNI slots via **`esimd::slm_block_store<uint32_t, 4>`** before `esimd::xmx::dpasw<8,8>`.
- **Measured performance (`5x`-batched L0 Regular Replay, `errors=0` vs CPU ref)**:
  - `NF4` fast path: **`763.9–764.1 us` (`22.49 TFLOPS`)** -- **`12.95x` faster** vs naive 8 B-strided + dynamic register-array LUT baseline (`9886.6 us / 1.74 TFLOPS`).
  - `INT4Sym` fast path: **`699.1–706.6 us` (`24.57 TFLOPS`)**.

Copy-ready core: [4-bit BK=gs=64 online dequant core](api/code-snippets.md#4-4-bit-nf4--int4sym-bkgs64-zero-branch-dequant--slm-lut-core).

### 1.2 Weight-Only Decode GEMV (`4096x4096, gs=64`, `9.4 MB` packed weights, `20x`-batched L0 Replay)

| Format | Batch (`M`) | Median Time | Effective BW | vs `bf16` GEMV (`100.5 us`) |
|---|---:|---:|---:|---|
| `NF4` | `1` | **`40.13–40.23 us`** | `235.2 GB/s` | **`2.50x–2.51x` faster** |
| `NF4` (weight reuse) | `4` | **`47.83–47.87 us`** | `197.3 GB/s` | `11.96 us/token` |
| `INT4Sym` | `1` | **`27.73–29.03 us`** | **`325.1–340.4 GB/s`** | **`3.47x–3.62x` faster** |

## 2. Accuracy-Matched `oneDNN` `u4` Baseline Contract `[DISPATCH]` `[TOOLCHAIN]`

When benchmarking against `oneDNN`'s `jit:gemm:any` `u4` path, use the **accuracy-matched `f16`-src configuration**:

- `src`: `{M, K}` **`f16` `ab`**; `weights`: `{K, N}` **`u4` `ba`** (`[N, K/2]`, low nibble first along `K`); `dst`: `{M, N}` `f32` or `bf16` `ab`.
- `scales`: `{K / group_size, N}` `f16` `ab` (`mask = (1<<0)|(1<<1)`, `groups = {group_size, 1}`); scalar `u8` zero-point `{1} = 8` (signed `[-8, 7]` stored as `u4 + 8`).
- `fpmath_mode`: **must explicitly set `strict` or `any` with `apply_to_int = true`** (leaving `fpmath_mode` unset is rejected on GPU with `dnnl_error` status 3 `[TOOLCHAIN]`).
- **Avoid `ab` weights**: Plain `ab` `u4` weights fall back to `ocl:ref:any` (`~110 ms` vs `0.033–0.034 ms` for `ba` on `1024x1536x512`).
- **`bf16`-src `oneDNN u4` accuracy trap `[CORRECTNESS]`**: `bf16`-src `u4` selects `jit:gemm:any` (`~0.14–0.32 ms` at `M=64`) but fails `rel=5e-2, abs=1e-2` (`~6,070/262,144` errors, `max_abs ~2.1–3.9`). Classify it as `fastest_only` (`comparable_for_speedup = false`), never as a matched baseline.

## 3. Why Host Pre-Dequantization Fails & Correctness Traps `[CORRECTNESS]` `[BUG]`

1. **Host pre-dequantization is not weight-only**: Converting `4096x4096` weights on CPU costs `~102 ms` (`~304 ms` for `8192x8192`) and expands VRAM footprint `4x` vs packed `u4`. Only acceptable for tiny static vectors (`u4` input vector `x` of `2 KB` in `GEMV-N1`).
2. **Packed byte offset trap**: Byte offset for K-block `bk` is `bk / 2`, never `bk * (BK / 2)` (silently reads wrong K blocks from block 2 onward).
3. **`M=1` DPAS row-guard**: Clamp row reads to `min(global_row, m - 1)` and guard stores `gr < m`; reading past a 1-row buffer can trigger `UR_RESULT_ERROR_DEVICE_LOST`.
4. **SLM `16 B` alignment**: Never store unpacked VNNI words with 8 B-aligned `slm_block_store<uint32_t, 2>`; DG2 truncates to `16 B` and silently overwrites neighbors.
5. **No mixed `f16 + u4` public DPAS**: SYCL `esimd::xmx::dpas` enforces `APrecision == BPrecision` for floating-point paths; unpack `4-bit -> bf16` in registers/SLM before `dpasw<8,8>`.
