# Dense Operator Best Practices and Measured Benchmarks on A770

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`, baseline `oneDNN 3.13.2` (`jit:gemm:any`, zero env vars, `errors=0`, `spill=0 B, priv=0 B`, main-loop `send.dc0 == 0`). Historical 2026.1/naive ladders have been retired in favor of current production best practices. Re-measure before transferring to other GPUs or stacks.

## 1. Domain II: Production ESIMD `GRF=256` GEMM & Fused MLP `[MEASURED]`

### 1.1 Recommended Architecture (`bf16` / `int8` / `SwiGLU`)

1. **Geometry & residency**: Enable per-kernel `sycl::ext::intel::experimental::grf_size<256>` with **`WG=32`** (`4x8` or `8x4`) and **`SLM <= 48 KB`** (`<= 49,152 B`) to sustain **`2 WGs/Xe-core = 64/64` HW threads** (never `WG=48 / 56 KB` or `WG=64` on RowMajor GEMM). Cap accumulators at **`<= 128 GRF`** (`acc[4][4]`).
2. **`cab4` 4-buffer circular SLM + `split_barrier` decoupling**: For `128x256, BK=16`, cycle four `12 KB` SLM slots (`A 4 KB + B 8 KB = 48 KB`) in a fixed decoupled beat:
   `16 uninterrupted dpasw -> sbar_wait() -> slm_fence_group() -> sbar_signal() -> store_buf(w_buf) -> load_slm_ops(next_r_buf) -> load_b() -> load_a()`
   Interleaving `store_buf` and `load_slm_ops` breaks SLM address GRF reuse chains (`45/48 -> 8/48`).
3. **`0-Dummy` `dpasw<8,8>` + A64 `bp_vec` + `-Xfinalizer`**:
   - Group `dpasw<8,8>` into `2x8` (4xB) or `3x4` (3xB) macros anchored on tail SLM load `a23` (`dummy_dpasw == 1`; see [dpas.md](../isa/dpas.md#4-igc-dummy_dpasw-nullf-injection-and-0-dummy-grouping-theorem-isa-measured)).
   - Vectorize 4x64-bit B pointers in `simd<uint64_t, 4> bp_vec` (`-94 ins`), use reverse-order `pack_a` (`-5 mov`), and coalesce B SLM writes into `4x64 B` (`store.slm.d64x8t`, `-24%–31%` write traffic).
   - Embed `-Xfinalizer -SBIDBundle -Xfinalizer -SWSBDepReduction -Xfinalizer -nopresched -Xfinalizer -twoSrcBundleBCR` at link time (`sync.allrd == 0`, `Short UGM waits <= 1`).
4. **Swizzle & cache hints**:
   - Single wave (`total_wgs <= 64`): linear `swizzle = 1`, `A` and `B` both `ca.ca`.
   - Multi-wave large grids: `2D 8x8 N-inner` serpentine walk, `swizzle <= 4` when `K >= 4096` (`-71 us` on `4096^3`), and `L1StreamB` (`st.ca` on `B`).
   - Output `C` store: `wb.wb` for standard / `Bias` / `SiLU` epilogues (`-42 us` on LM-Head vs `st.uc`); `st.uc` for `ResidualAdd` / `BiasResidualAdd` epilogues (`0.98x -> 1.06x` vs oneDNN).
5. **Single-wave medium `S1` (`1024x1536x512`) & `int8` `BK=64`**:
   - For `1024x1536x512`, `128-GRF` full-wave `BM=256 BN=192 WG=128` (`32` WGs, `BK=32` bf16 / `BK=64` int8, `57,344 B` SLM) + `64 B` coalesced staging + peeled K-loop + `dpasw<8,8>` beats `oneDNN 3.13.2` both zero-prepack (`68.30 vs 73.16 us` bf16, `1.07x`; `64.14 vs 68.71 us` int8, `1.07x`) and `Btile192`-prepacked (`66.08 us` bf16, `1.11x`; `57.31 us` int8, `1.20x`).
   - Standalone B VNNI prepack must use full-sector `64 B` (`col32`) loads (`+34%` faster vs `16 B` `col8`) and amortizes at `R >= 2` when `GRID_M = 4` or `R = 1` when `GRID_M >= 8`.

### 1.2 Production Scorecard vs `oneDNN 3.13.2` (`33–35 WIN | 4–6 TIE | 0 LOSE` across 39 shapes) `[MEASURED]`

| Operator / Shape (`M x N x K`) | Production ESIMD | Achieved Rate | `oneDNN 3.13.2` | vs oneDNN |
|---|---:|---:|---:|---|
| BF16 RowMajor `512x2560x2560` | `103.2–103.3 us` | `65.0 TFLOPS` | `107.7–108.8 us` | **`1.04x–1.05x WIN`** |
| BF16 RowMajor `512x4096x4096` | `206.0–206.7 us` | `83.1–83.4 TFLOPS` | `216.8–217.9 us` | **`1.05x WIN`** |
| BF16 RowMajor `512x4096x4096 + Bias+SiLU` | `211.7–212.3 us` | `80.9–81.2 TFLOPS` | `217.7–219.6 us` | **`1.03x–1.04x WIN`** |
| BF16 RowMajor `1024x4096x4096` | `394.2–394.9 us` | `87.0–87.2 TFLOPS` | `391.6–395.7 us` | **`0.99x–1.00x TIE`** |
| BF16 RowMajor `2048x4096x2048` | `374.7–375.2 us` (`373.1` peak) | `91.6–92.1 TFLOPS` | `365.5–366.8 us` | **`0.97x–0.98x TIE`** |
| BF16 RowMajor `4096x4096x4096` | **`1486.4–1495.3 us`** | **`91.9–92.5 TFLOPS`** | `1639.2–1647.6 us` | **`1.10x WIN`** |
| BF16 RowMajor `128x32768x2560` (LM-Head) | `414.3–415.0 us` | `51.7–51.8 TFLOPS` | `463.2–463.8 us` | **`1.12x WIN`** |
| BF16 `Btile192` `2048x3072x2048` | `290.0–292.0 us` | `88.3–88.9 TFLOPS` | `292.8–292.9 us` | **`1.00x–1.01x TIE`** |
| **Fused `SwiGLU`** `512x4096x2560` (`4/4 WIN`) | `269.6–270.1 us` | `79.5–79.7 TFLOPS` | `292.3–292.5 us` | **`1.08x WIN`** (`1.73x` on `1024x1536x1024`) |
| **`INT8` GEMM** `512x4096x4096` (`6/6 WIN`) | `132.5 us` (`GRF=256 BK=64`) | **`129.6 TOPS`** | `144.8 us` | **`1.09x WIN`** (`1.08x–2.29x` across 6 shapes) |

- **Horizontal `SwiGLU` fusion (`SiLU(A@W_gate) * (A@W_up)`)**: Because A770 exposes a single CCS compute engine (`0%` multi-queue compute concurrency), fuse Gate and Up GEMMs horizontally into one kernel sharing `A` in SLM and evaluating `SiLU(G) * U` in GRF to eliminate intermediate DRAM traffic.
- **`f32` GEMM**: `oneDNN` (`jit:gemm:any`) and `oneMKL` are performance ties within 3%; default to `oneDNN` (use `oneMKL` only if wall latency is the verified deployment metric; note `oneMKL` row-major requires a column-major view `gemm(N, M, K, B, N, A, K, C, N)` `[CORRECTNESS]`).

Copy-ready core: [GRF=256 cab4 + 0-Dummy + bp_vec skeleton](../api/code-snippets.md#1-grf256-cab4-48-kb--split_barrier--0-dummy-2x8--bp_vec-gemm-core).

## 2. Domain I: GEMV (`A*x` and `x*W^T`), RMSNorm, and Softmax `[MEASURED]`

All Domain I operators are DRAM stream-through (`79%–99.5%` pass-through, target `403.1 GB/s` read roof).

### 2.1 GEMV (`f32` GEMV-N1 & `bf16` Decode GEMV)

- **`bf16` Decode GEMV (`W[N,K]` RowMajor, `M in {1, 4}`)**: Sub-group `SG=16, WG=256, vec8` with float accumulators wins **`4/4` vs `oneDNN 3.13.2` (`1.04x–1.22x`)**: `1x4096x4096` `100.5 us` (`333.9 GB/s`, `1.08x`), `1x9216x2560` `127.1 us` (`371.4 GB/s`, `1.22x`), `1x32768x2560` **`423.0 us` (`396.8 GB/s` = `98.4%` roof, `1.10x`)** `[MEASURED]`.
- **`f32` GEMV-N1 (`y = A[M,K]*x[K]`, `4096x4096`)**:
  - Standard SYCL (one row per sub-group, `vec<float,16>`, direct L2 read of `x` without SLM staging, dynamic `per_lane = (K / 16) / sg.get_local_range()[0]`): **`215–216.6 us`** (`2.08x` faster vs `oneDNN Kx1` `450–456 us`); ESIMD 1-item-per-row `256 B` `block_load` reaches `201 us` (`334 GB/s`).
  - Library dispatch (`f32` GEMV-N1): `oneMKL` (`transpose::trans`) reaches `166 us` and wins wall latency across `M`; `oneDNN` (`Kx1`) wins tiny `M <= 64` device time (`>= 192` switches to `oneMKL`). Never use `oneDNN 1xK` for `A*x` -- it computes `x*A` (`A^T*x`) `[CORRECTNESS]`.
- **Correctness traps**: Never hardcode `SG=16` trip counts (compiler may pick `32`, silently computing half the row); never stage a `16 KB` `x` vector in SLM (adds barrier overhead; `229 vs 215 us`); always use `f32` accumulators for `bf16` GEMV at large `K`.

Copy-ready core: [f32 / bf16 GEMV core](../api/code-snippets.md#5-domain-i-gemv-core-sub-group-per-row-direct-l2).

### 2.2 RMSNorm (`f32` / `f16` / `bf16`) `[DISPATCH]` `[MEASURED]`

- **Structure**: For long rows (`hidden >= 4096`), stage one row per WG into SLM (`16 KB` at `4096 f32`, `WG=64–128`, `vec16`), read `x` from DRAM **once**, reduce via sub-group + local atomic (2 barriers total -- never store per-row `inv` in SLM for a 3rd barrier), keep `gamma` in L2 (do not stage `gamma` in SLM). Direct L2 re-read is `~55%` slower (`155 vs 90.5–99 us`).
- **Measured vs `oneDNN`**: `1024x4096 f32` hits **`90.5 us`** (`WG=64`) vs `oneDNN` (`layer_normalization_forward` with `use_scale | rms_norm`, `ocl:reusable:vectorized`) `112.9 us` (**`1.25x` faster vs oneDNN**).
- **Simplified production dispatch (`rows in 1..1024, hidden in 256..16384, hidden % 16 == 0`, 3% tie simplification rule)**:
  - `rows <= 8, hidden <= 1024`: `sg_direct` (`WG 32–128`) -- simpler and tied with SLM.
  - `rows <= 8, hidden >= 2048`: `slm` (`WG 64–256`).
  - `rows >= 16, hidden <= 2048`: `sg_direct` (`WG 32–128`).
  - `rows >= 16, hidden >= 4096`: `slm` (`WG 64–256`).
- **Trap**: When `oneDNN` RMSNorm uses `f16`/`bf16` src, `scale` (`gamma`) **must** remain `f32` (`T*` corrupts output with `max_abs_err ~2.68` `[CORRECTNESS]`).

Copy-ready core: [f32 RMSNorm SLM core](../api/code-snippets.md#62-standalone-rmsnorm-core-slm-row-tile-2-barriers).

### 2.3 Softmax (`f32`) `[MEASURED]`

- **Structure**: One WG per row with dynamic WG sizing (`16` threads for `cols <= 256`, `32` for `cols <= 1024`, `128–256` for `cols > 1024`). Stage up to `8,192` floats (`32 KB` SLM) in one shot (`vec<float,16>` on `64 B`-aligned USM, scalar-chunk SLM fallback for odd `cols`), requiring only `1` explicit barrier after load + `sycl::reduce_over_group(it.get_group(), ...)` for max and sum.
- **Ultra-long rows (`cols > 8192`)**: Compute `row_max` directly from global memory (never stage the max pass in SLM), then tile sum + normalize through SLM (`2` global reads instead of `3`, cutting `1024x16384` `0.66 -> 0.59 ms`).
- **Measured vs `oneDNN 3.13.2` (`4/4 WIN`, `1.58x–3.55x`)**: `1024x1024` `16.4 vs 54.1 us` (`3.29x`), `1024x4096` `89.0 vs 213.9 us` (`2.40x`), `32x32768` `56.7 vs 201.0 us` (`3.55x`), `16x151936` `483.9 vs 763.8 us` (`1.58x`).

Copy-ready core: [f32 Softmax SLM core](../api/code-snippets.md#7-domain-i-softmax-core-slm-row-tile--dynamic-wg).

### 2.4 Fused Row Reductions (`Add [+ Act] + LayerNorm / RMSNorm / Softmax`): Three-Tier Caching Rule `[DISPATCH]` `[MEASURED]`

When a pointwise prologue (`Add`, `BiasAdd`, `GELU`, `GeluErf`, `SiLU`) feeds a trailing row reduction (`LayerNorm`, `RMSNorm`, `Softmax`), re-reading global inputs and recomputing the prologue on Pass 2/3 destroys bandwidth (`~1.55x` slower on `2048x4096 f32`). Select the row-caching tier by hidden width `cols` (standard SYCL / OpenCL C, `SG=16`, `128-GRF`):

| Hidden Width (`cols`) | Caching Tier | Geometry & Budget | Why |
|---|---|---|---|
| **`cols <= 4096`** | **GRF register-cached row** | `WG=64` (`<=512`), `WG=128` (`<=1024`), **`WG=256`** (`<=4096`, `16` HW threads = `8 WGs/Xe-core`), `PER <= 16` floats (`<= 32 GRF` out of `128`, `0 B` SLM row buffer) | Single global read, single prologue eval, zero SLM bank/capacity pressure, zero spills when indexed with `#pragma unroll` compile-time constants (`spill == 0 B && priv == 0 B`) |
| **`4096 < cols <= 8192`** | **SLM row staging** | `WG=128–256`, `<= 32 KB` SLM (`vec16` aligned) | Avoids GRF spills (`> 32` floats/item) while keeping global reads single-pass |
| **`cols > 8192`** | **Streaming multi-pass fallback** | `WG=128–256`, `0 B` row buffer (or 2-pass global max + tiled SLM for Softmax) | Exceeds safe per-WG SLM residency ceiling (`<= 32 KB`) |

- **Exact two-pass variance in registers (for `LayerNorm`)**: Never use naive `E[x^2] - E[x]^2` (catastrophic cancellation at large mean shift `>= 200..1000`) and avoid per-element Welford `delta / count` divisions (`~50` cycles/div on DG2). Once the row sits in `float reg_vals[PER]`, Pass 1 sums `x` -> `row_mean`, Pass 2 scans **registers only** (`(reg_vals[i] - row_mean)^2`, zero DRAM reads, zero prologue recompute, zero loop divisions) -> `inv_std`, and Pass 3 writes normalized output with optional `gamma`/`beta`.
- **Measured `2048x4096` fused row reductions (vs `WG=128` uncached streaming and `oneDNN 3.13.2`, `errors=0`, `spill=0 B, priv=0 B`) `[MEASURED]`**:
  - `Add+LayerNorm (f32)`: `386.5 ->` **`249.5 us`** device (**`403.4 GB/s`** = `100%` read roof, **`1.55x`** faster vs streaming, **`2.24x`** vs `oneDNN` `560.1 us`).
  - `Add+GeluErf+LayerNorm (f32)`: `403.9 ->` **`257.8 us`** device (**`390.5 GB/s`** = `96.9%` roof, **`1.57x`** faster vs streaming, **`2.79x`** vs `oneDNN` `718.1 us`).
  - `Add+LayerNorm (bf16)`: `210.1 ->` **`168.3 us`** device (**`1.25x`** faster vs streaming, **`1.43x`** vs `oneDNN` `240.0 us`).
  - `Add+GeluErf+LayerNorm (bf16)`: `242.9 ->` **`191.6 us`** device (**`1.27x`** faster vs streaming, **`1.68x`** vs `oneDNN` `322.0 us`).

Copy-ready core: [Register-cached fused LayerNorm (WG=256, cols <= 4096)](../api/code-snippets.md#6-domain-i-rmsnorm-and-register-cached-fused-layernorm-cores).
