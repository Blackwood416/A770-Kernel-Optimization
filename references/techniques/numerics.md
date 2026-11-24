# Numerical Precision, Accumulators, Extended Math, and Tolerances on A770

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.1–2026.2, driver `32.0.101.9150`. Supported compute dtypes: `f32`, `bf16`, `f16`, `int8`, `int4` (dequantized or integer DPAS). `fp8` and `tf32` are **absent** on DG2.

## 1. Recommended Verification Tolerances (`err <= rel * |ref| + abs`) `[MEASURED]` `[HEURISTIC]`

Measured against `f64` CPU references (`GEMM 256^3`, `Softmax 256x1024`, `Reduction 256x4096`):

| Dtype / Accumulator | GEMM | Softmax | Reduction | Guidance |
|---|---|---|---|---|
| `int8` / `int32` | `rel=0, abs=0` | `abs=0.5` (quantized) | `rel=0, abs=0` | Exact integer match on small-int data |
| `f32` / `f32` | `rel=1e-4, abs=1e-5` | `rel=1e-6, abs=1e-8` | `rel=1e-3, abs=1e-3` | Standard baseline |
| `f16` / **`f32` acc** | `rel=1e-2, abs=1e-3` | `rel=1e-4, abs=1e-6` | `rel=1e-2, abs=1e-2` | Production default for `f16` |
| `bf16` / **`f32` acc** | `rel=5e-2, abs=1e-2` | `rel=2e-2, abs=1e-4` | `rel=2e-2, abs=0.5` | Production default for `bf16` (large GEMM cross-check: `4e-2 + 3e-2 * max(|a|,|b|)`) |
| `bf16` / `bf16` acc | `rel=0.5, abs=0.5` | N/A | `rel=1, abs=1` | **Avoid** -- `256x4096` reduction error reaches **`8.74`** (`f16/bf16`: `2.29`) |

**Accumulator rules `[CORRECTNESS]`**:
1. **Always accumulate `bf16` and `f16` in `f32`** (both for GEMM DPAS where DG2 requires `dst:f`, and for GEMV/reductions where `bf16` vector accumulators fail `535/4096` rows at `K=4096`).
2. Compare against a **same-precision** reference when auditing precision conversions (comparing a `bf16`-converted kernel against an `f16` float reference produced `max_abs_err ~3.7` even when the `bf16` kernel had `0` errors).
3. **FTZ on tiny inputs**: Subnormal products underflow to `0` under FTZ; use absolute tolerance or pre-scale inputs instead of pure relative tolerance.

## 2. Extended Math (`math.*`), `mad` vs `fma`, and Conversions `[ISA]` `[MEASURED]`

1. **Extended math (`math.exp` / `inv` / `sqt` / `log`)**:
   - Measured dependent latency of `math.exp` is **`~12.7 ns` (`~25 cycles`)**, but interleaved throughput is only **`~1.7 ns` (`~3.3 cycles/EU`)** -- `~4x` cheaper than one `dpas.8x8` (`~12.5 cycles/EU`).
   - **Rule**: Do **not** replace `exp` with multi-term polynomials to save throughput; hide its `~25`-cycle latency with independent lanes (`~12` in flight) and use ESIMD hardware `native_exp2` instead of scalar `sycl::exp2`.
   - `tanh` is not a native DG2 instruction (synthesized from `exp` + division); IGA XeHPG spellings are `exp`, `inv`, `sqt`, `log` (not `exp2`/`rcp`/`rsqrt`).
2. **`fma()` vs `a * b + c` contraction**:
   - DG2 has `mad` (opcode `0x5B`), not a separate `fma` mnemonic. Both `fma()` and `mad()` compile to `mad` (`< 0.3%` timing difference), whereas plain `a * b + c` may or may not contract depending on the surrounding expression. Write explicit `fma()` when guaranteed single rounding is required.
   - `dp4a` exists in the ISA table but is unreachable from standard OpenCL C on this stack; use ESIMD `dpas`/`dpasw` for `int8` dot products.
3. **`bf16` / `f16` conversion is free inline, expensive as a separate pass**:
   - DG2 has no `cvt` opcode; conversions compile to typed `mov` (`:bf` / `:hf`), measuring within `0.23%` of integer truncate (`226–227 GB/s`, completely hidden under memory traffic).
   - Running a separate `f32 -> bf16` pre-conversion pass costs `~52 us` (`~0.5x` of a full GEMM). Keep buffers in `bf16` or fuse conversion into the load/epilogue path.
4. **Denormals (`cr0.0` bit 7) cost zero throughput `[MEASURED]`**:
   - `-cl-denorms-are-zero` clears bit `7` (`0x80`) in the `cr0.0` prologue immediate (`0x4C0 -> 0x40`). Dependent loops run at identical speed (`108.1` vs `108.2 us`) with denorms enabled or flushed. Treat denormal flags as a **numerics switch only**, never a speedup lever.
