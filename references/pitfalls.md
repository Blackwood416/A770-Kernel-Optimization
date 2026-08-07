# A770 Pitfalls and Negative Results

## Table of Contents

- API / hardware gates
- Structural experiments that failed
- Behavioral traps
- Diagnostic traps

Cross-reference: the code that hits these gates lives in [code-snippets.md](code-snippets.md); API details and commands are in [api-usage.md](api-usage.md).

## API / Hardware Gates

| Item | Status on A770 | Evidence |
|---|---|---|
| `joint_matrix_prefetch` | Unsupported on DG2 | intel/llvm e2e test is marked `UNSUPPORTED: gpu-intel-dg2` |
| DPAS `ExecutionSize=16` | Compiles, wrong results | Smoke test: 78/128 errors |
| `load_2d` / `prefetch_2d` | PVC-only | A770 hangs; bf16 `Transposed` rejected at compile time (u32/u64 only) |
| `named_barrier` | PVC-only | `memory.hpp`: available only on PVC |
| `dpas<SystolicDepth=16>` | Compile-time rejected | `static_assert(SystolicDepth == 8)` |
| `block_load` 512 B | PVC-only | DG2 cap is 256 B |
| Large GRF `-ze-opt-large-register-file` | Negative or fails | joint_matrix: 211.2 / 180.3 / 193.9 ms vs 143.7 ms baseline; ESIMD run exits with code 1 |
| A `block_load` L1/L2 cache hint | Neutral | 0.0641 to 0.0659 ms vs 0.0632 to 0.0646 ms |
| `esimd::reduce(v, std::plus<float>{})` | Compiles, silently returns 0 | `reduce` only matches `std::plus<>`/`std::multiplies<>`; the typed functor hits an empty branch |
| Default sub-group size | Not fixed | A770 reports 8/16/32; a GEMV written for 16 lanes produced half sums when the compiler picked 32 |
| 2D sub-group mapping | Not along local dim 0 | oneAPI 2026.1 probe with local `(16,32)`: 32-lane sub-groups ran along dim 1 (`sg_gid` tracked `lid0`, `sg_lid` tracked `lid1`); use a 1D `nd_range` or probe the mapping |
| 64 KB SLM tile | Launch failure | 4 x 4096 float `local_accessor` (65536 B) exited with no diagnostic on A770; keep tiles at 32-48 KB |
| `group_barrier(it, fence_space)` | Compile error | oneAPI 2026.1 expects a `memory_scope`; use `it.barrier(access::fence_space::local_space)` |
| ESIMD SLM/`block_store` experiments (driver 32.0.101.8724) | Full system reboot | Bugcheck `0x00000116` VIDEO_TDR_FAILURE at 20:58:56; minidump `080526-7171-01.dmp` |
| oneDNN matmul `1xK` as a GEMV baseline | Wrong operator | Computes `x*A` (`A^T*x`); max_err 326 vs the CPU reference, so the 0.182 ms number is not comparable |
| `sycl::reduce_over_group(nd_item, ...)` | Compile error | oneAPI 2026.1 requires a group object; pass `it.get_group()` or a `sub_group`, not the `nd_item` |
| `vec<float,16>` on `std::vector`/buffer host memory | Alignment risk | 64 B loads need aligned USM; use `sycl::aligned_alloc_shared/device<float>(64, ...)` on the fast path and a scalar generic path for odd columns |
| oneDNN u4 matmul, `ab` weights, no scales | `ocl:ref:any` ~110 ms | Plain `ab` u4 did not select a GPU JIT path for f16/u4/bf16 on oneAPI 2026.1 |
| oneDNN u4 matmul, `ba` weights + scales | `jit:gemm:any` 0.033 to 0.034 ms | `ba` (`[N, K/2]`) + per-group f16 scales + zero-point 8 + `fpmath_mode::any` selects the fast path |
| oneDNN u4 GEMV, `Kx1` + `ba` weights + scales | `jit:gemm:any` 0.1456 ms | `{K,1}` u4 `ba` (`[1, K/2]`) + `{groups,1}` f16 scales + zp8 works; plain `ab` fails to create the primitive descriptor |
| SYCL `dpas` mixed f16/u4 | Compile-time rejected | The fp16/bf16 branch asserts `APrecision == BPrecision`; no mixed precision from the public API |
| Low-level `__esimd_dpas2<u4, fp16, ...>` | No working smoke | oneAPI 2026.1 expected unexpected A/B vector lengths (N1/N2 mismatch); treat as unverified |
| Repeated u8 ESIMD batch | Driver crash at `q.wait()` | One-shot u8 kernel was correct, but 100+ batched submissions crashed; use the stable joint_matrix u8 path for production |

## Structural Experiments That Failed

All variants were correct (`errors: 0/...`) and slower than the stated baseline.

| Experiment | Measured | Baseline | Why it failed |
|---|---:|---:|---|
| joint_matrix software prefetch | 0.6606 ms | 0.6397 ms SLM-only | Data fits 16 MB L2; prefetch only adds Send instructions |
| ESIMD A next-block prefetch | 0.0707 to 0.0715 ms | 0.0628 to 0.0646 ms | +8 prefetch Sends per pair outweigh hidden A latency |
| Double buffer at 32x32 tiles | 0.6695 ms | 0.6397 ms | Compute per block too short; barriers still wait on load |
| 3-level SLM (36 KB) | 0.1182 to 0.1197 ms | 0.0992 ms | SLM growth reduces resident work-groups |
| 4-level SLM + 2 GRF copies (ESIMD) | 0.282 ms | 0.0992 ms | Register pressure and SLM occupancy |
| GRF double copy (joint_matrix, unrolled) | 0.1248 to 0.1253 ms | 0.11831 ms | Explicit duplicate joint_matrix registers degrade IGC allocation |
| GRF double copy (joint_matrix, runtime branch) | 0.4176 ms | 0.11831 ms | Runtime branch plus register pressure; catastrophic |
| BK=32 + 4-level SLM (joint_matrix) | 0.1498 to 0.1512 ms | 0.11831 ms | Larger K block and SLM usage hurt in SYCL |
| BK=64 (ESIMD, 48 KB SLM) | 0.2639 to 0.2658 ms | 0.0992 ms | 48 KB SLM occupancy loss |
| 8-buffer B (32 KB SLM) | 0.0690 to 0.0703 ms | 0.0628 to 0.0646 ms | Occupancy loss beats barrier savings; 16 KB is the sweet spot |
| 16x8 per-thread + 64 threads (A direct) | 0.0832 to 0.0835 ms | 0.0628 to 0.0646 ms | A read amplification 4x -> 8x |
| 16x8 per-thread + 64 threads (A SLM relay) | 0.0780 to 0.0791 ms | 0.0613 to 0.0615 ms | A SLM read amplification 8x plus residency constraint |
| SLM bank padding (1056 B / 288 B slots) | 0.0698 to 0.0704 ms | 0.0613 to 0.0615 ms | 256 B LSC blocks already span banks; padding breaks alignment |
| A prepack for joint_matrix | 0.154 to 0.169 ms | 0.1437 ms | Host-side index math overhead beat layout benefit |
| Simple Boustrophedon walk | 0.1185 ms | 0.11831 ms | oneDNN's version needs fused linear ID + runtime bslice/bthresh |
| M-first walk | 0.106 ms | 0.0992 ms (N-first) | N-first has better L2 reuse |
| 16x32 tile (ESIMD) | 0.3375 ms | 0.0992 ms | Register pressure |
| 16x24 tile (ESIMD) | 0.1004 to 0.1013 ms | 0.0992 ms | Slightly worse; single-wave assumption failed |
| `dpasw` to save instructions | No gain | - | For 16x16 tile it emits the same number of instructions as `dpas` |
| A SLM relay + 4-buffer B (32 KB) | Run failure | - | Pair loop conflicts 2 A buffers with 2-block lookahead; abandoned |
| GEMV x SLM relay | 0.229 ms | 0.215 ms direct L2 | x is only 16 KB and L1/L2 already reuse it; SLM adds barrier and instruction overhead |
| GEMV forced `sub_group_size<16>` | 0.228 to 0.258 ms | 0.215 ms dynamic per-lane | The compiler's 32-lane default was faster for the row-per-sub-group shape |
| GEMV pointer hoisting | 0.258 to 0.323 ms | 0.215 ms index-math kernel | Compiler already converts `row*N+col` into increments; explicit hoisting added register pressure |
| GEMV `vec16` group reduce | 0.227 ms | 0.215 ms scalar `reduce_over_group` | No reduction win for one scalar per row |
| GEMV ESIMD full-row-per-lane + SLM partial exchange | TDR / reboot | - | Not isolated to a single API; abandoned after bugcheck 0x116 |
| RMSNorm direct L2 re-read | 0.153 to 0.157 ms | 0.0989 to 0.1000 ms SLM row tile | Each row is read twice; nominally-fitting 16 MB L2 did not save the second read |
| RMSNorm 2 rows/WG, wg512 | 0.108 to 0.110 ms | 0.0989 to 0.1000 ms | More SLM and barriers per work-group |
| RMSNorm 2 rows/WG, wg256 | 0.105 to 0.106 ms | 0.0989 to 0.1000 ms | Still slower than one row per wg128 |
| RMSNorm 1 row/WG, wg64 | 0.108 ms | 0.0989 to 0.1000 ms wg128 | Too little work per work-group |
| RMSNorm 1 row/WG, wg256 (2 barriers) | 0.1005 to 0.1029 ms | 0.0989 to 0.1000 ms wg128 | Larger work-group reduced residency |
| RMSNorm `vec16` square accumulator | 0.100 to 0.101 ms | 0.0989 to 0.1000 ms scalar loop | Neutral; compiler already vectorized the scalar loop |
| RMSNorm 4 rows/WG, 64 KB SLM | Launch failure | 0.0989 to 0.1000 ms | 64 KB local tile exceeded the usable SLM budget |
| Softmax one-shot SLM limit 4096 cols | 16x4097 0.0372 ms | 0.0227 ms after raising the limit to 8192 | 32 KB one-shot tiles launch fine and beat the three-pass tiled path for 4097/8192 |
| Softmax tiled max pass staged through SLM | 1024x16384 0.6609 ms | 0.5923 to 0.5985 ms with max read directly from global | The max pass never re-reads the row, so SLM staging only adds loads and barriers |
| Softmax fixed wg128 for short rows | 1024x256 0.0227 ms | 0.0112 to 0.0123 ms with dynamic 16/32/128 threads | Mostly-idle work-groups and group-reduce overhead dominate small shapes |
| f16/u4/bf16 host-dequantized bf16 DPAS | 0.060 to 0.061 ms | oneDNN u4 `jit:gemm:any` 0.033 to 0.034 ms | Native u4/f16 DPAS is not exposed to SYCL; dequantizing to bf16 adds the same bf16 GEMM cost as the bf16 campaign |
| u4->bf16 GEMV bf16 vector accumulator | 535/4096 rows wrong at 4096x4096 | 0/4096 with float accumulator | A bf16 `vec` accumulator passes 64x128 but loses too much precision at large K; keep the accumulator in float and round once |
| u4->bf16 GEMV manual 2/4-way unroll | 0.124 ms both variants | 0.088 ms simple loop | Explicit multi-accumulator unrolling added register pressure and address math; the compiler already schedules the 8-iteration per-lane loop |

## Behavioral Traps

- Runtime `if/else` inside an ESIMD kernel hangs on A770. Dispatch behavior at compile time with template `if constexpr` and instantiate both kernel paths from the host.
- When benchmarking `C = alpha*A*B + beta*C`, a timed loop accumulates beta effects across iterations. Restore C0 after timing and run a single-shot kernel for the correctness check.
- VNNI-packed B reference math must use the original B array, not the packed array. The packed uint32 layout is column-pair interleaved: word index `(k/2)*N + n`, low 16 bits `B[2k][n]`, high 16 bits `B[2k+1][n]`.
- `bit_cast_view<bf16>()` requires an lvalue `simd`; calling it on a temporary fails to compile. Store the loaded `simd` in a named variable first.
- Higher occupancy is not automatically faster. oneDNN at 28.7% occupancy beat an ESIMD variant at 58.4% because instruction volume was lower.
- Large GRF advice from official guides targets PVC (8x16x16, 32x64x16 blocks) and does not transfer to DG2.
- `esimd::reduce` must receive `std::plus<>{}` or `std::multiplies<>{}`; a typed functor such as `std::plus<float>{}` compiles and returns the default value, which looked like broken ESIMD stores until the reduction was fixed.
- Sub-group size is a correctness parameter, not just a tuning parameter. Query `sgrp.get_local_range()[0]` and derive per-lane loop bounds, or verify that the assumed size matches the compiled kernel.
- Initialize every local reduction accumulator before `atomic_ref::fetch_add`. An uninitialized RMSNorm `local_accessor` row-sum array produced `max_err ~1.48` and failed every element; the kernel otherwise looked correct.
- oneDNN RMSNorm is a normalization flag, not a separate primitive: use `layer_normalization_forward` with `use_scale | rms_norm` and `forward_inference`, then verify against the CPU reference.
- Library baselines must pass the same CPU reference. oneDNN `1xK` and a naive oneMKL gemv call both looked plausible but computed the transposed operator for row-major A.
- When timing a GEMV built from SYCL buffers, the first submission can include host-to-device copies. Allocate device/aligned USM once, warm up, then time the loop.
- For row-reduction kernels, treat `sycl::reduce_over_group` as a group API: pass `it.get_group()` for a whole work-group or a `sub_group`, never the `nd_item`. This is a compile-time gate, not a tuning choice.
- `vec<float,16>` fast paths are alignment-sensitive. If the input comes from `std::vector` or a SYCL buffer over host memory, do not assume 64 B alignment; either copy into aligned USM or use the scalar SLM path.
- oneDNN softmax baselines are shape-dependent: 1024x16384 measured 2.64 ms while 1024x4096 measured 0.217 ms on the same machine. Verify the library result against the CPU reference and re-measure per shape before drawing a conclusion.
- f16 -> bf16 activation conversion changes the result. Verify the optimized kernel against a same-precision bf16 reference; against the original f16 float reference the `max_abs_err` was about 3.7 for 1024x1536x512 even though the bf16-path reference reported `errors: 0`.
- For u4 -> bf16 GEMV, dequantize the u4 vector to bf16 on the host once before the timed loop. Re-unpacking u4 and scales inside the kernel keeps it instruction-bound (2.22 ms naive vs 0.0883 ms after host dequant).
- oneDNN u4 `Kx1` with `fpmath_mode::any` can differ from the f32/bf16 reference at K=4096 (39/4096 rows at 5% relative tolerance). Use it for timing, not as the correctness oracle; verify the SYCL kernel against the bf16-dequantized reference.
- bf16 host packing strides are element counts, not byte counts. A DPAS slice is 128 bf16 (256 B), so slice offsets are 0/128/256/384, and a K=32 block advances by 4096 A / 2048 B bf16 elements. Using byte-sized strides caused out-of-bounds writes and silent kernel crashes.
- On the joint-matrix pack+GEMM kernel, `sub_group_size<16>` and `reqd_sub_group_size(16)` trigger an IGC internal divide-by-zero on oneAPI 2026.1; use the measured 2D `(8, 64)` local mapping instead.
- The second 8-column B slice of a joint-matrix GEMM is 8 `uint32` words past the first slice, i.e. `+16 bf16` after casting to `bf16*`; `+8 bf16` silently produces half-wrong output columns.
- Fusing a pack kernel into GEMM is not always a win. Measured fused added 93 us of device time and removed only 46 us of launch gap plus host overhead, so it lost 47 us per iteration. Fuse only when `fused device delta < launch gap + host overhead saved`.

## Diagnostic Traps

- VTune `full-compute` fails with "Allow multiple runs" unless you pass `-allow-multiple-runs`; `instruction-count` does not need it.
- Without admin rights, VTune lacks EU-level detail (occupancy, barrier stalls); still useful for instruction-count and launch overhead.
- Wall clock and kernel time differ: our best kernel had about 1.7 us launch gap, already better than oneDNN's 4.3 us. Optimize launch only after measuring it.
- VTune `instruction-count` sampling inflated RMSNorm wall time by ~50-60x (0.154 ms to ~10 ms). Compare instruction mixes between variants, not profiled milliseconds.
- Device-side text LLVM IR is not supported on oneAPI 2026.1; keep LLVM bitcode, SPIR-V text via `llvm-spirv --to-text`, and GEN assembly via `ocloc compile`.
- On simple non-XMX bf16 loops, O3, fp-contract, fast-math, and large GRF do not change JIT instruction counts and are neutral or slightly negative; explicit 2-way unroll is the stable win.
- Large GRF has no universal verdict: it is negative for joint_matrix/XMX kernels but positive for AOT + unroll4 simple GEMM on the measured A770. Qualify every large-GRF claim with the kernel shape.
