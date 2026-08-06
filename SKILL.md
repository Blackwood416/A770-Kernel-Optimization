---
name: optimize-intel-gpu-kernels
description: "Optimize SYCL/ESIMD compute kernels for Intel Arc/Xe-HPG GPUs (A770/DG2). Use when writing or tuning high-performance kernels on Intel discrete GPUs: choosing tiling, register blocking, XMX/joint_matrix/dpas, SLM buffering, walk order, operand layouts, or row-reduction operators such as RMSNorm/layer normalization/softmax; interpreting VTune gpu-hotspots; checking whether an Intel API (prefetch, load_2d, large GRF, DPAS ES16, named_barrier) is supported on A770; or benchmarking against oneMKL/oneDNN."
---

# Intel GPU Kernel Optimization

Optimize SYCL/ESIMD kernels on Intel Arc A770 (Xe-HPG/DG2) with measured hardware facts, a proven technique ladder, known pitfalls, and API usage rules extracted from a complete bf16 GEMM campaign (naive 1.95 ms to 0.0614 ms, about 87% of oneMKL and 90% of oneDNN), a f32 GEMV campaign (naive 2.65 ms to 0.215 ms, faster than the same-operation oneMKL/oneDNN baselines), a f32 RMSNorm campaign (naive 5.382 ms to 0.098 ms, about 1.25x faster than the oneDNN RMSNorm baseline), and a f32 Softmax campaign (naive 5.55 ms to 0.107 ms for 1024x4096, about 2x faster than the oneDNN softmax baseline).

The f16/u4/bf16 GEMM campaign (M=1024, N=1536, K=512, group_size=128) documents a case where oneDNN's native u4 JIT path beats the best stable SYCL path: oneDNN `jit:gemm:any` 0.033 to 0.034 ms vs host-dequantized bf16 DPAS 0.060 ms, because SYCL `dpas` rejects mixed f16/u4 precision.

## Workflow

1. Build the correctness harness first: CPU reference, `errors: 0/...`, warmup + timed loop, then 3+ stable runs.
2. Measure the same benchmark with oneMKL/oneDNN to set the target, then verify the library output against the CPU reference. A fast library call can be a different operator: oneDNN matmul with `1xK` computes `x*A` (row-vector times matrix), not row-major `A*x`.
3. Profile with VTune before restructuring; start with `instruction-count`, escalate to `full-compute`.
4. Change exactly one structural variable per experiment; keep failed variants as files and record the negative result.
5. Verify every variant: exact reference compare, `C[0]` check, 3 stable values, and the VTune metrics that motivated the change.

## 1. A770 Hardware Parameters

| Parameter | Value |
|---|---|
| Hardware threads | 4096 total; 8 per vector engine (4 in large-GRF mode) |
| GRF | 128 (regular) or 256 (large) per thread; 256-bit (32 B) registers |
| SLM | 128 KB per Xe-core; 64 KB max per work-group |
| Cache | L1 192 KB per Xe-core; L2 16 MB |
| XMX bf16 DPAS | 8x8x16 on DG2; B must be VNNI-packed |
| `block_load` | 256 B max on DG2 (512 B is PVC-only) |
| Sub-group sizes | 8, 16, 32 reported by the device; the compiler may pick 32 even when the kernel was written for 16 |
| Measured sweet spot | 32 threads/WG, 16x16 per-thread tile, 24 KB SLM, total threads a multiple of 4096 |

oneDNN's fastest A770 path uses ngen codegen + Large GRF with 16x8 per-thread tiles, 4 SLM buffers, and DPASW chains; that exact geometry and ngen scheduling are not reproducible in SYCL/ESIMD. Full details: [hardware.md](references/hardware.md)

## 2. Optimization Ladder

Apply in this order; each step was measured on the same bf16 GEMM (1024x1536x512):

1. Tile into SLM and add register tiling so A/B slices are reused (outer-product accumulation).
2. Vectorize global<->SLM copies; move computation to XMX (`joint_matrix` or ESIMD `dpas`).
3. VNNI-pack B on the host; vectorize A copies; match work-group geometry to 4096 threads with GRF headroom.
4. Double-buffer SLM and pipeline the K loop, but only after per-block compute is large enough.
5. Prefer N-first walk order for L2 reuse; split first/last K blocks to remove branches.
6. Use VTune instruction-count to find Send/ALU bloat; widen SLM->GRF messages.
7. Prepack A/B on the host into DPAS operand layout so kernel loads are 256 B with zero `select`.
8. Hoist base pointers out of the K loop (fixed strides); add more SLM buffers to cut barrier frequency.
9. Route repeatedly-read global tiles through a cooperative SLM copy when direct per-thread reads are redundant.

Measured ladder: 1.95 ms naive -> 0.118 ms joint_matrix best -> 0.099 ms ESIMD -> 0.073 ms operand layout -> 0.0614 ms final (oneMKL 0.0529, oneDNN 0.0552). Every ladder step maps to an embedded code snippet in [Ladder to Snippet Map](references/code-snippets.md#ladder-to-snippet-map); details and reasons: [techniques.md](references/techniques.md).

For f16 activation + u4 weights + bf16 dst (`1024x1536x512`, `group_size=128`): naive 1.587 ms, oneDNN u4 `ba` + per-group f16 scales + zero-point 8 + `fpmath_mode::any` is `jit:gemm:any` at 0.0333 to 0.0344 ms, and the best stable SYCL path is host dequant u4 -> bf16 followed by the bf16 ESIMD DPAS kernel at 0.060 to 0.061 ms. The remaining ~1.8x gap is the native u4/f16 path that oneDNN's ngen codegen uses; SYCL `dpas` statically rejects the mixed precision and the low-level `__esimd_dpas2<u4, fp16, ...>` smoke did not produce a working A/B operand shape on oneAPI 2026.1.

For f32 GEMV (`y = A*x`, `4096x4096`, row-major A), the measured standard-SYCL ladder is: naive row-per-item 2.65 ms -> vec16 row + SLM x 0.50 ms -> one row per sub-group + SLM x 0.24 ms -> one row per sub-group + x direct from L2 + dynamic per-lane trip count 0.215 ms. Same-operation library baselines: oneMKL GPU gemv 0.329 ms, oneDNN GPU matmul `Kx1` 0.380 ms. Details and VTune instruction mix: [techniques.md](references/techniques.md#f32-gemv-ladder-4096x4096-row-major-a); copy-ready kernel: [code-snippets.md](references/code-snippets.md#f32-gemv-core-sub-group-per-row-direct-l2).

For f32 RMSNorm (`1024x4096`, x/gamma/y f32), the measured standard-SYCL ladder is: naive row-per-item 5.382 ms -> sub-group-per-row direct L2 0.153 to 0.157 ms -> SLM row tile with one row per 128-thread work-group and two barriers 0.098 to 0.100 ms. Same-operation baseline: oneDNN `layer_normalization_forward` with `rms_norm` flag, 0.1235 to 0.1242 ms. SLM staging wins here even though it lost for GEMV's x, because RMSNorm re-reads each row after the reduction. Details: [techniques.md](references/techniques.md#f32-rmsnorm-ladder-1024x4096-row-major-x-f32); copy-ready kernel: [code-snippets.md](references/code-snippets.md#f32-rmsnorm-core-slm-row-tile).

For f32 Softmax (`1024x4096`, row-major x/y f32), the measured standard-SYCL ladder is: naive row-per-item 5.55 ms -> one-shot SLM row tile (128 threads/WG) 0.1125 ms -> 32 KB one-shot limit (0.1125 ms at 4096 cols, 0.211 ms at 8192 cols) -> `vec16` exp accumulator and vector stores 0.107 ms -> dynamic WG size by column count (16/32/128 threads) 0.109 ms final, with oneDNN softmax at 0.217 ms. Rows larger than the SLM tile (16384 cols) use a tiled variant with max read directly from global, reaching 0.59 ms vs oneDNN 2.64 ms. Details: [techniques.md](references/techniques.md#f32-softmax-ladder); copy-ready kernel: [code-snippets.md](references/code-snippets.md#f32-softmax-core-slm-row-tile).

## 3. A770 Pitfalls

- `prefetch` (SYCL or ESIMD): negative on A770 for GEMM-sized data that fits the 16 MB L2.
- Large GRF (`-ze-opt-large-register-file`): negative for joint_matrix; ESIMD run failure. Only oneDNN's ngen 16x8 geometry benefits.
- DPAS ExecutionSize=16: compiles but produces wrong results on A770.
- `load_2d` and `named_barrier`: PVC-only; A770 hangs or rejects them.
- Deeper SLM pipelines, GRF double copies, 8 SLM buffers, 16x8 per-thread geometry, and SLM bank padding: all measured negative on A770.
- Runtime `if/else` inside an ESIMD kernel hangs; dispatch with template `if constexpr` instead.
- With `beta != 0`, restore C0 after the timing loop before the single-shot correctness check.
- Do not hardcode sub-group size 16: A770 also supports 8/32 and the compiler's default can be 32. A GEMV written for 16 lanes then computes only half of every row with `max_err` roughly half the reference magnitude. Query `sgrp.get_local_range()[0]` and derive per-lane trip counts, or force `sub_group_size<16>`; forcing can be slower than adapting to 32.
- Do not assume sub-groups map along local dimension 0. With a 2D local range `(16, 32)` on oneAPI 2026.1, A770 formed 32-lane sub-groups along dimension 1: `sg.get_group_linear_id()` tracked `lid0` and `sg.get_local_linear_id()` tracked `lid1`. Probe the mapping with a tiny kernel, or use a 1D `nd_range` where sub-groups are contiguous linear blocks.
- Do not allocate a full 64 KB SLM tile: a 4 x 4096 float `local_accessor` (65536 bytes) failed at launch on A770 with no diagnostic. Keep tiles at 32-48 KB and leave headroom for other local data.
- Initialize every local reduction accumulator before `atomic_ref::fetch_add`; an uninitialized `local_accessor` row-sum array produced `max_err ~1.48` with all elements failing.
- On oneAPI 2026.1, `sycl::group_barrier(it, sycl::access::fence_space::local_space)` no longer compiles; use `it.barrier(sycl::access::fence_space::local_space)` or pass a `memory_scope`.
- oneDNN has no separate RMSNorm primitive kind in 3.11.2: use `layer_normalization_forward` with `normalization_flags::use_scale | normalization_flags::rms_norm` and verify the output against the CPU reference.
- `esimd::reduce(v, std::plus<float>{})` silently returns 0 on A770 because the implementation only matches `std::plus<>`. Use `esimd::reduce(v, std::plus<>{})`.
- ESIMD experiments on driver 32.0.101.8724 caused a full system reboot with bugcheck `0x00000116` (VIDEO_TDR_FAILURE). Before running new ESIMD kernel shapes, isolate them in a small one-shot process with a watchdog and collect driver/minidump evidence; do not assume a hang is recoverable.
- oneDNN GPU matmul `1xK` (`src {1,K}` x `weights {K,N}`) is not a row-major `A*x` GEMV; it is `x*A`, i.e. `A^T*x`. It can look much faster but is wrong for the reference layout. Use `Kx1` for the same operator and always verify the library result.
- `sycl::reduce_over_group` takes a group object, not an `nd_item`: pass `it.get_group()` (or a `sub_group`) or oneAPI 2026.1 rejects the call at compile time.
- `vec<float,16>` loads/stores are 64 B operations; `std::vector` and buffer-backed host memory are not guaranteed 64 B aligned. Use `sycl::aligned_alloc_shared/device<float>(64, ...)` on the fast path and keep a scalar generic path for arbitrary column counts.
- Shrink the work-group size for small rows/columns. For softmax, 16 threads for cols <= 256, 32 for cols <= 1024, and 128 otherwise removed idle-thread waste; 1024x256 went from 1.34x the oneDNN time to faster than it.
- For rows larger than one SLM tile, compute the max pass directly from global and stage SLM tiles only for the sum and normalize passes. That cut 1024x16384 softmax from 0.66 ms to 0.59 ms by reducing global reads from three to two.
- oneDNN softmax baseline: use `softmax_forward` + `algorithm::softmax_accurate` with `axis=1` and verify against the CPU reference. Its 1024x16384 f32 path measured 2.64 ms, much slower than smaller shapes, so re-measure the baseline per shape before trusting it.
- oneDNN u4 matmul with a plain `ab` weights descriptor and no scales falls back to `ocl:ref:any` (~110 ms for 1024x1536x512). Use `ba` weights (`[N, K/2]` packed u4), per-group f16 scales, zero-point 8, and `fpmath_mode::any` to get the `jit:gemm:any` path at 0.033 to 0.034 ms.
- SYCL `dpas` does not accept mixed f16/bf16 + u4 operands: the fp16/bf16 branch asserts `APrecision == BPrecision`. Host-dequantize u4 to bf16 outside the timed loop and use the known-good bf16 DPAS path; that is stable but ~1.8x slower than oneDNN's native u4 JIT.
- Low-level `__esimd_dpas2<dpas_argument_type::u4, dpas_argument_type::fp16, ...>` compiles the intrinsic call only with unexpected A/B vector lengths on oneAPI 2026.1; the smoke did not run correctly. Treat mixed-precision u4 DPAS as not accessible from SYCL/ESIMD until a working layout is proven.
- When host-packing bf16 DPAS slices, all pointer strides are element counts: one slice is 128 bf16 (256 B), so the next slice is `+128`, not `+256`, and a 32-deep K block advances by 4096 A / 2048 B bf16 elements. Byte-sized strides caused out-of-bounds writes and silent kernel crashes.

Complete evidence table: [pitfalls.md](references/pitfalls.md)

## 4. API Usage

- SYCL `joint_matrix`: declare A as `layout::row_major`, B as `layout::ext_intel_packed`; `joint_matrix_load` stride is in elements (packed B rows are `BN*2` bf16).
- ESIMD: `esimd::dpas<8, RepeatCount, float>(C, B, A)` with VNNI B; `slm_block_load/store`, `barrier()`, `block_load` with `overaligned_tag<16>{}`.
- ESIMD reductions: pass `std::plus<>{}` (or `std::multiplies<>{}`) to `esimd::reduce`; typed functors such as `std::plus<float>{}` compile but hit an empty branch and return the default value.
- Standard SYCL GEMV: use a 2D `nd_range` with one row per sub-group, `vec<float,16>` loads, and a per-lane trip count derived from the actual `sub_group` size. Reading `x` directly from L2 beat staging it in SLM.
- Row-reduction operators (RMSNorm, layer norm, softmax): stage the full row in SLM and normalize from SLM. For 1024x4096 RMSNorm, direct L2 re-read after reduction was 0.153 to 0.157 ms while a 16 KB SLM row tile reached 0.098 to 0.100 ms. For softmax, shrink WG size for short rows and compute the max pass directly from global when the row exceeds the SLM tile.
- `bit_cast_view<bf16>()` requires an lvalue `simd`; calling it on a temporary fails to compile.
- Windows build: `cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && icx-cl /fsycl file.cpp /Fe:file.exe'`.
- VTune: `instruction-count` needs no admin; `full-compute` needs `-allow-multiple-runs`.

Full reference: [api-usage.md](references/api-usage.md). Tested, copy-ready building blocks (VNNI packing, joint_matrix core, dpas smoke, SLM double buffer, 16x16 GEMM core, barrier pipeline, alpha/beta template dispatch, verification harness): [code-snippets.md](references/code-snippets.md).

## References

- [hardware.md](references/hardware.md): specs, DPAS/SLM/cache constraints, occupancy math, oneDNN decode.
- [techniques.md](references/techniques.md): full measured ladder, VTune-driven instruction reduction, metric interpretation.
- [pitfalls.md](references/pitfalls.md): every measured negative result and behavioral trap with numbers.
- [api-usage.md](references/api-usage.md): joint_matrix/ESIMD API forms, build/profile commands, verification workflow.
- [code-snippets.md](references/code-snippets.md): actual kernel and host-side snippets extracted from the measured variants, plus a ladder-to-snippet map that keeps the skill self-contained.
