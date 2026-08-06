# Measured Optimization Ladder

## Table of Contents

- Benchmark context and baselines
- The full ladder with measured numbers
- Why each step works
- VTune-driven instruction reduction
- f32 GEMV ladder and baselines
- f32 RMSNorm ladder (1024x4096)
- f32 Softmax ladder
- f16/u4/bf16 GEMM (1024x1536x512, group_size=128)
- Interpreting VTune gpu-hotspots metrics

## Benchmark Context and Baselines

All numbers below come from bf16 GEMM, M=1024, N=1536, K=512, on Intel Arc A770, measured with the same harness: 100 warmup + 1000 timed kernel launches, `q.wait()`, average per run, exact CPU reference compare.

- Naive per-element global-memory kernel: 1.95174 ms (2.75% of oneMKL).
- oneMKL baseline: 0.0529 to 0.0537 ms.
- oneDNN baseline (same machine, same harness): 0.0552 to 0.0559 ms.

Every ladder row links to the embedded code that implements it in [code-snippets.md](code-snippets.md#ladder-to-snippet-map). Original campaign file names are kept there only as provenance; the snippets themselves are self-contained.

## The Full Ladder

| Step | Technique | Per-run | Relative to oneMKL | Snippet |
|---|---|---:|---:|---|
| 1 | Naive | 1.95174 ms | 2.75% | [naive-baseline](code-snippets.md#naive-baseline) |
| 2 | SLM tiling | 1.49231 ms | 3.59% | [slm-tiling](code-snippets.md#slm-tiling) |
| 3 | Register tiling | 0.430648 ms | 12.46% | [register-tiling](code-snippets.md#register-tiling) |
| 4 | SIMD vectorized loads/stores | 0.273737 ms | 19.61% | [simd-vectorized-copies](code-snippets.md#simd-vectorized-copies) |
| 5 | joint_matrix direct global | 0.430931 ms | 12.46% | [joint_matrix-kernel-core](code-snippets.md#joint_matrix-kernel-core) |
| 6 | joint_matrix + basic SLM | 0.629883 ms | 8.52% | [joint_matrix-kernel-core](code-snippets.md#joint_matrix-kernel-core) |
| 7 | + software prefetch | 0.660607 ms | 8.13% | negative, see [pitfalls.md](pitfalls.md) |
| 8 | + double buffer (small tile) | 0.669505 ms | 8.02% | [esimd-slm-double-buffer-skeleton](code-snippets.md#esimd-slm-double-buffer-skeleton) |
| 9 | + large tile 64x64 + 16x16 GRF block | 0.356372 ms | 15.07% | [joint_matrix-kernel-core](code-snippets.md#joint_matrix-kernel-core) |
| 10 | + hardware-tuned geometry | 0.307633 ms | 17.46% | [joint_matrix-kernel-core](code-snippets.md#joint_matrix-kernel-core) |
| 11 | + VNNI-packed B | 0.23417 ms | 22.93% | [host-side-vnni-packing-for-bf16-b](code-snippets.md#host-side-vnni-packing-for-bf16-b) |
| 12 | + vectorized A load | 0.143731 ms | 37.36% | [simd-vectorized-copies](code-snippets.md#simd-vectorized-copies) + [joint_matrix-kernel-core](code-snippets.md#joint_matrix-kernel-core) |
| 13 | + N-first walk + K first/last split | 0.11831 ms | 45.39% | [joint_matrix-kernel-core](code-snippets.md#joint_matrix-kernel-core) |
| 14 | ESIMD dpas, BK32, 16x16 tile | 0.0992 ms | about 54% | [esimd-dpas-smoke-test](code-snippets.md#esimd-dpas-smoke-test) + [esimd-slm-double-buffer-skeleton](code-snippets.md#esimd-slm-double-buffer-skeleton) |
| 15 | + wide SLM->GRF loads | 0.0836 ms | about 63% | [esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads](code-snippets.md#esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads) |
| 16 | + host-side operand layout, A direct read | 0.0734 ms | about 72.5% | [host-side-a-operand-layout-packing-esimd-16x16](code-snippets.md#host-side-a-operand-layout-packing-esimd-16x16) + [esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads](code-snippets.md#esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads) |
| 17 | + address slimming + 4-buffer B + barrier/2K | 0.0628 to 0.0646 ms | about 82% | [fewer-barriers-4-buffer-b-pipeline](code-snippets.md#fewer-barriers-4-buffer-b-pipeline) |
| 18 | + A SLM relay (cooperative copy) | 0.0613 to 0.0615 ms | about 87% (90% of oneDNN) | [esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads](code-snippets.md#esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads) |

Steps 5-8 were measured negative on that geometry and are recorded so nobody repeats them. Steps 9-13 are the joint_matrix path; steps 14-18 are the ESIMD path that replaced it.

## Why Each Step Works

### Tiling and register reuse

- SLM tiling replaces per-element global traffic with cooperative block copies.
- Register tiling keeps A/B slices in GRF and accumulates outer products, cutting SLM/global reads per FLOP.
- Per-thread C tile size controls the compute/load ratio. 16x16 was the ESIMD sweet spot; 16x32 (register pressure) and 16x8 with 64 threads (read amplification) both measured negative.

Code: [naive-baseline](code-snippets.md#naive-baseline), [slm-tiling](code-snippets.md#slm-tiling), [register-tiling](code-snippets.md#register-tiling).

### Vectorization and XMX

- Vectorized copies (`vec<bf16,4>`, uint32 VNNI words) cut element-wise load instructions by 2-4x.
- XMX via joint_matrix or ESIMD `dpas` is required to approach library GEMM. VNNI packing B on the host lets the loader copy uint32 words and feeds DPAS the exact operand layout.

Code: [simd-vectorized-copies](code-snippets.md#simd-vectorized-copies), [host-side-vnni-packing-for-bf16-b](code-snippets.md#host-side-vnni-packing-for-bf16-b), [joint_matrix-kernel-core](code-snippets.md#joint_matrix-kernel-core), [esimd-dpas-smoke-test](code-snippets.md#esimd-dpas-smoke-test).

### Geometry and scheduling

- Work-group count x size aligned to 4096 threads fills full waves; 24 full waves was best.
- GRF headroom matters: 64/128 used registers beat 96-112/128 by avoiding spills and scheduler pressure.
- N-first walk order swapped the M/N group indices so adjacent work-groups reuse A/B in L2, measured +17%.
- K first/last split removed a per-block branch; neutral alone but kept.

Code: [joint_matrix-kernel-core](code-snippets.md#joint_matrix-kernel-core) (N-first mapping and K split are in the loop).

### Software pipelining

- Double-buffer SLM: load K block n+1 while computing block n. Only wins when compute per block is large enough; with 32x32 tiles it was slower than the single-buffer baseline.
- 4-buffer B with one barrier per 2 K blocks halved barrier count (109M to 57M) and was -14%. 8 buffers at 32 KB lost more occupancy than barriers saved and was negative.

Code: [esimd-slm-double-buffer-skeleton](code-snippets.md#esimd-slm-double-buffer-skeleton), [fewer-barriers-4-buffer-b-pipeline](code-snippets.md#fewer-barriers-4-buffer-b-pipeline).

### VTune-driven instruction reduction

The instruction-count profile of the ESIMD 16x16 kernel showed Send 29.0%, Int32/SP Float 45.6% (mostly address math), Other 19.1%. Each subsequent step cut a measured instruction category:

| Variant | Total instructions/kernel | What changed |
|---|---:|---|
| tile16 | 26.55M | Baseline |
| wide | 24.31M | Merge SLM->GRF messages; Send -56% |
| abop | 12.15M | Host-side operand layout; A direct 4x256B, B 4x256B zero-select; ALU0 -80% |
| v6b4 | 8.58M | Hoist base pointers, fixed strides; ALU1 8.04B -> 4.80B; barriers 109M -> 57M |
| v8b | 10.83M | A SLM relay adds instructions but removes global read redundancy; wall time still -3.3% |

The v8b case is the key counterexample: instruction count went up 26% but time went down because global traffic and latency dropped. Judge a change by wall time plus the profile, not by instruction count alone.

Code: [esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads](code-snippets.md#esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads), [host-side-a-operand-layout-packing-esimd-16x16](code-snippets.md#host-side-a-operand-layout-packing-esimd-16x16).

## f32 GEMV Ladder (4096x4096, row-major A)

Measured on Intel Arc A770 with the same USM harness (50 warmup + 500 timed launches, `q.wait()`, CPU float reference, relative tolerance `1e-4 * (1 + max|ref|)`). All values are stable across three runs.

| Step | Technique | Per-run | Notes |
|---|---:|---|---|
| 1 | Naive row per work-item, scalar loop | 2.65 ms | Buffer-based first run includes host transfer; use USM for steady-state timing |
| 2 | `vec<float,16>` row + x staged in SLM | 0.50 ms | Vectorization helps; SLM still has staging overhead |
| 3 | One row per sub-group + x in SLM | 0.24 ms | 16 lanes cooperate on one row |
| 4 | One row per sub-group + x direct from L2 | 0.215 ms | Best measured standard-SYCL variant |

Library baselines for the same `y = A*x` operator: oneMKL GPU gemv 0.329 ms, oneDNN GPU matmul `Kx1` 0.380 ms. oneDNN matmul `1xK` measured 0.182 ms but computes `x*A`, which is `A^T*x` for the row-major input, and failed the reference check; always verify the library result before trusting a baseline.

Copy-ready kernel and fallback rule: [f32 GEMV core](code-snippets.md#f32-gemv-core-sub-group-per-row-direct-l2).

### Why the GEMV steps work

- Sub-group-per-row changes the A access pattern from one thread walking a full 16 KB row to 16/32 lanes covering one row in contiguous 1 KB/512 B chunks, which reduces load instruction count and improves memory-level parallelism.
- Staging x in SLM is not needed for this shape: x is 16 KB, L1/L2 hold it while A is streamed once, and the SLM copy adds a barrier and extra instructions.
- Hardcoding 16 lanes is a correctness trap. A770 reports sub-group sizes 8/16/32 and the compiler picked 32 for the fastest kernels; a kernel written for 16 lanes then computed only half of each row (`max_err` about half the reference magnitude). Derive `per_lane = (N / VEC) / sgrp.get_local_range()[0]` instead. Forcing `sub_group_size<16>` was correct but slower (about 0.258 ms for the pointer-hoisted variant).
- Pointer hoisting and explicit two-accumulator unrolling were slower (0.258 to 0.323 ms), so the compiler was already turning `a + row*N + col` into incrementing addresses. VTune showed the cost is FMA plus reduction, not address math.

## u4->bf16 GEMV Ladder (4096x4096, bf16 A, u4 x, group_size=128)

Operator: `y = A*x`, bf16 row-major A, u4-packed x with per-group f16 scales
and zero-point 8, bf16 y. Same USM harness as the f32 GEMV: 50 warmup + 500
timed launches, three runs, CPU reference against the bf16-dequantized x.

| Step | Technique | Per-run | Notes |
|---|---:|---|---|
| 1 | Naive row-per-item, u4 unpack + scale in kernel | 2.22 ms | Per-element unpacking and scale reads dominate |
| 2 | Host dequant x to bf16, one row per sub-group, `vec<bf16,16>` + float accumulator, WG128 | 0.0883 ms | Final; 1D `nd_range` with dynamic sub-group size |

Same-operation baseline: oneDNN GPU matmul `Kx1` with bf16 src `{M,K}`, u4
weights `{K,1}` in `ba`, f16 group scales `{groups,1}`, zero-point 8, and
`fpmath_mode::any`: 0.1456 ms (`jit:gemm:any`). The SYCL kernel is about
1.65x faster.

Copy-ready kernel and host dequantization: [u4->bf16 GEMV core](code-snippets.md#u4-bf16-gemv-core-sub-group-per-row-bf16-vector--float-accumulator).

### Why the u4->bf16 GEMV steps work

- The u4 vector is only 2 KB for n=4096, so host-dequantizing it to bf16
  once removes all per-row nibble unpack and scale lookup work from the timed
  kernel. A row-per-item kernel that re-unpacks u4 for every row is
  instruction-bound, not memory-bound.
- The remaining kernel is a bf16 GEMV: A is 32 MB and streamed once, xd is
  L1/L2-resident, and one row per sub-group with `vec<bf16,16>` loads keeps
  enough memory-level parallelism.
- Float vector accumulators are required for large K. A bf16 vector
  accumulator passes 64x128 but fails 535/4096 rows at 4096x4096 because
  intermediate products lose too much precision before the final bf16 round.
- Work-group size tuning is flat: WG32 0.0887 ms, WG64 0.0893 ms, WG128
  0.0883 ms. WG128 was kept as the final choice.
- Manual 2/4-way vector unrolling measured 0.124 ms for both variants, much
  slower than the simple loop. The compiler already schedules the 8-iteration
  per-lane loop; explicit multi-accumulator unrolling added register pressure
  and address math.

## f32 RMSNorm Ladder (1024x4096, row-major x, f32)

Measured on Intel Arc A770 with oneAPI 2026.1, USM harness (100 warmup + 1000 timed launches per run, three runs, `q.wait()`, CPU float reference, absolute tolerance `1e-4`). All values are stable across three runs.

| Step | Technique | Best avg per run | Notes |
|---|---:|---|---|
| 1 | Naive row per work-item, scalar loop | 5.382 ms | One work-item serially walks 4096 cols; 1024 rows underutilize 4096 threads |
| 2 | Sub-group per row, `vec16`, direct L2 re-read | 0.153 to 0.157 ms | Fused sum + normalize in one launch; each row is read from global twice |
| 3 | SLM row tile, 2 rows/WG, wg512 | 0.108 to 0.110 ms | x read from global once, normalized from SLM; gamma stays in L2 |
| 4 | SLM row tile, 2 rows/WG, wg256 | 0.105 to 0.106 ms | More 64 B chunks per thread than wg512 |
| 5 | SLM row tile, 1 row/WG, wg256 | 0.102 to 0.103 ms | 16 KB SLM/WG leaves more residency |
| 6 | SLM row tile, 1 row/WG, wg128 | 0.100 to 0.101 ms | Best geometry: 4 sub-groups, 2 chunks per thread |
| 7 | + remove `inv` local array and third barrier | 0.0989 to 0.1000 ms | Final champion; best measured 0.09797 ms |
| 8 | + `vec16` accumulator instead of scalar `v[e]*v[e]` loop | 0.100 to 0.101 ms | Neutral; compiler already vectorized step 7 |

Same-operation baseline: oneDNN `layer_normalization_forward` with `normalization_flags::use_scale | normalization_flags::rms_norm`, `prop_kind::forward_inference`, GPU engine, shared USM pointers: 0.1235 to 0.1242 ms, `errors: 0/4194304`, `max_abs_err=7.15e-07`. The final SYCL kernel is about 1.25x faster than oneDNN.

VTune instruction-count comparison (per kernel):

| Variant | Total instructions | Send | Int32 & SP Float | Notes |
|---|---:|---:|---:|---|
| Step 2 (direct L2 re-read) | 4.03M | 526K | 2.76M | Two global reads of x plus address math |
| Step 7 (SLM row tile) | 2.60M | 493K | 1.49M | One global read of x; SIMD utilization 92.8% |

VTune sampling inflated wall time by ~50-60x, so use the instruction mix for diagnosis, not the profiled run times.

### Why the RMSNorm steps work

- One work-group per row with 128 threads keeps all sub-groups on the same row, so sub-group size and 2D mapping cannot break correctness. Each thread handles two `vec<float,16>` chunks (128 B) of the 16 KB row.
- SLM staging removes the second global read of x. The direct L2 re-read variant was ~55% slower even though 16 MB of x nominally fits L2.
- Two barriers instead of three: with one row per WG, every thread can compute `inv_rms = 1/sqrt(row_sum/N + eps)` itself after the reduction barrier; a per-row `inv` local array plus an extra barrier was not needed.
- A 4-row 64 KB SLM tile failed at launch; 32 KB (2 rows) worked but was slower than 16 KB (1 row). Smaller SLM left more resident work-groups.
- Staging gamma in SLM is not useful for one row per WG: gamma is already read exactly once per row and stays in L2.

## f32 Softmax Ladder

Measured on Intel Arc A770 with oneAPI 2026.1, USM harness (50 warmup + 500 timed launches, 20 + 100 for cols > 4096, three runs per case, `q.wait()`, CPU float reference, tolerance `1e-5`). All values are 3-run averages and were stable across repeated benchmark invocations.

| Shape | Naive | Final SLM | oneDNN | SLM/oneDNN |
|---|---:|---:|---:|---:|
| 16x32 | 0.0224 ms | 0.0066 ms | 0.0116 ms | 0.57 |
| 16x256 | 0.0886 ms | 0.0061 ms | 0.0140 ms | 0.44 |
| 16x257 | 0.1707 ms | 0.0114 ms | 0.0184 ms | 0.62 |
| 16x1024 | 0.3407 ms | 0.0071 ms | 0.0112 ms | 0.63 |
| 16x1025 | 0.6786 ms | 0.0114 ms | 0.0178 ms | 0.64 |
| 16x4096 | 1.3922 ms | 0.0085 ms | 0.0147 ms | 0.58 |
| 16x4097 | 3.1981 ms | 0.0228 ms | 0.0229 ms | 1.00 |
| 1024x256 | 0.2267 ms | 0.0112 ms | 0.0170 ms | 0.66 |
| 1024x1024 | 0.8910 ms | 0.0345 ms | 0.0489 ms | 0.70 |
| 1024x4096 | 5.5716 ms | 0.1088 ms | 0.2169 ms | 0.50 |
| 1024x8192 | 11.7098 ms | 0.2113 ms | 0.4191 ms | 0.50 |
| 1024x16384 | 26.7873 ms | 0.5923 ms | 2.6444 ms | 0.22 |

The measured standard-SYCL ladder on the 1024x4096 shape:

| Step | Technique | Best avg per run | Notes |
|---|---:|---|---|
| 1 | Naive row per work-item, scalar loop | 5.55 ms | One work-item walks a full row; 1024 rows underutilize 4096 threads |
| 2 | One-shot SLM row tile, wg128 | 0.1125 ms | One global read, max + sum + normalize all served from SLM |
| 3 | + one-shot limit raised to 8192 cols (32 KB SLM) | 0.1125 ms (4096), 0.2113 ms (8192) | 16x4097 dropped from 0.037 ms tiled to 0.023 ms |
| 4 | + `vec16` exp accumulator and vector stores | 0.1074 ms | Fewer exp/ALU/store instructions; unlike RMSNorm, this was a small positive |
| 5 | + dynamic WG size by cols (16/32/128 threads) | 0.1088 ms | Fixed idle-thread waste on short rows; 1024x256 dropped from 0.0227 ms to 0.0112 ms |

Same-operation oneDNN baseline: `softmax_forward` with `algorithm::softmax_accurate`, `axis=1`, GPU engine, shared USM pointers. The final SYCL kernel is about 2x faster at 1024x4096 and up to about 4.5x faster at 1024x16384. The 1024x16384 oneDNN path measured 2.64 ms, far above its smaller-shape trend; always re-measure the oneDNN baseline per shape.

Generic column counts (257, 1025, 4097) use a scalar-load SLM path because `vec<float,16>` needs 64 B alignment; correctness stayed within `1e-5` and the times stayed close to oneDNN instead of falling back to the naive kernel.

### Why the Softmax steps work

- One work-group per row plus SLM staging gives softmax the same one-global-read property as RMSNorm: max and sum reductions plus the normalize pass all re-read the row from SLM.
- `sycl::reduce_over_group(it.get_group(), ...)` returns the reduction to every thread, so the one-shot kernel needs only one explicit barrier (after the SLM load); the group reductions provide the remaining synchronization.
- A 32 KB SLM tile (8192 floats) launches fine on A770 and beat the 32 KB-per-tile three-pass tiled variant for 4097 and 8192 columns; the earlier 4096-column threshold was too conservative.
- For rows that exceed one SLM tile, keep the max pass out of SLM entirely: threads read global chunks directly and reduce, then tiles are staged only for sum and normalize. That reduces global row reads from three to two and cut 1024x16384 from 0.66 ms to 0.59 ms.
- Dynamic work-group sizing matters most below 1024 columns. 16 threads for cols <= 256, 32 for cols <= 1024, and 128 otherwise removed mostly-idle work-groups; do not ship a fixed 128-thread kernel for every shape without checking small-row shapes.
- Alignment is a correctness gate for the vector fast path: allocate input/output with `sycl::aligned_alloc_shared<float>(64, ...)`. For arbitrary odd column counts, the generic scalar SLM path is still much faster than the naive row-per-item kernel.

## f16/u4/bf16 GEMM (1024x1536x512, group_size=128)

Measured on Intel Arc A770 with oneAPI 2026.1, USM harness (100 warmup + 1000 timed launches, three runs, `q.wait()`).

| Variant | Per-run | Notes |
|---|---:|---|
| Naive f16 x u4 -> bf16 | 1.587 ms | One work-item per C element, nibble unpack in the kernel |
| oneDNN `ab` u4, no scales | 110.62 ms | `ocl:ref:any`; measured with 5 warmup + 20 timed because the reference path is extremely slow |
| oneDNN `ba` u4 + f16 scales + zp8 | 0.0333 to 0.0344 ms | `jit:gemm:any`; `errors: 0/1572864` |
| SYCL bf16 DPAS after host dequant | 0.060 to 0.061 ms | u4 weights are dequantized to bf16 once on the host, then the standard bf16 ESIMD 16x16 kernel runs |

Why the SYCL path plateaus:

- The stable SYCL path is memory- and instruction-equivalent to the bf16 campaign, so it lands at ~0.060 ms, not at the u4 path's 0.033 ms.
- SYCL `dpas` statically rejects mixed f16 + u4 operands. The low-level `__esimd_dpas2<u4, fp16, ...>` intrinsic did not compile to a working A/B operand shape on oneAPI 2026.1.
- oneDNN reaches 0.033 ms because its ngen-generated u4 JIT path consumes the packed `ba` layout and fuses the group-scale dequantization natively.

To reproduce the oneDNN fast baseline, build a standalone C++ primitive (no PyTorch) with:

```cpp
src_md = memory::desc({M, K}, data_type::f16, format_tag::ab);
wei_md = memory::desc({K, N}, data_type::u4, format_tag::ba); // [N, K/2] bytes
dst_md = memory::desc({M, N}, data_type::bf16, format_tag::ab);
scale_md = memory::desc({G, N}, data_type::f16, format_tag::ab);
zp_md = memory::desc({1}, data_type::u8, format_tag::a);

attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                {group_size, 1}, data_type::f16);
attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, data_type::u8);
attr.set_fpmath_mode(fpmath_mode::any, true);
```

Weights are packed as `[N, K/2]` with the low nibble first along K. Scale `g` applies to `k / group_size`, and the signed value is `u4 - 8`. The full standalone harness is `bench_onednn_int4_gemm.cpp` from the QuantizedGEMM-Opti campaign.

## Interpreting VTune gpu-hotspots Metrics

Full-compute numbers to compare (all 5500 kernels, A770):

| Metric | Meaning | OneDNN reference |
|---|---|---:|
| GPU time per kernel | Real kernel cost | 50.9 us |
| ALU0 / ALU1 instructions | Integer/address and float math | 0.52B / 5.09B |
| Send instructions | Global/SLM messages | 2.17B |
| XMX instructions | DPAS issue count | 13.08B |
| GPU Barriers | Sync count | 38.3M |
| XMX pipeline active | How busy XMX stays | 19.1% |
| XVE Active / Stalled / Idle | Issue pipeline states | 18.6 / 42.3 / 39.0% |
| Occupancy | Thread residency | 28.7% |
| L3 Bandwidth Bound | Bandwidth saturation | 0.2% |

Diagnostic rules learned from the campaign:

- When L3 Bandwidth Bound is near 0, stop chasing bandwidth and look at instruction volume or issue rate.
- When one variant has higher occupancy but is slower, the bottleneck is instructions per kernel, not residency.
- XMX instruction count staying flat while ALU/Send drop means the remaining gap is operand feeding and scheduling, not math.
- Compare wall time to kernel time (`xpu-offload`) first; if launch overhead dominates, optimize the launch, not the kernel body.
