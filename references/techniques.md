# Measured Optimization Ladder

## Table of Contents

- Benchmark context and baselines
- The full ladder with measured numbers
- Why each step works
- VTune-driven instruction reduction
- f32 GEMV ladder and baselines
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
