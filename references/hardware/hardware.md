# A770 (Xe-HPG / DG2) Hardware Reference

## Table of Contents

- Specs
- DPAS / XMX constraints
- Memory system constraints
- GEMV notes
- Occupancy math and measured sweet spots
- oneDNN fastest path decode

## Specs

| Spec | Value | Implication |
|---|---|---|
| Hardware thread count | 4096 | Standard SYCL: align hardware-thread waves, not work-item counts; ESIMD occupancy is work-item-based |
| Hardware threads per vector engine | 8 | Regular GRF mode |
| Hardware threads per vector engine (large GRF) | 4 | Large GRF halves occupancy |
| GRF per thread | 128 regular / 256 large | Keep register budget with headroom; 64/128 was the 16x16 GEMM sweet spot |
| Register width | 256 bits (32 B) | A float 8x8 accumulator is 256 B = 8 GRF |
| Sub-group sizes | 8, 16, 32 reported by `info::device::sub_group_sizes` | The compiler can pick 32 even for a kernel written with a local dimension of 16 |
| L1 per Xe-core | 192 KB | On-chip reuse hierarchy starts here |
| SLM per Xe-core | 128 KB | Shared local memory pool |
| Max SLM per work-group | 64 KB | Hard per-WG cap |
| L2 | 16 MB | A+B working set of 2.5 MB is fully L2-resident; prefetch has nothing to hide |

## DPAS / XMX Constraints

- bf16 DPAS shape on DG2: 8x8x16 (RepeatCount 8, N 8, SystolicDepth 16). PVC differs (8x16x16), so PVC tuning advice such as the official 32x64x16 register block does not transfer directly.
- `SystolicDepth` is fixed at 8 by `static_assert` in `dpas.hpp`.
- DPAS `ExecutionSize=16` is allowed by the header but produces wrong results on A770 (78/128 errors); treat it as unsupported.
- B operand must be VNNI-packed: two consecutive K rows are packed into one uint32 (`{B[2k][n], B[2k+1][n]}`). A stays row-major.
- `dpasw` is ExecutionSize-8 only. For a 16x16 tile it emits the same number of instructions as `dpas`; it does not reduce XMX instruction count.
- The Intel joint_matrix path used 2x2 8x8 accumulators to form a 16x16 C tile per sub-group; oneDNN's 16x8 per-thread tile cannot be reproduced in the joint_matrix path (C tile is 16x16). ESIMD can express 16x8, but it measured negative on A770 from read amplification.

## Memory System Constraints

- DG2 `block_load` / `slm_block_load` max is 256 B; 512 B is PVC-only. Reduce message count by relayouting data, not by widening past 256 B.
- 256 B LSC block accesses already span SLM banks. Manual SLM bank padding broke 256 B alignment and measured negative.
- `load_2d` / `prefetch_2d`: PVC-only. A770 hangs; bf16 `Transposed` is rejected at compile time (only u32/u64).
- `named_barrier`: PVC-only.
- L2 is 16 MB. Use walk order (N-first) so adjacent work-groups share A/B tiles already in L2.

## GEMV Notes (f32, 4096x4096, row-major A)

- GEMV is memory-bound on A770: 64 MB of A plus a 16 KB x vector per launch. x fits in L2 and L1, so a cooperative SLM relay of x was slightly negative (0.229 ms vs 0.215 ms for direct L2 reads).
- Best measured standard-SYCL structure: 2D `nd_range`, one row per sub-group, `vec<float,16>` loads, per-lane trip count derived from the actual sub-group size, scalar `reduce_over_group`, 512-thread work-groups. Stable at about 0.215 ms.
- Same-operation library baselines on the same machine, device-time median with the unified protocol: oneMKL GPU gemv `transpose::trans` 0.166 ms, oneDNN GPU matmul `Kx1` `jit:gemm:any` 0.456 ms. The older 0.329 / 0.380 ms numbers were wall-time based and are superseded; see the ladder in [techniques.md](../techniques/techniques.md) for the full table. oneDNN `1xK` at 0.182 ms is `x*A` (`A^T*x`), not row-major `A*x`, and its output did not match the reference.
- VTune instruction-count mix for the 0.215 ms kernel: about 3.9M instructions/kernel, Int32/SP Float 61%, Other 18.8%, Send 13.7%, Control Flow 4.5%, Synchronization 1.7%, SIMD utilization 96.9%. The remaining cost is vector FMA plus reduction, not global bandwidth or address math.

## Occupancy Math and Measured Sweet Spots

- Standard SYCL mapping: a work-item is a SIMD lane, a sub-group is one SIMD
  hardware thread, and `HW threads per WG = WG_size / SG_size`. For full
  waves, align `WG_count * WG_size / SG_size` to 4096, not
  `WG_count * WG_size`. The compiler may pick SG=32 even when the kernel was
  written for 16.
- ESIMD mapping: an ESIMD work-item explicitly owns a SIMD vector, so do not
  reuse standard-SYCL occupancy arithmetic blindly. Count ESIMD work-items
  and confirm with VTune occupancy.
- Measured joint_matrix geometry (standard SYCL): BM=128, BN=64, BK=16,
  512-thread work-group, GRF usage about 64/128; 24 full waves was the best
  measured geometry with the actual sub-group mapping.
- Measured ESIMD geometry: 32 threads/work-group x 16x16 per-thread C tile,
  A 2x8 KB + B 2x4 KB SLM (24 KB total).
- With A read directly from global, 4-buffer B SLM (16 KB) and one barrier per 2 K blocks was best.
- SLM per work-group above about 32 KB starts losing resident work-groups (8-buffer B at 32 KB measured negative).
- Higher occupancy is not the goal by itself: oneDNN ran at 28.7% occupancy and beat an ESIMD variant at 58.4% because it issued fewer instructions.

## oneDNN Fastest Path Decode

- Entry gates in `jit_xe_hp_systolic.cpp`: `mayiuse_ngen_kernels()` and `mayiuse_large_grf_mode()`. A770 XMX GEMM is ngen code generation + Large GRF, not SYCL joint matrix.
- No-copy path when `M <= 1024 || N <= 1024`: no extra A/B packing kernel; the gemmstone kernel reads A/B layouts directly.
- Tuned configuration: SIMD8 threads, GRF 256 (4 threads/EU), 64 HW threads/work-group, work-group tile 128x64, per-thread C 16x8, K block 32, 48 KB SLM, 4 SLM buffers + 2 GRF copies, barrier per 32K, Boustrophedon walk via fused linear ID.
- The remaining 1.11x gap between our best ESIMD kernel and oneDNN is instruction volume from the A/B SLM relay plus ngen-level scheduling (DPASW chains, SWSB), which SYCL/ESIMD cannot express.
