# Occupancy and XMX Saturation on A770 (measured)

**Scope**: Intel Arc A770 (DG2), driver IGC `32.0.101.9150`, oneAPI 2026.1. The curves below were
measured by injecting a fixed dependent `dpas` chain into the runtime axpy binary and varying the
number of work-items (see `isa/dpas.md` for the mechanism, `probes/032-dpas` for the data).

## 1. The saturation curve `[MEASURED]`

Fixed work per thread: 512 `dpas.8x8`, i.e. every thread does the same instruction stream and only
the number of threads changes.

| Work-items | Hardware threads | Device µs with 0 dpas | with 512 dpas | Δ µs | Aggregate |
|---|---|---|---|---|---|
| 4096 | 128 | 4.2 | 9.2 | 5.0 | 13.1 Gdpas/s |
| 16384 | 512 | 4.6 | 14.5 | 9.9 | 26.5 Gdpas/s |
| 65536 | 2048 | 5.6 | 28.2 | 22.6 | 46.4 Gdpas/s |
| 131072 | 4096 | 6.7 | 32.3 | 25.6 | **81.9 Gdpas/s** |

- `[MEASURED]` Quadrupling the thread count from 512 to 2048 more than doubles the added time
  (9.9 → 22.6 µs): the EUs are filling up. From 2048 to 4096 the added time grows only 13% — the
  machine has reached its issue limit.
- `[MEASURED]` The plateau is the number to design against: **≈82 Gdpas/s aggregate ≈ 6.3 ns per
  `dpas.8x8` per EU**, independent of precision class and of per-thread ILP (`isa/dpas.md` §2).
- `[MEASURED]` Below ~1024 threads the same instruction stream costs 3–4× *less* per thread and
  the measurement says nothing about the machine: the per-thread dependency latency dominates
  (~10.7 ns per dpas for a lone chain).

## 1b. ESIMD work-groups are capped at 128 work-items `[ISA]`

In ESIMD one **work-item is one hardware thread**, and a work-group is dispatched to a single DSS
(2 subslices). With 8 hardware-thread slots per EU and 8 EUs per subslice, the ceiling is
**64 threads per subslice × 2 = 128 work-items per work-group**:

- `IGC/Compiler/CISACodeGen/Platform.hpp:982-992` — `getMaxNumberHWThreadForEachWG()` returns
  `getMaxNumberThreadPerSubslice() * 2` for Gen12..XeHPG ("Each WG is dispatched into one DSS
  which has 2 Subslices").
- `compute-runtime/shared/source/xe_hpg_core/hw_info_dg2.cpp:90` — `NumThreadsPerEu = 8`.

Consequences:

- A 256-work-item ESIMD work-group is **never scheduled**: the kernel hangs with no error, no TDR
  and no diagnostic, and the GPU stays healthy for other contexts (the process just spins).
  `info::device::max_work_group_size` still reports 1024 because that number is expressed in
  SIMD32 work-items, not in hardware threads — it is meaningless for ESIMD.
- To raise parallelism in an ESIMD kernel you can only **increase the per-thread tile** (more
  independent `dpas`/accumulators per thread), never the work-group size. A 128-thread work-group
  is 4 hardware threads, so a 4096-thread machine needs 1024 resident work-groups.

## 1c. Measured consequence: a memory-latency-bound shape lives or dies by thread count `[MEASURED]`

bf16 GEMM 1024x1536x512, one ESIMD work-group per output tile, driver `32.0.101.9150`.
Because the work-group is capped at 128 work-items (§1b) and there is one work-group per tile, the
hardware-thread count is `M*N/(256*TN)` — **independent of BM/BN** — so every attempt to cut memory
traffic with a bigger tile also cuts occupancy:

| tile (BM x BN x BK), TN | hw threads | naive traffic | device µs | TFLOPS |
|---|---|---|---|---|
| 128x64x32, TN=8 | 768 | 31 MB | 137.0 | 11.7 |
| **128x128x32, TN=16** | **384** | 25 MB | **101.1** | **15.9** |
| 128x128x64, TN=16 | 384 | 25 MB | 106.8 | 15.1 |
| 128x256x32, TN=32 | 192 | 19 MB | 122.2 | 13.2 |
| 256x128x32, TN=32 | 192 | 19 MB | 137.0 | 11.7 |
| 256x256x32, TN=64 | 96 | 12.6 MB | 120.0 | 13.4 |
| 256x256x16, TN=64 | 96 | 12.6 MB | 122.6 | 13.1 |

- `[WITHDRAWN]` The 101.1 us row cannot be reproduced by any current build (10 existing
  binaries x 2 protocols and `-O3` all measure >=103 us); the live tuned tile is 256x192x32
  TN=48 at 96.93-97.90 us (`docs/53` §2 in Arc-ISA-Research). Read the us/TFLOPS columns as
  relative shape only, never as an absolute baseline.

- `[MEASURED]` Time follows the **thread count**, not the traffic: cutting traffic 2.5x while
  cutting threads 8x makes it *slower*.
- `[MEASURED]` At the optimum the kernel moves 25 MB in 101 us = 250 GB/s, which is **below both
  roofs** (DRAM 291 GB/s, L2 855 GB/s): nothing is saturated, i.e. this is a **latency/occupancy**
  limit at ~9% occupancy, not a bandwidth limit. With N-first work-group order the A strip (1 MB)
  and B strip (1.5 MB) fit the 16 MB L2, so the naive traffic count overstates DRAM traffic.
- `[MEASURED]` Reference point: oneDNN `jit:gemm:any` does the same shape in 73.2 us (22.0 TFLOPS)
  with a standard SIMD8/16 kernel, where one work-item is 1/8..1/32 of a hardware thread and up to
  4096 threads can be resident. **That is the structural cost of ESIMD's 1:1 mapping on
  memory-latency-bound shapes.**

## 2. Occupancy is usually work-limited, not tile-limited `[ARCH]`

A770 has 512 vector engines × 8 hardware threads = **4096 hardware threads**, 32 lanes each. For a
GEMM the thread count is fixed by the output size and the per-thread tile:

```
hardware_threads = (M/BM) * (N/BN) * (threads_per_wg / 32)
```

For the 1024x1536 benchmark with BM=128, BN=64 and 128 threads per work-group:
`8 * 24 * 4 = 768` threads — **19% of the machine**, regardless of whether the work-group is 128
threads (8x8 grid) or 512 threads with a wider BN (both give 768). `[MEASURED]` This is why the
tiling debate matters less than it looks: with 768 threads the per-thread instruction stream must
carry its own ILP.

Practical consequences:

- If `hardware_threads` comes out below ~2048, **increase the per-thread tile** (8x16 or 16x16 =
  2 or 4 independent `dpas` per thread) before touching anything else.
- If it comes out above 4096, the problem is **wave quantization**: size the work-group so the
  total is an exact multiple of 4096 hardware threads, or choose a tile that divides the shape.
- `[MEASURED]` Adding ILP does not raise the plateau (§1): at 4096 threads 8 independent chains
  cost the same per instruction as one chain. ILP helps only in the *under-filled* regime.

## 3. Small kernels: check the launch term first `[MEASURED]`

Measured on the same box: wall−device = **~103 µs** (f32 GEMV, 217 µs device) and **~66 µs**
(f32 RMSNorm, 92 µs device). Any kernel under ~100 µs of device time must report both bases,
because host/queue overhead is the same order as the kernel.

## 4. Design checklist

1. Compute `hardware_threads` from the tiling before tuning anything.
2. If < 2048, raise per-thread work (more independent accumulators) or shrink BM/BN.
3. If ≥ 4096, fix wave quantization and tail effects instead.
4. Only then worry about instruction scheduling — the per-EU issue rate is already the wall.

## 256-GRF mode on DG2: not reachable, and not needed at sane pressures

The 128/256/512-GRF "static register sharing" modes are gated in IGC by
`Platform.hpp::supportsStaticRegSharing()`, which returns true only for `IGFX_XE_HP_SDV` (or when the
`ForceSupportsStaticRegSharing` regkey is set). DG2 is not XE_HP_SDV, so `getNumGRFPerThread()` never
returns anything but 128 at runtime. Measured (`probes/058-large-grf/grf_reachability.csv`): five
environment combinations (`IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread`, `IGC_ForceGRFModeUp`,
`IGC_ForceSupportsStaticRegSharing`, and both together) all left the runtime `grf_count` at 128, and
every AOT binary printed "recompiled from IR" - the 256-GRF code never executed.

AOT does honour the flag (`ocloc compile -options '-cl-intel-256-GRF-per-thread'` writes
`grf_count: 256`), but at 8-256 live accumulators the instruction count moves by +2 only (no spills
either way), so there is nothing to win even if it ran.

What actually pays on the reachable 128-GRF path, measured with a fixed 4096-mad body
(`occupancy_ilp.csv`):

| ILP \ threads (grid) | 64 | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| 1 | 57.19 us | 39.69 | 39.69 | 39.79 | 46.04 |
| 8 | 10.10 | 10.10 | 10.10 | 10.10 | 10.94 |
| 16 | 9.17 | 9.17 | 9.58 | 9.58 | 9.69 |
| 32 | 9.17 | 9.17 | 9.17 | 9.27 | 9.79 |
| 64 | 9.27 | 9.27 | 9.27 | 9.38 | 10.10 |

Rules:

1. Below ILP 8, add threads first (64 -> 256 work-items buys 1.44x on a latency-bound body).
2. At ILP >= 8, occupancy is irrelevant (<10% across a 256x thread range) and huge grids are 6%
   slower; cap ILP at 16-32 and keep the grid modest.
3. Do not plan a kernel around 256 GRF on A770: the mode is unreachable from OpenCL, so register
   pressure must be solved with ILP <= 32 plus rematerialisation inside 128 GRF.
