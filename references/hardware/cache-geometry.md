# Cache geometry on the A770 (DG2) `[MEASURED]`

Measured with `tools/cacheprobe` (pointer-chase and stride-scan microbenchmarks; kernel built from
source at runtime so no AOT recompile interferes). Grid: one work-item for latency, 64 for
bandwidth.

## 1. Latency ladder (dependent chase, 4 KB walk stride)

| working set | ns per access | level |
|---|---|---|
| 32 KB - 1 MB | 52 - 64 | L1 |
| 2 MB - 16 MB | 118 - 142 | L2 |
| 32 MB | 169 | transition |
| 64 MB | 229 | mostly DRAM |
| 128 MB | 366 | DRAM plateau |

Treat three numbers as the planning constants: **~52 ns L1, ~118-142 ns L2, ~366 ns DRAM**

## 2. Do not size tiles from "192 KB of L1"

A single hardware thread keeps L1-speed latency out to a **1 MB** working set; the step to L2
latency happens between **1 MB and 2 MB**. The commonly quoted 192 KB (per Xe-core / per subslice)
therefore under-estimates what one thread can keep hot by roughly 5x. Conversely L2 = **16 MB** is
confirmed: 16 MB still reads at L2 speed (142 ns) and 32 MB does not (169 ns).

## 3. L2 bandwidth and line granularity

Stride scan over an 8 MB L2-resident buffer, 64 threads:

| stride | GB/s |
|---|---|
| 4 B | 632 |
| 8 B | 996 |
| 16 B | 1406 |
| 32 B | 1768 |
| 64 B | 2024 |
| 128 B | 2187 |
| 256 B | 2276 |
| 512 B | 2323 |
| 1024 B | 2396 |

So **L2 peak is about 2.4 TB/s** (never quote it as DRAM bandwidth - DRAM-like contiguous reads are
in `dram-contiguous-roofline.md`), and bandwidth is essentially saturated at a **128 B stride,
consistent with 128 B line granularity**. Narrow strides are request-rate limited, not
bandwidth limited.

## 4. Capacity scales with active threads

Each work-item walking its own private 96 KB region:

| threads | total footprint | aggregate ns/access |
|---|---|---|
| 1 | 96 KB | 77.7 |
| 16 | 1.5 MB | 7.6 |
| 64 | 6 MB | 3.2 |
| 128 | 12 MB | 1.6 |

Aggregate cost falls almost linearly with thread count, i.e. near-side capacity/bandwidth is
per-subslice and the L2 is shared. Budget occupancy for bandwidth the same way you budget it for
XMX.

## 5. Dependent loads are ~2.5x slower than the ladder suggests `[MEASURED]`

A chain whose next address comes from the loaded value (classic pointer chasing, linked lists,
sparse gather) measures **~151 ns per access no matter how many lines it touches**: 150.41-152.47 ns over
25 dependent-chase configs (4-32 lines, 128 KB-2 MB spacing) and 156.80-157.45 ns over 12 independent-load
configs. That is ~2.5x the 52 ns L1 hit figure, so the L1/L2 "hit" numbers only apply to accesses
the compiler could issue ahead of time. Any loop with a real cross-iteration load dependency is
latency-bound at ~150 ns per hop: prefer restructuring (unroll + independent chains, prefetch into registers, tree instead
of list) over micro-tuning the arithmetic.

## 6. Associativity: not observable, and here is why [MEASURED]

Two independent methods, 37 configurations total:

* dependent chase over 4/8/16/24/32 lines at 128 KB - 2 MB spacing: **150.41-152.47 ns**, 25 configs;
* independent loads over a fixed set of N lines (no address dependency), N = 4..48 at 128 KB and
  1 MB spacing: **156.80-157.45 ns**, 12 configs.

Both converge to the same constant, which is the tell: a thread's loads are **serialised at
~152-157 ns per access** regardless of how many lines are in play. That floor is stronger than any
set-conflict effect, so L2 associativity cannot be resolved by software timing on this part. Do not
assume a low associativity (no cliff was ever seen), and do not build a loop that hopes for
intra-thread memory-level parallelism: bandwidth has to come from thread count.

## 7. Replacement behaviour: +16% the moment anyone else streams [MEASURED]

Inside one launch, work-item 0 chases a hot 64 KB region while others stream 64 MB:

| concurrent streamers | chaser ns/access |
|---|---|
| 0 | 52.54 |
| 1 | 61.20 |
| 4 | 60.30 |
| 16 | 61.33 |
| 32 | 62.20 |

One streamer costs +16%, and 32 streamers (32x traffic) add only 1.0 ns more. Step-then-flat:
bandwidth contention, not LRU eviction.

## 8. Whether the fast path is per-subslice [HEURISTIC]

Each work-item walking its own 1 MB region: 1 thread 83.5 ns/access, 16 threads (16 MB total) still
10.1 ns/access aggregate. Near-side capacity clearly scales with thread count, which is consistent with
per-subslice caches, but no subslice boundary was measured - treat 'per-subslice' as an inference.

## Instrument status on this box: VTune works, but still no associativity

The earlier note that VTune is unavailable is **outdated**: VTune 2026.2.0 (build 632324)
`-collect gpu-hotspots` runs on the A770 here and produces a report with GPU Time, GPU L3 Bandwidth
Bound, XVE Array Stalled/Idle and Occupancy (`probes/059-assoc-instrument/vtune_summary.csv`).
For the dependent-chase and conflict workloads it reports **GPU L3 bandwidth bound 0.0-0.2% with XVE
stalled 100%**, confirming they are latency-bound - which is exactly why timing-based conflict scans
never show an associativity cliff.

It does not expose L1/L2 hit-miss or per-set counters, so **L2 associativity remains unmeasurable**
(instrument-level negative: dependent-chain sweeps, fixed-line-set throughput sweeps, non-dependent
conflict loops and VTune GPU sampling have all been tried). What is needed is a GPU hardware counter
for per-set misses (not exposed by this driver) or Intel's L2 set-index hash documentation.

Per-subslice boundary, two-chaser experiment (cacheprobe mode 1, two threads, `--local 2` = one
work-group, `--local 1` = separate work-groups):

| per-chaser footprint | same work-group | separate work-groups |
|---|---|---|
| 24 KB | 27.21 ns | 26.26 ns |
| 48 KB | 27.37 ns | 26.95 ns |
| 96 KB | 29.42 ns | 29.01 ns |
| 192 KB | 35.32 ns | 34.87 ns |
| 384 KB | 49.77 ns | 49.38 ns |

Same-work-group versus separate-work-group changes the cost by <4%, so work-group co-residency is
**not** a usable proxy for subslice placement on this driver. The footprint curve itself (two chasers
stay at L1 speed up to 96 KB each = 192 KB combined, then degrade) is consistent with a ~192 KB
per-subslice budget but stays **[HEURISTIC]** - there is no counter evidence.
