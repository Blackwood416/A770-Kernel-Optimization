# Performance playbook: from a measured symptom to the ISA fact that explains it

This page is the decision layer of the skill: each row starts from an observation you can make with the
instruments in `instruments.md`, names the ISA/architecture fact that limits it (with the doc that proves it),
and gives the action plus the expected effect. Numbers are from `bench/records/` stamps and the docs cited.

## 1. Before changing anything

1. Reproduce the operator with the standard protocol (`bench/run_baselines.ps1`, warmup 20 / samples 20 /
   batch 50, per-operator tolerance, 0 errors) and record the stamp. One run is not a baseline: quote
   p10/median/p90 over at least two independent records.
2. Identify the binding constraint with **counters, not intuition**: `docs/50` gives VTune
   `instruction-count` (32-lane execution groups; only deltas are comparable) and `overview`
   (XVE stalled %, occupancy, L3-bound %).
3. Only then pick a row below.

## 2. Symptom to action

| Symptom (measured) | ISA fact behind it | Action | Expected effect |
|---|---|---|---|
| XVE Array Stalled/Idle high, L3-bound low, dpas count far below the issue ceiling | the kernel waits on load latency / barriers, not on math (`docs/53` §6b: 9.9 % of the dpas issue ceiling, 93 % stalled) | deepen the software pipeline while SLM budget allows, or raise work-group count; if SLM >= 32 KB per stage you cannot add a third stage (64 KB cap) | each extra pipeline stage removes one barrier's worth of exposure; a tile change that costs work-groups is usually a regression (`docs/53`: 512x256 with 12 WGs = 290 us) |
| Not enough work-groups to fill the DSSs | one ESIMD work-group is one DSS's 128 threads; `WG >= 256` never gets dispatched (I-003) | pick the tile with `WG=(BM/8)*(BN/TN)=128` and `M*N/(BM*BN)` equal to the DSS count (32 on the A770) | `docs/53`: 32 WGs = 96.9 us vs 24 WGs = 103-106 us vs 12 WGs = 290 us |
| Occupancy metric looks low but the kernel is already winning | VTune's occupancy is a sampled peak-XVE-thread ratio and spikes (31 % vs 71 % on the same kernel, `docs/50` §4) | do not tune against a single occupancy reading; use medians of >= 3 collections and prefer stall/instruction counters | avoids chasing a sampling artefact |
| Per-work-item data read from `__constant` | `send.dc0` costs 197-227 ns per message vs 0.91 ns/B for `load.ugm` (`docs/29`) | move the data to global and load it once per tile/row, or pass it as a kernel argument | removes a serialised constant fetch from the inner loop |
| Scalar loads/stores around a tile loop | LSC width is chosen from the access pattern; 32 B/128 B per message (`docs/29`, `docs/46`) | vectorise to at least 128 bits and keep accesses 32 B-aligned so IGC emits block loads | fewer messages and fewer SWSB tokens for the same bytes |
| Long dependent chain, send/load latency exposed | SWSB tokens are the scoreboard; 16 architectural token IDs (`docs/26`, `docs/48`) | with inline asm annotate `{$n}`/`{A@n}` per class; in source trust IGC, which already keeps ~8 load messages in flight for 16 loads (`docs/48` §3) | more in-flight loads without breaking dependencies |
| Many tiny work-groups, fixed per-group cost dominates | EOT/last-wave floor ~0.47 ns per work-group plus a thread-ladder intercept of ~1.5 us (`docs/36`) | raise the work per group (bigger tile or fewer, fatter groups) instead of adding groups | removes the launch/EOT floor from the measurement |
| bf16/f16 conversion shows up in the profile | conversion is one typed `mov` per 8 lanes, no `cvt` on DG2 (`docs/44`) | keep conversion in the epilogue; do not pre-convert whole buffers (a separate pack pass costs 0.44-0.51x the GEMM) | conversion disappears from the hot loop |
| Denormal inputs suspected of a throughput penalty | cr0 bit 7 is the denorm enable; denorms cost no throughput, the flag survives recompile (`docs/43`) | treat cr0 as a numerics switch only; do not expect speed | removes a dead end |
| Want to prefetch a future line | prefetch is unreachable from IGA, OpenCL and ESIMD on DG2 (`docs/47`) | improve locality/tiling instead; do not spend time on prefetch intrinsics | avoids a dead end |
| Need instruction-type counts for a dual-issue question | the CLI summary exposes only total `GPU Instructions Executed`; per-type data is in the result's `dicer.db` GT-PIN tables (`docs/50` §5-6) | parse `dd_gpu_exec_instruction_type` / `gpu_gtpin_data_agg_data`, or accept a documented negative | decides whether the question is answerable at all |
| Comparing two builds and one looks 15 % faster | run-to-run spread on this box can fake an 11 % "effect" in either direction (`bench/README`), and a retired baseline can be unreproducible (`docs/53` §2) | two independent records per build, same batch/protocol, quote ranges; retire unrecorded numbers | prevents a phantom baseline |
| Peak-thread occupancy low **and** bandwidth low (12.5 % of 4096 hardware threads, L3-bound 1.9 %, XVE stalled 92.5 %) | 128 work-groups x 32 lanes = 4096 work-items = 512 hardware threads at SIMD 8; a 32-lane group is one sub-group, so one head per group uses an eighth of the device (`docs/70`) | count `work-items x lanes / SIMD` against 4096 before touching the tile, then put several sub-groups on one head by splitting its reduction | `docs/70`: SDPA 557 -> 427 us, occupancy 12.5 % -> 50 %, 241 -> 314 GB/s, and the win is 0.00 us of barrier |
| More independent work exists and you want it in flight | heads and D-slices are both independent, but a work-group barrier synchronises **every** sub-group in the group (`docs/71`) | add sub-groups *within* one head (split the reduction), never heads *within* one group: that axis only widens the barrier domain | `docs/71`: hpw2 718 us vs base 557; hpw2_split4 456 vs split4 427; the wider group's barrier is 67.5 us vs 44.8 |
| The axis stops paying and the next wider shape will not launch | SLM is 64 KB per work-group, so 16 slices of staging need 90,112 B and the dispatch returns `UR_RESULT_ERROR_OUT_OF_RESOURCES` (`docs/71`) | record the launch failure as a result row (status plus required bytes), then change the structure - shared staging or register reduction - instead of the knob | `docs/71`: split16 = 82.4 % of the 403.1 GB/s roof, split32 unlaunchable, so the axis is closed with data |
| Bytes moved are already the minimum, the kernel is 15 % off the roof, and every thread/barrier change is spent | DRAM efficiency is set by **which addresses are in flight together**, not only by how many: `split16` walked a contiguous 128-key window per sub-group (16 streams x 128 heads apart) and reached 328.5 GB/s; the same bytes, partition, reduction and softmax with each slice owning **interleaved** key blocks reached 390.1 GB/s (`docs/72`) | change only the key-block ownership so the work-group's concurrent reads are adjacent - no arithmetic, no layout, no occupancy change - and re-measure before rebuilding anything | `docs/72`: 408.56 -> 344.02 us in one session, 64.54 us of a 75.60 us gap, then 343.26/343.58 us pinned at 96.9 % of roof; a cooperative-reduce rewrite in the same matrix was worth 11.73 us (18 %), so the order term is not the reduce |
| A GEMM is a wave or more behind and the tile/SLM/barrier knobs are already spent | the same 'which addresses are in flight together' quantity exists one level up: the work-group -> (m-band, n-band) map decides which tiles share a wave and therefore which rows/columns are concurrent. It pays **only** as a conjunction: more than one wave **and** a wide n-band (`GRID_N >= 11`); a wide band alone buys nothing, `GRID_N` alone buys nothing, one wave buys nothing (`docs/77`-`docs/81`) | set `GEMM_SWIZZLE` to a grouped map (`band = gid/(SWZ*GRID_N)`, `bm = band*SWZ + rem%SWZ`, `bn = rem/SWZ`) with SWZ dividing `GRID_M`, and check the wave count and band width first - if the grid fits in one wave, skip it entirely | `docs/81`: at 48 groups / 2 waves / tail 16 / identical 15.73 MB, band 12 gives +2.11 % / +3.28 % while band 8 gives +0.17 % / +0.36 %; `docs/79`: band 11-12 wins, 8-10 silent; `docs/80`: single-wave cells (32, 32 and 24 groups) are all within noise; `docs/81` F (72 groups, 3 waves, band 12) wins in both passes, so it is not a two-wave artefact |
| You measured a memory-order win and want to promise it to another operator | a pure-read microbenchmark with the total bytes held constant is the cheap test of whether the win is a streaming-bandwidth property at all - and here it **did not** reproduce an 18.75 % kernel win (`docs/75`) | run the microbenchmark before generalising; if it does not reproduce, the remaining candidates are the ones it cannot model - per-lane load width (kernel 8 B vs probe 16 B), per-thread outstanding depth, L2 interaction across the kernel's concurrent heads, and the issue rhythm between loads - and every one of them needs the real kernel, not a proxy | `docs/75`: 19 cells x 2 runs, stream-count axis flat to `<=0.62 %` per spacing, the only visible axis is 1.70 % and runs *opposite* to the kernel, link-back verdict `not-reproduced` (kernel +18.75 % vs probe -1.53 %); the same probe is the reason the DRAM byte counter is not quotable (21-23 % of true traffic) |


**Shipping the rule is a separate decision from proving it (P10).** After the rule transfers to a second
tile, the question "turn it on by default?" gets answered by the kernel's own shipped shape, not by the
sweep: here that shape was the quiet cell of the sweep, so the default went to the grouped map with the
adoption stated as *neutral where we run, pays where the grid is wide enough* — plus the condition
converted into the user's units (`N >= win_min x tile BN`) so a deployment below it is told it gets
nothing. Two traps: (1) an adoption page that quotes the sweep's best cell as if it were the shipped
shape is lying by juxtaposition, and (2) the revert path has to be written down as the sanctioned outcome
*before* it is needed, and asserted in the artifact script, or a future regression has no agreed response.

### Swizzling a GEMM's work-group -> tile map (P9, `docs/77`-`docs/81`)

The rule is a **conjunction**, and both halves were measured separately: `GEMM_SWIZZLE=2` pays only when the grid
crosses more than one wave **and** the n-band is wide (`GRID_N >= 11`). Three traps cost this ladder four rounds:

- **An interior optimum, not monotonicity.** At 1024x1536 the factor-2 map wins and factor 4 loses ~11 %, so "more
  compact is better" is false; overshoot is real and shape-dependent (`docs/77`, `docs/80`).
- **Factor 4's loss starts at a different band than factor 2's win** (band 9 vs band 11, `docs/79`): they are two
  different thresholds and must not share one sentence.
- **A wide band without a second wave buys nothing.** Holding groups, waves, tail and bytes fixed and moving only the
  band from 8 to 16 leaves the gain inside noise (`docs/80` A vs B); holding the band fixed and halving the groups
  also leaves it inside noise (`docs/80` C vs D).

The page that ships this also states its own limits: the third wave and a smaller tail are still entangled, and at a
fixed group count `GRID_M x GRID_N = const`, so band width and the M/N swap are inseparable in the best pair
(`docs/81` §6). Quote the conjunction, never the band alone.

The conjunction was then re-tested on a second tile (bf16, `BM=256 BN=192`, `docs/83`): both wide-band cells win in
both passes (+5.90/+6.02 % at 4x12 and +4.98/+4.87 % at 8x12) while both narrow-band cells stay inside noise
(+1.34/+1.21 % at 4x8, one wave; +0.87/+1.32 % at 8x8), so the rule is a property of the map, not of the int8
tile. Two things that round taught about how to phrase it: the axis being tested was the **tile**, not `GRID_M`
(int8 had already won at `GRID_M` 8, so "bigger `GRID_M` is untested" was a wrong worry), and the cleanest
separation holds the **tail** fixed rather than the wave count - the 8x8 / 8x12 pair runs tail 32/32 in both
cells and moves only the band.

P10–P11 (`docs/84`–`docs/89`, `probes/108`–`113`) close the bf16 adoption, hardware-counter mechanism, direct
threshold scan, tail/wave separation, and third-tile invariant:

1. **Conditional adoption on bf16 (`docs/84`, `docs/85`, `probes/108-bf16-adoption`)**: `GEMM_SWIZZLE=2` is the
   shipped default of `bench/bf16_gemm_esimd_v2.cpp`, pinned at 94.38/94.04 us (`bench/records/20261120-200816`,
   `20261120-200822`) vs 94.82/95.18 us before (`20261110-001933`, `20261110-001947`, +0.84 % against a 0.38 %
   noise floor -> `neutral` at the shipped 1024x1536 `GRID_N=8` shape `[MEASURED]`), while winning +5.10 % / +4.98 %
   at `S2` (2048x2304, `GRID_N=12`, 3 waves `[MEASURED]`).
2. **Hardware-counter mechanism (`docs/86`, `probes/110-gemm-swizzle-counters`)**: GT-PIN proves `Int16 & HP Float`
   (2,949,120/launch = 240/thread), `Send` (6,340,608/launch = 516/thread), `Synchronization` (897,024/launch =
   73/thread) and `Control Flow` (835,584/launch = 68/thread) differ by **0 instructions** between `SWZ=1` and
   `SWZ=2` `[MEASURED]`. At 0.05 ms OA sampling, per-launch DRAM reads stay pinned to the cold `A+B` footprint
   (`4.398` vs `4.375` MB against 4.46 MB cold, L3 hit rate 89.69 % vs 89.46 % `[MEASURED]`), falsifying cross-wave
   DRAM re-read. Instead, `SWZ=2` pairs adjacent DSSs onto the same `bn` column (100 % adjacent-pair `B`-column
   sharing vs 0 %, shrinking a 12-WG working set by 35.0 % from 2.621 MB to 1.704 MB), lifting resident L3 read
   bandwidth by +4.63 % (+5.32 % capture-calibrated, 268.93 -> 281.38 GB/s `[MEASURED]`), dropping
   `XVE Array:Stalled(%)` by 2.10 pp (57.60 % -> 55.50 % `[MEASURED]`), and raising resident `XMX active(%)` by
   +3.88 % (+4.69 % capture-calibrated `[MEASURED]`).
3. **Direct bf16 threshold scan, factor-4 overshoot, and tail/wave separation (`docs/87`, `docs/88`,
   `probes/111-bf16-swizzle-scan`, `probes/112-bf16-tail-wave`)**: Scanning `GRID_N in {8, 9, 10, 11, 12}` at
   `M=2048` (`GRID_M=8`) on the bf16 tile brackets `quiet_max=10` (`B8..B10` -0.46 %..+1.49 %) and `win_min=11`
   (`B11` +4.33 %/+4.56 %, `B12` +5.21 %/+4.83 % `[MEASURED]`). Factor `GEMM_SWIZZLE=4` overshoots a single
   32-DSS wave (`SWZ_BAND = 32..48 >= 32`) and trails factor 2 by 1.12..1.91 pp on every winning cell (`docs/87`
   `[MEASURED]`). Orthogonal pairs in `docs/88` prove both partial-tail `T3` (`6x12`, 3 waves, tail 8/32:
   +4.20 %/+4.45 %) and full-tail `T4` (`8x12`, 3 waves, tail 32/32: +5.36 %/+5.45 %) win `[MEASURED]`, while
   under-occupied single-wave `T1` (`512x2304`, `grid 2x12 = 24` WGs < 32 DSS, 8 DSS idle) suffers a
   **-3.11 % / -2.71 % `loss`** (`docs/88` `[MEASURED]`) — never enable `GEMM_SWIZZLE=2` when `WORK_GROUPS < 32`.
4. **Third-tile invariant proof (`docs/89`, `probes/113-gemm-third-tile`)**: On a third tile (bf16
   `BM=256 BN=128 BK=32 TN=16 TM=16`, SLM 49,152 B), the threshold tracks **topological column count
   `GRID_N = N / BN >= 11`** (`K11` N=1408 +5.86 %/+5.83 %, `K12` N=1536 +7.25 %/+6.83 % `[MEASURED]`) rather
   than physical width `N >= 2112`. The identical physical matrix `2048x1536x512` flips from `within-noise` on
   `BN=192` (`GRID_N=8`, `W3`/`B8` +0.50 %..+1.32 %) to a steady `win` on `BN=128` (`GRID_N=12`, `K12`
   +7.25 %/+6.83 % `[MEASURED]`).

### P12: B-operand VNNI prepack duality, SDPA 5-stage stream mechanism, and 6-operator taxonomy

1. **bf16 & int8 GEMM B-operand VNNI prepack (`GEMM_PREPACK=1`) duality, `GRID_M` break-even gate, and `GEMM_SWIZZLE=2` orthogonality (`docs/91`, `probes/115-bf16-prepack/accepted.json`, `docs/94`, `probes/118-operator-anatomy/accepted.json`)**:
   - On shipped `1024x1536x512` (`S1`, `GRID_M=4`), `bf16` `GEMM_PREPACK=1` collapses `3` B-tasks/thread x `2` strided `16` B loads into `1` contiguous `32` B `block_load<uint32_t, 8>`, dropping the GEMM kernel from `94.27` us to **`81.39` us** (breaking the `90.0` us budget by `8.61` us, `-12.88` us / `-13.66%` vs `adopted` baseline `[MEASURED]` `probes/115-bf16-prepack/accepted.json`). `int8` (`BN=128, BK=32, WG=128`, VNNI `k4` grouping) has `B_TASKS_PER_THREAD=1` and saves `3` loads/task (`4x8 B -> 1x32 B`), so across `NK=16` K-tiles both `bf16` (`1*3*16`) and `int8` (`3*1*16`) cut **exactly `48` `Send` instructions/thread** (`bf16` `-9.30%` from `516` to `468`, `int8` `-17.39%` from `276` to `228` vs `PREPACK=0` baseline; total dynamic instructions `-8.31%` / `-354` per thread on `bf16` S1 and `-10.41%` / `-215` per thread on `int8`, while `Int16 & HP Float`, `Synchronization`, and `Control Flow` differ by `0` `[MEASURED]` `docs/91`, `docs/94`).
   - Standalone prepack passes must widen to full or double sector: `bf16` `prepack_col32` (`64` B/load, `14.23` us) beats half-sector `prepack_col8` (`16` B/load, `21.68` us) by `+34.36%` vs `col8` baseline, and `prepack_col16` (`32` B/load, `17.04` us) beats `col8` by `+21.40%` vs `col8` baseline (`[MEASURED]` `probes/115-bf16-prepack/work/prepack.csv`). Single-use unamortized (`R=1`) break-even depends on intra-GEMM `B` row-band reuse `GRID_M = M / BM`: on `S1` (`GRID_M=4`, `B=1,572,864` B), `R=1` end-to-end is `95.62` us (`+1.35` us / `+1.43%` vs `adopted` `94.27` us baseline) and needs `R >= 2` (`88.50` us) to beat `adopted` and `90.0` us; on `S2` (`2048x2304x512`, `GRID_M=8`), `R=1` end-to-end (`18.41 + 160.63 = 179.04` us) directly beats `s2_swz2_nopack` (`184.07` us) by `-5.03` us (`-2.73%` vs nopack baseline `[MEASURED]` `docs/91`).
   - On `S2`, `GEMM_PREPACK=1` and `GEMM_SWIZZLE=2` compose near-orthogonally for a **`+16.80%`** combined speedup (`193.07 -> 160.63` us, `-32.44` us vs `s2_swz1_nopack` baseline; `PREPACK=1` adds `+12.73%` on top of `SWZ=2`, and `SWZ=2` adds `+3.35%` on top of `PREPACK=1` `[MEASURED]` `probes/115-bf16-prepack/work/orthogonal_s2.csv`). Remaining gap: `gemm_prepacked` (`81.39` us) still trails `oneDNN` (`71.96` us) by `9.43` us because per-K-tile `Synchronization` (`73` /thread) is untouched (`[MEASURED]` `docs/91`).
2. **SDPA access-order (`SDPA_ORDER=1`) 5-stage physical mechanism and `H-a` refutation (`docs/92`, `probes/116-sdpa-counters/accepted.json`, `docs/93`, `probes/117-sdpa-stream-mech/accepted.json`)**:
   - High-frequency `0.05` ms OA counters on `s2048_d128` measure per-launch DRAM reads of `129.989` MB (`s2048_ctrl`, `SDPA_ORDER=0`) vs `130.103` MB (`s2048_order`, `SDPA_ORDER=1`) in `overview` (`0.114` MB / `0.09%` diff vs ctrl baseline) and `129.608` vs `130.323` MB in `global-memory-accesses` (`0.715` MB diff vs ctrl baseline), well below `H-a` (`>= 148.7` MB) and refuting cross-wave DRAM re-read (`[MEASURED]` `docs/92`). GT-PIN proves `Send` (`1466`/HW-thread) and `Control Flow` (`66`/HW-thread) change by `0` instructions (`[MEASURED]` `docs/92`), while `SDPA_ORDER=1` lifts resident DRAM read bandwidth by `+20.62%` (`s2048`) / `+30.65%` (`s512`) and cuts `XVE Array:Stalled(%)` by `-2.95` pp / `-6.45` pp vs `ctrl` baseline (`[MEASURED]` `docs/92`). Conversely, `SDPA_REDUCE=2` (`slm_tree`) cuts total instructions `-50.20%` and `Send` `-63.03%` vs `ctrl` baseline, yet raises `XVE Array:Stalled(%)` by `+7.05` pp (`89.17%` resident stall) due to extra barriers and scattered `ORDER=0` streams (`[MEASURED]` `docs/92`).
   - Five-stage controlled microbenchmark `probes/117-sdpa-stream-mech/accepted.json` (`docs/93`) resolves why the `probes/099-dram-streams` pure-read probe (`WG=256`, single `512` MiB array) failed to reproduce (`-1.53%` vs `ORDER=0` baseline): co-residency of `16` slices in one `WG=512` (same DSS) alternating `2` KB blocks across `67.11` MB-separated `K`/`V` arrays already opens **`+9.42%`** (`367.3 -> 401.9` GB/s vs `ORDER=0` baseline in `M0` `[MEASURED]` `docs/93`); narrowing per-lane loads from `16` B to `8` B (`bf16x4`) adds **`+6.32` pp** (to `+15.73%` in `M1` `[MEASURED]`); SLM staging + `16`-slice `group_barrier` adds **`+2.98` pp** (to `+18.71%` in `M2` `[MEASURED]`); online Softmax `sycl::exp` rhythm adds **`+3.11` pp**, closing the full **`+21.82%`** bandwidth win in `M3` (`328.3 -> 399.9` GB/s, `408.85 -> 335.61` us, `-17.91%` time vs `ORDER=0` baseline `[MEASURED]` `docs/93`). At `D=64` (`4` B/lane, `M4`), the gain attenuates by `-5.44` pp yet holds **`+16.38%`** (`336.8 -> 391.9` GB/s vs `ORDER=0` baseline `[MEASURED]` `docs/93`).
3. **A770 six-operator two-domain orthogonal microarchitectural taxonomy (`docs/94`, `probes/118-operator-anatomy/accepted.json`)**:
   - **Domain I — DRAM Stream-Through Bandwidth Domain** (`f32_gemv`, `f32_rmsnorm`, `f32_softmax`, `bf16_sdpa`): `XMX active == 0.0%`, `DRAM/L3` read pass-through **`79.21%`–`99.55%`**, conserved effective total DRAM bandwidth **`310.80`–`400.99` GB/s** (`77.1%`–`99.5%` of the `403.1` GB/s read roof `[MEASURED]` `docs/94`). Custom SYCL leads `oneDNN` by `2.08x` on `f32_gemv` (`216.56` vs `449.97` us), `1.25x` on `f32_rmsnorm` (`90.53` vs `112.90` us), and `2.04x` on `f32_softmax` (`100.52` vs `205.44` us `[MEASURED]` `docs/94`); `f32_softmax` achieves lower DRAM bandwidth than `f32_rmsnorm` (`333.9` vs `371.5` GB/s) because `3`-pass SLM + `exp()` executes `2.62x` dynamic instructions/element vs `f32_rmsnorm` (`1.93` vs `0.74` `[MEASURED]` `docs/94`).
   - **Domain II — L3-Resident XMX Matrix Domain** (`bf16_gemm` S1/S2, `int8_gemm`): `XMX active` **`4.80%`–`18.25%`**, `DRAM/L3` read pass-through **`10.54%`–`19.25%`** (L3 hit rate **`80.75%`–`89.47%`**), separated from Domain I by a **`59.96` pp** empty gap (`[MEASURED]` `docs/94`). `GEMM_PREPACK=1` drops `XVE Array:Stalled(%)` by `4.20`–`6.05` pp across all matrix builds and lifts capture-calibrated L3 read bandwidth by `+14.84%`–`+16.87%` (`[MEASURED]` `docs/94`).

## 3. Operator-specific starting points

Numbers below are pinned `bench/records/*/record.json` entries (standard protocol warmup 20 / samples 20 /
batch 50, per-operator tolerance, 0 errors); the four P3 operators use the `20261107-1521*` set on oneAPI
2026.2 + 2026.1, the P4/P5 entries use their own stamps. Each operator doc (`docs/51`-`docs/61`) carries
the full p10/med/p90 table, `docs/59` is the P4 six-family leaderboard, `docs/60` the SDPA write-up,
`docs/61` the GEMM latency attribution and `docs/62` the P5 summary.

Roofline denominators: quote the **measured** ones from
[bandwidth.md](../hardware/bandwidth.md#measured-bandwidth-patterns) — on this card DRAM read
403.1 GB/s, DRAM write 464.6 GB/s, L2 read 3429.8 GB/s (float4 grid-stride, two independent runs each).
The earlier 291 GB/s figure was a *256 B-message* contiguous-read measurement, not this pattern's roof,
and using it understated every bandwidth percentage (the same SDPA kernel is 26.8 % of 403.1, not
37 % of 291).

- **bf16 GEMM** (`docs/56`, supersedes `docs/53` for the kernel): v2 `TM=16` dual M-chain, tile `BM=256 BN=192 BK=32
  TN=24`, WG=128, 32 work-groups, 95.11-96.00 us (pinned) vs oneDNN `jit:gemm:any` 72.22-72.81 us (1.31-1.33x
  behind). Paired A/B against the v1 `TM=8 TN=48` build: 5/5 rounds faster, mean -2.44 us (2.5 %), i.e. below the
  +-8 % single-run band - adopt only on paired repeats. Counters: -28.6 % operand bytes bought -4.7 % instructions
  and -1.4 % time, XVE stalled 92.5 %, so the residual is the staging->barrier->dpas dependency chain; 64 KB SLM
  blocks a third stage, 24-work-group variants are 6.7-9.1 % slower, and `WG>=256` still hangs (I-003).
- **f32 GEMV** (`docs/51`): 216.30-216.56 us vs oneDNN 447.79-449.97 us (2.07-2.08x ahead), same `A*x` decomposition,
  0/4096 errors, stable across 2026.1/2026.2; the parameter matrix rejected VEC=8 for wrong results.
- **f32 RMSNorm** (`docs/52`): 90.44-90.53 us vs oneDNN `ocl:reusable:vectorized` 112.75-112.90 us (1.24-1.25x ahead)
  after retuning WG to 64.
- **f32 Softmax** (`docs/54`): 100.38-100.52 us vs oneDNN `ocl:reusable` 205.21-205.44 us (2.04x ahead) after
  retuning WG to 256; the opponent has no vectorised kernel for this shape, so do not generalise the win to XMX.
- **bf16 SDPA decode** (`docs/57` for the P4 baseline, `docs/60` for the fix): B=8 H=16 S=2048 D=128.
  `[MEASURED]` the P4 build was 1240.48-1240.78 us, 0/16384 errors, 108.2 GB/s; the **P5 fix**
  (`SDPA_REDUCE=1`: stage the 32 lane shares in SLM, one work-group barrier, read back as `float4`)
  is **557.06/557.26 us, 0/16384 errors, 240.9 GB/s**, i.e. 2.22x over the P4 build - but it is two
  rewrites behind: the adopted build is **343.26/343.58 us** (`SDPA_ORDER=1`, see the parallel
  ladder below). Root cause of the P4 wrong-value variants: `reduce_over_group` is lowered per SIMD pass, so
  under register pressure IGC narrows the kernel to SIMD16/8 and each lane loses half or three
  quarters of the 128-dim dot; the tell is `reduction_width` 16 of 32 lanes in a per-lane dump and
  the missing float `add (16|M0)` stage in the listing — see
  [reductions-scan.md](../techniques/reductions-scan.md#sub-group-reduce-syclreduce_over_group--check-the-compiled-width-first).
  **P6 correction (`docs/65`)**: the "8-key unroll is 451-456 us but wrong" result of P4 was the
  *reduction* defect, not the block size. Re-swept on the fixed kernel, all nine (unroll, prefetch)
  points are 0/16384 errors and the best is still the adopted `UN=8, PREFETCH=0` at 557.91 us; the
  next point is 578.37 us. `SDPA_PREFETCH=1` is not even an axis — in the source it and the default
  branch are the same loop, so the kernel never prefetched anything. Exhaust the knob grid once, on
  a *correct* build, before believing any tuning story told against a wrong one.
  oneDNN 3.13.2 has no SDPA primitive, so this is a roofline, not a ratio. Cost accounting for the
  fix: the barrier alone is 35.31 us, and pinning the width with `reqd_sub_group_size(32)` instead
  costs 3705 us (6.6x).
  `[MEASURED]` **The SDPA parallel ladder (`docs/70`, `docs/71`) supersedes the 557 us build**: the
  adopted kernel is now the *same* math with the 128-dim reduction split across 16 sub-groups of one
  work-group (one head per group, 16 slices), at **402.91/403.85 us, 0/16384 errors, 332.3 GB/s =
  82.4 % of the 403.1 GB/s roof**, 1.21x the 332.96 us floor and 6.15 us inside a 410 us target that
  carries no attribution clause. The lever was not bandwidth: VTune reported XVE stalled 92.5 %,
  L3-bound 1.9 % and peak-thread occupancy 12.5 % on the 557 us build, i.e. the mapping had an eighth
  of the device resident. 557 -> 427 us came from `split4`, and 427 -> 403 us from `split16`
  (22.98 us, 1.057x), which spends the 4096-thread budget twice over. Two structural facts came out
  of it: (a) each shape was measured against its own barrier-removed twin, and `split4`'s own barrier
  is **43.81 us versus 34.29 us** on the old build, so the 129.56 us mapping win contains **0.00 us**
  of barrier saving - the claim survives because the twin is more expensive, not because it was
  argued; (b) the absolute barrier total falls as the split grows (44.8 -> 21.4 -> 20.0 us for
  split4/8/16) although the group gets wider - but only because each sub-group runs half as many
  barrier rounds (64 / 32 / 16), while the price of ONE round is flat from the 128- to the 256-thread
  group (0.70 / 0.67 us) and *dearer* at the 512-thread group (1.25 us); never read a falling total
  as a cheaper rendezvous.
  `[MEASURED]` **Slices must not share staging**: staging the prefix/sum shares once per group
  measured **7398 and 7050 wrong of 16384** values, while one staging buffer per sub-group measured
  0/16384 in both runs. A split structure is a correctness contract first, a bandwidth trade second.
  `[MEASURED]` **The last 51.21 us were DRAM access order, not extra bytes (`docs/72`)**: with the
  barrier removed the kernel still ran 384.17 us, and the byte audit says it moves exactly the
  134.2 MB of K+V (2 x 128 x 2048 x 128 x 2 B), so the floor is 332.96 us. Handing each sub-group a
  contiguous 128-key window made 2048 sub-group streams read 128-head-spaced addresses;
  **interleaving the key blocks** across the 16 slices - same bytes, same partition, same reduction,
  same softmax - moved the same kernel from 328.5 to **390.1 GB/s**. Adopted `SDPA_ORDER=1`:
  **343.26/343.58 us, 390.7 GB/s = 96.9 % of the 403.1 GB/s roof, 0/16384 errors**, 10.6 us above the
  floor, i.e. 38 % faster than the P5 build. A cooperative-SLM-reduce rewrite in the same matrix was
  worth 11.73 us (18 % of the order term), so the reduce was not the term either. Two transferable
  lessons: (a) when a streaming kernel is 15 % off the roof with the bytes already minimal, suspect
  the **set of addresses in flight together** before rebuilding the pipeline; (b) the DRAM byte
  counter on this box (`-knob collect-memory-bandwidth=true`) is unavailable without the sampling
  driver, and the user-mode counter under-reports to 21-23 % of the true traffic - it cannot
  adjudicate a byte question, so state the negative instead of quoting it.
  **(c) The rule is measured on four shapes, and it tracks stream count, not stream length** (`docs/74`):
  the same bytes-for-bytes A/B transfers everywhere - `S=2048/D=128` 402.23 -> 338.13 us (-15.9 %),
  `S=512` 113.24 -> 92.45 (-18.4 %), `S=8192` 1574.56 -> 1321.13 (-16.1 %), `D=64` 193.41 -> 172.66
  (-10.7 %) - so the effect survives a 16x change of the contiguous window (8 KB to 128 KB) and only
  attenuates when the per-lane load halves. Two of the four pre-registered predictions missed and
  both misses are what rule out the "window length" reading, which is why the axis is stated as *the
  number of concurrent key streams a head holds open*; no shape regressed, so `SDPA_ORDER=1` stays a
  compile-time constant with no shape-dependent branch. It is still not a dispatch for a different
  operator or a different key layout: `docs/75` shows a pure-read microbenchmark does **not**
  reproduce the 18.75 %, so a different access pattern must repeat the same-bytes A/B in the kernel
  itself, and the last 10.6 us of the adopted shape has no instrument that can attribute it.
- **int8 GEMM** (`docs/58`): ESIMD `dpas` 8x8x32 (`SystolicDepth 8, RepeatCount 8, ES 8, K = 32`), tile
  BM=256 BN=128 BK=32 TN=16 TM=16, 92.24/92.49 us vs oneDNN `jit:gemm:any` 68.38/68.50 us (1.35x behind),
  both sides exact against one CPU int64 accumulation (0/1572864). int8 is **not** dpas-throughput bound: it
  needs half the dpas of bf16 (393,216 vs 786,432) yet takes the same time, i.e. 5 % of the 82 Gdpas/s ceiling;
  the tile rules differ from bf16 because the VNNI group is four k values (BM and BN must both be multiples of 128,
  so 32 work-groups is unreachable - 48 was the measured best).
  `[MEASURED]` **P6 attribution (`docs/64`, `docs/67`)**: ablations price the 24.03 us gap one term at a time —
  the global->register->SLM feed 29.73 us, the global reload 25.36, the per-K-tile barrier 22.49, the dpas work
  21.92 — but those four sit on one dependency chain and cannot be summed, and none of them is removable. The one
  removable term is the **B-operand VNNI pack** (4 strided 8 B loads + 3 shifts/ors per dword): 12.43 us = 51.7 %
  of the gap. Paying it in a separate pass works, but only if the pass is widened: at the per-tile granularity
  (`cols=8`, 25 % of a 32 B sector per load) the pass costs 15.09 us and the end-to-end result is a 2.04 us
  **loss**; at `cols=32` it costs 11.10 us and the GEMM drops to 78.84 us, for 89.94 us end to end (-1.95 us vs
  91.89, still 1.32x oneDNN). So the lever is the *pass* (sector width), not the GEMM: pre-packing halves the
  packing work and can lose the whole win in the copy that feeds it.

`[MEASURED]` **P5 latency attribution for the bf16 GEMM** (`docs/61`): the adopted tile's worst pinned median is
95.18 us against a 90 us budget — 5.18 us short (5.8 %) — and the term that holds it there is the per-K-tile
work-group barrier, priced at **22.42 us = 4.33x the gap** by deleting it from a copy of the committed source
(numerically wrong by design, 1,498,300 bad outputs). Freeing the barrier is not a parameter: 57,344 B of
double-buffered staging is a whole DSS's SLM budget, so buffer reuse needs 16 barriers per work-group, and
one 128-thread group per DSS means the XVE array is stalled 92.4 % of the window. Five buildable alternatives
(BK=16 four-stage, BN=256, two 64-thread groups per DSS) all measured slower (97.7-118.7 us) and two more were
rejected by `static_assert`. So: price the gap with ablations before expanding the tile space — and do not
report a 5.8 % miss as a win because an attribution clause accepts it.

`[MEASURED]` **Do not pipeline a load across a work-group barrier without counting spills** (`docs/66`).
Hoisting the next SDPA key block into registers so the barrier waits on it instead of the load costs **3x**:
556.45 us becomes 1763.47 us (`pipe_reg`, 55 scratch-block writes / 30 reads) or 1567.12 us (`pipe_k`,
11/5), while the adopted build spills 0/0. The tell that the trick never applied: deleting the barrier from
the pipelined builds no longer buys anything (on `pipe_k` it is *slower*), i.e. the barrier was never waiting
on the memory it supposedly hid. Before believing any software-pipelining win on this card, read
`scratch_write`/`scratch_read` out of the AOT listing — a spill pair is 1-2 orders of magnitude more
expensive than the latency it was meant to hide.

## 4. What this playbook deliberately does not cover

- Dual issue, indirect-addressing cost and join/call stack depth: still blocked on per-type issue counters and
  cycles (`docs/22`, `docs/50` §6). Record them as negatives with the required instrument, do not rebuild the
  old vehicles.
- 256-GRF mode, bindless (B7) and L1/L2 associativity: measured as not worth retrying on this box; see `docs/22`.
