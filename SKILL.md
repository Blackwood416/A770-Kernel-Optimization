---
name: A770-Kernel-Optimization
description: "Optimize SYCL/ESIMD compute kernels on Intel Arc A770/DG2 (Windows). Use measured constants and dispatch rules only for A770; for other Intel GPUs transfer the methodology only and re-measure all hardware-specific assumptions. Covers dense GEMM/GEMV/RMSNorm/Softmax, flash attention / scaled dot-product attention (SDPA), irregular and sparse shapes, reductions and scans, attention and convolution, numerical tolerance, bandwidth and roofline, launch/fusion/graph overhead, compiler flags and codegen, VTune interpretation, robustness/TDR isolation, and oneMKL/oneDNN baselines."
---

# Arc A770 Kernel Optimization

Use this skill as a workflow and guidance layer for optimizing SYCL/ESIMD
kernels on Intel Arc A770 (Xe-HPG/DG2). Do not treat it as a full optimizer.

Use every measured value only inside the validity domain stated by its source
page. Early core campaigns (2026-09) used oneAPI 2026.1 and driver
`32.0.101.8724` (or `32.0.101.8860` for contiguous-DRAM and paired RMSNorm).
The P10-P14 ISA, operator-surpassing, and Level Zero/Graph runtime campaigns
(`docs/84`-`docs/102`, `probes/108`-`probes/126`, claims `C0321`-`C0355`) used
Intel Arc A770 (DG2), Windows 11 25H2, oneAPI 2026.2, driver `32.0.101.9150`,
and Level Zero SDK `1.33.1` (driver `1.17.39894+4`). For any other GPU, driver,
compiler, or library stack, transfer the methodology only and re-measure all
hardware-specific assumptions. Do not name a production kernel or transfer an
A770 dispatch boundary to a non-A770 device before measuring that target;
treat every A770 route only as a benchmark candidate there. For a non-A770
request that asks what to ship, state that this skill cannot make that
selection. Do not recommend either a library or custom route for production
until it passes target-device correctness and performance measurements.

## 1-Minute A770 Engineering Quick Sheet (P12-P14 + Aurex v0.2)

Use this before opening deep reference pages (`docs/94`-`docs/102`, `C0329`-`C0355`, `Aurex v0.2` `commit bf81332`):

- **Classify the kernel into Domain I or Domain II first (`docs/94`)**:
  - **Domain I (DRAM stream-through: GEMV, RMSNorm, Softmax, SDPA)**: `XMX=0%`,
    `DRAM/L3` pass-through `79%-99.5%` `[MEASURED]`. Target the `403.1 GB/s`
    float4 read roof (`probes/087-roof/work/roof.csv`) via wide `vec16` loads,
    interleaved concurrent address streams (`SDPA_ORDER=1`), and zero register
    spill; standard SYCL sub-group kernels suffice.
  - **Domain II (L3-resident XMX matrix: bf16/int8/4-bit GEMM & SwiGLU)**: `DRAM/L3`
    `10%-19%`, L3 hit `81%-89%` `[MEASURED]`. Unlock **`GRF=256` + `WG=32` +
    `cab4` (`4 x 12 KB = 48 KB` SLM = `2 WGs/Xe-core = 64/64` HW threads)** via
    `sycl::ext::intel::experimental::grf_size<256>`, `split_barrier` decoupling,
    `0-Dummy` `dpasw<8,8>` (`2x8`/`3x4` grouping, `dummy=1`), `bp_vec` (`-94 ins`),
    `L1StreamB` (`st.ca`, `K>=4096` `swizzle<=4`), `wb.wb` C store (keep `st.uc`
    for ResidualAdd), and ESIMD VC `-Xfinalizer` 4-flag optimum (`-SBIDBundle
    -SWSBDepReduction -nopresched -twoSrcBundleBCR`): reaches **`92.5 TFLOPS`** on
    `4096^3` (`1.10x` vs oneDNN) and **`33–35 WIN | 4–6 TIE | 0 LOSE`** across 39
    shapes vs `oneDNN 3.13.2` (including `SwiGLU` `79.6 TFLOPS`, `INT8` `129.6 TOPS`,
    and `4-Bit` `BK=gs=64` online dequant `22.49–24.57 TFLOPS`).
  - **Domain I Tier-3 fusion & Decode baselines**: In-process `ocloc64.dll` ->
    `zeModuleGetNativeBinary` `.zebin` SHA-256 cache (`ZE_MODULE_FORMAT_NATIVE`)
    fuses 5-op `1024x4096 f32` chains to `399.6 GB/s` (`99.1%` roof, `2.73x` vs
    unfused baseline `[MEASURED]`); `bf16` Decode SDPA (`ORDER=1`) hits `386.0 GB/s`
    (`95.7%` roof); `NF4`/`INT4Sym` Decode GEMV hits `40.1 / 27.7 us` (`2.50x / 3.62x`
    vs `bf16` GEMV).
  - **Production spill & SBID iron rule**: `KernelDiag.spillMemSize == 0` has
    false negatives (`acc[4][6]` spills `35–67` `send.dc0`; runtime `simd[]`
    indexing breaks SROA with `priv=576–1088 B`). Always verify `spill == 0 &&
    priv == 0` AND `ocloc disasm` main-loop `send.dc0 == 0` with `Short UGM waits <= 1`.
- **Runtime, Graph, Copy Engine, and Fusion rules on Windows WDDM (`docs/99`-`docs/102`)**:
  - **Never enable `UR_LOADER_USE_LEVEL_ZERO_V2=1` globally on uncaptured dynamic
    loops or dual-SYCL-queue Copy overlap**: uncaptured `L=256` chains slow down
    `7.93x` vs `sycl_v1_inorder` (`1236.70 -> 9810.75 us`) and dual-SYCL
    Compute+Copy hardware overlap collapses from `99.95%` to `<= 0%` `[MEASURED]`
    (`docs/99`, `docs/100`).
  - **Best zero-tax execution layer = Native Level Zero Regular Command List +
    explicit `ordinal 1` Copy Engine**: `l0_reg_replay` hits `317.15 us` at
    `L=256` (`1.239 us/kernel`, `0.484 us/edge` hardware gap, `1`-replay
    break-even; use `zeCommandListAppendBarrier`, not per-node `zeEvent`), and
    `ordinal 1` Copy Engine holds `91.6%-99.95%` Compute+Copy overlap `[MEASURED]`
    (`docs/99`, `docs/100`).
  - **Single-CCS `0.00%` compute concurrency**: A770 exposes `numQueues=1` on
    `ordinal 0`; multi-queue small kernels serialize strictly in hardware
    (`0.00%` timestamp overlap `[ARCH]` `docs/101`). Overlap small compute ops
    only via horizontal kernel fusion.
  - **Host-GPU doorbell & memory pools**: only `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`
    passes GPU-polled coherency (`10/10`, `6.00 us` RTT = `13.0x` faster than
    `78.00 us` event-split, `0` DMA bandwidth loss `[MEASURED]` `docs/99`,
    `docs/101`), but calibrate spin caps against `1.12 us/spin` to avoid WDDM
    TDR. Pre-allocate a Device Arena (avoiding UMF's `>= 16 MB` `636-1863 us`
    cliff) and reuse a static `zeEventPool` (`zeEventHostReset` `0.10-0.20 us`
    `[MEASURED]` `docs/101`).

## Scope Gate

Before selecting or changing a kernel:

1. Confirm the target is A770/DG2. Otherwise use only the workflow and require
   target-device measurements before making a production selection.
2. Write the exact operator semantics and layout: `A*x` versus `x*W`,
   `[M,K]` row-major versus packed VNNI, NCHW versus NHWC, causal versus full
   attention.
3. Record dtype, shape, alignment, device, driver, compiler, and library
   versions.
4. Choose device time, wall time, or full pipeline time as the deployment
   target. If the target is unspecified and champions differ, report both
   device and wall results.
5. Load the matching operator reference and only the applicable
   cross-cutting references below.
6. Apply the evidence, benchmark, and library-baseline contracts before
   reusing a value, promoting a rule, or reporting a speedup.
7. Separate sourced rules from new design proposals. If an implementation
   detail, threshold, tolerance, or dispatch rule is not stated in the loaded
   references, label it `[HYPOTHESIS]` and test it before recommending it.

## Route the Task

### Operator Router

| Task | Open first | Open next when needed |
|---|---|---|
| Dense GEMM ladder | [dense operator ladders](references/techniques/techniques.md) | [GEMM/GEMV shape crossover](references/techniques/gemm-shape-crossover.md) for production selection |
| GEMM/GEMV production dispatch | [GEMM/GEMV shape crossover](references/techniques/gemm-shape-crossover.md) | [boundary interpolation](references/techniques/gemm-crossover-boundary.md) for decision edges and unswept gaps |
| GEMV-N1, `A[M,K] * x[K]`, aligned fast-path shape | [GEMV-N1 ladder](references/techniques/techniques.md#f32-gemv-n1-ladder-4096x4096-row-major-a) | [GEMM/GEMV shape crossover](references/techniques/gemm-shape-crossover.md) for measured production selection |
| GEMV-N1 with short, odd, or non-16-aligned K | [irregular shapes](references/techniques/irregular-shapes.md#gemv-n1-shape-cost) | Use the [GEMV-N1 ladder](references/techniques/techniques.md#f32-gemv-n1-ladder-4096x4096-row-major-a) only to identify candidate families; keep any new padding/tail implementation `[HYPOTHESIS]` until measured |
| GEMV-M1, `x[M,K] * W[K,N]`, INT4/NF4 decode | [weight-only GEMV](references/weight-only-gemv.md) | [API usage](references/api/api-usage.md) for the oneDNN u4 form |
| RMSNorm | [RMSNorm shape sweep](references/rmsnorm-shape-sweep.md#dispatch-rule) | [paired/interleaved verification](references/rmsnorm-paired-bench.md) for noisy or close decisions; [dense ladder](references/techniques/techniques.md#f32-rmsnorm-ladder-1024x4096-row-major-x-f32) for the measured 1024x4096 path |
| Softmax | [Softmax ladder](references/techniques/techniques.md#f32-softmax-ladder) | [irregular shapes](references/techniques/irregular-shapes.md) for short or odd columns |
| Irregular gather/scatter or sparse GEMM | [irregular shapes and sparse GEMM](references/techniques/irregular-shapes.md) | [numerics](references/techniques/numerics.md) when dtype or tolerance changes |
| Full reduction or work-group scan | [reductions and scans](references/techniques/reductions-scan.md#selection-rules) | [numerics](references/techniques/numerics.md) for accumulator and tolerance constraints |
| Flash attention / SDPA, dense multi-row Q/K/V | [A770 SDPA port and optimization](references/techniques/sdpa-a770.md) | [attention and convolution](references/techniques/attention-conv.md) for the small prefill ladder; use the decode page only for `Q=1` |
| Prefill attention or direct convolution | [attention and convolution](references/techniques/attention-conv.md) | Use the separate decode page only for `Q=1` |
| Decode attention, `Q=1`, MHA/GQA/MQA or paged KV | [decode attention](references/techniques/attention-decode.md#decision-rules) | Do not substitute the prefill campaign for this route |

Keep the GEMV-N1/GEMV-M1 and prefill/decode distinctions explicit. They are
different operators and measurement campaigns.

### Cross-Cutting Router

| Situation | Open |
|---|---|
| Hardware topology, SLM/GRF/cache/XMX constraints | [hardware](references/hardware/hardware.md) |
| Bandwidth-pattern choice and empirical roofline | [bandwidth patterns](references/hardware/bandwidth.md), [contiguous DRAM roofline](references/hardware/dram-contiguous-roofline.md) |
| Tiny kernels, host overhead, fusion, queues, or graphs | [execution model](references/workflow/execution.md) |
| INT2/INT4 DPAS, limited/full graph, or dynamic parameter blocks | [DG2 low-bit and graph capability](references/workflow/dg2-lowbit-graph.md) |
| Flags, unrolling, AOT, large GRF, or IR/GEN artifacts | [codegen](references/workflow/codegen.md) |
| New or risky ESIMD shape, hang, device-loss, or TDR risk | [robustness protocol](references/workflow/robustness.md) |
| Known failures, correctness traps, or API gates | [pitfalls and negative results](references/workflow/pitfalls.md) |
| API forms, builds, profiling, or library primitives | [API usage](references/api/api-usage.md) |
| Copy-ready kernel cores | [code snippets](references/api/code-snippets.md) |
| Optional existing benchmark/correctness helpers | [automation helpers](references/workflow/automation.md) |
| Why a kernel behaves this way: DPAS limits, scoreboard, regions/banks, send encoding | [dpas/xmx ISA](references/isa/dpas.md) first, then [swsb](references/isa/swsb.md), [regions and banks](references/isa/regions-banks.md), [LSC messages](references/isa/lsc-messages.md), [addressing](references/isa/addressing.md) |
| Can I even measure this on the A770? | [references/workflow/instruments.md](references/workflow/instruments.md) |
| Given what the ISA says, what should I change? | [references/workflow/isa-to-decision.md](references/workflow/isa-to-decision.md) |
| I have a measured symptom (stall/occupancy/drop) and need the next action | [references/workflow/performance-playbook.md](references/workflow/performance-playbook.md) |
| Which ISA features can I actually write from source? | [references/workflow/isa-capability-matrix.md](references/workflow/isa-capability-matrix.md) |
| Barrier/fence structure or sync cost | [sync, fences, gateway](references/isa/sync-fences.md) |
| Shared-memory (SLM) layout, bank conflicts, padding | [SLM banks](references/hardware/slm-banks.md) |
| Texture/sampler or typed-surface message encoding | [typed/sampler channel](references/isa/typed-sampler.md) |
| Patching instruction bits / exec size / quarter / regions | [instruction encoding](references/codegen/encoding.md) |
| Which compiler flag changes the generated code (or does nothing) | [compiler knobs](references/codegen/knobs.md) |
| Hand-written or hand-edited GEN assembly on this box | [asm control loop](references/codegen/asm-control.md) |
| Occupancy looks fine but throughput does not scale | [occupancy and XMX saturation](references/hardware/occupancy-saturation.md) |
| Reading hardware ids (slice/EU/thread) or `sr0` | [sr0 fields](references/hardware/sr0-hwtid.md) |
| Cache sizing, tile footprint, pointer-chasing loops | [cache geometry](references/hardware/cache-geometry.md) |
| Cost of `math.exp`/`log`/`rsqrt` versus DPAS work | [extended math](references/isa/math-numerics.md) |
| Instruction-level microbenchmark design or a suspiciously silent run | [benchmarking method](references/workflow/benchmarking.md) |
| New-operator reasoning or a failed experiment | [methodology](references/workflow/methodology.md) |
| This skill is being audited or updated | [skill evaluation](references/workflow/evaluation.md) |

Use [the reference index](references/index.md) as the complete topic map.

## Evidence and Validity Contract

Label every claim:

| Label | Meaning |
|---|---|
| `[ARCH]` | Xe-HPG architectural constraint; expected to hold for DG2 unless toolchain behavior changes |
| `[MEASURED]` | A result measured on the stated A770 stack with an exact shape and protocol |
| `[HEURISTIC]` | A starting point derived from measured campaigns, not a dispatch rule |
| `[DISPATCH]` | A rule validated across a defined shape range and safe only inside its validity domain |
| `[BUG]` | Observed, version-specific toolchain or driver behavior |
| `[CORRECTNESS]` | Implementation correctness constraint or verified layout, stride, tail, offset, or lane-coverage trap |
| `[TOOLCHAIN]` | SYCL/ESIMD/API/compiler capability or limit that may change after an upgrade |
| `[HYPOTHESIS]` | Insufficiently validated; test before use |

Require every `[DISPATCH]` rule to state operator, dtype, shape range,
alignment, tested rows, device, oneAPI version, and confidence. Re-label a
`[MEASURED]` result as `[HEURISTIC]` when using it outside its measured
shape. After a toolchain upgrade, re-probe `[BUG]` and `[TOOLCHAIN]`,
re-check `[CORRECTNESS]`, and treat `[ARCH]` as expected to persist.

## Optimization and Verification Loop

1. Build the correctness harness first: CPU reference, `errors: 0/...`,
   warmup plus timed loop, then 3 or more stable runs.
2. Measure the same operator with oneMKL/oneDNN and verify the library output
   against the CPU reference. A fast library call may implement a different
   operator; for example, oneDNN `1xK` computes `x*A`, not row-major `A*x`.
3. Measure launch/host overhead and select the matching bandwidth/roofline
   model before restructuring.
4. Profile with VTune before restructuring. Start with `instruction-count`
   and escalate to `full-compute` only when needed.
5. Change exactly one structural variable per experiment and keep failed
   variants as standalone files.
6. Before promoting a discrete sweep into an inequality dispatch, sample
   boundary points around every decision edge. Keep unswept values
   `[HEURISTIC]`.
7. Verify every variant with the exact reference comparison, 3 stable
   values, and the VTune metrics that motivated the change. Record negative
   results.

Before changing compiler flags, read
[codegen.md](references/workflow/codegen.md). Before running a new or risky
ESIMD shape, follow [robustness.md](references/workflow/robustness.md) and run
one GPU pressure target at a time.

Keep the workflow light. The goal is evidence discipline, not an elaborate
per-experiment schema. The helpers live in `tools/harness/` of the research repo, not in this
skill, and are optional, not a
mandatory pipeline. Verify a helper exists before claiming it was executed.

### Timing Contract

Use this protocol for new experiments:

```text
warmup: operator-specific
samples: >= 20 batch measurements
reported: median
dispersion: p10/p90 or MAD
reject/flag: CV > threshold

device_time: SYCL event profiling (command_start/command_end)
wall_time: host submit -> wait completion
pipeline_time: preprocessing + operator + postprocess
```

For kernels near or below 10 us, report device time and wall time separately
because launch and host overhead can dominate. When reproducing a historical
campaign, preserve the protocol recorded on that campaign page rather than
silently relabeling its timing basis.

### oneDNN Baseline Contract

Record every oneDNN baseline with:

- operator semantics and primitive kind
- implementation string (`jit:gemm:any`, `ocl:ref:any`, and so on)
- format tags and src/weight/dst dtypes
- post-ops and `fpmath_mode`
- runtime/static dimensions and whether reorder/preprocessing is included
- device time, wall time, and CPU-reference correctness
- accuracy class (`matched`, `relaxed_matched`, `fastest_only`, `invalid`, or
  `unknown`)
- executable-reported `rel_tol`, `abs_tol`, `max_rel_err`, `reference`,
  `semantics_id`, `accuracy_mode`, and `comparable_for_speedup`

Enable `ONEDNN_VERBOSE=profile,dispatch` and preserve the implementation
string. Do not compute a speedup against a baseline that fails the required
tolerance. Classify it as a fastest-library performance lower bound only
when operator semantics are confirmed and the failure comes from a
deliberately relaxed math mode. A FAIL without confirmed semantics is
`invalid`, not `fastest_only`. Require the executable to report the tolerance
it actually used; a mismatch is `CORRECTNESS_CONTRACT_MISMATCH` and must not
be emitted as `[MEASURED]`.

## Safety Before Execution

- Run risky ESIMD shapes in isolated one-shot processes with the existing
  watchdog protocol. Do not run multiple GPU pressure processes together.
- On A770, keep ESIMD attention work-groups small: `WG=512` TDRs even for a
  trivial kernel, and any non-zero compiler spill with RPT>4 is a device-lost
  trigger. Follow [the SDPA reference](references/techniques/sdpa-a770.md) and
  [robustness.md](references/workflow/robustness.md) before trying new
  attention geometry.
- Treat hangs, non-zero exits, new minidumps, TDRs, and device loss as
  evidence. Stop stacking risk probes and record the versioned negative
  result.
- Re-probe version-specific `[BUG]` and `[TOOLCHAIN]` claims after an upgrade.
- Block performance comparisons when operator semantics, correctness, or the
  requested/executable tolerance contract does not match.
- Preserve failed variants and negative results; do not delete them merely
  because they failed.

## Implementation Resources

| Resource | Use |
|---|---|
| [API usage](references/api/api-usage.md) | API forms, build/profile commands, oneMKL/oneDNN baselines, graph/dual-queue submission, IR dump, and watchdog usage |
| [Code snippets](references/api/code-snippets.md) | Copy-ready building blocks for the measured campaigns |
| [Automation helpers](references/workflow/automation.md) | Optional benchmark, correctness, baseline, VTune, watchdog, and record helpers, which live in `tools/harness/` of the research repo |
| [Reference index](references/index.md) | Complete topic and campaign map |
| [Performance playbook](references/workflow/performance-playbook.md) | Symptom -> ISA fact -> action -> expected effect, plus the four operators' current numbers and dead ends |
| [Methodology](references/workflow/methodology.md) | New-operator thinking, profiling-driven next experiments, and model updates after failures |
| [Skill evaluation](references/workflow/evaluation.md) | Behavior evaluation and applied text audit for maintaining this skill |

Do not assume full campaign sources are bundled. The skill contains thin
helpers and minimal examples; campaign-specific runners remain outside the
skill where their reference page says so.
