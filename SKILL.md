---
name: A770-Kernel-Optimization
description: "Optimize SYCL/ESIMD compute kernels on Intel Arc A770/DG2 (Windows). Use measured constants and dispatch rules only for A770; for other Intel GPUs transfer the methodology only and re-measure all hardware-specific assumptions. Covers dense GEMM/GEMV/RMSNorm/Softmax, flash attention / scaled dot-product attention (SDPA), irregular and sparse shapes, reductions and scans, attention and convolution, numerical tolerance, bandwidth and roofline, launch/fusion/graph overhead, compiler flags and codegen, VTune interpretation, robustness/TDR isolation, and oneMKL/oneDNN baselines."
---

# Arc A770 Kernel Optimization

Use this skill as a workflow and guidance layer for optimizing SYCL/ESIMD
kernels on Intel Arc A770 (Xe-HPG/DG2). Do not treat it as a full optimizer.

Use every measured value only inside the validity domain stated by its source
page. Most core campaigns used oneAPI 2026.1 and driver `32.0.101.8724` on
Windows. The contiguous-DRAM and paired RMSNorm pages explicitly record later
measurements on driver `32.0.101.8860`. For any other GPU, driver, compiler,
or library stack, transfer the methodology only and re-measure all
hardware-specific assumptions. Do not name a production kernel or transfer an
A770 dispatch boundary to a non-A770 device before measuring that target;
treat every A770 route only as a benchmark candidate there. For a non-A770
request that asks what to ship, state that this skill cannot make that
selection. Do not recommend either a library or custom route for production
until it passes target-device correctness and performance measurements.

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
| Flags, unrolling, AOT, large GRF, or IR/GEN artifacts | [codegen](references/workflow/codegen.md) |
| New or risky ESIMD shape, hang, device-loss, or TDR risk | [robustness protocol](references/workflow/robustness.md) |
| Known failures, correctness traps, or API gates | [pitfalls and negative results](references/workflow/pitfalls.md) |
| API forms, builds, profiling, or library primitives | [API usage](references/api/api-usage.md) |
| Copy-ready kernel cores | [code snippets](references/api/code-snippets.md) |
| Optional existing benchmark/correctness helpers | [automation helpers](references/workflow/automation.md) |
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
per-experiment schema. The helpers under `scripts/` are optional, not a
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
| [Automation helpers](references/workflow/automation.md) | Optional benchmark, correctness, baseline, VTune, watchdog, and record helpers already present under `scripts/` |
| [Reference index](references/index.md) | Complete topic and campaign map |
| [Methodology](references/workflow/methodology.md) | New-operator thinking, profiling-driven next experiments, and model updates after failures |
| [Skill evaluation](references/workflow/evaluation.md) | Behavior evaluation and applied text audit for maintaining this skill |

Do not assume full campaign sources are bundled. The skill contains thin
helpers and minimal examples; campaign-specific runners remain outside the
skill where their reference page says so.
