# Reference Index

Start with [`SKILL.md`](../SKILL.md) for task routing, evidence labels, the
optimization loop, timing rules, and the oneDNN baseline contract. Then load
only the operator page and cross-cutting pages that apply to the task.

Treat the environment and validity domain on each reference page as
authoritative. Early core campaigns used Intel Arc A770 (DG2), Windows 11,
oneAPI 2026.1, and driver `32.0.101.8724` (with `32.0.101.8860` on the
contiguous-DRAM and paired RMSNorm pages). The P10-P14 ISA, operator-surpassing,
and Level Zero/Graph runtime campaigns (`docs/84`-`docs/102`, claims
`C0321`-`C0355`) used Windows 11 25H2, oneAPI 2026.2, driver `32.0.101.9150`,
and Level Zero SDK `1.33.1` (`1.17.39894+4`). Re-measure before transferring a
conclusion to another GPU, driver, compiler, library build, shape, or layout.

## Operator Decisions and Measured Ladders

| Need | Open | Contains |
|---|---|---|
| Dense GEMM/GEMV/RMSNorm/Softmax techniques | [techniques/techniques.md](techniques/techniques.md) | Measured optimization ladders and VTune interpretation |
| GEMM/GEMV selection across shapes | [techniques/gemm-shape-crossover.md](techniques/gemm-shape-crossover.md) | f32/bf16 routes, dispatch map, validity domains, and library contract |
| Continuous GEMM/GEMV decision edges | [techniques/gemm-crossover-boundary.md](techniques/gemm-crossover-boundary.md) | Boundary interpolation and explicitly unswept gaps |
| RMSNorm selection across dtype/shape | [rmsnorm-shape-sweep.md](rmsnorm-shape-sweep.md) | Wall/device champion maps, tie handling, and production dispatch |
| RMSNorm close or noisy comparisons | [rmsnorm-paired-bench.md](rmsnorm-paired-bench.md) | Paired/interleaved deltas, champion flips, and tie evidence |
| Weight-only GEMV-M1 | [weight-only-gemv.md](weight-only-gemv.md) | INT4/NF4 routes, accuracy-aware oneDNN baseline, and negative paths |
| Odd, short, gathered, scattered, or sparse shapes | [techniques/irregular-shapes.md](techniques/irregular-shapes.md) | Fast/fallback gates and sparse GEMM measurements |
| Reduction or scan | [techniques/reductions-scan.md](techniques/reductions-scan.md) | Tree/atomic selection, contention evidence, and scan choices |
| Dense SDPA / flash attention | [techniques/sdpa-a770.md](techniques/sdpa-a770.md) | ESIMD sidecar port ladder (v1-v4.1), fused/multi-kernel dispatch, measured tables, and SDP-specific negative results |
| Prefill attention or direct convolution | [techniques/attention-conv.md](techniques/attention-conv.md) | Separate measured attention and convolution ladders |
| `Q=1` decode attention | [techniques/attention-decode.md](techniques/attention-decode.md) | MHA/GQA/MQA, paged KV, wall/device results, and dispatch domain |
| Precision, accumulators, math modes, or tolerance | [techniques/numerics.md](techniques/numerics.md) | Error/performance tables and tolerance guidance |

## Hardware and Optimization Workflow

| Need | Open | Contains |
|---|---|---|
| Low-bit DPAS and graph capability | [workflow/dg2-lowbit-graph.md](workflow/dg2-lowbit-graph.md) | Audited functional evidence, parameter blocks, and limits on performance claims |
| DG2 topology and hard constraints | [hardware/hardware.md](hardware/hardware.md) | Threads, GRF, SLM/cache, DPAS/XMX, memory-message, and occupancy facts |
| Access-pattern bandwidth | [hardware/bandwidth.md](hardware/bandwidth.md) | DRAM-like, L2, SLM, message-width, stride, and empirical roofline data |
| Contiguous DRAM constants | [hardware/dram-contiguous-roofline.md](hardware/dram-contiguous-roofline.md) | Named bandwidth patterns and validated/empirical ridges |
| New-operator or failed-experiment reasoning | [workflow/methodology.md](workflow/methodology.md) | Operator classification, bottleneck-driven next tests, and model updates |
| Launch, host, fusion, queues, Copy Engine, UR v1/v2, or L0 graphs | [workflow/execution.md](workflow/execution.md) | P14 Level Zero topology, 5-mode PCIe/D2D bandwidth, Copy Engine overlap, 7-path submit/graph ladder, batch cliff, doorbell coherency, UMF/event pools, and hybrid fusion architecture |
| Compiler flags and generated code | [workflow/codegen.md](workflow/codegen.md) | O2/O3, unroll, AOT, large GRF, IR artifacts, and migration checks |
| Risky ESIMD execution | [workflow/robustness.md](workflow/robustness.md) | One-process watchdog/TDR protocol and versioned probes |
| Known failures and traps | [workflow/pitfalls.md](workflow/pitfalls.md) | API gates, structural negatives, correctness traps, and diagnostic traps |
| Which ISA features can be written from source | [workflow/isa-capability-matrix.md](workflow/isa-capability-matrix.md) | 16-feature source/asm/unreachable matrix with the probe behind each row |
| What can actually be measured here | [workflow/instruments.md](workflow/instruments.md) | Works / does-not-work instrument tables and the required instrument for each gap |
| ISA fact to design decision | [workflow/isa-to-decision.md](workflow/isa-to-decision.md) | The 20-row table that turns an ISA fact into the design decision it licenses, plus the decisions it forbids |
| Measured symptom to next action | [workflow/performance-playbook.md](workflow/performance-playbook.md) | 17-row symptom -> ISA fact -> action -> expected-effect table, with the four operators' current stamps and the dead ends (prefetch, dual issue, 256-GRF); compute, message, scheduling and programming-model rules with measured numbers |
| **Where a number comes from** | [workflow/evidence-map.md](workflow/evidence-map.md) | Operator family -> pinned number -> doc -> probe directory, the gap-pricing answers, the roofline denominators by access pattern, and the ISA fact -> probe table; the inverse of the repo's generated `docs/69-probe-index.md` |
| **Keeping a page of numbers true** | [workflow/evidence-integrity.md](workflow/evidence-integrity.md) | The ten rules for generated measurement pages - assert the pinned artifact not the live code, one derived value one home, supersede by name, never type what the renderer holds, prove it by perturbation, make acceptance a predicate, keep un-interpolatable sources (comments, READMEs) free of derived numbers, print the constant that scores a threshold, compute percentages from unrounded quantities, amend a pre-registration append-only - plus the incident behind each rule and the four checks to run before believing a page |

## Instruction-Level / ISA (new)

These pages are instruction- and encoding-level, derived from offline IGA plus hand-edited GEN asm
executed on the A770 (driver `32.0.101.9150`). They are the reference for *why* a kernel behaves
the way it does, not a replacement for the operator pages.

| Need | Open | Contains |
|---|---|---|
| DPAS/XMX legality, throughput, latency | [isa/dpas.md](isa/dpas.md) | Legal type pairs, tf32/region/predication gates, measured 82 Gdpas/s per-EU issue limit, why ILP does not help |
| Scoreboard (SWSB) encoding and cost | [isa/swsb.md](isa/swsb.md) | Token/distance encodings, class-matching rules, silent no-op annotations, measured +33% token starvation and dependency-stripped hangs |
| Operand regions and register banks | [isa/regions-banks.md](isa/regions-banks.md) | Arity-dependent region syntax, legal stride sets, PRM restrictions are warnings, DG2 has no compiler-side bank mitigation |
| Which compiler flags actually change the ISA (and which are dead) | [codegen/knobs.md](codegen/knobs.md) | vISA -options are unreachable from ocloc; only CL-level flags move instructions: O0 (+463 instr), correctly-rounded div/sqrt (+307), fast-math (-30); precision flags only rewrite cr0 |
| 128-bit instruction encoding: exec size, quarter, region, modifiers | [codegen/encoding.md](codegen/encoding.md) | ExecSize `[18:16]` = log2(width); compaction makes the instruction 8 bytes; verified by patching bytes and observing 1024/4096 wrong results |
| Typed/sampler channel (`send.smpl`, `send.dc0`, `send.dc1`) fields | [isa/typed-sampler.md](isa/typed-sampler.md) | Sampler SIMD encoding is part-specific (DG2 SIMD8+iret vs Xe2 SIMD16+iret); typed channel mask at `[11:8]` is 1 = DISABLED, opposite of LSC quad |
| SLM bank behaviour and padding rules | [hardware/slm-banks.md](hardware/slm-banks.md) | No 32-bank cliff; only 64 B-multiple lane strides cost (up to 3.3x); broadcast is 0.69x; 8 B spacing beats 4 B |
| Barriers, fences, and their costs | [isa/sync-fences.md](isa/sync-fences.md) | Barrier = fence message + gateway signal + `sync.bar`; `barrier(0)` emits no fence; hand-injected barriers deadlock |
| Send messages and cache policy | [isa/lsc-messages.md](isa/lsc-messages.md) | Cache policy lives in the descriptor, ExBSO needs a register descriptor, JIT `bti[]` vs AOT `bss[]` divergence |
| Addressing mode (stateful BTI vs bindless ExBSO) | [isa/addressing.md](isa/addressing.md) | Runtime emits `a32 + bti[n]`; AOT bindless (`a0 + ExBSO`) is recompiled away; the two are not interchangeable by field patching (a hand-swapped kernel reads zeros while looking 1.50x-1.59x faster) |
| Hand-written GEN asm control loop | [codegen/asm-control.md](codegen/asm-control.md) | Driver-native dump → disasm → edit → `ocloc asm` → run, why ocloc-built binaries get recompiled away, IGA standalone build |
| Instruction-level measurement method | [workflow/benchmarking.md](workflow/benchmarking.md) | Straight-line amplification, separate timing/correctness runs, "missing bench ≠ 0", compute-bound gating |
| Extended math (`math.*`) legality and cost | [isa/math-numerics.md](isa/math-numerics.md) | Subfunction names IGA accepts, destination-region rule, measured exp latency/throughput vs dpas |
| Occupancy, XMX saturation, and wave arithmetic | [hardware/occupancy-saturation.md](hardware/occupancy-saturation.md) | Measured thread-scaling plateau, work-limited occupancy math, per-thread tile decision rules |
| sr0 field layout and hardware bits (DG2) | [hardware/sr0-hwtid.md](hardware/sr0-hwtid.md) | slice/subslice/EU/TID bit map, sr0.2/.3 unimplemented (all-ones), AOT patches are recompiled away |
| L1/L2 geometry, latency ladder, L2 bandwidth | [hardware/cache-geometry.md](hardware/cache-geometry.md) | ~55 ns L1 (hot out to 1 MB), ~120-140 ns L2, ~366 ns DRAM; L2 peak ~2.4 TB/s at 128 B lines; dependent loads ~152 ns |

## APIs, Existing Helpers, and Skill Maintenance

| Need | Open | Contains |
|---|---|---|
| API and command forms | [api/api-usage.md](api/api-usage.md) | SYCL/ESIMD forms, oneDNN baselines, build/profile commands, and verification rules |
| Copy-ready implementation cores | [api/code-snippets.md](api/code-snippets.md) | Kernel and helper snippets for the measured campaigns |
| Optional bundled helpers | [workflow/automation.md](workflow/automation.md) | Existing build, compare, benchmark, oneDNN, VTune, watchdog, and record interfaces |
| Skill behavior audit | [workflow/evaluation.md](workflow/evaluation.md) | Historical skill evaluation, applied text audit, and known limitations |

The skill bundles thin scripts and minimal GEMV examples, not every campaign
runner or full optimizer source. Verify that a referenced script exists before
claiming execution; where a campaign page says its runner remains in the
campaign checkout, treat its commands as reproduction context rather than a
skill-root command.
