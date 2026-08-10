# Reference Index

Start with [`SKILL.md`](../SKILL.md) for task routing, evidence labels, the
optimization loop, timing rules, and the oneDNN baseline contract. Then load
only the operator page and cross-cutting pages that apply to the task.

Treat the environment and validity domain on each reference page as
authoritative. Most core campaigns used Intel Arc A770 (DG2), Windows 11,
oneAPI 2026.1, and driver `32.0.101.8724`. The contiguous-DRAM and paired
RMSNorm pages explicitly record later measurements on driver
`32.0.101.8860`. Re-measure before transferring a conclusion to another GPU,
driver, compiler, library build, shape, or layout.

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
| Prefill attention or direct convolution | [techniques/attention-conv.md](techniques/attention-conv.md) | Separate measured attention and convolution ladders |
| `Q=1` decode attention | [techniques/attention-decode.md](techniques/attention-decode.md) | MHA/GQA/MQA, paged KV, wall/device results, and dispatch domain |
| Precision, accumulators, math modes, or tolerance | [techniques/numerics.md](techniques/numerics.md) | Error/performance tables and tolerance guidance |

## Hardware and Optimization Workflow

| Need | Open | Contains |
|---|---|---|
| DG2 topology and hard constraints | [hardware/hardware.md](hardware/hardware.md) | Threads, GRF, SLM/cache, DPAS/XMX, memory-message, and occupancy facts |
| Access-pattern bandwidth | [hardware/bandwidth.md](hardware/bandwidth.md) | DRAM-like, L2, SLM, message-width, stride, and empirical roofline data |
| Contiguous DRAM constants | [hardware/dram-contiguous-roofline.md](hardware/dram-contiguous-roofline.md) | Named bandwidth patterns and validated/empirical ridges |
| New-operator or failed-experiment reasoning | [workflow/methodology.md](workflow/methodology.md) | Operator classification, bottleneck-driven next tests, and model updates |
| Launch, host, fusion, queue, or graph overhead | [workflow/execution.md](workflow/execution.md) | Submission-form measurements and fusion threshold |
| Compiler flags and generated code | [workflow/codegen.md](workflow/codegen.md) | O2/O3, unroll, AOT, large GRF, IR artifacts, and migration checks |
| Risky ESIMD execution | [workflow/robustness.md](workflow/robustness.md) | One-process watchdog/TDR protocol and versioned probes |
| Known failures and traps | [workflow/pitfalls.md](workflow/pitfalls.md) | API gates, structural negatives, correctness traps, and diagnostic traps |

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
