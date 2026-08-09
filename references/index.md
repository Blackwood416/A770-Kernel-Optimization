# Reference Index

Evidence labels (`[ARCH]`, `[MEASURED]`, `[HEURISTIC]`, `[DISPATCH]`, `[BUG]`,
`[CORRECTNESS]`, `[TOOLCHAIN]`, `[HYPOTHESIS]`) are defined in `SKILL.md`.

All measured numbers in this skill come from an Intel Arc A770 (DG2) running
Windows 11 with oneAPI 2026.1 and driver `32.0.101.8724`. Re-measure before
transferring conclusions to another GPU, driver, or compiler.

Start with `SKILL.md` for the decision flow; open the reference below for the
measured tables, rules, and negative results.

| Topic | Reference | Measured campaign |
|---|---|---|
| Hardware specs and measured bandwidth patterns | [hardware/hardware.md](hardware/hardware.md), [hardware/bandwidth.md](hardware/bandwidth.md) | GEMM and bandwidth campaigns |
| Dense operator ladders (GEMM/GEMV/RMSNorm/Softmax) | [techniques/techniques.md](techniques/techniques.md) | Dense operator campaigns |
| GEMM/GEMV shape crossover (M 1-1024, N/K 4096-14336, f32/bf16) | [techniques/gemm-shape-crossover.md](techniques/gemm-shape-crossover.md) | GEMM-shape crossover campaign |
| RMSNorm shape-dispatch sweep (rows 1-1024, hidden 256-16384, f32/f16/bf16) | [rmsnorm-shape-sweep.md](rmsnorm-shape-sweep.md) | RMSNorm shape sweep |
| Weight-only INT4/NF4 decode GEMV-M1 (M=1/64, N/K=4096/8192, gs=32-256) | [weight-only-gemv.md](weight-only-gemv.md) | Weight-only GEMV campaign |
| Irregular shapes and sparse GEMM | [techniques/irregular-shapes.md](techniques/irregular-shapes.md) | Irregular-shapes campaign |
| Numerical precision and tolerance | [techniques/numerics.md](techniques/numerics.md) | Numerics campaign |
| Reduction and scan selection | [techniques/reductions-scan.md](techniques/reductions-scan.md) | Reduction-scan campaign |
| Attention and convolution | [techniques/attention-conv.md](techniques/attention-conv.md) | Attention-conv campaign |
| Decode attention Q=1, GQA/MQA, paged KV (KV 512-32768, D 64/128) | [techniques/attention-decode.md](techniques/attention-decode.md) | Decode attention campaign |
| Launch, fusion, graph, host overhead | [workflow/execution.md](workflow/execution.md) | Execution-model campaign |
| Compiler behavior and codegen | [workflow/codegen.md](workflow/codegen.md) | Codegen campaign |
| Robustness, watchdog, TDR protocol | [workflow/robustness.md](workflow/robustness.md) | Robustness campaign |
| Bundled benchmark/verify/record harness | [workflow/automation.md](workflow/automation.md) | Automation harness campaign |
| Skill behavior evaluation and text audit | [workflow/evaluation.md](workflow/evaluation.md) | SkillEval campaign |
| Pitfalls and negative results | [workflow/pitfalls.md](workflow/pitfalls.md) | All campaigns |
| API usage forms | [api/api-usage.md](api/api-usage.md) | All campaigns |
| Copy-ready core kernels | [api/code-snippets.md](api/code-snippets.md) | All campaigns |

The skill keeps condensed measured facts and small copy-ready cores; full
compilable sources are intentionally not bundled.
