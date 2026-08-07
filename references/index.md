# Reference Index

Evidence labels (`[ARCH]`, `[MEASURED]`, `[HEURISTIC]`, `[DISPATCH]`, `[BUG]`,
`[HYPOTHESIS]`) are defined in `SKILL.md`.

All measured numbers in this skill come from an Intel Arc A770 (DG2) running
Windows 11 with oneAPI 2026.1 and driver `32.0.101.8724`. Re-measure before
transferring conclusions to another GPU, driver, or compiler.

Start with `SKILL.md` for the decision flow; open the reference below for the
measured tables, rules, and negative results.

| Topic | Reference | Measured campaign |
|---|---|---|
| Hardware specs and measured ceilings | [hardware/hardware.md](hardware/hardware.md), [hardware/bandwidth.md](hardware/bandwidth.md) | GEMM and bandwidth campaigns |
| Dense operator ladders (GEMM/GEMV/RMSNorm/Softmax) | [techniques/techniques.md](techniques/techniques.md) | Dense operator campaigns |
| Irregular shapes and sparse GEMM | [techniques/irregular-shapes.md](techniques/irregular-shapes.md) | Irregular-shapes campaign |
| Numerical precision and tolerance | [techniques/numerics.md](techniques/numerics.md) | Numerics campaign |
| Reduction and scan selection | [techniques/reductions-scan.md](techniques/reductions-scan.md) | Reduction-scan campaign |
| Attention and convolution | [techniques/attention-conv.md](techniques/attention-conv.md) | Attention-conv campaign |
| Launch, fusion, graph, host overhead | [workflow/execution.md](workflow/execution.md) | Execution-model campaign |
| Compiler behavior and codegen | [workflow/codegen.md](workflow/codegen.md) | Codegen campaign |
| Robustness, watchdog, TDR protocol | [workflow/robustness.md](workflow/robustness.md) | Robustness campaign |
| Pitfalls and negative results | [workflow/pitfalls.md](workflow/pitfalls.md) | All campaigns |
| API usage forms | [api/api-usage.md](api/api-usage.md) | All campaigns |
| Copy-ready core kernels | [api/code-snippets.md](api/code-snippets.md) | All campaigns |

The skill keeps condensed measured facts and small copy-ready cores; full
compilable sources are intentionally not bundled.
