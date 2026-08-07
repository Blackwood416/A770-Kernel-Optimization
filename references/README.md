# Reference Index

All measured numbers in this skill come from one machine: Intel Arc A770
(DG2), oneAPI 2026.1, driver `32.0.101.8724`, Windows 11. Re-measure before
transferring conclusions to another GPU, driver, or compiler.

Start with `SKILL.md` for the decision flow; open the reference below for the
measured tables, rules, and negative results.

| Topic | Reference | Source project |
|---|---|---|
| Hardware specs and measured ceilings | [hardware/hardware.md](hardware/hardware.md), [hardware/bandwidth.md](hardware/bandwidth.md) | original GEMM campaign; `E:\RiderProjects\BandWidth-Opti` |
| Dense operator ladders (GEMM/GEMV/RMSNorm/Softmax) | [techniques/techniques.md](techniques/techniques.md) | original campaigns |
| Irregular shapes and sparse GEMM | [techniques/irregular-shapes.md](techniques/irregular-shapes.md) | `E:\RiderProjects\IrregularShapes-Opti` |
| Numerical precision and tolerance | [techniques/numerics.md](techniques/numerics.md) | `E:\RiderProjects\Numerics-Opti` |
| Reduction and scan selection | [techniques/reductions-scan.md](techniques/reductions-scan.md) | `E:\RiderProjects\Reduction-Opti` |
| Attention and convolution | [techniques/attention-conv.md](techniques/attention-conv.md) | `E:\RiderProjects\AttentionConv-Opti` |
| Launch, fusion, graph, host overhead | [workflow/execution.md](workflow/execution.md) | `E:\RiderProjects\KernelExec-Opti` |
| Compiler behavior and codegen | [workflow/codegen.md](workflow/codegen.md) | `E:\RiderProjects\Codegen-Opti` |
| Robustness, watchdog, TDR protocol | [workflow/robustness.md](workflow/robustness.md) | `E:\RiderProjects\Robustness-Opti` |
| Pitfalls and negative results | [workflow/pitfalls.md](workflow/pitfalls.md) | all campaigns |
| API usage forms | [api/api-usage.md](api/api-usage.md) | all campaigns |
| Copy-ready core kernels | [api/code-snippets.md](api/code-snippets.md) | all campaigns |

Full compilable sources and raw results stay in the E:\ projects listed
above; the skill keeps condensed measured facts and small copy-ready cores.
