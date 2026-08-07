# A770 Compiler Behavior Reference

Measured on Intel Arc A770, oneAPI 2026.1 (`icpx`, build 20260617),
driver `32.0.101.8724`, VTune 2026.2. Source project:
`E:\RiderProjects\Codegen-Opti` (`references\codegen.md`).

## Fixed Kernels

The sweep uses deliberately simple, non-XMX standard-SYCL kernels, so
`XMX=0` and `Barrier=0` are expected:

| Kernel | Shape | Structure |
|---|---|---|
| `gemm_bf16<U>` | 1024x1536x512 | one output per work-item, float accumulator, optional explicit 1/2/4-way unroll |
| `rmsnorm_bf16<U>` | 1024x4096 | one row per work-item, float accumulator, gamma, optional explicit unroll |

## Key Wall Times

Wall time is 100 warmup + 300 timed iterations, 3 repeats, min taken.

| Config | GEMM ms | RMS ms |
|---|---:|---:|
| O0 | 28.572 | 6.566 |
| O2 baseline | 1.923 | 3.343 |
| O3 | 2.032 | 3.354 |
| O2 fp-contract=off | 2.072 | 3.359 |
| O2 fast-math | 2.102 | 3.375 |
| O2 large GRF | 2.077 | 3.355 |
| O2 AOT | 2.108 | 3.329 |
| O2 unroll=2 | 1.481 | 2.770 |
| O2 unroll=4 | 1.595 | 2.990 |
| O3 fast AOT u4 | 1.598 | 3.002 |
| O3 fast AOT GRF u4 | 1.390 | 2.939 |

## Compiler Done / Neutral / Harmful Matrix

| Factor | GEMM | RMS | Verdict |
|---|---|---|---|
| O0 | 28.6 ms, SIMD8 | 6.57 ms | harmful |
| O2 auto-vectorization | SIMD32, 100% util | SIMD32, 100% util | compiler already does it |
| O3 vs O2 | identical JIT instructions, wall +5.7% | identical, +0.3% | neutral |
| fp-contract=off | identical instructions, +7.7% | identical, +0.5% | neutral |
| fast-math | identical instructions, +9.3% | identical, +1.0% | neutral |
| large GRF (JIT u1) | identical instructions, +8.0% | identical, +0.4% | neutral on simple loops |
| AOT (O2 u1) | instr -6.6%, wall +9.6% | instr +15%, wall -0.4% | neutral |
| unroll=2 | instr -16.6%, wall -23.0% | instr -8.5%, wall -17.1% | beneficial |
| unroll=4 | instr -21.6%, wall -17.0% | instr -17.9%, wall -10.6% | beneficial, worse alone than u2 |
| O3+fast+AOT+u4 | instr -28.2% | wall about -10% | beneficial combination |
| O3+fast+AOT+GRF+u4 | wall -27.7%, best | wall -12.1% | beneficial combination |

## Rules

1. On simple bf16 loops, O2 already vectorizes to 32-lane SIMD. O3,
   fp-contract, fast-math, and large GRF do not change JIT instruction counts
   and are neutral or slightly negative.
2. Explicit 2-way unroll is the only stable source-level win in this sweep;
   unroll=4 alone is worse than unroll=2 but combines well with AOT/fast-math.
3. Large GRF is context-dependent: negative for joint_matrix/XMX kernels
   (see hardware pitfalls) but positive for AOT + unroll4 simple GEMM. Do not
   write a single universal large-GRF verdict.
4. AOT and JIT share the same SPIR-V. O3, large GRF, and AOT produced the
   same SPIR-V as the O2 baseline; instruction differences come from the IGC
   backend or runtime compile path.
5. `-S -emit-llvm` does not produce device text IR on this toolchain. Keep
   LLVM bitcode, SPIR-V text (`llvm-spirv --to-text`), and GEN assembly
   (`ocloc compile` for `dg2-g10-a0`).
6. VTune `full-compute` ALU0/ALU1 numbers are single-run samples with high
   variance; prefer wall time plus `instruction-count` for cross-config
   conclusions.

## Version Migration Checklist

1. Record the environment fingerprint first: oneAPI version, driver, Level
   Zero, OpenCL NEO, VTune build, and AOT device name.
2. Rebuild all configs and confirm `icpx` GNU-style flags still compile.
3. Re-run the wall-time sweep and require `errors: 0` plus 3 stable runs.
4. Confirm O0 is still far slower than O2, or the optimization-flag mapping
   changed.
5. Confirm VTune column names and `ocloc` device names still match.
6. Verify the IR dump paths (bitcode, SPIR-V text, GEN asm) still work.
7. Re-check the key baselines: O2 vs O3 instruction identity, unroll=2 faster
   than simple loop, and AOT+GRF+u4 behavior on GEMM.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_sweep.ps1 -Kernel gemm
powershell -ExecutionPolicy Bypass -File scripts\vtune_sweep.ps1 -Kernel gemm -Mode instruction-count
powershell -ExecutionPolicy Bypass -File scripts\dump_ir.ps1
```

Environment details: `E:\RiderProjects\Codegen-Opti\references\environment.md`.
