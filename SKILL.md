---
name: optimize-intel-gpu-kernels
description: "Optimize SYCL/ESIMD compute kernels for Intel Arc/Xe-HPG GPUs (A770/DG2). Use for dense GEMM/GEMV/RMSNorm/Softmax, irregular and sparse shapes, reductions and scans, attention and convolution, numerical tolerance, bandwidth and roofline analysis, launch/fusion/graph overhead, compiler flags and codegen, VTune interpretation, robustness/TDR isolation, and oneMKL/oneDNN baselines."
---

# Intel GPU Kernel Optimization

Optimize SYCL/ESIMD kernels on Intel Arc A770 (Xe-HPG/DG2) with measured
hardware facts, proven technique ladders, known pitfalls, and API usage rules.
All numbers are single-machine conclusions from oneAPI 2026.1 and driver
`32.0.101.8724`; re-measure before transferring them to another GPU, driver,
or compiler.

Measured campaign results include: bf16 GEMM 1.95 ms to 0.0614 ms (about 87%
of oneMKL, 90% of oneDNN), f32 GEMV 2.65 ms to 0.215 ms, RMSNorm 5.382 ms to
0.098 ms, Softmax 5.55 ms to 0.107 ms, plus bandwidth/roofline, execution
model, compiler, irregular-shape, numerics, reduction/scan, robustness, and
attention/conv campaigns.

## How to Use This Skill

1. Classify the operator first:
   - Dense GEMM / GEMV / RMSNorm / Softmax: [techniques.md](references/techniques/techniques.md)
   - Irregular shapes, gather/scatter, sparse GEMM: [irregular-shapes.md](references/techniques/irregular-shapes.md)
   - Full reduction, atomic contention, work-group scan: [reductions-scan.md](references/techniques/reductions-scan.md)
   - Flash attention, direct convolution: [attention-conv.md](references/techniques/attention-conv.md)
   - Precision, accumulator, tolerance, math modes: [numerics.md](references/techniques/numerics.md)
2. Decide whether the kernel is compute-bound or memory-bound with the
   bandwidth microbenchmark and roofline ridge: [bandwidth.md](references/hardware/bandwidth.md).
3. Before tuning a small kernel, measure launch gap and host overhead:
   [execution.md](references/workflow/execution.md).
4. Before changing compiler flags, unrolling, AOT, or large GRF, read the
   codegen matrix: [codegen.md](references/workflow/codegen.md).
5. Before running new ESIMD shapes, isolate them with the robustness
   watchdog: [robustness.md](references/workflow/robustness.md).
6. Verify every change with the workflow below, and record negative results.

## Workflow

1. Build the correctness harness first: CPU reference, `errors: 0/...`,
   warmup + timed loop, then 3+ stable runs.
2. Measure the same operator with oneMKL/oneDNN and verify the library output
   against the CPU reference; a fast library call can be a different operator
   (e.g. oneDNN `1xK` computes `x*A`, not row-major `A*x`).
3. Profile with VTune before restructuring; start with `instruction-count`,
   escalate to `full-compute`.
4. Change exactly one structural variable per experiment and keep failed
   variants as standalone files.
5. Verify every variant: exact reference compare, 3 stable values, and the
   VTune metrics that motivated the change.

## Hardware Quick Facts

| Parameter | Value |
|---|---|
| Hardware threads | 4096 total; 8 per vector engine (4 in large-GRF mode) |
| GRF | 128 regular / 256 large per thread; 256-bit registers |
| SLM | 128 KB per Xe-core; 64 KB max per work-group, keep tiles at 32-48 KB |
| Cache | L1 192 KB per Xe-core; L2 16 MB |
| XMX bf16 DPAS | 8x8x16 on DG2; B must be VNNI-packed |
| `block_load` | 256 B max on DG2 (512 B is PVC-only) |
| DRAM-like read | 277 GB/s measured |
| L2-resident read | 855 GB/s measured |
| SLM read | 4.1 TB/s measured |

Full hardware details and measured bandwidth/stride tables:
[hardware.md](references/hardware/hardware.md),
[bandwidth.md](references/hardware/bandwidth.md).

## Decision Branches

- Dense GEMM: bf16 16x16 ESIMD DPAS with host-side operand layout was the
  best stable SYCL path; joint_matrix is the portable fallback. See
  [techniques.md](references/techniques/techniques.md#the-full-ladder).
- GEMV: sub-group-per-row with `vec<float,16>` loads and per-lane trip counts
  derived from the actual sub-group size. See
  [techniques.md](references/techniques/techniques.md#f32-gemv-ladder-4096x4096-row-major-a).
- Row reductions (RMSNorm/Softmax): stage the row in SLM and normalize from
  SLM; shrink WG size for short rows. See
  [techniques.md](references/techniques/techniques.md#f32-rmsnorm-ladder-1024x4096-row-major-x-f32).
- Irregular/sparse: CSR for >=90% sparsity, BSR B4 for about 50%, scalar-chunk
  fallback for non-16-aligned shapes. See
  [irregular-shapes.md](references/techniques/irregular-shapes.md).
- Reduction/scan: global atomic (one per sub-group) or tree for full sums;
  WG64 Hillis-Steele for scans. See
  [reductions-scan.md](references/techniques/reductions-scan.md).
- Attention/conv: naive three-kernel attention wins on small shapes; NHWC
  conv wins only with OC cache blocking. See
  [attention-conv.md](references/techniques/attention-conv.md).
- Numerics: keep f32 accumulators for bf16/f16; use combined relative/absolute
  tolerances. See [numerics.md](references/techniques/numerics.md).
- Execution: fuse only when `fused device delta < launch gap + host overhead
  saved`; reuse graphs for repeated small pipelines. See
  [execution.md](references/workflow/execution.md).
- Codegen: O2 auto-vectorizes simple bf16 loops; try explicit unroll=2 before
  flags. See [codegen.md](references/workflow/codegen.md).
- Risk isolation: `load_2d` hangs, `named_barrier` is rejected at device JIT,
  and large SLM geometries can trigger device loss. See
  [robustness.md](references/workflow/robustness.md).

## Curated Pitfalls

- `prefetch` is negative on A770 for GEMM-sized data that fits the 16 MB L2.
- DPAS `ExecutionSize=16` compiles but produces wrong results on A770.
- `load_2d`, `prefetch_2d`, and `named_barrier` are PVC-only; on A770 they
  hang, are rejected at device JIT, or both.
- Do not hardcode sub-group size 16; A770 can compile 32-lane sub-groups and
  a GEMV then covers only half of each row.
- Do not allocate full 64 KB SLM tiles; keep tiles at 32-48 KB.
- Large GRF has no universal verdict: negative for joint_matrix/XMX kernels,
  positive for AOT + unroll4 simple GEMM.
- One global atomic per thread is wrong for full reductions: it has fewer
  instructions but is 3.8x slower from address serialization.
- Keep bf16/f16 accumulators in f32; low-precision accumulators can push
  reduction error to O(1).
- `BM=64` flash-attention tiles can trigger `UR_RESULT_ERROR_DEVICE_LOST`;
  keep BM=32 on the measured shape.
- Fusing a pack kernel into GEMM is not always a win; apply the measured
  threshold before fusing.
- Device-side text LLVM IR is unsupported; keep bitcode, SPIR-V text, and GEN
  assembly.
- Host preprocessing belongs outside the timed loop when inputs are static.
- The second 8-column B slice of a joint-matrix GEMM is `+16 bf16`, not `+8`.
- Verify library baselines against the CPU reference; oneDNN `1xK` is
  `x*A`, not `A*x`.
- VTune `full-compute` ALU0/ALU1 numbers are single-run samples with high
  variance; compare wall time and `instruction-count` for conclusions.

Complete evidence: [pitfalls.md](references/workflow/pitfalls.md).

## API and Code

- API forms, build/profile commands, oneMKL/oneDNN baselines, graph/dual-queue
  submission, IR dump, and watchdog usage:
  [api-usage.md](references/api/api-usage.md).
- Copy-ready building blocks for every campaign:
  [code-snippets.md](references/api/code-snippets.md).

## References

- Topic map and source project provenance:
  [references/README.md](references/README.md).
