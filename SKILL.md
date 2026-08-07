---
name: optimize-a770-kernels
description: "Optimize SYCL/ESIMD compute kernels on Intel Arc A770/DG2 (Windows). Use measured constants and dispatch rules only for A770; for other Intel GPUs transfer the methodology only and re-measure all hardware-specific assumptions. Covers dense GEMM/GEMV/RMSNorm/Softmax, irregular and sparse shapes, reductions and scans, attention and convolution, numerical tolerance, bandwidth and roofline, launch/fusion/graph overhead, compiler flags and codegen, VTune interpretation, robustness/TDR isolation, and oneMKL/oneDNN baselines."
---

# Arc A770 Kernel Optimization

Optimize SYCL/ESIMD kernels on Intel Arc A770 (Xe-HPG/DG2) with measured
hardware facts, proven technique ladders, known pitfalls, and API usage rules.
All numbers are measured on Intel Arc A770 (DG2) with oneAPI 2026.1 and
driver `32.0.101.8724` on Windows; re-measure before transferring them to
another GPU, driver, or compiler.

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

## Evidence Levels

Label every claim so a future agent does not promote a single-shape result
into a universal rule:

- `[ARCH]`: Xe-HPG architectural constraint. Expected to hold for DG2 unless
  toolchain behavior changes.
- `[MEASURED]`: A770 / oneAPI 2026.1 / driver `32.0.101.8724` with an exact
  shape and result.
- `[HEURISTIC]`: derived from one or more measured campaigns. Use as a
  starting point, not a dispatch rule.
- `[DISPATCH]`: validated across a defined shape range. Safe only when every
  condition in its validity domain holds.
- `[BUG]`: observed toolchain/driver behavior. Version-specific.
- `[HYPOTHESIS]`: not sufficiently validated. Must be tested before use.

Every `[DISPATCH]` rule must carry a validity domain: operator, dtype, shape
range, alignment, tested rows, device, oneAPI version, and confidence. A
`[MEASURED]` result that is used outside its measured shape must be
re-labeled `[HEURISTIC]`.

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

## Benchmark Protocol

Use one timing protocol for new experiments so campaigns can be compared:

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

For kernels near or below 10 us, always report device time and wall time
separately; launch/host overhead can dominate.

## oneDNN Baseline Contract

Every oneDNN baseline must record:

- operator semantics and primitive kind
- implementation string (`jit:gemm:any`, `ocl:ref:any`, ...)
- format tags and src/weight/dst dtypes
- post-ops and `fpmath_mode`
- runtime/static dims, reorder/preprocessing included
- device time, wall time, and CPU-reference correctness

Enable `ONEDNN_VERBOSE=profile,dispatch` and keep the implementation string;
it distinguishes "won against a JIT kernel" from "oneDNN fell back to a
reference path".

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

- Dense GEMM: `[MEASURED]` bf16 1024x1536x512: ESIMD DPAS with host-side
  operand layout was the best stable SYCL path; joint_matrix is the portable
  fallback. See
  [techniques.md](references/techniques/techniques.md#the-full-ladder).
- GEMV: `[MEASURED]` f32 4096x4096 standard-SYCL champion is
  sub-group-per-row with `vec<float,16>` loads and per-lane trip counts
  derived from the actual sub-group size; the ESIMD crossover is pending
  measurement. See
  [techniques.md](references/techniques/techniques.md#f32-gemv-ladder-4096x4096-row-major-a).
- Row reductions (RMSNorm/Softmax): `[MEASURED]` f32 1024x4096: stage the row
  in SLM and normalize from SLM; shrink WG size for short rows. Low-row
  decode shapes are pending measurement. See
  [techniques.md](references/techniques/techniques.md#f32-rmsnorm-ladder-1024x4096-row-major-x-f32).
- Irregular/sparse: `[HEURISTIC]` from `M=K=512, N=8, f32`: CSR for >=90%
  sparsity, BSR B4 for about 50%, scalar-chunk fallback for non-16-aligned
  shapes. See
  [irregular-shapes.md](references/techniques/irregular-shapes.md).
- Reduction/scan: `[MEASURED]` 4096x4096 f32 full sums and 1M f32
  one-element-per-lane scans: global atomic (one per sub-group) or tree;
  WG64 Hillis-Steele is a starting point. See
  [reductions-scan.md](references/techniques/reductions-scan.md).
- Attention/conv: `[MEASURED]` small f32 prefill shape `B=4 H=16 Q=128
  KV=256 D=64`: naive three-kernel attention wins; decode attention is
  pending. NHWC conv wins only with OC cache blocking. See
  [attention-conv.md](references/techniques/attention-conv.md).
- Numerics: `[MEASURED]` small f32/bf16/f16/int8 shapes: keep f32 accumulators
  for bf16/f16 and use combined relative/absolute tolerances. See
  [numerics.md](references/techniques/numerics.md).
- Execution: `[MEASURED]` pack+GEMM 1024x1536x512: fuse only when
  `fused device delta < launch gap + host overhead saved`; reuse graphs for
  repeated small pipelines. See
  [execution.md](references/workflow/execution.md).
- Codegen: `[MEASURED]` simple non-XMX bf16 loops: O2 auto-vectorizes; try
  explicit unroll=2 before flags. See [codegen.md](references/workflow/codegen.md).
- Risk isolation: `[BUG]` driver/oneAPI-version-specific: `load_2d` hangs,
  `named_barrier` is rejected at device JIT, and large SLM geometries can
  trigger device loss. See
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

- Topic map and campaign index:
  [references/index.md](references/index.md).
