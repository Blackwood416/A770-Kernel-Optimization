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

Measured campaign results include: bf16 GEMM 1.95 ms to 0.0614 ms
(1024x1536x512; about 87% of oneMKL, 90% of oneDNN), f32 GEMV 2.65 ms to
0.201 ms device median (4096x4096; see
[techniques.md](references/techniques/techniques.md#f32-gemv-ladder-4096x4096-row-major-a)),
RMSNorm 5.382 ms to 0.098 ms (f32 1024x4096; see
[rmsnorm-shape-sweep.md](references/rmsnorm-shape-sweep.md)), Softmax 5.55 ms
to 0.107 ms (f32 1024x4096), plus bandwidth/roofline, execution model,
compiler, irregular-shape, numerics, reduction/scan, robustness, and
attention/conv campaigns.

## How to Use This Skill

1. Classify the operator first:
   - Dense GEMM / GEMV / RMSNorm / Softmax:
     [techniques.md](references/techniques/techniques.md),
     [gemm-shape-crossover.md](references/techniques/gemm-shape-crossover.md)
   - Irregular shapes, gather/scatter, sparse GEMM: [irregular-shapes.md](references/techniques/irregular-shapes.md)
   - Full reduction, atomic contention, work-group scan: [reductions-scan.md](references/techniques/reductions-scan.md)
   - Flash attention, direct convolution: [attention-conv.md](references/techniques/attention-conv.md)
   - Decode attention (Q=1, GQA/MQA, paged KV): [attention-decode.md](references/techniques/attention-decode.md)
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
7. Reuse the measured harness for build, verify, benchmark, baseline, VTune,
   watchdog, and record workflows. The core scripts are bundled under
   `scripts/`; if a deployment lacks them, verify script existence before
   claiming execution and treat `automation.md` as an interface contract:
   [automation.md](references/workflow/automation.md).

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
- `[CORRECTNESS]`: implementation-level correctness constraint or verified
  trap (packed offset, stride, layout, tail, lane coverage). Usually still
  applies after toolchain changes.
- `[TOOLCHAIN]`: oneAPI / IGC / SYCL / ESIMD API or compiler capability or
  limit. It may change with the toolchain and must be re-probed after an
  upgrade.
- `[HYPOTHESIS]`: not sufficiently validated. Must be tested before use.

Every `[DISPATCH]` rule must carry a validity domain: operator, dtype, shape
range, alignment, tested rows, device, oneAPI version, and confidence. A
`[MEASURED]` result that is used outside its measured shape must be
re-labeled `[HEURISTIC]`. After a toolchain upgrade, re-probe `[BUG]` and
`[TOOLCHAIN]`; re-check `[CORRECTNESS]`; treat `[ARCH]` as expected to persist.

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
- accuracy class (`matched` / `fastest` / `unknown`), reference tolerance,
  baseline correctness status, and `comparable_for_speedup`

Enable `ONEDNN_VERBOSE=profile,dispatch` and keep the implementation string;
it distinguishes "won against a JIT kernel" from "oneDNN fell back to a
reference path". Do not compute speedup ratios against a baseline that fails
the required tolerance; classify it as a fastest-library performance
lower-bound only.

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
  fallback. At this shape the custom path reaches about 90% of oneDNN; the
  M-dependent crossover below decides when the library is the better target.
  See
  [techniques.md](references/techniques/techniques.md#the-full-ladder).
- GEMM/GEMV shape crossover: `[DISPATCH]` bf16 GEMM/GEMV uses oneDNN
  `jit:gemm:any` except `N=14336, K=4096, M<=16` where DPAS8 wins; f32 GEMM
  is a oneMKL/oneDNN tie; f32 GEMV on device events uses oneDNN for `M<=64`
  and oneMKL for `M>=256`, while wall time favors oneMKL at every M. See
  [gemm-shape-crossover.md](references/techniques/gemm-shape-crossover.md).
- GEMV: `[MEASURED]` f32 4096x4096 standard-SYCL champion is
  sub-group-per-row with `vec<float,16>` loads and per-lane trip counts
  derived from the actual sub-group size (0.215 ms device median). The ESIMD
  one-work-item-per-row path with 256 B block loads measured faster on
  `4096x4096` (0.201 ms), `1024x4096` (0.091 ms vs 0.127 ms), and `64x128`
  (0.0039 ms vs 0.0075 ms). All crossover values are device-time medians
  (SYCL events); wall medians are recorded separately and can change the
  champion (at `64x128`, ESIMD also has the lowest wall time). oneMKL still
  wins device time (0.166 ms at `4096x4096`); oneDNN `Kx1` used
  `jit:gemm:any` at 0.456 ms event / 0.685 ms verbose on the same shape. See
  [techniques.md](references/techniques/techniques.md#f32-gemv-ladder-4096x4096-row-major-a).
- Weight-only decode GEMV: `[MEASURED]` M=64, N=8192, K=8192, gs=128:
  packed load -> unpack -> FMA 41.47 ms, device unpack -> VNNI SLM -> DPAS
  5.65 ms, host predequant + DPAS 0.656 ms, accuracy-matched oneDNN (f16 src,
  strict/any, f32 dst, `jit:gemm:any`) 0.209 ms at gs=128. Device ratio
  R3/oneDNN is about 3.1x; bf16-src oneDNN remains fastest-only and fails the
  tolerance. M=1 DPAS is a negative path. See
  [weight-only-gemv.md](references/weight-only-gemv.md).
- Row reductions (RMSNorm/Softmax): `[MEASURED]` f32 1024x4096: stage the row
  in SLM and normalize from SLM; shrink WG size for short rows. For RMSNorm
  rows 1-1024 x hidden 256-16384 in f32/f16/bf16, use the measured
  shape-dispatch map instead of one champion. Wall-time and device-time
  champions differ per cell; ask which target the deployment needs, or report
  both. See
  [rmsnorm-shape-sweep.md](references/rmsnorm-shape-sweep.md) and
  [techniques.md](references/techniques/techniques.md#f32-rmsnorm-ladder-1024x4096-row-major-x-f32).
- Irregular/sparse: `[HEURISTIC]` sparse GEMM `M=K=512, N=8, f32`: CSR for
  sparsity >= 90%, BSR B4 for about 50%. `[HEURISTIC]` GEMV/softmax fallback
  (f32, M=64/96, cols 256-4096) uses the scalar-chunk SLM path for
  non-16-aligned shapes. See
  [irregular-shapes.md](references/techniques/irregular-shapes.md).
- Reduction/scan: `[MEASURED]` 4096x4096 f32 full sums and 1M f32
  one-element-per-lane scans: global atomic (one per sub-group) or tree;
  WG64 Hillis-Steele is a starting point. See
  [reductions-scan.md](references/techniques/reductions-scan.md).
- Attention/conv: `[MEASURED]` small f32 prefill shape `B=4 H=16 Q=128
  KV=256 D=64`: naive three-kernel attention wins. `[MEASURED]` f32 Q=1
  decode `B=4 Hq=8 Hkv=8/2/1 KV=512..32768 D=64/128`: naive wins only at
  KV=512; `kv_cache_chunk_layout`/`online_causal_fused` take over at
  KV>=2048, with paged KV close behind. Wall-time and device-time champions
  may differ; report both (tables in `attention-decode.md`); host overhead is
  104-279 us per decode call and dominates wall time at small KV. NHWC conv
  wins only with OC cache blocking. See
  [attention-conv.md](references/techniques/attention-conv.md) and
  [attention-decode.md](references/techniques/attention-decode.md).
- Numerics: `[MEASURED]` small f32/bf16/f16/int8 shapes: keep f32 accumulators
  for bf16/f16 and use combined relative/absolute tolerances. See
  [numerics.md](references/techniques/numerics.md).
- Execution: `[MEASURED]` pack+GEMM 1024x1536x512: fuse only when
  `fused device delta < launch gap + host overhead saved`; reuse graphs for
  repeated small pipelines. See
  [execution.md](references/workflow/execution.md).
- Codegen: `[MEASURED]` simple non-XMX bf16 loops (GEMM 1024x1536x512,
  RMSNorm 1024x4096): O2 auto-vectorizes. `[HEURISTIC]` start with explicit
  unroll=2, then re-measure. See
  [codegen.md](references/workflow/codegen.md).
- Risk isolation: `[BUG]` driver/oneAPI-version-specific: `load_2d` hangs,
  `named_barrier` is rejected at device JIT, and large SLM geometries can
  trigger device loss. See
  [robustness.md](references/workflow/robustness.md).

## Curated Pitfalls

- `[HEURISTIC]` `prefetch` is negative on A770 for GEMM-sized data that fits
  the 16 MB L2.
- `[BUG]` DPAS `ExecutionSize=16` compiles but produces wrong results on A770.
- `[ARCH]` `load_2d`, `prefetch_2d`, and `named_barrier` are PVC-only APIs on
  Xe-HPG.
- `[BUG]` / `[TOOLCHAIN]` on A770 with oneAPI 2026.1 / driver
  `32.0.101.8724`, those APIs hang or are rejected at device JIT; re-probe
  after any toolchain upgrade.
- `[CORRECTNESS]` Do not hardcode sub-group size 16; A770 can compile
  32-lane sub-groups and a GEMV then covers only half of each row.
- `[MEASURED]` Do not allocate full 64 KB SLM tiles; keep tiles at 32-48 KB.
- `[MEASURED]` Large GRF has no universal verdict: negative for
  joint_matrix/XMX kernels, positive for AOT + unroll4 simple GEMM.
- `[MEASURED]` One global atomic per thread is wrong for full reductions: it
  has fewer instructions but is 3.8x slower from address serialization.
- `[CORRECTNESS]` Keep bf16/f16 accumulators in f32; low-precision
  accumulators can push reduction error to O(1).
- `[BUG]` `BM=64` flash-attention tiles can trigger
  `UR_RESULT_ERROR_DEVICE_LOST`; keep BM=32 on the measured shape.
- `[HEURISTIC]` Fusing a pack kernel into GEMM is not always a win; apply the
  measured threshold before fusing.
- `[TOOLCHAIN]` Device-side text LLVM IR is unsupported; keep bitcode,
  SPIR-V text, and GEN assembly.
- `[HEURISTIC]` Host preprocessing belongs outside the timed loop when inputs
  are static.
- `[CORRECTNESS]` The second 8-column B slice of a joint-matrix GEMM is
  `+16 bf16`, not `+8`.
- `[CORRECTNESS]` Packed u4 byte offset for K offset `bk` is `bk / 2`, not
  `bk * (BK/2)`; the latter corrupts every K block after the first.
- `[CORRECTNESS]` M=1 DPAS kernels must replicate row 0 into the 8 DPAS rows
  and guard output stores; reading rows 1..7 of a one-row buffer can trigger
  device loss.
- `[CORRECTNESS]` Decode attention should use a deterministic global
  two-phase reduce; a single-kernel local-memory online combine produced
  `1952/2048` errors.
- `[CORRECTNESS]` Verify library baselines against the CPU reference; oneDNN
  `1xK` is `x*A`, not `A*x`.
- `[MEASURED]` VTune `full-compute` ALU0/ALU1 numbers are single-run samples
  with high variance; compare wall time and `instruction-count` for
  conclusions.

Complete evidence: [pitfalls.md](references/workflow/pitfalls.md).

## API and Code

- API forms, build/profile commands, oneMKL/oneDNN baselines, graph/dual-queue
  submission, IR dump, and watchdog usage:
  [api-usage.md](references/api/api-usage.md).
- Reusable benchmark/verify/record/watchdog CLI and record schema:
  [automation.md](references/workflow/automation.md).
- Copy-ready building blocks for every campaign:
  [code-snippets.md](references/api/code-snippets.md).

## References

- Topic map and campaign index:
  [references/index.md](references/index.md).
- Skill behavior eval and applied text audit:
  [workflow/evaluation.md](references/workflow/evaluation.md).
