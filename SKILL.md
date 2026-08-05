---
name: optimize-intel-gpu-kernels
description: "Optimize SYCL/ESIMD compute kernels for Intel Arc/Xe-HPG GPUs (A770/DG2). Use when writing or tuning high-performance kernels on Intel discrete GPUs: choosing tiling, register blocking, XMX/joint_matrix/dpas, SLM buffering, walk order, or operand layouts; interpreting VTune gpu-hotspots; checking whether an Intel API (prefetch, load_2d, large GRF, DPAS ES16, named_barrier) is supported on A770; or benchmarking against oneMKL/oneDNN."
---

# Intel GPU Kernel Optimization

Optimize SYCL/ESIMD kernels on Intel Arc A770 (Xe-HPG/DG2) with measured hardware facts, a proven technique ladder, known pitfalls, and API usage rules extracted from a complete bf16 GEMM campaign (naive 1.95 ms to 0.0614 ms, about 87% of oneMKL and 90% of oneDNN).

## Workflow

1. Build the correctness harness first: CPU reference, `errors: 0/...`, warmup + timed loop, then 3+ stable runs.
2. Measure the same benchmark with oneMKL/oneDNN to set the target.
3. Profile with VTune before restructuring; start with `instruction-count`, escalate to `full-compute`.
4. Change exactly one structural variable per experiment; keep failed variants as files and record the negative result.
5. Verify every variant: exact reference compare, `C[0]` check, 3 stable values, and the VTune metrics that motivated the change.

## 1. A770 Hardware Parameters

| Parameter | Value |
|---|---|
| Hardware threads | 4096 total; 8 per vector engine (4 in large-GRF mode) |
| GRF | 128 (regular) or 256 (large) per thread; 256-bit (32 B) registers |
| SLM | 128 KB per Xe-core; 64 KB max per work-group |
| Cache | L1 192 KB per Xe-core; L2 16 MB |
| XMX bf16 DPAS | 8x8x16 on DG2; B must be VNNI-packed |
| `block_load` | 256 B max on DG2 (512 B is PVC-only) |
| Measured sweet spot | 32 threads/WG, 16x16 per-thread tile, 24 KB SLM, total threads a multiple of 4096 |

oneDNN's fastest A770 path uses ngen codegen + Large GRF with 16x8 per-thread tiles, 4 SLM buffers, and DPASW chains; that exact geometry and ngen scheduling are not reproducible in SYCL/ESIMD. Full details: [hardware.md](references/hardware.md)

## 2. Optimization Ladder

Apply in this order; each step was measured on the same bf16 GEMM (1024x1536x512):

1. Tile into SLM and add register tiling so A/B slices are reused (outer-product accumulation).
2. Vectorize global<->SLM copies; move computation to XMX (`joint_matrix` or ESIMD `dpas`).
3. VNNI-pack B on the host; vectorize A copies; match work-group geometry to 4096 threads with GRF headroom.
4. Double-buffer SLM and pipeline the K loop, but only after per-block compute is large enough.
5. Prefer N-first walk order for L2 reuse; split first/last K blocks to remove branches.
6. Use VTune instruction-count to find Send/ALU bloat; widen SLM->GRF messages.
7. Prepack A/B on the host into DPAS operand layout so kernel loads are 256 B with zero `select`.
8. Hoist base pointers out of the K loop (fixed strides); add more SLM buffers to cut barrier frequency.
9. Route repeatedly-read global tiles through a cooperative SLM copy when direct per-thread reads are redundant.

Measured ladder: 1.95 ms naive -> 0.118 ms joint_matrix best -> 0.099 ms ESIMD -> 0.073 ms operand layout -> 0.0614 ms final (oneMKL 0.0529, oneDNN 0.0552). Every ladder step maps to an embedded code snippet in [Ladder to Snippet Map](references/code-snippets.md#ladder-to-snippet-map); details and reasons: [techniques.md](references/techniques.md).

## 3. A770 Pitfalls

- `prefetch` (SYCL or ESIMD): negative on A770 for GEMM-sized data that fits the 16 MB L2.
- Large GRF (`-ze-opt-large-register-file`): negative for joint_matrix; ESIMD run failure. Only oneDNN's ngen 16x8 geometry benefits.
- DPAS ExecutionSize=16: compiles but produces wrong results on A770.
- `load_2d` and `named_barrier`: PVC-only; A770 hangs or rejects them.
- Deeper SLM pipelines, GRF double copies, 8 SLM buffers, 16x8 per-thread geometry, and SLM bank padding: all measured negative on A770.
- Runtime `if/else` inside an ESIMD kernel hangs; dispatch with template `if constexpr` instead.
- With `beta != 0`, restore C0 after the timing loop before the single-shot correctness check.

Complete evidence table: [pitfalls.md](references/pitfalls.md)

## 4. API Usage

- SYCL `joint_matrix`: declare A as `layout::row_major`, B as `layout::ext_intel_packed`; `joint_matrix_load` stride is in elements (packed B rows are `BN*2` bf16).
- ESIMD: `esimd::dpas<8, RepeatCount, float>(C, B, A)` with VNNI B; `slm_block_load/store`, `barrier()`, `block_load` with `overaligned_tag<16>{}`.
- `bit_cast_view<bf16>()` requires an lvalue `simd`; calling it on a temporary fails to compile.
- Windows build: `cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && icx-cl /fsycl file.cpp /Fe:file.exe'`.
- VTune: `instruction-count` needs no admin; `full-compute` needs `-allow-multiple-runs`.

Full reference: [api-usage.md](references/api-usage.md). Tested, copy-ready building blocks (VNNI packing, joint_matrix core, dpas smoke, SLM double buffer, 16x16 GEMM core, barrier pipeline, alpha/beta template dispatch, verification harness): [code-snippets.md](references/code-snippets.md).

## References

- [hardware.md](references/hardware.md): specs, DPAS/SLM/cache constraints, occupancy math, oneDNN decode.
- [techniques.md](references/techniques.md): full measured ladder, VTune-driven instruction reduction, metric interpretation.
- [pitfalls.md](references/pitfalls.md): every measured negative result and behavioral trap with numbers.
- [api-usage.md](references/api-usage.md): joint_matrix/ESIMD API forms, build/profile commands, verification workflow.
- [code-snippets.md](references/code-snippets.md): actual kernel and host-side snippets extracted from the measured variants, plus a ladder-to-snippet map that keeps the skill self-contained.
