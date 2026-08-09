# Optimization Methodology and Mental Model

This skill is a workflow and guidance layer, not a full optimizer. The
measured case studies below are evidence to reason from; they do not require
every new shape to become a formal `[DISPATCH]`.

## New Operator: First Thoughts

1. Write the operator mathematically and name its layout:
   `A*x` vs `x*W`, NCHW vs NHWC, `[M,K]` row-major vs packed VNNI, causal or
   full attention. The same word "GEMV" can mean two different memory
   problems (`GEMV-N1` vs `GEMV-M1`).
2. Estimate arithmetic intensity and pick the matching bandwidth pattern:
   `B_DRAM_contiguous` for coalesced streaming, `B_DRAM_strided` for
   strided, `B_L2_reuse` for reused tiles, `B_SLM` for staged data.
3. Build a CPU reference first. Every later number is meaningless until the
   operator and tolerance are fixed.
4. Benchmark the library baseline, but verify semantics first: oneDNN
   `1xK` can compute `x*A`, and a fast path can be `invalid` rather than
   `fastest_only`.
5. Start with the simplest correct kernel. Then profile before restructuring:
   launch/host time, instruction count, Send, ALU, occupancy, bandwidth.
6. For tiny kernels, use paired/interleaved comparisons when a champion
   matters; otherwise treat small margins as ties and pick the simpler
   variant.
7. Choose the next experiment from the bottleneck, not from a fixed list of
   variants.

## Why Optimizations Work / When They Fail

| Pattern | Why it works | When it fails |
|---|---|---|
| SLM staging of rows/tiles | Replaces repeated global reads with one read + on-chip reuse | Tiny shapes where launch/barriers dominate, or data already fits L2 |
| Wider vector/block messages | Fewer messages and better DRAM efficiency | Alignment/tail cases, launch-bound kernels, small payloads |
| DPAS/XMX | Fits compute-bound GEMM with packed operands | Tiny M, memory-bound loops, mixed u4/f16 unsupported on this stack |
| Host preprocessing | Removes per-call unpack/pack work | Inputs change per call; GEMV-M1 weight-change cost can dominate |
| Kernel fusion | Removes launch gap + host overhead | Added device time exceeds the overhead saved |
| Library baseline | JIT/ngen path is mature and accurate | Wrong operator, `ocl:ref` fallback, or fastest-only accuracy |

## Profiling-Driven Next Experiment

- Host/wall dominates: measure `execution.md` launch gap; try graph or
  dual-queue, or reduce host transforms.
- Instruction-bound: use `codegen.md` flags/unroll, remove redundant address
  math, vectorize loads.
- Memory-bound: reduce bytes moved or improve pattern; if fully coalesced and
  near `B_DRAM_contiguous`, do not chase bandwidth, reduce work.
- Barrier/occupancy limited: shrink SLM tile or work-group; check VTune
  occupancy/barrier counts.
- Library far faster: keep the implementation string. If `jit:gemm:any`, the
  gap is real; if `ocl:ref`, the comparison is not a JIT baseline.

## From a Failed Experiment to a Model Update

1. Keep the failed variant and record the exact shape, measurement, and the
   hypothesis it tested.
2. Separate structural failure from measurement noise:
   - Structural: bottleneck did not change or got worse.
   - Noise: paired/interleaved delta is small or flips.
3. Update the mental model, not just the table. Examples:
   - `prefetch` is negative when the working set fits L2.
   - Large GRF is context-dependent, not universally positive/negative.
   - A 3% independent gap is a simplification threshold, not significance.
4. Re-run the affected measured case study before promoting the update to a
   `[DISPATCH]` or `[HEURISTIC]`.

## Scope Guard

Do not formalize every discrete shape into a dispatch. Use the measured
campaigns to derive regions, keep unswept gaps `[HEURISTIC]`, and prefer
simple stable variants when margins are small.
