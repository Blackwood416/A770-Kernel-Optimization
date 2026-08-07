# A770 Execution Model Reference

Measured on Intel Arc A770, Level-Zero backend, driver `32.0.101.8724`,
oneAPI 2026.1, VTune 2026.2.

> Evidence: `[MEASURED]` pack + bf16 joint-matrix GEMM, `1024x1536x512`,
> CPU f32 reference, 30 warmup + 200 timed iterations, 3 stable runs.

## Environment and Task

- Two-stage task: VNNI pack of B plus a joint-matrix bf16 GEMM,
  `M=1024, N=1536, K=512`.
- CPU f32 reference over bf16 inputs, `errors: 0/1572864`; 30 warmup +
  200 timed iterations, 3 stable runs.
- Wall time is measured around submission and `q.wait()`. Device times come
  from SYCL profiling events; graph node times were cross-checked with VTune
  `xpu-offload` because graph does not expose per-node SYCL events.

## Submission Forms

| Form | Wall avg us | Pack us | GEMM us | Launch gap us | Host overhead us |
|---|---:|---:|---:|---:|---:|
| same-queue | 288.23 | 17.62 | 148.65 | 18.36 | 121.96 |
| dual-queue + event | 271.13 | 17.97 | 149.00 | 15.34 | 104.17 |
| fused single kernel | 335.81 | 0 | 241.92 | 0 | 93.88 |
| graph batch submit | 260.64 | 16.92 VTune | 161.25 VTune | n/a | ~93.7 est / ~82.6 VTune |
| host-pack-out | 235.51 | 0 | 153.63 | 0 | 81.86 |
| host-pack-in | 1311.30 | 0 | 153.27 | 0 | 383.31 + CPU 774.73 |

## Rules

1. Measure launch and host overhead before optimizing a small kernel. On this
   shape the same-queue host overhead is about `122 us`, close to the GEMM
   kernel time of `149 us`.
2. Fuse only when `fused device delta < launch gap + host overhead saved`.
   Measured here: fused adds `241.92 - 148.65 = 93.27 us` of device time but
   removes only `18.36 + 28.08 = 46.44 us`, so it loses about `47 us` per
   iteration.
3. Use graph when the same graph is reused: it was fastest at about `260.6 us`,
   saving roughly `28 us` per invocation versus dual-queue or same-queue.
4. Use dual-queue + event when two queues already exist: it saved about
   `17 us` versus same-queue.
5. Use host preprocessing only outside the timed loop when inputs are static.
   Host pack outside was fastest at `235.51 us`; inside the loop it reached
   `1311 us` because CPU pack plus USM visibility overhead dominates.
6. Cross-check graph node times with VTune `xpu-offload`; SYCL graph does not
   expose per-node profiling events on this stack.

## Negative Results

- Requesting 16-lane sub-groups with `sub_group_size<16>` or
  `[[intel::reqd_sub_group_size(16)]]` on this 1D/2D joint-matrix kernel
  triggers an IGC internal divide-by-zero error on oneAPI 2026.1. Use the
  stable 2D `(8, 64)` local mapping.
- The second 8-column B slice must be addressed as 8 `uint32` words past the
  first slice, i.e. `+16 bf16` after casting to `bf16*`; `+8 bf16` silently
  produces half-wrong output columns.

## Reproduction

Build the two-stage benchmark with the standard oneAPI Windows commands in
[api-usage.md](../api/api-usage.md), then profile each submission form with
SYCL events and cross-check graph node times with VTune `xpu-offload`.

Copy-ready cores: [execution graph/dual-queue cores](../api/code-snippets.md#execution-graph-dual-queue-cores).
