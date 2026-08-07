# Numerical Precision and Tolerance Reference

Measured on Intel Arc A770 (Level Zero, driver `32.0.101.8724`), oneAPI
2026.1, `icx-cl /fsycl /O2 /EHsc`.

## Method and Scope

- Shapes: GEMM `256x256x256`; Softmax `256x1024`; row reduction `256x4096`.
- Every result is compared to a f64 CPU reference; timing is 20 warmup +
  100 timed launches, 3 runs, median.
- Reference is computed from original f64 data, so error includes storage
  quantization and output quantization.
- A770 supports f32, bf16, f16, int8. There is no SYCL fp8 type and no
  Xe-HPG fp8 DPAS path on oneAPI 2026.1, so fp8 is excluded.
- Cell format below: `max_abs_err / time(ms) / fail@1e-3` where
  `fail` means `err > 1e-3 * max(|ref|, 1e-3)`.

## GEMM Error-Performance (normal data)

| Precision / accumulator | default | fastmath | oneDNN strict | oneDNN any |
|---|---|---|---|---|
| f32 / f32 | 9.10e-06 / 0.0964 / 3 | 9.10e-06 / 0.0889 / 3 | 3.53e-06 / 0.4239 / 1 | 3.53e-06 / 0.4114 / 1 |
| bf16 / f32 | 5.05e-02 / 0.1351 / 52671 | same | 5.05e-02 / 0.6922 / 52671 | 5.05e-02 / 0.4494 / 52671 |
| bf16 / bf16 | 5.84e-01 / 0.1547 / 63053 | same | 5.05e-02 / 0.4516 / 52671 | 5.05e-02 / 0.4640 / 52671 |
| f16 / f32 | 9.29e-05 / 0.1337 / 12416 | same | 9.29e-05 / 0.4190 / 12418 | 9.29e-05 / 0.4294 / 12418 |
| f16 / f16 | 1.28e-03 / 0.1356 / 47010 | same | 9.29e-05 / 0.6577 / 12418 | 9.29e-05 / 0.4291 / 12418 |
| int8 / int32 | 0 / 0.1191 / 0 | 0 / 0.1172 / 0 | 0 / 0.4282 / 0 | 0 / 0.5614 / 0 |

## Softmax Error-Performance (normal data)

| Precision / accumulator | default | fastmath |
|---|---|---|
| f32 / f32 | 4.95e-10 / 0.0174 ms / 0 | 4.95e-10 / 0.0155 ms / 0 |
| bf16 / f32 | 1.09e-05 / 0.0261 ms / 134496 | 1.09e-05 / 0.0137 ms / 134496 |
| f16 / f32 | 5.09e-07 / 0.0148 ms / 0 | 5.09e-07 / 0.0146 ms / 0 |
| int8 / f32 | 0 / 0.0155 ms / 0 | 0 / 0.0148 ms / 0 |

## Reduction Error-Performance (normal data)

| Precision / accumulator | default | fastmath |
|---|---|---|
| f32 / f32 | 1.52e-04 / 0.766 ms / 0 | 1.52e-04 / 0.715 ms / 0 |
| bf16 / f32 | 2.50e-01 / 0.700 ms / 204 | same |
| bf16 / bf16 | 8.74e+00 / 0.681 ms / 254 | same |
| f16 / f32 | 4.01e-03 / 0.632 ms / 23 | same |
| f16 / bf16 | 2.29e+00 / 0.715 ms / 252 | same |
| int8 / int32 | 0 / 0.413 ms / 0 | 0 / 0.411 ms / 0 |

## Tolerance Recommendations

Use `err <= rel * |ref| + abs`. Recommended bounds for the measured shapes:

| Precision | GEMM | Softmax | Reduction |
|---|---|---|---|
| int8 | `rel=0, abs=0` | `abs=0.5` on quantized output | `rel=0, abs=0` |
| f32 | `rel=1e-4, abs=1e-5` | `rel=1e-6, abs=1e-8` | `rel=1e-3, abs=1e-3` |
| f16 + f32 acc | `rel=1e-2, abs=1e-3` | `rel=1e-4, abs=1e-6` | `rel=1e-2, abs=1e-2` |
| f16/bf16 acc | relax to `rel=0.2` or use integer/fixed-point checks | - | relax to `rel=1, abs=1` |
| bf16 + f32 acc | `rel=5e-2, abs=1e-2` | `rel=2e-2, abs=1e-4` | `rel=2e-2, abs=0.5` |
| bf16/bf16 acc | `rel=0.5, abs=0.5` | - | `rel=1, abs=1` |

## Math Mode and FTZ Observations

- On these naive SYCL kernels, `-ffast-math` barely changes GEMM/reduction
  results and only shifts the f32 softmax tail error distribution. Do not
  assume fast-math changes codegen on simple bf16 loops (see
  [codegen.md](../workflow/codegen.md)).
- oneDNN `fpmath_mode::any` keeps the same bf16 GEMM error as strict, and its
  time is slower than the naive SYCL kernel; this is a different optimization
  level, so it is not a math-mode performance comparison.
- On tiny data, FTZ flushes f32/bf16 products that underflow to f32 subnormals
  to zero. If the application allows underflow, use an absolute tolerance or
  pre-scale the data instead of a relative one.
- f16 with a f32 accumulator keeps FTZ impact small; true f16 subnormal
  behavior appears in f16-accumulator variants.

## Rules

1. Keep the accumulator in f32 for bf16/f16 GEMM and reduction kernels; a
   bf16 or f16 accumulator can push reduction error to O(0.1)-O(1).
2. int8 paths were exact for the measured small-integer data; they can use
   tight integer tolerances.
3. Before switching math modes, compare against the same-precision reference,
   not a higher-precision one, and keep the error histogram.
4. Treat fp8 as unavailable on this stack until a SYCL fp8 type and DPAS fp8
   path exist.

## Negative Results

- bf16/bf16 and f16/bf16 accumulators raise reduction error to O(0.1)-O(1);
  keep the accumulator in f32 for GEMM and reduction kernels.
- On tiny data, FTZ clears f32/bf16 products that underflow to f32 subnormals,
  so relative tolerance can report large failures where absolute tolerance is
  the correct check.
- fp8 is unusable on A770 + oneAPI 2026.1: no SYCL fp8 type and no Xe-HPG fp8
  DPAS path.

## Reproduction

Build and run the precision sweep with the standard oneAPI Windows commands
in [api-usage.md](../api/api-usage.md).

Copy-ready tolerance helper: [precision tolerance helper](../api/code-snippets.md#precision-tolerance-helper).
