# GEMM/GEMV Shape Crossover on A770

> `[MEASURED]` Intel Arc A770 (DG2), oneAPI 2026.1.0, Level-Zero
> `1.15.37669`, driver `32.0.101.8724`, oneDNN `3.11.2`. Operator:
> `C[M,N] = A[M,K] * B[K,N]`, row-major A/B, f32 or bf16 inputs, f32 C;
> GEMV-N1 is the `N=1` specialization `A[M,K] x[K] -> y[M]`. GEMV-M1
> (`x[1,K] W[K,N] -> y[1,N]`, weight-only decode) is a separate operator
> covered in `weight-only-gemv.md`. Re-measure before transferring to another
> GPU, driver, compiler, or oneDNN build.

## Routes

| Route | Supports |
|---|---|
| `gemv_sycl_f32` / `gemv_esimd_f32` | f32 GEMV-N1, `N=1` |
| `dpas_bf16` (DPAS8, padded rows + guard stores) | bf16, `N=1` or `N%8=0`, `K%32=0` |
| `mkl_f32` | f32 GEMV/GEMM |
| `onednn_f32` / `onednn_bf16` | f32 / bf16 GEMV/GEMM, `jit:gemm:any` |

## Protocol

- CPU reference; f32 tolerance `rel=1e-3, abs=1e-3`; bf16 tolerance
  `rel=5e-2, abs=1e-2`. Every accepted run was `errors: 0/<total>`.
- Warmup 5, samples 20 per case; reported values are SYCL-event device
  medians with p10/p90, CV, and wall median.
- oneDNN ran with `ONEDNN_VERBOSE=profile,dispatch`; all swept shapes
  selected `jit:gemm:any` with `accuracy_class=matched`,
  `comparable_for_speedup=true`.

## Headline Findings

1. f32 GEMV-N1 device map: oneDNN wins `M<=64`; oneMKL wins `M>=256`;
   `M=128` is a tie. Wall-time map differs: oneMKL has the lowest wall time
   at every M on both swept K values.
2. f32 GEMM: oneMKL and oneDNN are performance ties on every swept M/shape.
   Use oneDNN for f32 GEMM because the same `jit:gemm:any` path also covers
   bf16. oneMKL remains a reasonable wall-latency alternative.
3. bf16 GEMM: oneDNN is the champion at every M for every swept shape except
   `N=14336, K=4096, M<=16`, where the DPAS8 path wins by 11-18%.
4. DPAS8 padded-row + guard-store is correct for `M=1,2,4,8` on every swept
   GEMM shape and the bf16 `N=1` GEMV subset.
5. f32 GEMM champion alternation between oneMKL and oneDNN is within noise;
   bf16 GEMM champions are stable.

## Dispatch Map

### bf16 GEMM/GEMV

`[DISPATCH]`

- `N=14336`, `K=4096`, `M<=16`: `dpas_bf16`.
- Otherwise: `onednn_bf16` (`jit:gemm:any`).

Measured anchors:

| Shape | DPAS8 | oneDNN bf16 |
|---|---:|---:|
| `8x14336x4096` | 533.41 us | 597.10 us |
| `8x4096x4096` | 255.91 us | 196.49 us |
| `64x4096x4096` | 307.25 us | 115.83 us |
| `1x1x4096` (GEMV) | 193.90 us | 29.72 us |

### f32 GEMM

`[DISPATCH]` oneDNN for all swept M/shapes; oneMKL and oneDNN are tied within
the 3% / p10-p90 overlap rule. Choose oneMKL only when wall latency is the
measured deployment target and its wall advantage is reproduced.

### f32 GEMV-N1 (`N=1`)

`[DISPATCH]` device-event rule:

- `M<=64`: `onednn_f32`.
- `M>=256`: `mkl_f32`.
- `M=128`: tie; either path.

Wall-time rule for deployment: `mkl_f32` at all M. oneDNN's tiny-M device win
is offset by 70-200 us of host/wall overhead. Streaming SYCL and ESIMD GEMV
routes are not dispatch champions anywhere in this sweep.

### Discrete-to-Inequality Caveat

The sweep measured M values `{1,2,4,8,16,32,64,128,256,512,1024}`. Rules that
read as intervals (`M<=16`, `M<=64`, `M>=256`) are `[DISPATCH]` only on those
exact rows and `[HEURISTIC]` for unswept integers such as 3, 7, 12, 48, or
192. Before promoting an interval to a continuous dispatch, run a boundary
interpolation sweep around each decision edge.

## Validity Domains

- bf16: operator `gemm`/`gemv`; dtype `bf16`; M in swept subset; N/K in
  `{4096,8192,11008,14336}x{4096,14336}` plus `N=1`; row-major A/B; f32 C;
  A770 / oneAPI 2026.1.0 / driver `32.0.101.8724` / oneDNN 3.11.2.
- f32 GEMM: dtype `f32`; M list `1..1024`; swept N/K pairs; same device.
- f32 GEMV-N1: `N=1`; K `{4096,14336}`; M list `1..1024`; same device.

Confidence: `[MEASURED]` on the exact swept rows, `[HEURISTIC]` outside them.

## oneDNN Baseline Contract

| Field | Value |
|---|---|
| Primitive | `dnnl::matmul` |
| Implementation | `jit:gemm:any` for every swept shape/dtype |
| Tags | src/weights `ab`, dst `ab` |
| `semantics_id` | `gemm_mkn_rowmajor` / `gemv_n1_mkn_rowmajor` |
| Post-ops | none |
| `fpmath_mode` | default (no attribute set) |
| Accuracy class | matched |
| Correctness | PASS for all swept runs |
| Comparable | true |

## Negative Results

- `[BUG]` ESIMD 16x16 SLM-staged DPAS GEMM triggers reproducible
  `UR_RESULT_ERROR_DEVICE_LOST` on this stack; keep it as a standalone
  negative path.
- `[TOOLCHAIN]` ESIMD 16x16 global-memory DPAS GEMM crashes with Windows
  access violation `0xC0000005`; the shipped DPAS route uses the proven
  DPAS8 tile form.
- `[MEASURED]` `gemv_sycl_f32` and `gemv_esimd_f32` are slower than the
  library baselines at every M in this sweep.
- `[MEASURED]` `dpas_bf16` is non-champion on most GEMM shapes; it is useful
  only in the wide `N=14336,K=4096,M<=16` region.
- `[CORRECTNESS]` oneMKL row-major A/B must be called through a column-major
  view (`gemm(N,M,K,B,N,A,K,C,N)` and `gemv(trans,K,M,...)`); passing
  row-major dimensions directly caused device loss and wrong results.

## Reproduction

```powershell
python build.py
python sweep.py --samples 20 --warmup 5
```

Full per-cell tables are maintained in the campaign's machine-readable
results; this page keeps the condensed dispatch surface. Packaging the
campaign sweep runner under `scripts/sweeps/gemm_crossover.py` is pending; the
commands above currently run from the campaign checkout, not from this skill
root.
