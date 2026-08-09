# GEMM/GEMV Boundary Interpolation Sweep

> `[MEASURED]` Intel Arc A770 / oneAPI 2026.1.0 / driver `32.0.101.8724` /
> oneDNN `3.11.2`. Paired/interleaved runs are preferred for champion
> decisions; absolute runs cover the non-champion streaming routes.

## Protocol

- Paired cases: AB/BA interleaved rounds, 10 samples per run, 2 rounds per
  route; warmup 3.
- Every executable emitted the correctness JSON contract; accepted rows have
  `errors: 0`, matching requested/executable tolerance, and
  `relaxed_accuracy=false`.

## Continuous Dispatch Decisions

### bf16 GEMM `N=14336, K=4096`

`[DISPATCH]`

- `M <= 16`: `dpas_bf16`.
- `M >= 24`: `onednn_bf16`.
- `17 <= M <= 23`: not measured; `[HEURISTIC]`.

Representative paired anchors:

| M | DPAS8 | oneDNN | champion |
|---:|---:|---:|---|
| 16 | 495.07 us | 589.09 us | DPAS8 |
| 24 | 917.57 us | 306.56 us | oneDNN |
| 32 | 1.039 ms | 310.65 us | oneDNN |
| 64 | 2.324 ms | 317.80 us | oneDNN |

### f32 GEMV-N1 (`N=1`)

`[DISPATCH]`

- `M <= 64`: `onednn_f32`.
- `M >= 192`: `mkl_f32`.
- `65 <= M <= 191`: not covered by a single continuous rule; `[HEURISTIC]`.
  K=4096 switches at M=96; K=14336 switches between M=128 and M=192.

Representative paired anchors (K=4096):

| M | oneMKL | oneDNN | champion |
|---:|---:|---:|---|
| 64 | 12.13 us | 8.15 us | oneDNN |
| 96 | 12.12 us | 15.40 us | oneMKL |
| 128 | 13.17 us | 15.25 us | oneMKL |
| 192 | 16.24 us | 21.91 us | oneMKL |

## Rules

- Streaming SYCL/ESIMD GEMV routes remain non-champion everywhere.
- Do not claim `[DISPATCH]` for the two unmeasured gaps (`17-23` and
  `65-191`); sample those before shipping a boundary dispatch.
- The original discrete-sweep table remains valid for its exact rows; this
  boundary sweep upgrades only the intervals explicitly listed above.

Full per-M tables are maintained in the campaign; this page keeps the
continuous dispatch surface.
