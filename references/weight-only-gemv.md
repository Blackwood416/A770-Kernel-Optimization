# Weight-Only Decode GEMV: INT4 / NF4 on Arc A770

Operator: `y = x * W`, where `x` is bf16/f16 `[M,K]` with M in `{1,64}`, `W`
is INT4/NF4 `[K,N]`, and `y` is f32/bf16. Packed layout is `[N,K/2]`, low
nibble first along K; scales are f16 `[G,N]`; INT4 uses zero point 8.

Evidence scope: `[MEASURED]` on Intel Arc A770 / oneAPI 2026.1 / driver
`32.0.101.8724`, N/K in `{4096,8192}`, group size in `{32,64,128,256}`.
Re-measure before transferring to other GPUs, drivers, or shapes.

## Three Routes Compared

| Route | Idea | M=64 representative |
|---|---:|
| R1 | packed load -> unpack/dequant -> FMA | 13.7-41.5 ms |
| R2 | packed load -> device unpack -> VNNI SLM tile -> ESIMD DPAS | 2.85-5.65 ms |
| R3 | host dequant + VNNI pack -> DPAS control | 0.25-0.65 ms |
| oneDNN | u4 `ba` + f16 scales + zp8 + `fpmath_mode::any` | 0.14-0.32 ms |

`[MEASURED]` M=64 mean device-time ratios over the 16 swept INT4 shapes:
R1/R2 = 5.96x, R2/R3 = 9.91x, R1/R3 = 59x, R3/oneDNN = 1.99x.

Group-size crossover: `[DISPATCH]` none within `{32,64,128,256}`; all routes
are flat within about 5%. `gs=128` is a `[HEURISTIC]` default, not a measured
dispatch rule.

M=1 crossover: `[MEASURED]` R2 is a negative path (`4.47 ms` at
`1x4096x4096`) because only 4 work-items per work-group are used; R1 is
`1.76 ms`, R3 is `0.126 ms`, oneDNN is `0.036 ms`.

## oneDNN Baseline

`[DISPATCH]` matmul config that selects `jit:gemm:any`:

- src `{M,K}` bf16/f16 `ab`
- weights `{K,N}` u4 `ba` (`[N,K/2]`, low nibble first)
- dst `{M,N}` bf16/f32 `ab`
- scales `{G,N}` f16 `ab`, mask `(1<<0)|(1<<1)`, dims `{group_size,1}`
- scalar u8 zero point = 8
- `fpmath_mode::any`

`[MEASURED]` correctness caveat: M=1 passes a 5% precast comparison; every
M=64 INT4 shape fails 5% (representative `6072/262144`, max abs error `3.87`
at `64x4096x4096`, gs=128). Use the oneDNN number as a timing baseline, not an
exact-output oracle.

## Why Host Pre-Dequant Is Not a Weight-Only Solution

`[MEASURED]` one-shot host dequant+VNNI pack costs `101.8 ms` at
`4096x4096` and `304.3 ms` at `8192x8192`, versus 0.25-0.65 ms kernel time.
Dequantized bf16 VNNI weights are 4x the packed size. The u4-vector GEMV trick
does not transfer because weights are O(K*N), not a `[1,K]` vector shared by
every row.

## Negative Results and Pitfalls

- `[BUG]` packed W byte offset for K offset `bk` is `bk / 2`, not
  `bk * (BK/2)`. The latter silently reads into the next K block from block
  two onward.
- `[BUG]` M=1 DPAS kernels must read row `min(global_row, m-1)` and skip
  stores with `gr >= m`; otherwise they read rows 1..7 of a one-row buffer and
  can trigger `UR_RESULT_ERROR_DEVICE_LOST`.
- `[MEASURED]` NF4 R1/R2 are slower than INT4 (28%/18% at M=64) because the
  codebook lookup is scalar in the device unpack path.
- `[ARCH]` direct mixed u4/f16 DPAS is not available on stable oneAPI 2026.1;
  R2 unpacks to bf16/f16 VNNI before DPAS.
