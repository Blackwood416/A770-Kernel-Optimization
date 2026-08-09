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
R1/R2 = 5.96x, R2/R3 = 9.91x, R1/R3 = 59x. Against the accuracy-matched
oneDNN baseline below, R3 device-time ratio median is 2.071x (min 1.482,
max 3.142), and wall-time ratio median is 1.170x. If one-shot host
predequant+VNNI pack is included, R3 pipeline is about 731x slower per
weight change; that is a weight-change cost, not a kernel speedup.

Group-size crossover: `[MEASURED]` none within `{32,64,128,256}`; all routes
are flat within about 5%. `gs=128` is a `[HEURISTIC]` default, not a measured
dispatch rule.

M=1 crossover: `[MEASURED]` R2 is a negative path (`4.47 ms` at
`1x4096x4096`) because only 4 work-items per work-group are used; R1 is
`1.76 ms`, R3 is `0.126 ms`, oneDNN is `0.036 ms`.

## oneDNN Baseline

### oneDNN-Matched (selected)

`[DISPATCH]` The accuracy-matched baseline for M=64 speedup calculation:

- src `{M,K}` f16 `ab`
- weights `{K,N}` u4 `ba` (`[N,K/2]`, low nibble first)
- dst `{M,N}` f32 `ab`
- scales `{G,N}` f16 `ab`, mask `(1<<0)|(1<<1)`, dims `{group_size,1}`
- scalar u8 zero point `{1}` = 8
- `fpmath_mode::strict` with `apply_to_int=true` (`any` also passes)
- implementation `jit:gemm:any`
- `accuracy_class=matched`, `comparable_for_speedup=true`

`[MEASURED]` 16 f16-src configurations (strict/any x f16/f32 scales x
scalar/group zp x bf16/f32 dst) all passed `rel=5e-2, abs=1e-2` on all 16
swept shapes, and all within about 3% on the representative
`64x4096x4096, gs=128` case. No fpmath/scale/zp dispatch rule is claimed; the
selected form is the simplest stable member.

Representative matched timings (device median):

| Shape | oneDNN | R3 | ratio |
|---|---:|---:|---:|
| `64x4096x4096, gs=128` | 0.1577 ms | 0.2544 ms | 1.61x |
| `64x8192x8192, gs=128` | 0.2089 ms | 0.6563 ms | 3.14x |

### oneDNN-Fast Only (bf16 src)

`[MEASURED]` every bf16-src configuration selected `jit:gemm:any` but failed
the required tolerance on all 16 shapes. Representative
`64x4096x4096, gs=128, bf16->bf16`: `6072/262144` errors, max abs `3.873`;
`bf16->f32`: `6066/262144` errors, max abs `2.099`. Classify bf16-src oneDNN
as a fastest-library lower bound, not an accuracy-matched baseline.

### Toolchain Gate

`[TOOLCHAIN]` on oneDNN 3.11.2, leaving `fpmath_mode` unset rejects the u4
primitive descriptor on GPU (`dnnl_error`, status 3). Working forms are
`strict` or `any` with `apply_to_int=true`. `fpmath_mode::high` does not exist
in oneDNN 3.11.2.

Record these fields in every baseline artifact:
`accuracy_class`, `reference_tolerance`, `baseline_correctness_status`,
`comparable_for_speedup`.

## Why Host Pre-Dequant Is Not a Weight-Only Solution

`[MEASURED]` one-shot host dequant+VNNI pack costs `101.8 ms` at
`4096x4096` and `304.3 ms` at `8192x8192`, versus 0.25-0.65 ms kernel time.
Dequantized bf16 VNNI weights are 4x the packed size. The u4-vector GEMV trick
does not transfer because weights are O(K*N), not a `[1,K]` vector shared by
every row.

## Negative Results and Pitfalls

- `[CORRECTNESS]` packed W byte offset for K offset `bk` is `bk / 2`, not
  `bk * (BK/2)`. The latter silently reads into the next K block from block
  two onward.
- `[CORRECTNESS]` M=1 DPAS kernels must read row `min(global_row, m-1)` and skip
  stores with `gr >= m`; otherwise they read rows 1..7 of a one-row buffer and
  can trigger `UR_RESULT_ERROR_DEVICE_LOST`.
- `[CORRECTNESS]` the oneDNN-matched result is tied to the f16-precast CPU
  reference and `rel=5e-2, abs=1e-2`; it does not imply bitwise equality.
- `[MEASURED]` NF4 R1/R2 are slower than INT4 (28%/18% at M=64) because the
  codebook lookup is scalar in the device unpack path.
- `[TOOLCHAIN]` direct mixed u4/f16 DPAS is not available on stable oneAPI 2026.1;
  R2 unpacks to bf16/f16 VNNI before DPAS.
- `[TOOLCHAIN]` default `fpmath_mode` (attribute unset) is rejected on this u4
  path; use `strict` or `any` with `apply_to_int=true`.
