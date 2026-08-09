# Irregular Shapes and Sparse GEMM on A770

Measured on Intel Arc A770, oneAPI 2026.1, Level-Zero, f32 64 B aligned USM.

> Evidence: `[MEASURED]` f32 softmax rows=64 cols 256-4096; GEMV M=64/M=96
> cols 256-4096; sparse GEMM `M=K=512, N=8`. The sparse and fallback rules
> are `[HEURISTIC]` outside these domains.

## Softmax Shape Cost

Rows fixed at 64. `fast` is the 64 B aligned `vec<float,16>` SLM path,
`fallback` is the scalar-chunk SLM path, `naive` is one work-item per row.
oneDNN uses `softmax_accurate`, axis=1. Times are wall-time means per run:
100 warmup + 1000 timed launches with `q.wait()`, repeated 3 times.

| cols | fast vec | fallback scalar | naive row | oneDNN |
|---:|---:|---:|---:|---:|
| 256 | 17.827 | 14.708 | 136.027 | 28.307 |
| 512 | 10.513 | 15.152 | 266.288 | 14.145 |
| 1024 | 15.413 | 24.901 | 557.034 | 23.360 |
| 1025 | N/A | 18.039 | 1099.013 | 18.362 |
| 2048 | 11.522 | 23.500 | 1103.938 | 35.781 |
| 2011 | N/A | 23.302 | 2205.647 | 21.617 |
| 4096 | 17.930 | 37.416 | 2201.133 | 31.789 |
| 1536 runtime | 14.295 | 20.256 | 825.093 | 33.938 |

Rules:

1. `[HEURISTIC]` fast path only for `cols % 16 == 0 && cols >= 512`. For
   `cols <= 256`, the vector path can be slower than fallback because
   launch/wave overhead dominates.
2. WG sizing: 16 threads for `cols <= 256`, 32 for `cols <= 1024`, 128
   otherwise.
3. Non-16-aligned or short rows use the scalar-chunk SLM fallback, never the
   naive row-per-item kernel.
4. Rows above 8192 cols: read max directly from global and stage only sum +
   normalize tiles in SLM.

## GEMV Shape Cost

Default `M=64`; dynamic case `M=96, N=1536`. `fast` is sub-group-per-row with
`vec<float,16>` and per-lane trip counts from the actual sub-group size.
oneDNN baseline is `Kx1` matmul, verified against the CPU reference.
Times are wall-time means per run: 100 warmup + 1000 timed launches with
`q.wait()`, repeated 3 times. Device/pipeline medians were not recorded in
this campaign and must not be mixed with SYCL-event tables.

| cols | fast vec16 | scalar row | oneDNN Kx1 |
|---:|---:|---:|---:|
| 256 | 12.609 | 49.162 | 37.215 |
| 512 | 10.919 | 93.486 | 37.562 |
| 1024 | 16.895 | 183.849 | 33.812 |
| 1025 | N/A | 586.166 | 23.845 |
| 2048 | 28.239 | 355.590 | 42.048 |
| 2011 | N/A | 1159.231 | 58.487 |
| 4096 | 51.445 | 709.487 | 40.890 |
| 1536 runtime | 24.702 | 268.388 | 42.696 |

Rules:

1. `[HEURISTIC]` fast path requires `M % 32 == 0 && N % 16 == 0`; never
   hardcode a 16-lane per-lane trip count because A770 may compile with
   32-lane sub-groups.
2. `[MEASURED]` f32 M=64 on A770: N=1025 is about `34.7x` slower than N=1024
   and N=2011 is about `41.1x` slower than N=2048. Re-measure for other M/N
   before reuse. If such shapes are common, add a padded/tail-vector fast
   path instead of staying scalar.

## Gather / Scatter

At lengths 1024/1025/2048/2011/1536 with unique permutation indices, gather
and scatter times sit at about 17-20 us with no repeatable random-vs-sequential
gap. This size is launch-bound; use hundreds of thousands of elements plus
`vec`/coalesced control variables before drawing memory-behavior conclusions.

## Sparse GEMM

Operator: `C[M,N] = A[M,K] * B[K,N]`, `M=K=512`, `N=8`, f32. CSR direct
traverses nnz, dense filter scans all elements, BSR tiles use
`vec<float,8>` multiply-add.

| sparsity | CSR direct | dense filter | BSR B4 | BSR B8 | BSR B16 |
|---:|---:|---:|---:|---:|---:|
| 50% | 297.958 | 245.994 | 168.290 | 260.401 | 1359.107 |
| 90% | 65.121 | 246.569 | 143.852 | 255.997 | 1359.031 |
| 99% | 30.945 | 153.391 | 38.005 | 166.977 | 1402.758 |

Rules:

1. `[HEURISTIC]` (`M=K=512, N=8, f32`) `sparsity >= 90%`: CSR direct
   traversal.
2. `[HEURISTIC]` (`M=K=512, N=8, f32`) `sparsity ~50%`: BSR B4 with
   `vec<float,8>` output accumulation.
3. Dense filter is a last resort: it scans the full dense matrix even at 99%
   sparsity.
4. `[MEASURED]` avoid BSR tiles that leave too few work-items: at `M=512`,
   B16 has only 32 row-block work-items and is launch-bound. Split large row
   blocks across work-groups instead.

## Negative Results

- Non-16-aligned GEMV fallback is the largest gap at the measured f32 M=64
  pairs: N=1025 is 34.7x slower than N=1024 and N=2011 is 41.1x slower than
  N=2048.
- BSR B16 at M=512 is launch-bound with only 32 row-block work-items and is
  not a fair tile-size comparison.
- At the tested gather/scatter lengths (~1-2K), there is no repeatable
  random-vs-sequential gap; the measurements are launch-bound and do not yet
  characterize irregular memory behavior.

## Reproduction

Build and run the shape/fallback/sparse benchmarks with the standard oneAPI
Windows commands in [api-usage.md](../api/api-usage.md).

Copy-ready cores: [softmax fast/fallback](../api/code-snippets.md#softmax-fast-fallback-cores),
[GEMV fast core](../api/code-snippets.md#gemv-fast-core),
[sparse CSR/BSR cores](../api/code-snippets.md#sparse-csr-bsr-cores).
