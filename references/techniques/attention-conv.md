# Attention and Convolution Ladders on A770

Measured on Intel Arc A770, oneAPI 2026.1, standard SYCL + USM, f32.

> Evidence: `[MEASURED]` attention `B=4, H=16, Q=128, KV=256, D=64` f32;
> conv `N=4, IC=32, IH=IW=64, OC=64, 3x3` f32. Rules are `[HEURISTIC]` for
> decode attention, GQA/MQA, paged KV, and other conv shapes.

## Attention Ladder

Shape: `B=4, H=16, Q=128, KV=256, D=64, kv_start=128`, causal except
`online_full_kv`. Q/K/V and output are f32. All variants correct,
`errors: 0/524288`; warmup 20 + timed 200, 3 runs averaged.

| Variant | Avg ms | Notes |
|---|---:|---|
| `naive_3kernel` | 1.8152 | QK^T to global S, row softmax to P, then P*V |
| `slm_tile_online` | 2.0132 | Q/K/V/S tiles in SLM, online max/sum |
| `online_full_kv` | 2.0338 | scores in registers, single QK^T |
| `online_causal_fused` | 2.0595 | skip masked KV blocks at block boundaries |
| `online_direct_l2` | 3.2116 | no SLM, direct L2 re-reads of K/V |
| `kv_cache_chunk_layout` | 2.0705 | K/V pre-packed as `[head][chunk][BN][D]` |

On this small f32 shape the naive three-kernel path is still the fastest
because it has the most parallelism and S/P fit in L2. The flash-style
variants prove the online/causal/KV-cache structures are correct and stable,
and the earlier one-row-per-thread version was reduced from about 23 ms to
about 2 ms. `[HEURISTIC]` for `seq <= 256` without an O(QKV) memory bound,
prefer naive or the library baseline; decode attention needs its own sweep.

### Attention Negative Results

- `BM=64, BN=32, VLEN=8, WG=512, ~40 KB SLM` triggered Level Zero
  `UR_RESULT_ERROR_DEVICE_LOST` and required a manual GPU reset. Keep BM=32
  for this shape and do not re-run the failed geometry without a watchdog.
- `online_direct_l2` (3.21 ms) is slower than the SLM online path (2.06 ms):
  each d-slice of a query row redundantly re-reads K/V, so cooperative SLM
  staging wins, consistent with the RMSNorm/softmax SLM-vs-direct evidence.
- Row-internal d-slice scan: `VLEN=16` ~5.3 ms, `VLEN=8` ~3.7 ms, `VLEN=4`
  ~4.0 ms; 8 d-slices per row was the sweet spot for this shape.

## Conv Ladder

Shape: `N=4, IC=32, IH=IW=64, OC=64, KH=KW=3, stride=1`, output
`4x64x62x62`, `errors: 0/984064` for all variants.

| Variant | Avg ms | Notes |
|---|---:|---|
| `direct_nchw_naive` | 0.2234 | NCHW output, channel stride OC |
| `direct_nhwc_naive` | 1.3352 | NHWC output without cache blocking |
| `direct_nhwc_cacheblock_oc16` | 0.2976 | `[KH][KW][IC][OC]` weights |
| `direct_nhwc_cacheblock_oc64` | 0.1605 | full OC cache block, best SYCL |
| `im2col_gemm_tile32` | 0.3330 | SLM tile32 + GEMM |
| `im2col_gemm_tile64` | 2.3308 | SLM tile64 loses on occupancy/remainder |
| `im2col_gemm_tile32_direct` | 0.3349 | register blocked, direct global |
| `im2col_gemm_tile64_direct` | 0.4251 | still worse than tile32 |
| `onednn_conv_f32` | 0.0935 | strongest baseline |

### Conv Rules

1. `[MEASURED]` (`N=4, IC=32, 64x64, OC=64, 3x3, f32`) the SYCL winner is
   `direct_nhwc_cacheblock_oc64` at `0.1605 ms`; oneDNN `0.0935 ms` remains
   the same-operator baseline.
2. `[HEURISTIC]` NHWC only pays off with OC cache blocking and
   `[KH][KW][IC][OC]` weight layout; bare NHWC direct was about 6x slower
   than NCHW direct in this campaign.
3. `[MEASURED]` enlarging the OC cache block from 16 to 64 cut time from
   `0.298` to `0.160 ms` by increasing input reuse per thread.
4. `[MEASURED]` im2col+GEMM did not beat OC64 direct on this shape. Use GEMM
   tile32, not SLM tile64; the latter loses to SLM occupancy, barriers, and M
   remainder.
5. The GEMM matrix width is `OC=64`, not batch `N=4`; im2col K is
   `[KH][KW][IC]`, matching `w_hwio`. These are the two classic correctness
   traps.
6. Output channel stride must equal the output channel count: NCHW output
   uses `OC`; NHWC output uses `OC`. Input layout uses `IC`.

### Conv Negative Results

- Bare NHWC direct conv is about 6x slower than NCHW direct; NHWC only wins
  with OC cache blocking and the `[KH][KW][IC][OC]` weight layout.
- im2col+GEMM did not beat the OC64 direct kernel; SLM tile64 was especially
  bad (2.33 ms) due to occupancy, barriers, and M remainder.
- `im2col_gemm_tile64_direct` still lost to tile32, so the winner is not
  simply "SLM vs direct"; tile size dominates on this shape.

## Reproduction

Build the attention and conv ladders with the standard oneAPI Windows
commands in [api-usage.md](../api/api-usage.md); include oneDNN as the conv
baseline.

Copy-ready cores: [attention online core](../api/code-snippets.md#attention-online-core),
[conv cache-block core](../api/code-snippets.md#conv-cacheblock-core).
