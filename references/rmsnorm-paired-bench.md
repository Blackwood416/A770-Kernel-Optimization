# RMSNorm Paired/Interleaved Tie Verification

> `[MEASURED]` Intel Arc A770 (DG2), oneAPI 2026.1, driver `32.0.101.8860`, Windows 11, Level Zero. Same-process A/B/B/A bursts of 1000 launches; 5 warmup launches per variant, then 10 rounds x 2 pair samples = 20 paired deltas per comparison. `device_time` is the SYCL event duration; `wall_time` is host submit to wait; `pipeline_time` equals wall time because inputs are static and no per-call host transform exists.

The original shape sweep was measured on driver `32.0.101.8724`; this paired campaign was measured on the current host with driver `32.0.101.8860`. The paired protocol conclusions are valid for the current toolchain/driver and selected cells. Differences versus the shape-sweep champion table may include driver drift as well as reduced wall noise.

No new oneDNN primitive was executed here; the oneDNN baseline contract is inherited from [rmsnorm-shape-sweep.md](rmsnorm-shape-sweep.md).

## Correctness Contract

All 51 paired comparisons ran CPU f64 reference checks on the stored dtype. Every variant reported `errors: 0`; no `CORRECTNESS_CONTRACT_MISMATCH` occurred.

| Field | Value |
|---|---|
| comparisons | 51 |
| correctness failures | 0 |
| contract mismatches | 0 |
| reference | `cpu_f64_stored_dtype` |
| semantics_id | `rmsnorm_row_scale_ab` |
| accuracy_mode | `strict` |
| relaxed_accuracy | `false` |
| f32 tolerance | `rel=1e-3, abs=1e-3` |
| f16/bf16 tolerance | `rel=2e-2, abs=1e-2` |

## Selection and Protocol

Selected cells: `f32/f16/bf16` x rows `{1, 8, 64, 1024}` x hidden `{256, 1024, 4096, 16384}`. This covers the requested high-CV domain from the shape sweep; most selected cells have wall CV > 10% and some have device CV > 10%.

Each comparison timed the exact wall-time champion from the shape sweep (`A`) against a second or simplified candidate (`B`). A negative paired delta means `B` is faster. Pair delta is `(T_B - T_A) / T_A`; MAD is normalized 1.4826 MAD.

## Simplified Production Choice by Cell

`same` means the simplified production region already selects the exact champion candidate. Where they differ, the paired wall delta is shown.

### f32

| rows | hidden | exact | simplified | same | ind margin | paired wall | paired dev |
|---|---|---|---|---|---|---|---|
| 1 | 256 | sg@64 | sg@64 | yes | 0.00 | n/a | n/a |
| 1 | 1024 | slm@16 | sg@64 | no | 3.70 | -3.81 | -12.41 |
| 1 | 4096 | slm@128 | slm@128 | yes | 0.00 | n/a | n/a |
| 1 | 16384 | slm@128 | slm@128 | yes | 0.00 | n/a | n/a |
| 8 | 256 | multi(R2x32) | sg@128 | no | 1.59 | -7.77 | -34.56 |
| 8 | 1024 | sg@128 | sg@128 | yes | 0.00 | n/a | n/a |
| 8 | 4096 | slm@128 | slm@128 | yes | 0.00 | n/a | n/a |
| 8 | 16384 | slm@256 | slm@256 | yes | 0.00 | n/a | n/a |
| 64 | 256 | multi(R4x64) | sg@128 | no | 0.90 | -14.51 | -54.19 |
| 64 | 1024 | sg@32 | sg@32 | yes | 0.00 | n/a | n/a |
| 64 | 4096 | multi(R4x32) | slm@256 | no | 1.95 | -7.71 | -21.75 |
| 64 | 16384 | slm@256 | slm@256 | yes | 0.00 | n/a | n/a |
| 1024 | 256 | sg@64 | sg@64 | yes | 0.00 | n/a | n/a |
| 1024 | 1024 | sg@128 | sg@128 | yes | 0.00 | n/a | n/a |
| 1024 | 4096 | slm@128 | slm@128 | yes | 0.00 | n/a | n/a |
| 1024 | 16384 | slm@256 | slm@256 | yes | 0.00 | n/a | n/a |

### f16

| rows | hidden | exact | simplified | same | ind margin | paired wall | paired dev |
|---|---|---|---|---|---|---|---|
| 1 | 256 | sg@128 | sg@128 | yes | 0.00 | n/a | n/a |
| 1 | 1024 | sg@64 | sg@64 | yes | 0.00 | n/a | n/a |
| 1 | 4096 | slm@256 | slm@256 | yes | 0.00 | n/a | n/a |
| 1 | 16384 | slm@256 | slm@256 | yes | 0.00 | n/a | n/a |
| 8 | 256 | slm@16 | sg@32 | no | 7.31 | -3.46 | -10.10 |
| 8 | 1024 | sg@128 | sg@128 | yes | 0.00 | n/a | n/a |
| 8 | 4096 | slm@64 | slm@64 | yes | 0.00 | n/a | n/a |
| 8 | 16384 | slm@64 | slm@64 | yes | 0.00 | n/a | n/a |
| 64 | 256 | sg@128 | sg@128 | yes | 0.00 | n/a | n/a |
| 64 | 1024 | sg@64 | sg@64 | yes | 0.00 | n/a | n/a |
| 64 | 4096 | sg@32 | slm@128 | no | 13.69 | -2.46 | -19.40 |
| 64 | 16384 | slm@128 | slm@128 | yes | 0.00 | n/a | n/a |
| 1024 | 256 | sg@32 | sg@32 | yes | 0.00 | n/a | n/a |
| 1024 | 1024 | sg@32 | sg@32 | yes | 0.00 | n/a | n/a |
| 1024 | 4096 | slm@256 | slm@256 | yes | 0.00 | n/a | n/a |
| 1024 | 16384 | slm@256 | slm@256 | yes | 0.00 | n/a | n/a |

### bf16

| rows | hidden | exact | simplified | same | ind margin | paired wall | paired dev |
|---|---|---|---|---|---|---|---|
| 1 | 256 | sg@64 | sg@64 | yes | 0.00 | n/a | n/a |
| 1 | 1024 | slm@128 | sg@32 | no | 15.47 | +0.77 | +18.40 |
| 1 | 4096 | slm@64 | slm@64 | yes | 0.00 | n/a | n/a |
| 1 | 16384 | slm@128 | slm@128 | yes | 0.00 | n/a | n/a |
| 8 | 256 | slm@16 | sg@32 | no | 16.35 | -3.26 | -7.61 |
| 8 | 1024 | slm@128 | sg@128 | no | 5.85 | -1.41 | +4.76 |
| 8 | 4096 | slm@128 | slm@128 | yes | 0.00 | n/a | n/a |
| 8 | 16384 | slm@128 | slm@128 | yes | 0.00 | n/a | n/a |
| 64 | 256 | sg@32 | sg@32 | yes | 0.00 | n/a | n/a |
| 64 | 1024 | sg@128 | sg@128 | yes | 0.00 | n/a | n/a |
| 64 | 4096 | sg@128 | slm@64 | no | 3.02 | +1.82 | +0.33 |
| 64 | 16384 | multi(R4x32) | slm@128 | no | 11.64 | -11.87 | -23.57 |
| 1024 | 256 | sg@32 | sg@32 | yes | 0.00 | n/a | n/a |
| 1024 | 1024 | sg@64 | sg@64 | yes | 0.00 | n/a | n/a |
| 1024 | 4096 | sg@32 | slm@256 | no | 10.08 | +2.51 | +2.47 |
| 1024 | 16384 | slm@256 | slm@256 | yes | 0.00 | n/a | n/a |

## Paired Simplified-Comparison Results

12 of 48 selected cells had a different simplified candidate. In every one, the simplified candidate was faster or within 2.6% slower on paired wall time; no simplified candidate lost by more than 3%.

| dtype | rows | hidden | exact | simplified | ind margin | wall delta | wall MAD | dev delta | verdict |
|---|---|---|---|---|---|---|---|---|---|
| f32 | 1 | 1024 | slm@16 | sg@64 | 3.70 | -3.81 | 1.72 | -12.41 | candidate faster |
| f32 | 8 | 256 | multi(R2x32) | sg@128 | 1.59 | -7.77 | 3.04 | -34.56 | candidate faster |
| f32 | 64 | 256 | multi(R4x64) | sg@128 | 0.90 | -14.51 | 1.60 | -54.19 | candidate faster |
| f32 | 64 | 4096 | multi(R4x32) | slm@256 | 1.95 | -7.71 | 1.20 | -21.75 | candidate faster |
| f16 | 8 | 256 | slm@16 | sg@32 | 7.31 | -3.46 | 0.80 | -10.10 | candidate faster |
| f16 | 64 | 4096 | sg@32 | slm@128 | 13.69 | -2.46 | 1.28 | -19.40 | within 3% |
| bf16 | 1 | 1024 | slm@128 | sg@32 | 15.47 | +0.77 | 1.00 | +18.40 | tie (MAD) |
| bf16 | 8 | 256 | slm@16 | sg@32 | 16.35 | -3.26 | 0.38 | -7.61 | candidate faster |
| bf16 | 8 | 1024 | slm@128 | sg@128 | 5.85 | -1.41 | 0.96 | +4.76 | within 3% |
| bf16 | 64 | 4096 | sg@128 | slm@64 | 3.02 | +1.82 | 2.48 | +0.33 | tie (MAD) |
| bf16 | 64 | 16384 | multi(R4x32) | slm@128 | 11.64 | -11.87 | 0.68 | -23.57 | candidate faster |
| bf16 | 1024 | 4096 | sg@32 | slm@256 | 10.08 | +2.51 | 1.09 | +2.47 | within 3% |

## Exact Champion Flips

A flip is a consistent paired win for `B` (`median < 0`, at least 75% of pair samples negative, and `p90 < 0`). Hard flips exceed 3%; soft flips are below 3% but still consistent.

| dtype | rows | hidden | exact | candidate | second? | ind margin | wall delta | wall MAD | dev delta | hard/soft |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 8 | 256 | slm@16 | sg@32 | no | 16.35 | -3.26 | 0.38 | -7.61 | hard |
| bf16 | 8 | 1024 | slm@128 | sg@128 | yes | 5.85 | -1.41 | 0.96 | +4.76 | soft |
| bf16 | 64 | 16384 | multi(R4x32) | slm@128 | no | 11.64 | -11.87 | 0.68 | -23.57 | hard |
| f16 | 8 | 256 | slm@16 | sg@32 | yes | 7.31 | -3.46 | 0.80 | -10.10 | hard |
| f16 | 64 | 4096 | sg@32 | slm@128 | yes | 13.69 | -2.46 | 1.28 | -19.40 | soft |
| f32 | 1 | 1024 | slm@16 | sg@64 | yes | 3.70 | -3.81 | 1.72 | -12.41 | hard |
| f32 | 8 | 256 | multi(R2x32) | sg@128 | yes | 1.59 | -7.77 | 3.04 | -34.56 | hard |
| f32 | 64 | 256 | multi(R4x64) | sg@128 | yes | 0.90 | -14.51 | 1.60 | -54.19 | hard |
| f32 | 64 | 4096 | multi(R4x32) | slm@256 | yes | 1.95 | -7.71 | 1.20 | -21.75 | hard |

Total: 9 flips; 7 hard and 2 soft. 7 of the flips are against the second-best family; the remaining flips are against a lower-ranked simplified candidate.

## Tie-Rule Evidence

The current 3% rule is useful as a production simplification threshold, but it is not a statistical significance threshold. Every selected cell that the shape sweep called a 3% tie (8 comparisons) resolved to a paired delta outside 3%; 0 were within 3%, 1 were within 5%, and 7 exceeded 5%. None had a MAD interval containing zero.

| Aggregate | value |
|---|---:|
| paired comparisons | 51 |
| exact-vs-second pairs | 48 |
| simplified pairs | 12 |
| wall paired `|delta| <= 3%` | 5 |
| wall paired `|delta| <= 5%` | 13 |
| wall paired `|delta| > 5%` | 38 |
| wall paired MAD overlap | 2 |
| device paired MAD overlap | 0 |
| shape-sweep 3% tie cells | 8 |
| 3% tie cells with paired `|delta| <= 3%` | 0 |
| 3% tie cells with paired `|delta| <= 5%` | 1 |
| 3% tie cells with paired `|delta| > 5%` | 7 |
| 3% tie cells with MAD overlap | 0 |
| 3% tie cells where simplified flipped | 3 |
| exact-vs-second wall flips | 7 |
| exact-vs-second device flips | 6 |

Raising the threshold to 5% is not supported: all 8 paired deltas in the 3-5% band had MAD intervals that excluded zero. A 5% rule would treat measured 4-5% differences as ties even though paired samples were consistent.

## All Paired Comparisons

`kind=simplified` means the candidate is the simplified production choice; `kind=second` means the candidate is the second-best family from the shape sweep. A cell can appear once when both definitions select the same candidate.

| dtype | rows | hidden | kind | exact | candidate | ind wall | pair wall | wall MAD | verdict |
|---|---|---|---|---|---|---|---|---|---|
| f32 | 1 | 256 | second | sg@64 | slm@128 | 8.18 | +4.75 | 1.61 | exact faster |
| f32 | 1 | 1024 | simplified | slm@16 | sg@64 | 3.70 | -3.81 | 1.72 | candidate faster |
| f32 | 1 | 4096 | second | slm@128 | sg@64 | 12.83 | +10.30 | 2.97 | exact faster |
| f32 | 1 | 16384 | second | slm@128 | multi(R1x64) | 15.50 | +12.80 | 1.74 | exact faster |
| f32 | 8 | 256 | simplified | multi(R2x32) | sg@128 | 1.59 | -7.77 | 3.04 | candidate faster |
| f32 | 8 | 1024 | second | sg@128 | multi(R2x32) | 8.61 | +6.41 | 1.36 | exact faster |
| f32 | 8 | 4096 | second | slm@128 | multi(R2x32) | 8.96 | +16.41 | 1.35 | exact faster |
| f32 | 8 | 16384 | second | slm@256 | multi(R1x64) | 20.26 | +24.57 | 2.22 | exact faster |
| f32 | 64 | 256 | simplified | multi(R4x64) | sg@128 | 0.90 | -14.51 | 1.60 | candidate faster |
| f32 | 64 | 1024 | second | sg@32 | slm@256 | 1.60 | +8.70 | 1.34 | exact faster |
| f32 | 64 | 4096 | simplified | multi(R4x32) | slm@256 | 1.95 | -7.71 | 1.20 | candidate faster |
| f32 | 64 | 16384 | second | slm@256 | multi(R4x32) | 15.67 | +12.50 | 0.62 | exact faster |
| f32 | 1024 | 256 | second | sg@64 | slm@32 | 47.57 | +29.95 | 1.37 | exact faster |
| f32 | 1024 | 1024 | second | sg@128 | slm@128 | 17.91 | +8.25 | 2.05 | exact faster |
| f32 | 1024 | 4096 | second | slm@128 | sg@32 | 13.26 | +16.64 | 1.21 | exact faster |
| f32 | 1024 | 16384 | second | slm@256 | multi(R2x64) | 28.72 | +29.82 | 1.09 | exact faster |
| f16 | 1 | 256 | second | sg@128 | slm@32 | 8.02 | +4.28 | 1.00 | exact faster |
| f16 | 1 | 1024 | second | sg@64 | multi(R1x32) | 7.55 | +5.91 | 1.19 | exact faster |
| f16 | 1 | 4096 | second | slm@256 | sg@32 | 8.65 | +16.76 | 2.21 | exact faster |
| f16 | 1 | 16384 | second | slm@256 | multi(R1x64) | 28.97 | +24.84 | 1.00 | exact faster |
| f16 | 8 | 256 | simplified | slm@16 | sg@32 | 7.31 | -3.46 | 0.80 | candidate faster |
| f16 | 8 | 1024 | second | sg@128 | multi(R8x32) | 0.91 | +11.20 | 2.17 | exact faster |
| f16 | 8 | 4096 | second | slm@64 | sg@64 | 1.11 | +10.16 | 1.43 | exact faster |
| f16 | 8 | 16384 | second | slm@64 | multi(R4x64) | 26.77 | +21.61 | 0.74 | exact faster |
| f16 | 64 | 256 | second | sg@128 | multi(R4x32) | 16.21 | +11.01 | 1.04 | exact faster |
| f16 | 64 | 1024 | second | sg@64 | slm@32 | 21.13 | +10.41 | 2.29 | exact faster |
| f16 | 64 | 4096 | simplified | sg@32 | slm@128 | 13.69 | -2.46 | 1.28 | within 3% |
| f16 | 64 | 16384 | second | slm@128 | multi(R4x32) | 12.47 | +13.06 | 0.81 | exact faster |
| f16 | 1024 | 256 | second | sg@32 | slm@256 | 23.09 | +37.27 | 1.72 | exact faster |
| f16 | 1024 | 1024 | second | sg@32 | slm@64 | 35.93 | +28.34 | 2.34 | exact faster |
| f16 | 1024 | 4096 | second | slm@256 | sg@64 | 2.89 | +3.85 | 1.24 | exact faster |
| f16 | 1024 | 16384 | second | slm@256 | multi(R4x64) | 51.94 | +52.46 | 3.36 | exact faster |
| bf16 | 1 | 256 | second | sg@64 | slm@64 | 10.74 | +4.90 | 1.53 | exact faster |
| bf16 | 1 | 1024 | simplified | slm@128 | sg@32 | 15.47 | +0.77 | 1.00 | tie (MAD) |
| bf16 | 1 | 1024 | second | slm@128 | multi(R1x32) | 8.63 | +7.84 | 1.45 | exact faster |
| bf16 | 1 | 4096 | second | slm@64 | sg@128 | 35.09 | +14.42 | 3.00 | exact faster |
| bf16 | 1 | 16384 | second | slm@128 | sg@32 | 73.91 | +60.26 | 5.73 | exact faster |
| bf16 | 8 | 256 | simplified | slm@16 | sg@32 | 16.35 | -3.26 | 0.38 | candidate faster |
| bf16 | 8 | 256 | second | slm@16 | multi(R2x32) | 10.34 | +5.81 | 1.04 | exact faster |
| bf16 | 8 | 1024 | simplified | slm@128 | sg@128 | 5.85 | -1.41 | 0.96 | within 3% |
| bf16 | 8 | 4096 | second | slm@128 | multi(R4x64) | 12.05 | +22.04 | 1.47 | exact faster |
| bf16 | 8 | 16384 | second | slm@128 | multi(R1x64) | 42.26 | +18.95 | 1.23 | exact faster |
| bf16 | 64 | 256 | second | sg@32 | slm@32 | 1.76 | +7.37 | 0.90 | exact faster |
| bf16 | 64 | 1024 | second | sg@128 | slm@64 | 20.53 | +3.34 | 0.92 | exact faster |
| bf16 | 64 | 4096 | simplified | sg@128 | slm@64 | 3.02 | +1.82 | 2.48 | tie (MAD) |
| bf16 | 64 | 16384 | simplified | multi(R4x32) | slm@128 | 11.64 | -11.87 | 0.68 | candidate faster |
| bf16 | 64 | 16384 | second | multi(R4x32) | sg@64 | 8.25 | +8.58 | 2.74 | exact faster |
| bf16 | 1024 | 256 | second | sg@32 | slm@256 | 23.11 | +33.85 | 2.30 | exact faster |
| bf16 | 1024 | 1024 | second | sg@64 | slm@256 | 22.35 | +27.11 | 3.93 | exact faster |
| bf16 | 1024 | 4096 | simplified | sg@32 | slm@256 | 10.08 | +2.51 | 1.09 | within 3% |
| bf16 | 1024 | 16384 | second | slm@256 | multi(R4x64) | 51.88 | +54.63 | 2.33 | exact faster |

## Negative Results and Caveats

- `[MEASURED]` The shape sweep's independent-absolute 3% tie threshold was not a statistical tie in the selected cells: all 8 independent `<=3%` cells produced paired deltas outside 3%.
- `[MEASURED]` A 5% tie threshold is too loose; 3-5% paired deltas were reproducible and their MAD intervals excluded zero.
- `[MEASURED]` The simplified production regions remain useful: none of the 12 differing simplified candidates lost by more than 2.6% on paired wall time, and 9 were faster.
- `[MEASURED]` Exact champion cells are not stable against the paired protocol on the current driver: 9 selected exact champions flipped, including 7 against the second-best family.
- `[MEASURED]` Device deltas are even less tie-like than wall deltas: 0 of 51 device comparisons had a MAD interval containing zero.
- `[HYPOTHESIS]` The exact champion tables from the shape sweep should be re-checked with this paired protocol after any driver or toolchain change; the flips here may include driver drift from `32.0.101.8724` to `32.0.101.8860`.

## Validity Domain

`[DISPATCH]` operator=RMSNorm forward with per-row `gamma`; dtype=f32/f16/bf16; rows in {1, 8, 64, 1024}; hidden in {256, 1024, 4096, 16384}; device=A770/oneAPI 2026.1/driver `32.0.101.8860`; paired A/B/B/A with 1000-launch bursts and 20 pair samples; confidence=medium for selected cells, `[HEURISTIC]` outside those cells. The oneDNN baseline contract remains the shape-sweep record.

