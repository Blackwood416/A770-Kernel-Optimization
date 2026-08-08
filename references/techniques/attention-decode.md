# A770 Decode Attention Ladder (Q=1)

> `[MEASURED]` Intel Arc A770 (DG2), oneAPI 2026.1, driver
> `32.0.101.8724`, Windows, Level Zero, standard SYCL + USM, f32. Q/K/V and
> output are f32. Every variant is verified against an f64 CPU reference with
> tolerance `abs=1e-3`; every reported row has `errors: 0/<total>`.

## Benchmark Protocol

- Config matrix: `B=4`, `Q=1`, `KV in {512, 2048, 8192, 32768}`, `D in {64,
  128}`, `Hq=8`, `Hkv in {8, 2, 1}` for MHA/GQA/MQA, `pos=-1` (active = full
  KV cache). A causal-mask spot check runs `KV=2048, D=128, GQA, pos=1023`
  with active=1024.
- Warmup 10 launches, then 30 timed samples per run, 3 runs, total 90 samples
  per variant. Reported values are sample medians; p10/p90 and CV are
  summarized below.
- `device_time` is the sum of SYCL `command_start`/`command_end` event
  durations. `wall_time` is host submit to wait. `pipeline_time` equals wall
  time because inputs are static and all host layout prep (chunk pack, paged
  table/pack) happens outside the timed loop.
- CV flag threshold is 10%. No oneDNN baseline was added to this decode
  campaign; the oneDNN Baseline Contract applies if a future library baseline
  is measured.

## Variants

| Variant | Structure |
|---|---|
| `naive_3kernel` | Global S, row softmax to P, then P*V |
| `online_causal_fused` | One fused online max/sum/PV scan per query head |
| `kv_cache_chunk_layout` | K/V pre-packed as `[head][chunk][tokens][D]` |
| `paged_kv_layout` | K/V stored in 64-token physical pages behind a shuffled page table |
| `split_d_vlen4/8/16` | Optional d-slice split scan |

## MHA Wall and Device Median (us)

`pipeline = wall` in this static-input benchmark.

| KV | D | naive wall/device | online_causal wall/device | kv_chunk wall/device | paged_kv wall/device |
|---|---:|---:|---:|---:|---:|
| 512 | 64 | 706.9 / 557.2 | 997.9 / 862.6 | 1000 / 863.4 | 995.5 / 866.6 |
| 512 | 128 | 793.7 / 654.6 | 915.5 / 790.3 | 911.9 / 791 | 916.6 / 793.1 |
| 2048 | 64 | 2185.6 / 2067.9 | 1188.8 / 1070.3 | 1200.7 / 1068 | 1187.1 / 1071.4 |
| 2048 | 128 | 2269.1 / 2138.6 | 1675.8 / 1562 | 1669.7 / 1562.6 | 1684 / 1569.8 |
| 8192 | 64 | 8015.6 / 7807.5 | 1948.9 / 1810.7 | 1948.3 / 1815.1 | 1939.8 / 1820.3 |
| 8192 | 128 | 8419 / 8276.7 | 4718.5 / 4591 | 4712.2 / 4583.4 | 4769.8 / 4632.3 |
| 32768 | 64 | 30671.4 / 30417 | 4903.1 / 4765.4 | 4905.3 / 4767.3 | 4952.7 / 4822.8 |
| 32768 | 128 | 32729.1 / 32486.7 | 16143.4 / 15950.9 | 16088.1 / 15904.4 | 16205.7 / 15990.9 |

## GQA Wall and Device Median (us)

`pipeline = wall` in this static-input benchmark.

| KV | D | naive wall/device | online_causal wall/device | kv_chunk wall/device | paged_kv wall/device |
|---|---:|---:|---:|---:|---:|
| 512 | 64 | 646.2 / 492.8 | 986.4 / 850.8 | 993.4 / 853.4 | 986.7 / 854.8 |
| 512 | 128 | 640 / 518 | 850.2 / 731.2 | 857.4 / 737.9 | 848.6 / 739.3 |
| 2048 | 64 | 2117.4 / 1969.7 | 1123.1 / 1001.4 | 1115.9 / 1006.9 | 1137.2 / 1005.9 |
| 2048 | 128 | 2319.6 / 2192.1 | 1427.8 / 1306.4 | 1418.1 / 1307.6 | 1516.4 / 1395.2 |
| 8192 | 64 | 7879.6 / 7737.4 | 1678.6 / 1533.4 | 1689.9 / 1541.5 | 1700.4 / 1551.9 |
| 8192 | 128 | 8214 / 8059.5 | 3670 / 3552.8 | 3655.8 / 3546.6 | 3679.3 / 3570.9 |
| 32768 | 64 | 30963.7 / 30685.2 | 3829.8 / 3690.8 | 3841.3 / 3693.4 | 3840.2 / 3722.9 |
| 32768 | 128 | 32346.4 / 32147.9 | 12223 / 12090.7 | 12095.4 / 11960.9 | 12258.3 / 12122.9 |

## MQA Wall and Device Median (us)

`pipeline = wall` in this static-input benchmark.

| KV | D | naive wall/device | online_causal wall/device | kv_chunk wall/device | paged_kv wall/device |
|---|---:|---:|---:|---:|---:|
| 512 | 64 | 627.2 / 491.7 | 976 / 847.3 | 983 / 850.8 | 983 / 850.4 |
| 512 | 128 | 632.8 / 519 | 828.7 / 714.8 | 844.6 / 721.8 | 833.8 / 719.1 |
| 2048 | 64 | 2122 / 2004.5 | 1102 / 988 | 1105.6 / 993.4 | 1106.6 / 994.8 |
| 2048 | 128 | 2185.3 / 2043.5 | 1361.5 / 1250.7 | 1360.4 / 1252.2 | 1361.4 / 1256.6 |
| 8192 | 64 | 7906.4 / 7761.6 | 1623.6 / 1500.7 | 1628.2 / 1505.8 | 1637.6 / 1517.4 |
| 8192 | 128 | 8216 / 8027.7 | 3574.8 / 3437.9 | 3542.5 / 3429.6 | 3574.2 / 3451.7 |
| 32768 | 64 | 31109.4 / 30872.1 | 3694.8 / 3574.1 | 3707.4 / 3583.5 | 3769.2 / 3627.2 |
| 32768 | 128 | 32430.6 / 32271.2 | 11758.8 / 11614.7 | 11704.3 / 11564.7 | 11897.2 / 11741.6 |

## Host Overhead (wall median - device median, us)

| Mode | KV | D | naive | online_causal | kv_chunk | paged_kv |
|---|---|---:|---:|---:|---:|---:|
| MHA | 512 | 64 | 149.7 | 135.4 | 136.6 | 128.9 |
| MHA | 512 | 128 | 139.1 | 125.2 | 120.9 | 123.6 |
| MHA | 2048 | 64 | 117.7 | 118.6 | 132.7 | 115.7 |
| MHA | 2048 | 128 | 130.5 | 113.8 | 107.1 | 114.2 |
| MHA | 8192 | 64 | 208.2 | 138.2 | 133.2 | 119.5 |
| MHA | 8192 | 128 | 142.4 | 127.5 | 128.8 | 137.5 |
| MHA | 32768 | 64 | 254.3 | 137.7 | 138 | 129.9 |
| MHA | 32768 | 128 | 242.4 | 192.5 | 183.7 | 214.8 |
| GQA | 512 | 64 | 153.4 | 135.7 | 140.1 | 131.9 |
| GQA | 512 | 128 | 122 | 119 | 119.5 | 109.3 |
| GQA | 2048 | 64 | 147.7 | 121.7 | 109 | 131.3 |
| GQA | 2048 | 128 | 127.4 | 121.4 | 110.6 | 121.2 |
| GQA | 8192 | 64 | 142.1 | 145.1 | 148.4 | 148.5 |
| GQA | 8192 | 128 | 154.6 | 117.2 | 109.3 | 108.4 |
| GQA | 32768 | 64 | 278.5 | 139 | 147.9 | 117.3 |
| GQA | 32768 | 128 | 198.6 | 132.3 | 134.5 | 135.4 |
| MQA | 512 | 64 | 135.5 | 128.7 | 132.2 | 132.6 |
| MQA | 512 | 128 | 113.8 | 113.9 | 122.7 | 114.6 |
| MQA | 2048 | 64 | 117.4 | 113.9 | 112.3 | 111.8 |
| MQA | 2048 | 128 | 141.8 | 110.8 | 108.2 | 104.8 |
| MQA | 8192 | 64 | 144.8 | 122.9 | 122.4 | 120.2 |
| MQA | 8192 | 128 | 188.3 | 136.8 | 112.9 | 122.5 |
| MQA | 32768 | 64 | 237.3 | 120.7 | 123.9 | 142 |
| MQA | 32768 | 128 | 159.3 | 144 | 139.6 | 155.5 |

## Split-d Scan (KV=2048, D=128, GQA, full KV)

| Variant | Wall | Device | Pipeline | Host | errors |
|---|---:|---:|---:|---:|---:|
| `split_d_vlen16` | 21434.2 | 21288.1 | 21434.2 | 146.1 | 0 |
| `split_d_vlen4` | 21132.2 | 20983.6 | 21132.2 | 148.6 | 0 |
| `split_d_vlen8` | 21129.7 | 20985.5 | 21129.7 | 144.2 | 0 |

## Causal Mask Spot Check (KV=2048, D=128, GQA, pos=1023, active=1024)

| Variant | Wall | Device | Pipeline | Host | errors |
|---|---:|---:|---:|---:|---:|
| `kv_cache_chunk_layout` | 1058.4 | 934 | 1058.4 | 124.4 | 0 |
| `naive_3kernel` | 2161.6 | 2022.8 | 2161.6 | 138.8 | 0 |
| `online_causal_fused` | 1060 | 926.1 | 1060 | 133.9 | 0 |
| `paged_kv_layout` | 1073.1 | 932.6 | 1073.1 | 140.5 | 0 |

## Stability Flags (CV > 10%)

Only small-KV naive rows were flagged:

| Shape | Variant | wall CV % | device CV % |
|---|---:|---:|---:|
| MQA KV512 D128 pos-1 | `naive_3kernel` | 9.7 | 10 |
| GQA KV512 D64 pos-1 | `naive_3kernel` | 10.2 | 11.7 |
| MHA KV512 D64 pos-1 | `naive_3kernel` | 9.3 | 10.4 |
| MQA KV512 D64 pos-1 | `naive_3kernel` | 10.5 | 8.7 |

## Decision Rules

1. `[MEASURED]` At `KV=512`, the three-kernel naive path wins wall and device
   time for MHA/GQA/MQA at both D values (`492-655 us` device). At `KV=2048`,
   `kv_cache_chunk_layout` or `online_causal_fused` take over depending on
   mode; the difference from naive is already visible in device time.
2. `[MEASURED]` At `KV=8192, D=64`, the online causal fused path wins device
   time (`1501-1811 us`); at `KV=8192, D=128`, `kv_cache_chunk_layout` wins
   (`3430-4583 us`). The same split continues at `KV=32768`: D=64 prefers
   `online_causal_fused` (`3574-4765 us`), D=128 prefers
   `kv_cache_chunk_layout` (`11565-15904 us`).
3. `[MEASURED]` Paged KV adds only a small device-time penalty versus the
   chunk layout on this 64-token page table sweep (`+4-88 us` at `KV=2048,
   D=128` and `+87-177 us` at `KV=32768, D=128`), while remaining correct
   across the shuffled page table.
4. `[MEASURED]` Host/runtime overhead is `104-279 us` per decode call and
   dominates the wall time at small KV; always report device and wall
   separately for Q=1.
5. `[MEASURED]` Split-d scan is not competitive for decode: VLEN 4/8/16 are
   all about `21.1-21.4 ms` at `KV=2048, D=128, GQA` versus `1.4-1.5 ms` for
   the work-group parallel online/chunk/paged paths. It duplicates the QK^T
   reduction once per d-slice and should not be dispatched.
6. `[HEURISTIC]` For other B/Hq/Hkv/KV/D values or bf16/DPAS, re-measure
   before reuse; these rules are tied to the validity domain below.

## Dispatch Validity Domain

`[DISPATCH]` operator=decode attention forward, `Q=1`, online softmax over
K/V with causal prefix (`active = KV` or `active = pos+1`); dtype=f32 with
f32 accumulators; `B=4`, `Hq=8`, `Hkv in {8, 2, 1}`, `KV in {512, 2048, 8192,
32768}`, `D in {64, 128}`, `KV % 64 == 0`, `D % 8 == 0`, 64-byte aligned
USM; device=A770/oneAPI 2026.1/driver `32.0.101.8724`; confidence=medium
(single machine and seed). Outside this domain, re-label as `[HEURISTIC]`.

## Negative Results

- `BM=64` flash tiles are a preserved device-lost failure:
  `UR_RESULT_ERROR_DEVICE_LOST` on A770 with `BM=64, BN=32, VLEN=8, WG=512`,
  about 40 KB SLM. Do not reintroduce that geometry without a watchdog.
- The first decode work-group reduction used a single-kernel local-memory
  combine and produced `1952/2048` errors at `KV=512, D=64`; the harness now
  uses a deterministic global two-phase combine. The failed local-reduce
  path is a preserved negative result.
- Split-d VLEN 4/8/16 is correct but about 15x slower than the
  online/chunk/paged paths at `KV=2048, D=128, GQA`; keep it as a negative
  control.

## Reproduction

Build and run the decode-ladder harness with the standard oneAPI Windows
commands, then run the full config matrix and collect `wall_median_us`,
`device_median_us`, `pipeline_median_us`, `host_median_us`, and `errors`.
Keep per-config raw output with p10/p90 and CV so a later report can be
regenerated without re-running the GPU.
