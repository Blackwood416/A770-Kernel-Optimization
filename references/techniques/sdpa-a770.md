# A770 SDPA / Flash Attention Reference (ESIMD Sidecar Port)

> `[MEASURED]` Intel Arc A770 (DG2-G10), Windows, driver
> `32.0.101.8860`, oneAPI 2026.1, Torch 2.13.0+xpu, Python 3.13, ESIMD
> attention sidecar, measured 2026-08-10..2026-08-12. Re-measure before
> transferring any number or dispatch rule to another GPU, driver, compiler,
> Torch, or library stack.

## Contents

- Operator scope
- Porting root causes
- Correctness baseline and regression gate
- Kernel design ladder (v1 -> v4.1)
- Measured performance tables
- Dispatch rules and validity domain
- VTune findings
- Cache and integration lessons
- Negative results
- Not measured / open hypotheses
- Provenance

## Operator Scope

This page covers dense, multi-row scaled dot-product attention (SDPA):

```text
O = softmax(Q * K^T / sqrt(D)) * V
```

with `[B=1, L, H, D]` (or `[1, H, L, D]` after permute), fp16/bf16, no mask.
It is a separate campaign from the small prefill ladder in
[attention-conv.md](attention-conv.md) and from `Q=1` decode attention in
[attention-decode.md](attention-decode.md). Padding shapes such as `kv_len`
not a multiple of 32 are inside the measured domain.

The port target was an ESIMD attention sidecar integrated with ComfyUI by
patching `optimized_attention` to call the sidecar's `sdp(q, k, v)`.
MiniMax H3 GGUF (Q4_K_M) calls the custom path with `[1, 56, seq, 128]`
fp16/bf16 and no mask.

## Porting Root Causes

The upstream Xe2-oriented SDP kernel family could not be enabled on A770:

1. `[TOOLCHAIN]` `dpas.hf.hf.8.8` (fp16 SxV accumulator) is rejected by the
   DG2 VISA validator with `Raw Operand ... incorrect type hf`; DG2 only
   allows fp32 accumulators. An fp16->fp32-accum variant compiled for
   `-device dg2`.
2. `[BUG]` The compiled variant still raised
   `UR_RESULT_ERROR_DEVICE_LOST` on the first call at the minimal
   `B=1, L=16, H=8, D=128, fp16`, with or without `-doubleGRF`.
3. `[BUG]` The root cause was isolated with a standalone SYCL harness plus
   Windows LiveKernelEvent 141 (VIDEO_TDR): **ESIMD work-group size 512
   triggers an A770 TDR**, even for a trivial kernel that only calls
   `slm_init` and performs one store. The Xe2 family also combined 2D LSC,
   about 17 KB of compiler spill, and fp16 DPAS accumulation.
4. `[BUG]` Non-deterministic outputs observed earlier under the torch path
   were stale memory after TDR recovery, not a kernel logic fault.

Consequence: A770 ESIMD attention must use a small work-group (measured
`WG=32`), keep per-thread state in registers, use fp32 accumulation, avoid
2D LSC and named barriers, and compile with zero spill. See
[robustness.md](../workflow/robustness.md) for the isolation protocol.

## Correctness Baseline and Regression Gate

- Early v1 acceptance: `B=1, H=1..8, D in {64,128}`, fp16/bf16, q/kv
  non-multiple-32 and 1-token boundary: 40/40 against Torch SDPA; fp16
  `max_abs <= 2.4e-4`, bf16 `<= 2e-3`; 10 repeated calls with `max diff = 0`.
- Later 81-case matrix including 65/129/300 padding boundaries, fp16/bf16:
  all pass; worst vs ground truth about `1e-3` (`8.4e-4` reported in v3/v4
  runs).
- A regression gate requiring finite output plus an error threshold is the
  entry check before any new DG2 SDP geometry can be dispatched.
- Padding handling: v1 masked padded rows when the K row was all zero;
  v4.1 passes the original `kv_len` to the kernel and masks by row number,
  removing the `kvZero` flag path entirely.

## Kernel Design Ladder (v1 -> v4.1)

### v1 FMA baseline

`flash.attn.b.mha.dg2.h`: `WG=32`, one query row per thread, whole head dim
in registers, fp32 accumulator, SLM K/V tile `BN=32`, online softmax, no 2D
LSC, no named barrier.

`[MEASURED]` Correct and stable, but 5-10x slower than Torch SDPA (H3
`H=56/D128`, wall median): `S=1024` 10.6 ms vs 2.2 ms; `S=16384` 2.97 s vs
0.30 s. D64 still uses this v1 FMA path in the final v4.1 build.

### v2 DPAS

`flash.attn.b.mha.dg2.dpas.h`: K/V are packed into VNNI layout while copied
to SLM; both QK^T and SxV use `dpas<8,8,float>`.

- `[CORRECTNESS]` `dpas<8,1,float>` (M=1) produced noisy scores (softmax
  weight correlation only 0.86). Replicating the single query row across 8
  DPAS rows and reading only row 0 passed 22/22 with fp16 error about
  `1e-4` and bf16 about `5e-4`.
- `[MEASURED]` Still 5-10x slower than Torch SDPA: every thread serially
  walked all K/V rows, XMX row utilization was 1/8, and pack/softmax scalar
  overhead was high. Lesson: the bottleneck was data reuse and parallelism
  structure, not the DPAS instruction choice.

Copy-ready: [DG2 DPAS 8x8 fp32 core](../api/code-snippets.md#dg2-dpas-8x8-fp32-core).

### v3 two-kernel packed-KV

`flash.attn.b.mha.dg2.dpas3.h`:

- pack kernel: one pass over K/V per head writes global VNNI-packed
  `packedK` / `packedV` / `kvZero`, so every query work-group reuses the
  same packed data.
- attn kernel: `WG=32`, each thread owns 4 query rows (`RPT=4`, `QGRP=128`),
  DPAS M=8 with 4 distinct rows (50% XMX rows), packed K/V tile staged
  cooperatively into SLM once per `(query-group, kv-tile)`.
- Per-thread fp32 accumulator, query rows, scores, and softmax values live
  in registers; SLM holds only shared K/V tiles.
- SLM layout: K and V in separate regions; `BN=128` uses the full 64 KB
  per-work-group budget (K 32 KB + V 32 KB) and removes the QK -> SxV
  barrier; `kvZero` flags load per tile from global into registers.

Three correctness bugs were found and fixed:

1. `[CORRECTNESS]` Per-thread state written to shared SLM raced across the
   32 lanes (q rows, scores, accumulators overwritten). Fix: move all
   per-thread state to registers; keep SLM for shared K/V tiles only.
2. `[CORRECTNESS]` The pack kernel double-applied the tile offset to
   `kvZero`, so tiles >= 1 wrote flags to the wrong location, everything was
   masked, and softmax degenerated to a uniform distribution.
3. `[CORRECTNESS]` `BN=128` padding rows were not masked (the wrapper passed
   the padded `kv_len`); with no SLM room for a flags region, load
   `kvZero` per tile from global into registers.

`[MEASURED]` L=8192 / H=32 / D=128 fp16, wall median x30:

| Variant | Time |
|---|---:|
| v3 BN=128 (production) | ~79-81 ms |
| v3 BN=64 | 105 ms |
| v2 DPAS | 110 ms |
| Torch SDPA | 43 ms |

The pack kernel was only about 2.4 ms; the bottleneck was the attn kernel.

Copy-ready: [packed K/V VNNI operand layout core](../api/code-snippets.md#packed-kv-vnni-operand-layout-core).

### v3 VTune-driven optimization

`[MEASURED]` Initial attn profile (12 launches): 404G instructions, mix
Other 60% / Int32+SP 32% / Send 3.5%; full-compute showed XVE 57% Stalled,
XMX pipeline only 19.5% active, SLM read 5.56 TB/s near the bandwidth roof,
and DRAM only 4 GB/s (pack preprocessing gave good L2 hits). Conclusion:
not an XMX or DRAM-bandwidth problem; instruction volume and operand relay
dominate.

One-variable optimizations:

1. `[MEASURED]` Vectorize softmax `exp2` with ESIMD hardware
   `native_exp2` instead of scalar `sycl::exp2`: SP instructions
   130G -> 58G, 74.5 -> 59.6 ms.
2. `[MEASURED]` Vectorize max reduction, sum, and p writes: 59.6 -> 52.6 ms.
3. `[MEASURED]` Keep scores directly in per-row SIMD vectors
   (`svecArr`); QK writes and softmax reads skip scalar staging:
   -> 48.1 ms.
4. `[MEASURED]` Stage K and V into separate SLM regions simultaneously;
   `BN=128` fits exactly K 32 KB + V 32 KB and drops the QK -> SxV barrier
   (3 -> 2 per tile): isolated 76.4 -> 74.5 ms.
5. `[MEASURED]` Do not zero-fill DPAS A rows 4..7 (their C rows are never
   read): saves 4 GRF fills per DPAS, 48 -> 40 ms, overtaking torch.

Final v3 state: attn instructions 404G -> 194G; 81-case matrix passed
(worst 8.4e-4); see the v3 final table below. The kernel header comment
summarizes the combined softmax vectorization as SP instructions
130G -> 41G.

### v4 fused + async dispatch

- `[MEASURED]` Small shapes were dominated by fixed overhead, not the
  kernel: `queue.wait()` 250-450 us per call (GPU only about 100 us),
  Python sidecar `glob` about 0.5 ms per call, and two submissions instead
  of one. Pure host submit measured about 40 us; `_C`-direct async 512 was
  0.55-0.61 ms.
- Fixes: fused single kernel for `qLen <= 1024` (pack raw K/V into SLM per
  work-group, `BN=64` + flags, one submit); large shapes stay two-kernel
  `BN=128`; release builds drop internal `queue.wait()` (caller torch stream
  synchronizes); sidecar glob cached once.
- `[CORRECTNESS]` `#if FUSED` preprocessor trap: template parameters are
  constant 0 at preprocessing time, so the fused path took the
  `kvZero == nullptr` branch and TDR'd. Use `if constexpr`.
- `[CORRECTNESS]` Fused V-pack SLM offset missed a `*4` (word vs byte).

Copy-ready: [stable online-softmax rescale core](../api/code-snippets.md#stable-online-softmax-rescale-core)
and [host-side dispatch and packed-buffer LRU core](../api/code-snippets.md#host-side-sdp-dispatch-and-packed-buffer-lru-core).

### v4.1 packed-Q RPT8

`flash.attn.b.mha.dg2.dpas4.h`:

- Non-fused path prepacks Q into the DPAS A-operand layout (one contiguous
  256 B block per QK operand), enabling `RPT=8` with 100% XMX rows and zero
  spill: `packQDg2 + packKvDg2 + attnDg2<false>` (three submits), `BN=64`,
  `RPT=8`, `WG=32`.
- Small shapes keep the fused `RPT=4` / `BN=64` single kernel.
- Host passes the original `kv_len`; padded rows are masked by row number,
  so the `kvZero` flag writes are removed.
- Cleanups: dead `kvZero` plumbing removed (1024x4096 about
  3.18 -> 2.94 ms); SDP buffer cache key now includes the KV head count.

Copy-ready: [packed Q DPAS A-operand core](../api/code-snippets.md#packed-q-dpas-a-operand-core).

## Measured Performance Tables

All tables are A770 / driver `32.0.101.8860` / oneAPI 2026.1, D=128 fp16,
H as noted, wall median; protocols are labeled per table.

### v3 final (wall median x30, H=32)

| L | v3 final | Torch SDPA |
|---|---:|---:|
| 512 | 1.44 ms | 0.66 ms |
| 1024 | 1.95 ms | 1.20 ms |
| 2048 | 3.71 ms | 3.78 ms |
| 4096 | 10.9 ms | 13.5 ms |
| 8192 | 41.6 ms | 42.2 ms |

### v4 core (interleaved A/B, H=32)

| L | v4 | Torch SDPA | ratio |
|---|---:|---:|---:|
| 512 | 0.57 ms | 0.60 ms | 0.96 |
| 1024 | 1.24 ms | 1.05 ms | 1.18 |
| 2048 | 2.92 ms | 3.47 ms | 0.84 |
| 4096 | 11.3 ms | 12.6 ms | 0.89 |
| 8192 | 43.0 ms | 43.8 ms | 0.98 |

### v4 headline range (wall median)

| L | v4 | Torch SDPA | result |
|---|---:|---:|---|
| 512 | 0.57-0.62 ms | 0.66 ms | ~10-14% faster |
| 1024 | 1.34-1.36 ms | 1.36 ms | tie / marginal |
| 2048 | 2.9-3.0 ms | 3.78 ms | ~22% faster |
| 4096 | 9.8-10.3 ms | 13.5 ms | ~25% faster |
| 8192 | 40-42 ms | 42.2 ms | ~3-7% faster |

### v4 extended shapes (interleaved A/B, 5 rounds each, median)

Wins: 512/H16 (0.90), 2048/H16/H48 (0.83-0.92), 8192/H16/H48 (0.97),
256/1024 (0.82-0.93), 8192 bf16 (about 1.00).

Losses at that point: 1024/H32 (about 1.18), 1024/H48 (about 1.12),
512/2048 (about 1.13-1.16), 16384/H32 (about 1.02-1.04).

### v4.1 extended matrix (interleaved A/B, same process)

| Shape | v4.1 | Torch | ratio |
|---|---:|---:|---:|
| 512x512 H16 | 0.54 ms | 0.59 ms | 0.91 |
| 512x512 H48 | 0.63 ms | 0.64 ms | 0.98 |
| 1024x1024 H16 | 0.70 ms | 0.70 ms | 1.00 |
| 1024x1024 H48 | 1.44 ms | 1.42 ms | 1.02 |
| 2048x2048 H16 | 1.83 ms | 1.90 ms | 0.96 |
| 2048x2048 H48 | 3.79 ms | 4.85 ms | 0.78 |
| 8192x8192 H32 | 34.7 ms | 41.4 ms | 0.84 |
| 16384x16384 H32 | 136.9 ms | 155.3 ms | 0.88 |
| 1024x4096 H32 | 3.19 ms | 3.07 ms | 1.04 |
| 8192x8192 bf16 | 37.0 ms | 42.3 ms | 0.88 |

The 1024-class cells fluctuate with Torch (absolute about 1.2 ms, Torch
1.04-1.36 ms), so single runs can show ratio 0.97-1.18.

### v4.1 40-sample interleaved anchors

Recorded in the campaign's build/install record:

| Shape | v4.1 | Torch |
|---|---:|---:|
| L=512 (H32, D128) | 0.54 ms | 0.60 ms |
| L=2048 (H32, D128) | 2.75 ms | 3.29 ms |
| L=4096 (H32, D128) | 8.8 ms | 12.6 ms |
| L=8192 (H32, D128) | 34.7 ms | 41.4 ms |
| 1024x4096 H32 | 2.91 ms | 3.20 ms |
| 1024x1024 H48 | 1.43 ms | 1.49 ms |
| 512x512 H48 | 0.64 ms | 0.65 ms |

At 40 samples the 1024 series also wins; at 25 samples Torch fluctuation
occasionally flips it to about 1.00-1.04.

### H3 adapter contracts (bf16, H=56)

| seq | old v4 (dpas3) | v4.1 | Torch |
|---|---:|---:|---:|
| 891 | 1.79 ms | 2.01 ms | 1.36 ms |
| 20685 | 421 ms | 394 ms | ~416 ms |

Newer v4.1 wins the large sequence and is slightly slower on the small one;
the measured custom-vs-torch difference is only sub-millisecond per call.

## Dispatch Rules and Validity Domain

`[DISPATCH]` on A770 / driver `32.0.101.8860` / oneAPI 2026.1 / Torch
2.13.0+xpu / ESIMD sidecar:

- Fused single kernel when `qLen <= 1024` AND
  `qTilesFused * kvTilesFused <= 128`; otherwise use the non-fused
  `packQ + packKv + attn` three-submit path.
- D=128 uses the DPAS path; D=64 stays on the v1 FMA kernel.
- fp16/bf16 are measured; fp32 inputs fall back to PyTorch SDPA (also when
  ComfyUI GGUF `dequant_dtype` is `float32`).
- The integration adapter defaults to PyTorch SDPA on Windows; the custom
  path must be enabled through the adapter's attention-backend switch.
- A770 adapter fallbacks for measured slow H3 contracts: `h=128` RMSNorm
  with row count >= 65536 -> torch; bf16 attention with `q_len < 1024` ->
  torch; large `seq` keeps the custom SDP path.
- MiniMax H3 contract validated: `[1, 56, seq, 128]` fp16, no mask; max
  error vs Torch SDPA 0.000122 (seq 256), 0.000061 (seq 512), 0.000061
  (seq 1024).

Copy-ready: [host-side dispatch and packed-buffer LRU core](../api/code-snippets.md#host-side-sdp-dispatch-and-packed-buffer-lru-core).

Validity domain: `B=1`, D in `{64,128}`, H in `{16,32,48,56}`, fp16/bf16,
seq and kv up to 16384 (and 20685 in the H3 adapter path), `kv_len` padding
handled, single machine/stack; confidence medium (one GPU, one toolchain
generation, multiple interleaved rounds). Outside this domain, re-label
rules `[HEURISTIC]` and re-measure.

Structural limit at 1024-class shapes: oneDNN's no-copy GEMM reaches 100%
DPAS row utilization, while the fused custom path is limited to RPT=4 (50%
rows); the custom path still ties or wins most measured cells but is not a
universal champion.

## VTune Findings

- `[MEASURED]` Initial attn kernel: 404G instructions per 12 launches,
  Other 60% / Int32+SP 32% / Send 3.5%.
- `[MEASURED]` full-compute: XVE 57% Stalled, XMX pipeline 19.5% active,
  SLM read 5.56 TB/s near the bandwidth roof, DRAM 4 GB/s.
- `[MEASURED]` xpu-offload on 1024x4096: attn about 73% of GPU time,
  packKv about 23%, packQ about 3%; remaining attn cost is K/V SLM staging
  and operand relay.
- `[MEASURED]` Instruction volume is not the only metric: v8-style SLM
  relay in the earlier GEMM campaign also increased instructions while
  reducing time (see [techniques.md](techniques.md)); the same principle
  applies to SDP.

## Cache and Integration Lessons

- `[MEASURED]` Python sidecar directory glob was about 0.5 ms per call;
  cache it once at module load. Copy-ready:
  [sidecar glob cache core](../api/code-snippets.md#sidecar-glob-cache-core-python).
- `[MEASURED]` The packed Q/K/V buffer cache was initially unbounded. One
  H3 shape at `seq ~= 20685` cached about 0.9 GB. Bound it with an LRU and
  include the KV head count in the key; `queue.wait()` before freeing USM so
  an async kernel cannot read freed buffers. Final build keeps only the most
  recent shape. Copy-ready:
  [host-side dispatch and packed-buffer LRU core](../api/code-snippets.md#host-side-sdp-dispatch-and-packed-buffer-lru-core).
- `[MEASURED]` First call in a process was 286 ms (cache/alloc); steady
  state about 2.1 ms; no progressive in-process degradation.
- `[MEASURED]` The second-run ComfyUI slowdown was not the SDP kernel: event
  counts stayed equal (attention 200/200, RMSNorm 486/411) but median kernel
  intervals doubled (attention 1080 -> 2548 ms, RMSNorm 354 -> 1553 ms).
  MiniMaxH3 residency changed from 5859 MB loaded / 5462 MB offloaded to
  11010 MB / 311 MB; the whole pipeline slowed under VRAM pressure, and the
  remaining sidecar-side action was to reduce packed-buffer cache pressure.

## Negative Results

All on the same A770 / driver `32.0.101.8860` / oneAPI 2026.1 stack. Keep
each as a standalone reproduction; do not re-run without a watchdog.

- `[BUG]` ESIMD WG=512 triggers A770 TDR (LiveKernelEvent 141) even for a
  trivial `slm_init` + store kernel.
- `[BUG]` RPT=6 / RPT=8 with any compiler spill (7.5-16 KB) raises
  `UR_RESULT_ERROR_DEVICE_LOST`; RPT=6 with fp16 score array reduced spill
  to 2.6 KB and still crashed. Treat any non-zero spill with RPT>4 as a TDR
  trigger.
- `[BUG]` RPT=6 / BN=128 compiled with zero spill but still DEVICE_LOST on
  the first real launch.
- `[BUG]` WG=64 for the non-fused attn kernel produced wrong outputs even
  on a single KV tile (no TDR); staging/barrier geometry stays WG=32.
- `[CORRECTNESS]` Fused BN=128 is numerically wrong on this stack
  (single-tile error about 0.12); fused stays BN=64.
- `[TOOLCHAIN]` Fused RPT=6 spills at BN=64 (1.7-2.0 KB) and BN=32
  (0.9-1.3 KB); never launched.
- `[TOOLCHAIN]` fp16 DPAS accumulator: `ExecutionSize=8` rejected by
  `dpas.hpp`; fp16 C requires N=16, which is numerically wrong on A770.
- `[MEASURED]` Reading packedV directly from global instead of SLM staging:
  172 ms vs 105 ms at BN=64 (cooperative SLM staging wins).
- `[MEASURED]` Chunk-array (6.6 KB spill) variant was about 2.3x slower.
- `[MEASURED]` Manually hoisting the mask check as a loop-invariant was 18%
  slower; the compiler's scheduling was better.
- `[MEASURED]` Unified packQ + fused RPT=8 was slower on small shapes; the
  small/large split was kept.
- `[MEASURED]` Storing fp32 scores back into `svecArr` in the fused path
  helped 512/H48 (0.82) but hurt 1024 variants; fp16 `pChunk` stayed.
- `[CORRECTNESS]` Online-softmax rescale overflow: `exp2((m_old - m_new) *
  log2e)` explodes when the new tile max is lower; H3-scale score spans
  produced inf/NaN and black video frames. Keep rescale <= 1 in both
  directions (scale the old accumulator when the new max is higher; scale
  the current tile when it is lower). Stress test q/k x10, v x10 no longer
  saturates to 65504; 81-case matrix still passes. Copy-ready:
  [stable online-softmax rescale core](../api/code-snippets.md#stable-online-softmax-rescale-core).

## Not Measured / Open Hypotheses

`[HYPOTHESIS]` Until measured on this stack:

- RPT=8 with register compression (e.g., half accumulator in SLM per
  thread, or pChunk in registers) to remove the 50% row waste without
  spill.
- Split-KV / two-kernel global-direct packing for `kv >> q` shapes.
- BN=256 with SLM tile looping for 16K sequences.
- Small-shape fused path with L1 direct reads or pack fusion.

## Provenance

Extracted from an ESIMD SDP sidecar port campaign (2026-08-10..12) on the
stack listed at the top of this page. The v1..v4.1 labels are campaign
versions; the extended benchmark used 8 warmup + 30 samples, median plus
p10/p90, interleaved ours/torch/ours. The source project's build/install
record and kernel headers were used to cross-check the numbers.
