# SLM (Shared Local Memory) Banking on DG2

**Scope**: A770 (DG2), icpx 2026.1, driver `1.17.39894+4`. Measured with 64 work-groups x 64 lanes
(= 4096 hardware threads, full occupancy), 16 KB of SLM per work-group, one 4-byte SLM word read
per lane per iteration, address `(lane * STRIDE + r) & 4095`. The loop-overhead reference
(`acc += r`, same loop shape, no SLM traffic) is subtracted. Receipts:
`probes/040-slm-banks/`, `docs/15-slm-banks.md`.

## The numbers `[MEASURED]`

| STRIDE | lane spacing | ns per iteration | vs stride 1 |
|---|---|---|---|
| 0 (all lanes same word) | 0 B | **43.8** | **0.69x** |
| 1 (consecutive) | 4 B | 63.7 | 1.00x |
| 2 | 8 B | 54.3 | 0.85x |
| 3 | 12 B | 56.0 | 0.88x |
| 4 | 16 B | 57.6 | 0.90x |
| 8 | 32 B | 62.5 | 0.98x |
| 16 | 64 B | 79.6 | 1.25x |
| 32 | 128 B | 106.4 | 1.67x |
| 33 | 132 B | 57.9 | 0.91x |
| 48 | 192 B | 79.8 | 1.25x |
| 64 | 256 B | **209.6** | **3.29x** |
| 96 | 384 B | 106.5 | 1.67x |

## What this means

1. **There is no classic 32-bank x 4-byte conflict cliff.** A 32-bank model predicts 2x at stride 2
   and 32x at stride 32; the measurement shows stride 2 is *15% faster* than consecutive and stride
   32 is only 1.67x. DG2's SLM path has a much wider fetch granule and absorbs conflicts well.
2. **Cost tracks 64 B multiples, not bank indices.** 64 B (1.25x), 128 B (1.67x), 192 B (1.25x),
   256 B (3.29x) and 384 B (1.67x) all cost extra; **odd strides cost nothing** (3: 0.88x,
   33: 0.91x). 256 B is twice as expensive as 128 B or 384 B.
3. **Broadcast is cheap.** All lanes reading the same word costs 0.69x of the consecutive case, so
   hardware merges same-address accesses. Patterns like "every lane reads the same weight" are fine.
4. **4-byte-consecutive lanes are not optimal.** An 8 B lane spacing is 15% *cheaper* than 4 B, so
   `smem[thread_id]` is not the fastest SM layout.

## Practical rules

1. Never make the per-lane stride a multiple of 64 B (16/32/48/64/96 words) — that is the only
   region where SLM costs real time (up to 3.3x).
2. Prefer odd strides, or an 8 B lane spacing; both beat the naive contiguous 4 B layout.
3. Transposed tiles and column-major SLM reads are the usual source of 64 B-multiple strides —
   pad the row pitch by one element (odd stride) and the penalty disappears.
4. Do not "spread out" broadcast reads; same-address access is the cheapest case.

## Still open

- Store (write) side banking, `d8`/`d16` elements.
- The exact `send.slm` form IGC emits for these accesses (block load vs per-lane gather) — both
  appear in `probes/040-slm-banks`, not yet inspected.
- Whether the 3.29x at 256 B is an arbitration artifact or a true bank-count effect.
