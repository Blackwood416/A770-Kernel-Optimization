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

## Store (write) side banking, alignment, and SLM read wall `[MEASURED]`

Measured on A770 (oneAPI 2026.2, driver `32.0.101.9150`, ESIMD `GRF=256 + WG=32`) via `ocloc disasm` and VTune `full-compute`:

1. **Write-side 256 B stride triggers an 8-way conflict**: Using `esimd::slm_scatter<uint32_t, 64, 4>` / `lsc_slm_scatter` with a 256 B stride (`256 B = 0 mod 128 B`) causes a hardware 8-way SLM store bank conflict (`mov` packing rises `143 -> 218`, `1024x4096` regresses `454 -> 470 us`, and `4096x4096x4096` slows by `+79 us` from `1733 -> 1812 us`) `[MEASURED]`.
2. **Coalesced `4x64 B` (`store.slm.d64x8t`) store reduction**: Coalescing `8x32 B` B-tile SLM writes into `4x64 B` (`store.slm.d64x8t`, packing adjacent `uint16_t` pairs into `32x8 B` block writes) plus `1x128 B` A store cuts per-`BK=16` `store.slm` ops from `9` to `5`, reducing SLM write traffic by `24%–31%` (`0.395 GB -> 0.301 GB`) and freeing SBID tokens `[MEASURED]`.
3. **`slm_block_store` enforces strict 16 B (OWord) hardware alignment `[BUG]` `[CORRECTNESS]`**: Passing an 8 B-aligned offset (e.g., `slm_block_store<uint32_t, 2>`) does **not** fault; DG2 silently truncates the address down to 16 B granularity and overwrites adjacent elements. Always align offsets to 16 B and emit `slm_block_store<uint32_t, 4>` or wider.
4. **Physical `128 KB` SLM co-residency cliff at `GRF=256 + WG=32`**: With 64 HW threads/Xe-core at `GRF=256`, **`SLM <= 48 KB` (`<= 49,152 B`, e.g., `cab4` `4 x 12 KB = 48 KB`) sustains `2 WGs/Xe-core = 64/64` HW threads** (`2 x 48 KB = 96 KB <= 128 KB`). On `Btile192` (`2048x3072x2048`), `WG=32 / 48 KB` hits **`290–300 us`**, far outperforming `WG=24 / 40 KB` (`362 us`, under-occupied) and `WG=48 / 56 KB` (`452 us`, `2 x 56 > 128 KB` forces 1 WG/Xe-core barrier stalls) `[MEASURED]`.
5. **Measured DG2 SLM read bandwidth wall**: VTune `full-compute` measures practical/peak SLM read throughput at **`3.29–3.77 TB/s`** (`Btile192` `3,288 GB/s`, oneDNN `cab4` `3,769 GB/s`). Once near `3.3+ TB/s`, further gains require higher M/N reuse per SLM byte (e.g., `cab4` `BK=16` vs `BK=32`), not raw `store`/`load` micro-tweaks `[MEASURED]`.

## Still open

- Exact hardware arbitration mechanism behind the 256 B bank cliff (arbitration artifact vs physical bank count) and `d8`/`d16` scattered sub-word writes.
