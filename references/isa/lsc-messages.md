# LSC / Send-Message Encoding on DG2 (XeHPG)

**Scope**: Intel Arc A770 (DG2), driver IGC `32.0.101.9150`, ocloc IGC `32.0.101.8857`, IGA
`1.1.0-640a706f`. The bit maps below were established **empirically** by flipping single descriptor
bits on a real driver-emitted `send.ugm`, reassembling and reading back IGA's own message printout
(receipts: `probes/035-lsc/desc_bits.csv`, `desc_fields.csv`, `docs/10-lsc-descriptor.md`), then
cross-checked against IGA's XeHPG decoder and IGC's encoder (`docs/13-lsc-descriptor-source.md`).

**Do not trust `iga64 -Xdsd` on XeHPG.** It decodes with a newer (E64) field table and reports
nonsense for this part (e.g. `AddrSize=invalid`, `DataSize=8b` for a message that is `d32.a32`).
The tables below are what the assembler/disassembler and the XeHPG decoder source agree on.

## 1. Message descriptor (`Desc`, the second immediate dword)

| Field | Bits | Encoding |
|---|---|---|
| Opcode | `[5:0]` | a **sequential** enum, not powers of two: `0x00` load, `0x01` load_strided, `0x02` load_quad, `0x03` load_block2d, `0x04` store, `0x05` store_strided, `0x06` store_quad, `0x07` store_block2d, `0x08`–`0x1A` atomics (iinc, idec, load, store, iadd, isub, smin, smax, umin, umax, icas, fadd, fsub, fmin, fmax, fcas, and, or, xor), `0x1B` load_status, `0x1C` store_uncompressed, `0x1D` ccs, `0x1E` rsi, `0x1F` fence, `0x20` store_uncompressed_quad, `0x28`–`0x2A` atomic_apnd*, `0x31`/`0x32` quad_msrt |
| AddrSize | `[8:7]` | 0 = invalid (prints `gathering load` / `scattering store`), 1 = a16, **2 = a32**, 3 = a64. For block2d this bit becomes `Block2dVnniTransform` (address size is implied a64) |
| DataSize | `[11:9]` | element type only: 0 d8, 1 d16, **2 d32**, 3 d64, 4 d8u32, 5 d16u32, 6 d16u32h |
| VecSize | `[14:12]` | vector width: 0 V1, 1 V2, 2 V3, 3 V4, 4 V8, 5 V16, 6 V32, 7 V64 (not printed for block2d) |
| DataOrder | `[15]` | 0 = non-transposed, 1 = transposed (`d32x1t`); a transposed message forces **exec size 1** |
| Channel mask | `[15:12]` | quad ops only (`load_quad`/`store_quad`/`*_uncompressed_quad`/atomics): 4-bit mask, **1 = channel enabled** (inverse of the legacy ChMask). Measured over all 16 values: bit0=x, bit1=y, bit2=z, bit3=w; **mask 0 selects no channel** and IGA degenerates to `quad load (a.k.a. load_cmask)` with no operands |
| Caching | `[19:17]` | see §2 |
| AddrType | `[30:29]` | 0 = flat, 1 = bss, 2 = ss, 3 = bti |
| `[6] [16] [28] [31]` | — | flipping them changes nothing that is printed (reserved, or Xe2+-only fields) |

I.e. `d8x16` is DataSize=0 (d8) plus VecSize=5 (V16): element type and width are two separate
fields, which is why the field-wise walk looked like one 6-bit enum.

## 2. Cache policy (`Caching`, 3 bits at `[19:17]`)

| Value | Load prints | Store prints |
|---|---|---|
| 0 | *(nothing — "use state settings")* | *(same)* |
| 1 | `uc.uc` | `uc.uc` |
| 2 | `uc.ca` | `uc.wb` |
| 3 | `ca.uc` | `wt.uc` |
| 4 | `ca.ca` (**default**) | `wt.wb` |
| 5 | `st.uc` | `st.uc` |
| 6 | `st.ca` | `st.wb` |
| 7 | `ri.ca` | `wb.wb` (**default**) |

Tokens: `uc` uncached, `ca` cache-all, `st` streaming, `ri` read-invalidate, `wt` write-through,
`wb` write-back; the first token is the L1 policy and the second the L2/L3 policy. Xe2+ adds a
4-bit variant at `[19:16]` (`cc` = constant-cached) which does not exist on DG2.
Measured value of these bits (plus the trap that they only matter past cache capacity) is in
`docs/11-cache-policy.md`: for a pure streaming axpy, `st.uc` is 14% faster than the default.

## 3. Extended descriptor (`ExDesc`, the first immediate dword)

| Field | Bits | Evidence |
|---|---|---|
| **BTI** (surface index) | `[31:24]` | base `0x01000000` prints `bti[1]`; flipping bits 24..31 yields `bti[0] [3] [5] [9] [17] [33] [65] [129]` — a plain 8-bit unsigned index |
| **Src1.Length** | `[10:6]` | flipping these makes the assembler reject the line with `mismatch of Src1.Length suffix and ExDesc[10:6]` |
| `[3:0]` | — | not an SFID on this part: the SFID is encoded in the instruction |

## 4. What the driver's own code looks like

```
send.ugm (16|M0)  r19  r15  null:0  0x1000000  0x64280500  {A@4,$3}   // load.ugm.d32.a32.ca.ca.bti[1]
send.ugm (16|M0)  null r11  r23:2   0x0        0x640E0504  {A@2,$7}   // store.ugm.d32.a32.wb.wb.bti[0]
```

- Load: `Desc=0x64280500` → AddrType 3 (bti), Caching 4 (`ca.ca`), DataSize 2 (`d32`), AddrSize 2 (`a32`).
- Store: `Desc=0x640E0504` → Caching 7 (`wb.wb`), AddrType 3; `bti[0]` comes from `ExDesc=0`.
- Driver-JIT code uses `bti[N]`, while AOT ocloc output uses `bss[a0.0]` + `{ExBSO}` bindless
  addressing. Register maps read off an AOT disassembly do **not** transfer to the runtime binary.

## 5. Practical rules

1. A hand-written `bti[N]` send needs **both** halves: `AddrType=3` in `Desc[30:29]` *and*
   `ExDesc[31:24] = N`. Getting only one right silently addresses the wrong surface.
2. `a32` is `AddrSize=2`. Writing 0 does not error; IGA just prints `gathering load` — a quiet trap.
3. Defaults are `ca.ca` for loads and `wb.wb` for stores. Use `st` for streaming, `uc` to bypass L1,
   `ri` to read-and-invalidate.
4. Compare descriptors against the `ocloc disasm` comment, never against `-Xdsd`.

## 5b. Alignment cost of the payload message

A `d32x4` message sent at a non-16-byte address does not fault, it just splits into more sectors.
Measured on a 32 MB buffer, 8 passes [MEASURED]:

| base offset | median | effective bandwidth | vs aligned |
|---|---|---|---|
| 0 B | 110.4 us | 2431 GB/s | 1.00x |
| +4 B | 132.2 us | 2031 GB/s | **1.20x** |
| +8 B | 129.4 us | 2075 GB/s | 1.17x |
| +12 B | 130.0 us | 2065 GB/s | 1.18x |

Byte/word-granular accesses are alignment-insensitive (u8 aligned 844.3 us vs u8+1 B 855.9 us, 1.4%),
and scalar throughput is request-rate limited rather than bandwidth limited (u8 318 / u16 609 /
u32 1187 GB/s, i.e. a constant requests-per-second). The 2431 GB/s of the aligned vector case is an
L2-resident artefact of the 32 MB buffer, not a DRAM figure.

## 6. DG2 has no immediate-offset field

IGA gates `exDescHasImmOff` on `platform() >= XE2`, so on XeHPG an LSC message **cannot** carry an
`[+imm]` address offset and `ExDesc[11:0]` is required to be zero. Offsets must be folded into the
address register. This is why flipping `Desc` bits 28/31 changes nothing: those bits are the
immediate offset (or, on Xe2+ block2d, the X/Y offset pair) only on newer parts.

## 7. 2D block messages: encodable, but they DO NOT WORK on DG2

IGC happily emits `load_block2d.ugm.d32.a64` for `-device dg2` and IGA disassembles it, but
**the hardware never completes the message**: a `load_2d` of 2 floats / 1 row hangs the kernel
forever (no TDR, no error code, GPU stays healthy for other contexts), and every block shape from
2 to 256 elements fails the same way. JIT and AOT (`-device dg2`) both hang.
**Do not use 2D block IO on A770.** Receipts: `docs/14-block2d.md`, `probes/039-block2d/`.

Encoding, for the record (shared with parts that do support it):

| field | bits | note |
|---|---|---|
| Opcode | `[5:0]` | `0x03` load_block2d / `0x07` store_block2d |
| VNNI transform | `[7]` | prints as suffix `v` |
| element size | `[11:9]` | d32 = 2 |
| transpose | `[15]` | prints `t` |
| **block size in GRFs** | `[24:20]` | `bw*bh*elemsize/32`, min 1, saturates at 31; depends only on the element count, not on the width/height split (8x4 and 16x2 both give `0x02400403`) |
| constant | `[25]` | always 1 for block2d |

Payload = 8 dwords (2 GRFs), read off compiler output:

```
d0,d1  surface base (64-bit)
d2     SurfaceWidth  - 1   (bytes)
d3     SurfaceHeight - 1   (rows)
d4     SurfacePitch  - 1   (bytes)
d5     X (elements)   d6  Y (rows)
d7     constant 0x307 (purpose unknown)
```

The message is always exec size 1, the payload lives in a GRF-aligned register block, and IGA does
not validate the payload length (it accepts `srcLen` 2/4/6/8/16), so this layout can only be read
off generated code. Three independent sources agree the capability is absent here: the ESIMD
header marks `load_2d`/`store_2d` as `Supported platforms: PVC`; oneDNN's `block_2d_utils.hpp`
covers only XeHPC/Xe2/Xe3/Xe3p and sends XeHPG to `gpu_error_not_expected()`; and ocloc's OpenCL
front end does not expose `intel_sub_group_2d_block_read_*`.

## 7b. Atomics `[MEASURED]`

All 22 atomic opcodes assemble and print as listed in the opcode table above (`0x08` iinc ... `0x1A` xor,
`0x28`-`0x2A` print as `acadd/acsub/acstore` even though IGA's source calls them `apnd*`).
Every atomic message defaults to `wt.wb` caching (atomics bypass L1), unlike loads which default to `ca.ca`.

Cost with 4096 hardware threads all hitting the **same** dword:

| operation | time | per op | scaling |
|---|---|---|---|
| fp32 atomic add | 551 us (K=64) / 4375 us (K=512) | **2.09 ns** | linear, trustworthy |
| uint CAS loop | 316 ms (K=64) / 3166 ms (K=512) | **1.2-1.5 us** | linear, trustworthy |
| uint fetch_add | 14.8 us / 62.6 us | 0.03-0.06 ns | **not trustworthy** |

The integer `fetch_add` rows are folded by the compiler: 8x more work costs only 4.2x more time, and
contended is *faster* than uncontended (33 vs 22 Gops/s, i.e. 66 same-address atomics per clock, which is
physically impossible). **Rule: before quoting any atomic cost, verify it scales linearly with the op count.**

Engineering consequences:

1. Never use a CAS loop as a counter. Under full contention it costs 1.2 us per successful exchange,
   roughly 600x an fp32 atomic add.
2. fp32 atomic add is cheap (2.09 ns/op) and linear, so histograms and accumulators are fine.
3. Reduce contention first: aggregate inside the work-group, then issue one global atomic per group
   (4096 contenders -> 64).

## 8. Still open

- The `[6] [16] [28] [31]` bits: no effect on the printed message, but they may be functional for
  opcodes other than load/store (atomics, cmask, block reads).
- The `0x307` dword in the block2d payload.
- `oword block-read` / typed (`send.tgm`) message widths versus LSC `d32x2/x4` on SLM and global loads.

## Message issue cost and in-flight behaviour `[MEASURED]`

K identical `store.ugm.d32` messages (64 B each) cloned into one thread body, 128 hardware threads,
time relative to K=1:

| K | device time | delta |
|---|---|---|
| 1 | 3.1 us | - |
| 16 | 3.3 | +0.2 |
| 64 | 4.1 | +1.0 |
| 256 | 7.4 | +4.3 |
| 1024 | 21.0 | +17.9 |

Linear from K=64 on: **17.2-17.7 ns per extra 64 B message per thread**, i.e. ~465 GB/s aggregate at
128 threads. Within the store/one-address probe (K <= 1024) there is no stall knee and the limiter is
delivery rate. Two further families were generated from OpenCL source (the asm-cloning route silently
drops them): writes at distinct addresses cost **169 ns** per message and reads at distinct addresses
**51 ns**, both linear with no knee up to 250 messages per thread.

A store send that waits on a token nobody sets (`{$5.dst}` / `{$5.src}`) does **not** hang: its dst is
null, so the wait has nothing to wait for. Hangs belong to the barrier path (injected barriers
deadlock 8/8, see `isa/sync-fences.md`) and to load sends whose dst token never arrives.

Tooling trap found here: `ocloc disasm` output ends each line with a `// ...` comment, so an
annotation appended to the whole line lands *inside the comment* and is silently dropped. Split the
line at `//`, append the annotation to the code part, then re-attach the comment.


### Message price depends on the address pattern `[MEASURED]`

Same 64 B store message, three families (source-generated, 128 hardware threads):

| family | send count swept | marginal cost |
|---|---|---|
| store, same address | 4 - 1024 | **17.6 ns** per message per thread |
| store, distinct addresses (128 B apart) | 4 - 250 | **169 ns** per message per thread |
| load, distinct addresses | 4 - 250 | **51 ns** per message per thread |

All three stay linear with no stall knee, i.e. within 250 (distinct) to 1024 (same address) outstanding
messages per thread the thread never blocks on a queue - the cost is a per-message delivery price.
That price is ~9.6x higher for distinct addresses than for the same address, because same-address
stores get merged; quoting the same-address figure for a real kernel underestimates by an order of
magnitude. Reads are ~3.3x cheaper than distinct-address writes.

Tooling note on how the sweep was produced: cloning sends in the disassembly text fails for
distinct addresses and for reads (the inserted instructions silently vanish), so these two families
are generated from OpenCL source instead - the compiler emits 2 sends per source-level store/load
and stops unrolling past K=124, which caps the method at 250 sends per thread.

Two different bounds must not be conflated: the **architectural scoreboard width is 16 tokens**
(SBID is 4 bits) while the **number of in-flight messages is not token-bound** - unannotated sends
consume no token, and no stall knee appeared up to 250 (distinct address) or 1024 (same address)
messages per thread.

## Prologue argument fetch: what `send.dc0` costs

The arg block is not free, and the cost is visible as `send.dc0` "oword aligned block read" messages
in the kernel prologue. Measured on A770 (`probes/055-oword-block-read`, device profiling, median of
15, grid 64 unless noted):

| arg block | payload | dc0 messages | descriptor | device time |
|---|---|---|---|---|
| 4 args | 16 B | 0 | — | 2.187 us |
| 8 args | 32 B | 1 | 0x021842FD (x2) | 2.083 us |
| 16 args | 64 B | 1 | 0x022843FD (x4) | 2.187 us |
| 32 args | 128 B | 1 | 0x024844FD (x8) | 2.187 us |
| 64 args | 256 B | 2 | 0x024844FD | 2.291 us |
| 128 args | 512 B | 4 | 0x024844FD | 2.708 us |
| 256 args | 1 KB | 8 | 0x024844FD | 3.645 us |

Rules extracted:

1. **<=16 B of arguments costs no dc0 at all** - the values arrive in the thread payload.
2. Above that the driver stages args in the constant/arg block and the prologue reads them with
   x2 / x4 / x8 block reads, then in multiples of x8.
3. **Each dc0 message costs 212.3 ns on the mixed-width fit; x8-only it is 226.9 ns (1.77 ns/B)** (least squares over all six dc0>=1 rows of the committed CSV: dc0 = 1,1,1,2,4,8;
   including the dc0=0 baseline row gives 198 ns, taking only first occurrences gives 224 ns), i.e. ~0.21 us per
   extra 128 B of arguments. This tax is per launch, not per work-item: growing the grid 64x (64 -> 4096
   work-items) moved the 1 KB case only from 3.645 to 3.854 us.
4. Per byte, x8 dc0 is **~1.9x more expensive than `load.ugm`** (1.77 ns/B vs 0.91 ns/B; 1.66 ns/B on the mixed fit; one unrolled
   `send.ugm` load of 64 B costs ~58 ns). For large read-only tables prefer a global pointer.
5. A file-scope `__constant float C[N] = {...}` **never produces dc0** - the compiler folds the
   literals into immediates (dc0 = 0 and a flat 2.187 us for N = 4..256). A `__constant float*`
   parameter, by contrast, is still a global pointer and does not use the constant buffer.
6. dc0 width lives in `Desc[11:8]` (DataElements): 0x021842FD / 0x022843FD / 0x024844FD differ only
   in that field, each writes 1+ GRF and declares rd:1/2/4, and every message carries its own
   `{A@1,$n}` token - the prologue fetch is a parallel, token-tracked message stream, not a
   dependency chain.

## Prefetch: no encodable instruction and no working source path on DG2

Three routes were measured (`probes/073-prefetch`, raw rows in `prefetch_attempts.csv`):

| route | what was tried | result | error |
|---|---|---|---|
| IGA | `lsc_prefetch.ugm` | REJECT | line 1.1: error: invalid mnemonic lsc_prefetch.ugm (16\|M0) r10.0:u32 r20.0:u32 ^ |
| IGA | `lsc_prefetch` | REJECT | line 1.1: error: invalid mnemonic lsc_prefetch (1\|M0) null:ud r10.0:uq ^ |
| IGA | `send.prefetch` | REJECT | line 1.6: error: invalid subfunction send.prefetch (16\|M0) null:ud r10:ud 0x0:ud 0x0:ud ^ |
| IGA | `lsc_prefetch_stateless` | REJECT | line 1.1: error: invalid mnemonic lsc_prefetch_stateless (8\|M0) null:ud r10.0:uq ^ |
| OpenCL __builtin_prefetch | `cl_prefetch.cl` | COMPILE_FAIL | InvalidFunctionCall: Unexpected llvm intrinsic: llvm.prefetch.p1i8 |
| SYCL/ESIMD esimd::prefetch | `prefetch(x+k+4096) with L1/L2 hints` | COMPILES_BUT_NO_ISA | prefetch instructions in ISA: 0 |

The local IGA (XeHPG model) has **no prefetch mnemonic at all** — `lsc_prefetch`, `lsc_prefetch.ugm`,
`lsc_prefetch_stateless` and `send.prefetch` are all rejected, and no `Model*.hpp` mentions prefetch — so asm
injection is impossible with this toolchain. On the source side, OpenCL `__builtin_prefetch` fails compilation
(`Unexpected llvm intrinsic: llvm.prefetch.p1i8`) and ESIMD `esimd::prefetch` compiles only with explicit L1/L2
cache hints, then contributes **zero instructions** to the ISA.

Practical consequence: do not design around explicit prefetch on DG2. Get locality from blocking, sequential access
and data reuse instead; whether the hardware has an automatic prefetcher is not measurable here (needs EU/L2
counters or IGC pass logs).
