# Typed / Sampler Channel Messages on DG2 (`send.smpl`, `send.dc0`, `send.dc1`)

**Scope**: A770 (DG2), ocloc IGC `32.0.101.8857`. Three real messages taken from ocloc output,
then single-bit flipped and reassembled to read back IGA's own printout; cross-checked against
IGA's XeHPG decoders (`MessageDecoderOther.cpp:tryDecodeSMPL`, `MessageDecoderHDC.cpp`).
Receipts: `probes/041-typed/typed_bits.csv`, `docs/16-typed-sampler.md`.

The reference messages (d32):

```asm
send.smpl (16|M0) r30 r6   r26:2 a0.0 0x028D80FC {ExBSO,A@1,$3}  ; simd16 sample, lod forced to 0, sampler 0
send.dc0  (8|M0)  r1  r127 null:0 0x0 0x024844FD {A@1,$0}       ; oword aligned block read x8
send.dc1  (8|M0)  null r15 r35:4 a0.0 0x020350FC {ExBSO,$5}     ; typed surface write, xyzw, u
```

## 1. Sampler (`send.smpl`) `[ISA]`

| Bits | Flipping it prints | Field |
|---|---|---|
| `[11:8]` | `sampler index 1/2/4/8` | **SamplerIndex**, 4-bit, LSB aligned |
| `[16:12]` | bit12 `sample compare+…`, bit13 `sample load …`, bit14 `ld2 multi-sample wide`, bit15 `gather4+compare`, bit16 `gather4` | **SamplerMessageType**, 5-bit; base `0x18` = sample with lod forced to 0 |
| `[18:17]` | bit18 `simd8 with integer return`, bit17 `unknown simd mode` | **SIMD[1:0]** |
| `[29]` | `simd16 with half precision parameters` | **SIMD[2]** |
| `[30]` | `simd16 16b …` | **ReturnFormat** (0 = 32b, 1 = 16b) |
| `[31]` | `simd16 ?` | message-type bit 5 (positional-offset sampler messages only) |
| `[7:0]` | no printed change | mlen / parameter count area (IGA does not print it) |

### The SIMD encoding is part-specific `[ISA]`

| SIMD[2:0] | XeHPG (DG2) | Xe2 |
|---|---|---|
| 0 | **SIMD8 + integer return** | SIMD16 + integer return |
| 1 | SIMD8 | SIMD16 |
| 2 | **SIMD16** | SIMD32 |
| 5 | SIMD8H (half-precision params) | SIMD16H |
| 6 | SIMD16H | SIMD32H |

The same 3-bit value means a different width on DG2 versus Xe2, so a sampler descriptor copied from
an Xe2 kernel computes the wrong return size on A770. DG2's integer-return form is SIMD8+iret,
not SIMD16+iret.

## 2. `send.dc0` block / gather reads `[ISA]`

| Bits | Flipping it prints | Field |
|---|---|---|
| `[10:8]` | bit8 `x16`, bit9 `x0`, bit10 `x1` | **DataElements**, non-monotonic: 0→x1, 4→x8, 5→x16, 6→x0 |
| `[13]` | `hword aligned block read x16` | **BlockMessageSubtype** (0 = OWord, 1 = HWord) |
| `[14]` | `oword block read x8` (loses "aligned") | alignment / type bit |
| `[15]` | `dword gathering read x2` | block ↔ gather family |
| `[16]` | `qword gathering read` | element width (dword/qword) |
| `[17]` | `hword aligned block write x8` | **read/write direction** |
| `[18]` | `hword scratch block read x1` | **scratch** semantics |

The whole type lives at `[18:14]`, matching `addField("MessageType", 14, 5, …)` in the HDC decoder.
This is the channel every kernel uses at entry to read its argument payload (the three
`send.dc0` instructions at the top of any dump).

## 3. `send.dc1` typed surface writes `[ISA]`

| Bits | Flipping it prints | Field |
|---|---|---|
| `[11:8]` | bit8 `yzw`, bit9 `xzw`, bit10 `xyw`, bit11 `xyz` | **channel disable mask** (1 = disabled; all-zero = xyzw) |
| `[18:14]` | bit14 `typed atomic 16-bit counter …` | **MessageType**, 5-bit |
| `[16]` | `untyped surface write` | typed ↔ **untyped** |
| `[17]` | `typed surface read` | **read/write direction** |
| `[25]` | loses `; u` | `u` flag |
| `[28:26]` | `; u,v,r` / `; ???` | extra flags |

**Polarity warning**: LSC quad messages use `Desc[15:12]` as a channel mask where **1 = enabled**,
while typed writes use `Desc[11:8]` where **1 = disabled**. Mixing the two conventions silently
drops channels.

## 4. Typed vs untyped at a glance

| Aspect | untyped (`send.ugm` / `send.slm`) | typed (`send.dc0` / `dc1` / `smpl`) |
|---|---|---|
| Field layout | Opcode `[5:0]`, DataSize `[11:9]`, VecSize `[14:12]` | MessageType `[18:14]`, subtype `[13]`, data elements `[10:8]` |
| Channel mask | `[15:12]`, **1 = enabled** | `[11:8]`, **1 = disabled** |
| SIMD width | from the instruction's exec size | from `[18:17]` + `[29]`, and **part-specific** |
| Surface | ExDesc BTI, or `bss[a0.0]` + ExBSO | ExBSO as well, but through the sampler/DC channel |
| Immediate offset | none on DG2 | no address offset here either; `[7:0]` carries mlen/param count |

## 5. Practical notes

1. Do not hand-write sampler descriptors from a newer-part reference; the SIMD field changes
   meaning between DG2 and Xe2.
2. `send.dc0` "oword aligned block read" is the cheapest way the compiler fetches kernel arguments
   and constant data — it is not an LSC message and has no cache-policy field.
3. When reading typed writes, remember the mask is inverted relative to LSC quad messages.
