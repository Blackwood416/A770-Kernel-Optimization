# DG2 addressing modes: stateful BTI vs bindless ExBSO

Scope: measured on A770 (DG2) with driver IGC 32.0.101.9150 / ocloc 2026.1, from
`probes/056-bindless-vs-stateless` in the Arc-ISA-Research repo.

## 1. What the runtime actually emits

An ordinary OpenCL kernel (global pointer arguments, `cl_mem`) is compiled by the driver into the
**stateful BTI** form:

```
send.ugm (16|M0) r19 r15 null:0 0x0      0x64280500 {A@4,$3}  // load.ugm.d32.a32.ca.ca.bti[0]
send.ugm (16|M0) r23 r11 null:0 0x1000000 0x64280500 {A@2,$5} // load.ugm.d32.a32.ca.ca.bti[1]
```

- address = the per-lane 32-bit offset held in the payload GRF (`r15`, `r11`)
- buffer identity = `ExDesc[31:24]` (BTI index: `0x0` for the first buffer, `0x1000000` for the second)
- `Desc` bit 30 = 0; the buffer base is added by per-lane `add` instructions in the address chain
  (4 such adds for an axpy body).

Passing the same buffers as **SVM pointers** does not change this: the runtime binary is
byte-identical (MD5 match), so SVM does *not* buy a flat a64 kernel on this stack.

`ocloc` AOT compilation of the same source produces the **bindless** form instead:

```
mov (1|M0) a0.0<1>:ud r8.6<0;1,0>:d {A@1}
mov (1|M0) a0.2<1>:ud r8.7<0;1,0>:d {A@1}
send.ugm (16|M0) r19 r15 null:0 a0.0 0x24280500 {ExBSO,A@4,$3} // load.ugm.d32.a32.ca.ca.bss[a0.0]
```

but the driver rejects that binary and recompiles from IR, so bindless code is inspect-only here.

## 2. Field map

| Item | stateful BTI | bindless ExBSO |
|---|---|---|
| `Desc` bit 30 | 0 (`0x64xxxxxx`) | 1 (`0x24xxxxxx`) |
| `ExDesc` | immediate, `BTI[31:24]` | address register form (`a0.0` / `a0.2`) |
| address | payload GRF, per-lane a32 | `a0.x` base + per-lane offset |
| IGA comment | `.bti[n]` | `.bss[a0.n]` |
| base field in arg payload | `r8.4` / `r8.5` | `r8.6` / `r8.7` (different slots) |

## 3. They are not one bit apart

Rewriting a BTI kernel by hand into the ExBSO form (move the base into `a0`, clear the BTI in
`ExDesc`, set `Desc` bit 30, add `{ExBSO}`) **assembles and runs** - no fault, no IR recompile - but
every load returns 0 (bad = 4096/4096, y keeps its original value) while the kernel looks 1.50x-1.59x
faster. ExBSO needs a *bindless surface offset* in `a0`, which a BTI kernel's payload does not
carry, so the two modes are not interchangeable by field patching. Never quote a "faster ExBSO"
number without checking correctness first.

## 4. SWSB and address-computation instructions

The two forms differ in the dependency graph, not just in the descriptor bits (both dumps are in
`probes/056-bindless-vs-stateless`):

| item | stateful BTI | bindless ExBSO |
|---|---|---|
| `mov a0.x` writes | 0 | 2 (`a0.0` <- `r8.6`, `a0.2` <- `r8.7`), each marked `{A@1}` |
| per-lane address `add` | 4 (16-wide, base folded into every lane) | 0 in the hand-written variant, 4 in the AOT form |
| sends | 6 x `{A@n,$m}` | 6 x `{ExBSO,A@n,$m}` |

Bindless replaces per-lane base adds with **address-register writes**: an `a0.x` write is an ARF
dependency, and the compiler model puts ARF latency at 16 cycles (`LatencyTable.h: ARF = 16`), so each
buffer-base switch costs at least one ARF write before its first send. The per-message token stays a
`$m` token in both forms - ExBSO changes where the base comes from, not how sends are tracked.
Whether a switch pays therefore depends on "4 vector adds in the address chain" versus "2 serialising
ARF writes", never on the message cost itself.

## 5. Practical rules

1. On this driver stack, budget address arithmetic for `a32 + bti[n]` plus the per-lane base adds;
   do not copy bindless codegen assumptions from other platforms.
2. To measure a real bindless path you need a runtime that allocates a bindless heap (Level Zero
   device allocation with bindless descriptors, or oneDNN's bindless path); OpenCL here does not.
3. Keep the three knobs consistent: `Desc` bit 30 (ExBSO), `ExDesc[31:24]` (BTI), and the `a0`
   base are a single choice, not independent flags.

## GRF-indirect operands (`r[a0.0]`): encoding differs, cost not measurable here

`mov r40, r[a0.0]` and `mov r40, r64` differ only in the source-operand field (4095 disassembly lines
each, `probes/064-indirect/addressing_direct.txt` vs `addressing_indirect.txt`), but the injected-block
vehicle cannot price the difference: `nop`, direct, indirect and indirect+offset blocks all land at
12.39-12.45 ns per instruction because the 16-byte straight-line blocks
are fetch-bound (`nop` costs the same as `mov`), while a real kernel issues one instruction every
2.85 ns per `docs/31`. That gap is two orders of magnitude larger than the variant spread, so the
experiment neither shows nor excludes a 1-5 cycle addressing surcharge: the mode is *encoded* for free,
its *cost* is unmeasured. The loop-resident vehicle was tried in E38 and failed its controls
(`docs/41`), so EU counters are the remaining instrument. Source:
`docs/38-indirect-addressing.md`.
