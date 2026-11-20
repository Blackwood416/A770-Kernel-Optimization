# Operand Regions and Register Banks on DG2 (XeHPG)

**Scope**: Intel Arc A770 (DG2 / XeHPG), offline IGA `1.1.0-640a706f` + ocloc IGC `32.0.101.8857`.
The legality numbers come from a 3317-candidate matrix (`probes/021-region/legality.csv`,
2475 accepted / 842 rejected); the timing numbers come from 7 whole-kernel variants executed on
the A770 (`probes/021-region/results-*.csv`).

## 1. Region syntax depends on instruction arity `[ISA]`

| Instruction class | Source region form | Example |
|---|---|---|
| 3-source (`mad`) | `src0`/`src1` take `<V;H>`, `src2` takes `<H>` | `r23.0<1;0>:f  r19.0<1;0>:f  r8.2<0>:f` |
| 2-source (`add`, `mul`, `mov`) | `<V;W,H>` | `r5.0<8;8,1>:f` |

- `[ISA]` Applying the mad form `<1;0>` to `add`/`mov` is a hard error: `invalid region width`.
- `[ISA]` Applying the 2-source form to a 3-source instruction fails with `expected >`.
- `[ISA]` IGA prints the offending line with a caret column, which makes `iga64 -a -p XeHPG` the
  fastest syntax oracle to run on a snippet before touching a whole kernel.

## 2. What the fields can encode

| Field | Legal values | Label |
|---|---|---|
| Source horizontal stride | `{0, 1, 2, 4}` | [ISA] |
| Destination horizontal stride | `{1, 2, 4}` | [ISA] |
| `h = 8` | not encodable — all 560 candidates rejected | [ISA] |
| 3-source vertical stride | `{0, 1, 4, 8}` only (2-bit field 0/1/4/8); `V=2` has no encoding and GED rejects it for XeHPG | [ISA] |
| PRM "source region restrictions" (1.1/2.1/2.2/2.4-2.6/2.8) | **warnings, not errors** — 2475/3317 candidates still assemble; R2.8 fires iff `subReg + (W-1)*H >= 8` for `:d`, independent of the register number | [ISA] |

## 3. Assembler acceptance is not hardware legality `[ISA]`

IGA does **not** bound-check GRF numbers: `r128`..`r255` assemble and round-trip cleanly on a
DG2 target even though DG2 has only `r0`..`r127`. Any "the assembler accepted it" argument about
register legality is therefore void; use the ISA limits, not the assembler.

## 4. Register banks

### 4.1 Compiler side `[SRC]`

IGC/vISA do **no** register-bank-conflict reduction for DG2 by default:

- `hasBankCollision()` excludes `Xe_DG2` (`sources/igc/visa/HWCaps.inc:388-395`)
- the forced-BCR path excludes `IGFX_DG2` (`IGC/Compiler/CISACodeGen/CISABuilder.cpp:4494-4506`)
- `oneGRFBankDivision()` is false (`HWCaps.inc:397-402`)
- vISA's DG2 bank model is 4 classes, `bank = ((reg >= 64) << 1) | ((reg >> 1) & 1)`,
  `bundle = (reg >> 2) & 15` per 64-GRF half (`visa/G4_BB.cpp:378-392`, `visa/LocalRA.h:250`);
  the pair-based analysis is Xe_HPSDV-only.
- IGA and vc-intrinsics model no banks at all.

So the compiler will not save you from a bank conflict, and the assembler cannot even tell you
that you have one.

### 4.2 Hardware side `[MEASURED]`

7 variants that move the `mad` source registers (same register, into the other 64-GRF half, across
bank classes) all measured **4.2 µs** on axpy, with overlapping p10–p90. The baseline cannot price
bank conflicts because it is memory-bound; bank pricing has to be done on a compute-bound kernel
(see `dpas.md` §2.2 for the vehicle).

## 5. Practical rules

1. Emit the arity-correct region form; run `iga64 -a` on the snippet first.
2. Stay inside `h ∈ {0,1,2,4}` and `V ∈ {0,1,4,8}`; treat anything else as unencodable rather than
   "probably fine".
3. Do not use register numbers ≥ 128 as a scratch trick — they assemble and then fail on hardware.
4. Assume no bank-conflict mitigation from the compiler on DG2; if a kernel is near the issue
   limit, spread operands across banks by hand and measure on a compute-bound shape.


## GRF bank conflicts: measured zero cost on DG2 `[MEASURED]`

vISA models DG2 banks as 4 classes with `bank = ((reg>=64)<<1) | ((reg>>1)&1)` (adjacent register pairs
share a bank). Measuring that on hardware: replacing the axpy body with K independent `mad`s whose three
source registers are placed either **all in one bank** (r20/r21/r28 -> bank0) or **spread over three banks**
(r20/r22/r68) gives identical time at K = 128 / 512 / 2048:

| configuration | banks | K=128 | K=512 | K=2048 |
|---|---|---|---|---|
| spread | 0,1,2 | 4.4 us | 9.2 us | 28.2 us |
| all same | 0,0,0 | 4.4 | 9.2 | 28.2 |
| pair | 0,0,1 | 4.4 | 9.2 | 28.2 |
| high half | 2,2,3 | 4.4 | 9.2 | 28.0 |

The probe is compute-bound (time tracks mad count: 12.4-12.5 ns per mad per thread at fixed problem
size), so the absence of a difference is not memory noise. This matches the compiler side: IGC performs
**no** bank-conflict reduction for DG2 (`hasBankCollision()` excludes Xe_DG2, `oneGRFBankDivision()` is
false), i.e. it is not missing an optimization - there is nothing to optimize.

Practical consequences:

1. Do not waste registers scrambling for bank spread; allocate for live range and alignment only.
2. The SLM rule from `hardware/slm-banks.md` (avoid 64 B-multiple lane strides) does **not** transfer to
   GRF - they are different structures.
3. Hand-written register numbers are preserved verbatim by ocloc (verified by disassembly), so asm-level
   register experiments are reliable.
