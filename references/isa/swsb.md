# SWSB (scoreboard) on DG2 (XeHPG) — Encoding, Traps, Measured Cost

**Scope**: Intel Arc A770 (DG2 / XeHPG), driver IGC `32.0.101.9150`, offline IGA
`1.1.0-640a706f`. Facts come from an IGA legality matrix over 71 candidate annotations
(`probes/020-swsb/legality.csv`) plus 10 whole-kernel variants executed on the A770
(`probes/020-swsb/results-*.csv`).

This page matters for two reasons: SWSB is **functionally required** (a stream with no
dependencies can fail to complete), and it is a **first-class performance resource** (starving the
token pool costs 33%).

## 1. Encoding

| Item | Value | Label |
|---|---|---|
| Model | `ThreeDistPipe` — 16 scoreboard tokens, maximum register distance 7 | [ISA] |
| Field location | instruction byte 1, same for 8 B and 16 B forms (not part of the opcode word) | [ISA] |
| Verbatim encodings | `{@1}`=0x01, `{A@1}`=0x09, `{F@1}`=0x11, `{I@1}`=0x19, `{L@1}`=0x51, `{$0}`=0x40, `{$0.dst}`=0x20, `{$0.src}`=0x30, `{@1,$0.dst}`=0x90 | [ISA] |
| Combination rule | a distance and a token combine only in the **class-matching** forms: `{@n,$m.dst}` for normal ALU, `{A@n,$m}` for send, `{@n,$m}` for math/DPAS. Everything else is `invalid SWSB`. All three combined forms use the same packed field `0x80|(n<<4)|m` (m=0 gives 0x90) | [ISA] |
| `sync` | only `nop`, `allrd`, `allwr`, `bar`, `host` exist. `sync.all` / `sync.sb` / `sync.thread` do not exist in IGA; `sync.grfsb` has a hardware encoding (1) that the assembler cannot express | [ISA] |
| Dependency reporting | `iga64 -Xprint-defs` / `-Xprint-deps` report register def/use sets only; they do **not** resolve `$n` to the producing instruction, so a hand-written schedule must be reasoned about from the encodings | [TOOLCHAIN] |

## 2. Silent traps `[BUG]`

- `{$0.inc}` assembles to **0x00** — IGA's `assert(0)` is compiled out, so the no-dependency
  annotation is silently dropped.
- `{$16}` assembles to **0x50** (`REG_DIST_LONG | 0`) because the range check is
  `sbid > getMaxSWSBTokenNum()` with max 16 — an off-by-one that turns token 16 into "no token".
- `$17..$31` are correctly rejected.
- Neither of the first two produces any warning; both quietly invalidate a schedule.

## 3. Measured cost (axpy, n=4096, 200 launches)

| Variant | dev µs (median) | wrong elements | Reading |
|---|---|---|---|
| baseline (byte-identical to driver binary) | 4.2 | 0 | control |
| all token ids collapsed to `$0` | **5.6** | 0 | **token starvation costs +33%** |
| 29 extra `sync.nop` inserted | 4.3 | 0 | sync slots are nearly free (+2%, in the noise) |
| all `sync.nop` deleted | 4.2 | 3936 | races, no speedup |
| one coarse `sync.allrd` replacing two fine waits | 4.2 | 4032 | the wait sets are **not** equivalent |
| every register distance +3 | 4.2 | 2784 | distance is not a simple "≥ producer gap" |
| **all tokens/distances stripped** | — | — | **kernel never completes: `CL_OUT_OF_RESOURCES`** |
| **every distance pinned to 7** | — | — | **kernel never completes: `CL_OUT_OF_RESOURCES`** |

Labels: all rows `[MEASURED]`.

Takeaways:

- `[MEASURED]` Reallocating tokens so many producers share one id is a real 33% regression even
  when the result stays correct — token assignment is a budget, not a hint.
- `[MEASURED]` Stripping dependencies is not "optimization", it is a way to hang the kernel.
  A failed dependency stream returns `CL_OUT_OF_RESOURCES` from `clFinish`; the process exits and
  there is **no** device-level TDR event. Never treat "no bench line" as "0 µs".
- `[MEASURED]` `sync.nop` slots are cheap on a memory-bound kernel, so "delete syncs for speed"
  has nothing to win here.
- `[CORRECTNESS]` A straight-line body injected into asm needs dependency anchoring: without it
  only ~N/4+2 of the inserted operations reach the final store and the two SIMD halves disagree;
  inserting the body **before** the original `mad` pair (which keeps the `$3.dst`/`$4.dst` chain
  intact) makes every lane agree.
- `[MEASURED]` The timer cannot distinguish distance-field edits at this shape — the kernel is
  memory-bound. Price SWSB on a compute-bound base (`dpas.md` §2), not on axpy.

## 4. Practical rules

1. Never hand-edit away a producer's token when the consumer still needs the value; if you must
   re-annotate, keep the class-matching form (§1) and verify by running.
2. Budget tokens deliberately: collapse only producers that are never live at the same time.
3. Do not rely on the assembler to reject a bad annotation (§2) — assemble, then run, then check
   correctness, in that order.


## DG2 byte-level SWSB encoding `[ISA]` `[MEASURED]`

On A770 the whole SWSB annotation lives in byte 1 of the instruction (ThreeDistPipe). Verified by
assembling one variant per annotation with `iga64 -a -p XeHPG` and reading the byte back:

| written | byte1 | meaning |
|---|---|---|
| (nothing) | `00` | no dependency |
| `{@1}`..`{@7}` | `01`..`07` | minimum distance, any pipe |
| `{A@n}` / `{F@n}` / `{I@n}` / `{L@n}` | `08+n` / `10+n` / `18+n` / `50+n` | per-pipe distance |
| `{$n.dst}` / `{$n.src}` | `20+n` / `30+n` | wait for token n on write / read |
| `{$n}` | `40+n` | allocate / set token n |
| `{A@n,$m}` | `80 | (n<<4) | m` | distance + token, **send only** |
| `{@n,$m}` | `80 | (n<<4) | m` | distance + token, **math/DPAS only** |
| `{@n,$m.dst}` | `80 | (n<<4) | m` | distance + token, **normal ALU only** |

SBID is 4 bits (`$0`-`$15`), distance is 3 bits (0-7).

Traps, all reproduced:

* **`{$n.inc}` assembles to `00`** - the counter-increment has no encoding slot on DG2, so
  "releasing" a token this way is a silent no-op.
* **`{$16}` assembles to `50`** (decodes as `{L@0}`): out-of-range SBIDs fold into the prefix
  field instead of failing. `{$255}` and `@8`/`A@8` do fail ("Invalid SWSB ID/distance number").
* `{S@n}` and `{M@n}` are rejected ("invalid SWSB") - DG2 has only A/F/I/L. `NoAccSBSet` is
  rejected too (Xe2+ only).
* A distance + token combination **is** legal, but the spelling is fixed per instruction class -
  IGA enforces this in `SWSB::verify<ThreeDistPipe>` (`iga_types_swsb.cpp:292-309`) and hardware-side
  assembly agrees (`probes/051-swsb-token/swsb_classes.csv`, 24 combos over 4 classes):

  | class | legal form | measured |
  |---|---|---|
  | normal ALU | `{@n,$m.dst}` | `{@1,$2.dst}` -> `0x92` (accept); `{@1,$2}` rejected |
  | math / DPAS | `{@n,$m}` | `{@1,$2}` -> `0x92` (accept); `{A@1,$2}` rejected |
  | send | `{A@n,$m}` | `{A@1,$2}` -> `0x92` (accept); `{@1,$2.dst}` rejected |

  All three land in the same packed field `0x80|(n<<4)|m`; `@0` collapses to the plain token, and
  `{I@n,$m}` / `{F@n,$m}` / `{L@n,$m}` do not exist in any class.

Hardware check (512-mad body, n=4096): every annotation variant runs at 9.1-9.2 us, identical to
the unannotated control, i.e. annotations are free - the cost of SWSB is the waiting, not the
encoding. Re-setting the same token on 512 instructions without ever consuming it does **not**
deadlock: tokens are set/overwrite, not a queue. Only waiting on a token nobody sets hangs.

## Calibrating the compiler's latency model against hardware (DG2)

vISA's LocalScheduler carries the compiler's idea of instruction latency
(`sources/igc/visa/LocalScheduler/LatencyTable.h/.cpp`): `FPU = 10` cycles, `FPU_ACC = 6`,
`DELTA = 1` per wider SIMD step, `MATH = 17` (`DELTA_MATH = 4`), and DPAS on DG2 = 21 / 22 / 32
cycles for repeat 1 / 2 / 8. Measured on A770 with dependent chains (K = 256..2048, slope of a
least-squares fit, 15 repeats, 2.4 GHz nominal clock; `probes/057-latency-model`):

| instruction class | measured | model | model is low by |
|---|---|---|---|
| `mad` (fma) | 9.17 ns ≈ 22 cycles | 10 cycles | 2.2x |
| `math.exp` | 32.4 ns ≈ 78 cycles | 17 cycles | 4.6x |
| `1/(v+b)` | 21.1 ns ≈ 51 cycles | no EM entry (uses 10) | 5.1x |

ILP sweep at fixed work (2048 mads): 1 / 2 / 4 / 8 / 16 independent chains cost
10.22 / 5.65 / 4.83 / 3.05 / 2.85 ns per mad - the knee is at **8 chains** (16 buys only 7% more).

Practical rules for hand scheduling:

1. Do not size SWSB distances from the model's numbers: a mad needs ~22 cycles on hardware, so
   `@4` (model-style `ceil(10/1)`) leaves the consumer stalling. **Use `{@8}` for ALU chains.**
2. For transcendental chains use distance ~12 (`math.exp` ≈ 32 ns vs a 2.85 ns issue interval).
3. Integer/float division is not in the table at all - budget ~50 cycles per `1/(v+b)` yourself.
4. Going past 8-12 chains costs registers without buying throughput; the ILP plateau is real.
5. The DPAS latency numbers (21/22/32) remain **unverified on hardware**: a single-work-item ESIMD
   chain produced non-monotone device times (3.75-72.4 us across configurations, see
   `probes/057-latency-model/dpas_chain.csv`), so only the XMX *issue rate* (~12.5-13 cycles) is
   measured. Treat dpas latency as "at least 21 cycles, probably more".
