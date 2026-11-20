# DPAS / XMX on DG2 (XeHPG) — Instruction-Level Facts

**Scope**: Intel Arc A770 (DG2 / g10 / XeHPG), Windows 11, oneAPI 2026.1, driver IGC
`32.0.101.9150`, offline assembler IGA `1.1.0-640a706f` (bundled in intel-graphics-compiler
`visa/iga`). Everything here is either a **legality/encoding fact** read out of the IGA model or
an **executed measurement** on the A770 through hand-edited GEN asm; the receipt for each row is
in the Arc-ISA-Research repo (paths given inline, repo-relative).

Labels: `[ISA]` = instruction/encoding fact, `[MEASURED]` = executed on the A770, `[SRC]` =
repo source fact, `[HYPOTHESIS]` = not yet verified.

## 1. What `dpas` accepts on DG2

| Item | Rule | Label |
|---|---|---|
| Legal shape | `dpas.8x8` only; `SystolicDepth` fixed at 8. Exec size 16 assembles but produces wrong results | [ISA] |
| bf16 | `dst:f src0:f src1:bf src2:bf` — the form IGC/ESIMD actually emits | [ISA] |
| f16 | `dst:f src0:f src1:hf src2:hf` | [ISA] |
| i8 | `dst:d src0:d src1:b src2:b` | [ISA] |
| Type pairing | accumulator and source precision must be **same class**: `:f` + `:b` is rejected by GED (`GED_SetSrc1Precision: invalid value`), `:si` is not a legal accumulator type (`invalid type`) | [ISA] |
| `tf32` | **illegal on XeHPG** — GED precision enum only has tf32 for XE.HPC/XE2/XE3, and `ModelXeHPG.hpp`'s DPAS type list has no tf32 row | [ISA] |
| Regioning | operands take **no** regions; `src2` with `<1;1,0>` fails `invalid src2` | [ISA] |
| src1 alignment | must be GRF-aligned with subreg 0 (`r1.16` fails `GED_SetSrc1SubRegNum`) | [ISA] |
| Predication | **silently dropped**: `(f0.0) dpas ...` assembles but encodes PredCtrl=Normal, byte-identical to the unpredicated instruction | [ISA][BUG] |
| sub-byte accumulators | `dst:hf` / `dst:bf` assemble but ngen and vISA both restrict XeHP/XeHPG to `dst=f` + `src0=f`; treat as DG2-illegal | [ISA] |

Receipts: `probes/022-dpas/legality.csv` (50 rows, 35 accept / 15 reject), `probes/032-dpas/i8probe/`.

## 2. Throughput and latency (measured)

The measurement vehicle is an axpy kernel disassembled out of the **runtime** driver binary,
with dpas chains injected as straight-line code (`scripts/inject_repeat.py`,
`probes/032-dpas/`). No control flow is introduced, so the kernel always terminates.

### 2.1 Per-instruction cost is fixed

| Configuration | dpas per thread | Device µs (median) |
|---|---|---|
| 1 dependent chain, 0 / 512 / 2048 dpas | – / 512 / 2048 | 4.2 / 9.2 / 26.1 |
| 1 / 2 / 4 / 8 independent chains (4096 threads) | 128 / 256 / 512 / 1024 | 11.9 / 18.7 / 32.4 / 59.7 |
| bf16 chain vs i8 chain (same instruction count) | 512 | 9.2 vs **9.2** (n=4096), 32.3 vs **32.3** (n=131072) |

- `[MEASURED]` Time is linear in the **number of dpas instructions** and nothing else.
- `[MEASURED]` Adding **ILP does not help**: 8 independent accumulator chains cost the same per
  instruction as 1 dependent chain (40.6 → 51.8 ns per dpas per thread; it gets slightly worse,
  not better).
- `[MEASURED]` **Precision class does not matter**: replacing bf16 sources with i8 sources leaves
  the time unchanged.
- `[MEASURED]` A single dependent dpas chain costs ~10.7 ns per instruction when only one thread
  runs per EU (low occupancy): that is the raw **latency** (~21 cycles at ~2 GHz).

### 2.2 The number to quote

`[MEASURED]` Thread scaling at a fixed 512 dpas/thread: Δ(over the same kernel with 0 dpas) =
5.0 / 9.9 / 22.6 / 25.6 µs for 128 / 512 / 2048 / 4096 hardware threads. It saturates from ~2048
threads, giving an aggregate **≈82 Gdpas/s**, i.e. **≈6.3 ns (≈12.5–13 cycles) per `dpas.8x8`
per EU** (512 EUs).

**Converting to TFLOPS.** One `dpas.8x8` with `:bf` sources covers K=16, i.e. 2048 FLOP:
an ESIMD probe that declares an 8x8x16 multiply through `xmx::dpas` compiles to **exactly one**
`dpas.8x8 (8|M0) r20:f r20:f r12:bf r3.0:bf {$2}`, and that encoding is bit-identical to the
injected form outside the register/precision fields. So

| Quantity | Value | Label |
|---|---|---|
| Pure-dpas aggregate limit | **82 Gdpas/s ≈ 168 TFLOPS bf16** | [MEASURED] |
| Best measured bf16 GEMM in this skill | 100.63 TFLOPS | [MEASURED] (kernel-level, memory included) |
| Theoretical peak at ~2.1 GHz | 137.63 TFLOPS | [ARCH] |
| Hardware model | 64 MAC/cycle/EU (8 lanes × 8 systolic) → 16 cycles per 1024-MAC dpas; measured 12.5–13 cycles ≈ 79 MAC/cycle/EU (±20%, clock/median uncertainty) | [HEURISTIC] |

**The useful reading: the best measured bf16 GEMM reaches only ~60% of the XMX instruction limit, so
there is ~1.6× of headroom left for a GEMM that removes memory and wave-quantization stalls.**

### 2.3 What this means for kernels

- `[MEASURED]` You cannot buy XMX throughput with instruction-level parallelism, wider tiles per
  thread, or a cheaper precision class. The per-EU issue rate is the wall.
- The only lever is **occupying more EUs**: raise occupancy / fix wave quantization, do not
  micro-tune the dpas sequence.
- Consequences for tiling: prefer giving each thread *fewer, longer-lived* accumulators so
  occupancy stays high, rather than stacking many independent dpas per thread.

## 3. Related pages

- `swsb.md` — scoreboard/dependency encoding, and why a dependency-free instruction stream can
  fail to complete at all.
- `regions-banks.md` — what operand regions and register banks do and do not cost.

## Legal envelope of dpas on DG2 (assembler-verified, 100 combinations)

| dimension | legal | rejected |
|---|---|---|
| shape dpas.RCxSD | RC, SD in {1,2,4,8} (10 shapes) | 16x*, 32x* -> invalid subfunction |
| precision pair | f+hf, f+bf, d+b, d+ub | f+f -> GED_SetSrc1Precision invalid |
| exec size | any of 1/2/4/8/16/32 | - |
| quarter mask | M0/M8/M16/M24 | - |
| subregister on any operand | accepted (.0-.4) | - |
| regioning on any operand | none | <1>, <2>, <1;1>, <2;1> -> does not support regioning |
| compaction | never | {Compacted} -> GED unable to compact instruction |
| hand-written SWSB | not allowed | {A@1,$2} -> invalid SWSB |
| register alignment / range | not checked by IGA | (so assembler acceptance is not legality) |

Register footprint of one dpas.8x8: dst 8 GRFs + src1 8 + src0 8 + src3 8 = **32 GRFs**, so a 128-GRF
file holds about 4 independent dpas at a time. Combined with the ban on regioning (no register reuse),
that is the structural reason more independent chains stop paying off.

## dpasw: the "wide" sibling (DG2)

`dpasw` is a **separate opcode**, not an operand-swapped `dpas`: with identical operands the only
difference in the 16-byte encoding is the first byte (`0x59` for dpas, `0x5A` for dpasw, IGA
`ModelXeHPG.hpp:302`), and IGA names it "Dot Product Accumulate Systolic **Wide**".

What "wide" changes (all three sources agree):

1. **A operand is half size**: for the same M×K, ESIMD's `xmx::dpasw` requires `AN` elements where
   `dpas` needs `2*AN` (`AFactorForDPASW = 2`, `xmx/dpas.hpp:136-138`).
2. **SIMD8 only**: the ESIMD static assert allows execution size 8 or 16 for dpas but only 8 for
   dpasw (`dpas.hpp:147-148`), and IGC's OpenCL table says the same ("split matrix (dpasw, simd8
   only)").
3. **Cannot share a compaction macro with dpas**: IGA's `RegDeps.cpp:834` states "DPAS and DPASW
   should not be in the same macro", so mixing them splits scheduler slots.

Assembler-level legality (IGA oracle, `probes/060-dpasw`, 45 rows):

| aspect | dpasw result |
|---|---|
| shapes | 1x1, 1x8, 2x2, 2x8, 4x2, 8x1, 8x2, 8x4, 8x8 accepted; 16x8 rejected (`invalid subfunction`) |
| type pairs | `f <- hf`, `f <- bf`, `hf <- hf`, `d <- b/ub/u4/s4/u2/s2` |
| operand alignment | no 8-GRF enforcement (src1 r21/r22/r24 and src2 r31/r34/r36 all assemble) |
| aliasing | dst==src1, dst==src2, src1==src2 all assemble |
| `{Compacted}` | rejected, same as dpas (`GED unable to compact instruction`) |
| SWSB | `{@1,$2}` and `{$2}` accepted; `{A@1,$2}` and `{@1,$2.dst}` rejected (DPAS is a "math-class" instruction: use `{@n,$m}`) |

Runtime status on this box: `xmx::dpasw` compiles and executes (same accumulator result as dpas for
the test input), but single-work-item device times scatter over 11.146-54.271 us across eight
configurations, so **no throughput number is claimed**. To measure it, dump the JIT zebin and confirm
opcode 0x5A, then run an unwrapped chain of >=1024 dpasw at a pinned clock.

## Precision surface: f16 accumulation and 4/2-bit inputs

IGA oracle (`probes/061-dpas-types`, 27 rows): the type table belongs to the instruction *format*, so
`dpas` and `dpasw` and every legal shape share it.

| question | answer |
|---|---|
| f16 accumulation (`hf` dst) | **IGA/GED accepted** (`dpas.8x8 hf <- hf,hf`, `hf <- hf,bf`, `hf <- bf,bf`) **but DG2-illegal in practice**: ngen and vISA restrict XeHP/XeHPG to `dst=f` + `src0=f` (see the sub-byte accumulator row above), so treat hf accumulation as assembler-only |
| hf dst with f32 input | **rejected**: `GED_SetSrc1Precision: invalid value` (`IGALibrary/GED/Encoder.cpp:701`) |
| 4-bit / 2-bit inputs | **accepted** with an integer destination: `u4+s4`, `u4+u2`, `s2+s2`, `u2+s4`, ... |
| low-bit mixed with 8-bit | **accepted**: `u8+s2`, `s2+u8`, `u4+u8`, `u4+s2` |
| low-bit with `f` destination | **rejected**, same `GED_SetSrc1Precision` message; `d` and `ud` both work |
| dpasw / 2x2 shapes | same surface (`hf<-hf,hf`, `d<-u4,s2`, `d<-u8,s8`) |

Consequences for kernel design: an f16 accumulator forces hf/bf inputs (no f32 inputs with an hf
accumulator), and low-bit inputs force an integer accumulator (`d`/`ud`), i.e. a low-bit GEMM must
accumulate in integers and dequantise afterwards. The hf-accumulator half of that is an IGA-level fact only:
DG2 codegen refuses `dst:hf` (C0025), so in OpenCL/SYCL you always end up with an `f` accumulator. Element counts per operand change with the bit
width, so throughput must be derived from packed widths, never from instruction counts.

## GEMM tile and occupancy for dpas on DG2 (P3, `docs/53`)

The bf16 `xmx::dpas` GEMM (`bench/bf16_gemm_esimd.cpp`, 8x8x16, f32 accumulator) was swept across every
legal resident tile; the constraints are structural, not tunable:

- one ESIMD work-group is one DSS's 128 threads, so `WG = (BM/8) * (BN/TN)` must be 128 (I-003: `WG >= 256`
  is never dispatched and hangs);
- SLM staging needs `2 * (BM*BK + BK*BN) * 2 B <= 65,536`, which caps `BK` once the tile grows;
- the per-thread staging split must divide exactly: `BM*BK/16` and `BN*BK/16` must both be multiples of 128
  (BK=16 forces BM and BN to multiples of 128, BK=32 relaxes it to multiples of 64).

Best point: **BM=256 BN=192 BK=32 TN=48** (SLM 57,344 B, 32 work-groups, one per DSS) = 96.9-105.2 us over four
pinned records, versus 118.4 us for the old 128x128/BK32/TN16 tile, and oneDNN `jit:gemm:any` at 71.4-72.5 us.

What the counters say (`docs/53` §6b-6d): 786,432 dpas at 8.1 Gdpas/s is only **9.9 %** of the DG2 issue ceiling
(~82 Gdpas/s from `docs/07`), XVE stalled 93 %, L3-bound 0.8 %, so the kernel waits on load/barrier latency rather
than on XMX issue. The gap to oneDNN decomposes exactly into "our kernel executes 2.54x more instructions, each
2.16x cheaper" (`docs/53` §6c). Two work-groups per DSS (WG=64) was measured and is slower (108-116 us), and a
third SLM pipeline stage does not fit at 256x192 (would need ~86 KB). So further gains need a kernel change
(two M-chains per thread, separate A/B pipelines), not another `-D` value.

## GEMM v2: two M-chains per thread, and why it is only worth 2.5 % (P4, `docs/56`)

The first structural change from that list is now measured (`bench/bf16_gemm_esimd_v2.cpp`, `probes/081-gemm-v2`).
`TM=16` gives every thread **two** 8x8 DPAS tiles in M, so one B operand load feeds two `dpas`; the per-thread
operand read per K-tile falls from `(TM+TN)*BK*2 = 3584 B` to `2560 B` (-28.6 %) at the same WG=128, SLM 57,344 B
and 32 work-groups. Adopted configuration: **BM=256 BN=192 BK=32 TN=24 TM=16**, plus issuing the next SLM stage
store before the dpas loop.

- Paired A/B in one session, 5 alternating rounds: **5/5 rounds faster**, mean -2.44 us (ratio 0.9752).
- Pinned records: **95.11-96.00 us** vs oneDNN 72.22-72.81 us (1.31-1.33x behind, still ~0.76 of the library).
- The gain is *below* the +-8 % spread of the v1 record set (96.9-105.2 us), so adopt on the paired sign test and
  the instruction count, never on one median.

What did **not** pay, both measured: 24 work-groups (two variants, 101.7-104.2 us) and the early store on its own
(~0.5 %). And the counter check that explains the small win: v2 executes only 4.7 % fewer instructions
(18.56e9 -> 17.69e9) and runs 1.4 % faster, with XVE stalled 92.3/92.5 % - so this kernel is neither LSC-message
bound nor instruction bound. Feeding XMX is a dependency chain (global load -> SLM store -> barrier -> operand
read), and breaking it needs more pipeline stages than 64 KB of SLM allows.

Roofline caution: use the **XMX bf16** throughput (`docs/07`: ~82 Gdpas/s) as the denominator, never the A770
39.32 TFLOPS figure - that one is the FP32 vector peak and inflates this shape's floor from 9.6 us to 41 us.
At 95.11 us the kernel is at 10.1 % of the measured dpas ceiling, at 72.22 us oneDNN is at 13.3 %.
