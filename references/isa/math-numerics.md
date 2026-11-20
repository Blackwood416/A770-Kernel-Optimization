# Extended Math (`math.*`) on DG2 — Legality and Measured Cost

**Scope**: Intel Arc A770 (DG2 / XeHPG), driver IGC `32.0.101.9150`. Legality from IGA
`1.1.0-640a706f` (`ModelXeHPG` + GED encoder), cost measured by chaining the instruction in the
runtime axpy binary (`probes/033-math`).

## 1. Which math subfunctions exist `[ISA]`

IGA's XeHPG assembler accepts these subfunction names:

`exp`, `inv`, `sqt`, `log` — with a single source.

Rejected as `invalid subfunction`: `exp2`, `rcp`, `rsqrt`, `log2`, `sqrt`, `sqr`, `rsq`. This is a
naming difference from classic GEN assembly, not a hardware difference — write `math.exp`, not
`math.exp2`, when hand-assembling for DG2 through IGA.

Rejected for a different reason (they need a **second source**, so one-source forms fail inside
GED with `GED_SetMathFC: invalid value` + `GED_SetSrc1DataType: invalid value`): `pow`, `fdiv`.

`tanh` is not encodable at all on XeHPG (`GED_SetMathFC: invalid value` plus a cascade of
`invalid field` for the source operands) — it has to be synthesized from exp and division.

## 2. Operand form `[ISA]`

```asm
math.exp  (8|M0)  r32.0<1>:f  r33.0:f        // ok
math.exp  (16|M0) r32.0<2>:f  r33.0:f        // ok: destination region = exec_size / 8 for :f
math.exp  (8|M0)  r32.0<8>:f  r33.0<8>:f     // invalid destination region
math.exp  (8|M0)  r32.0<1>:f  r33.0<1>:f     // error: the source takes no <W> region here
```

- The destination region width is `exec_size / 8` for `:f` (one GRF per 8 lanes).
- The source accepts a broadcast region (`<0;1,0>`) or no region at all, but not a plain `<1>`
  width in that position.
- All of this is easiest to settle by running `iga64 -a -p XeHPG` on the snippet: the assembler
  prints the caret column and the exact error text.

## 3. Measured cost `[MEASURED]`

Vehicle: a dependent chain `math.exp (8|M0) r32.0<1>:f r32.0:f` injected into the runtime axpy
binary, repeated N times, launched with 4096 hardware threads (n=131072):

| exps per thread | device µs (median) | p10–p90 |
|---|---|---|
| 0 | 6.7 | 6.4–7.0 |
| 512 | 12.3 | 11.8–12.8 |
| 2048 | 32.8 | 32.4–33.2 |

- `[MEASURED]` Slope: **≈12.7 ns per `math.exp` per thread** (0 → 2048), i.e. **≈25 cycles of
  latency** at ~2 GHz for a dependent chain.
- `[MEASURED]` With 8 threads per EU interleaved this is **≈1.7 ns per `math.exp` per EU ≈ 3.3
  cycles of throughput**, so the extended-math pipe retires roughly one transcendental per 3
  cycles per EU.

## 4. What this means for kernels

- `[MEASURED]` A transcendental is **cheap compared with a DPAS**: `math.exp` costs ~3.3 cycles/EU
  of throughput versus ~12.5 cycles/EU for one `dpas.8x8` (see `isa/dpas.md` §2.2). Aggressively
  replacing `exp` with polynomial approximations is therefore usually the wrong trade on DG2 —
  removing the *dependent latency* (≈25 cycles) matters more than removing the op.
- `[HEURISTIC]` For softmax-style kernels the useful levers are therefore: fewer serialized
  transcendentals per lane, more independent lanes in flight, and keeping the reduction off the
  math pipe. Precision-mode tricks (`math.exp` has no `.mme`-style dual-issue on DG2) are not
  available.
- Do not assume `exp2`/`rcp`/`rsqrt` spellings survive a hand-edit round trip; use the names in §1.

## mad vs fma vs dp4a on DG2

| question | answer | evidence |
|---|---|---|
| is there an `fma` instruction? | **no** - `fma` is an invalid mnemonic in IGA; the instruction is `mad` (XeHPG opcode 0x5B) | `ModelXeHPG.hpp:453`, `iga_matrix.csv` |
| what does `fma()` / `mad()` compile to? | `mad` (kernel body: 2 and 2 `mad`, 0 `fma`) | `isa_state.csv` |
| does plain `a*b+c` get contracted? | **depends on the expression**: the register-only `A*B+C` numeric kernel compiles to 2 `mad` (fused), while the `a*x*2 + y[i]` kernel compiles to 0 `mad`, 5 `mul` + 6 `add` (not fused) | `isa_state.csv` |
| is that observable numerically? | yes, with A=1+2^-23, B=1-2^-23, C=-1: fused keeps -2^-46 and two-step rounds to 0; measured y[0]=0.0 means the two agree, i.e. the plain register expression **was fused** | `numeric.csv` |
| cost of the three spellings | 97.2 us, 97.5 us, 97.4 us over 4M elements (<0.3% apart) | `timing.csv` |
| `dp4a` (4x int8 dot) | in the ISA model (0x58, `UD,D <- UD,D`, 32-bit accumulator) but **unreachable from OpenCL C** here: unknown `cl_khr_integer_dot_product`, no `dot(char4/uchar4)` overload, IGA text parser rejects it | `ModelXeHPG.hpp:274`, `isa_state.csv` |

Two practical rules: write `fma()`/`mad()` when you need a guaranteed single rounding - plain `a*b+c`
may or may not be contracted depending on the expression (both outcomes were measured here), so it is
not something to rely on; and when counting multiply-adds in a disassembly, drop the prologue (the
`r127` address arithmetic contains `mad` too, which inflates whole-kernel counts).
Source: `docs/40-mad-fma-dp4a.md`.

## bf16/f16 conversion: one typed `mov`, free per instruction, expensive as a separate pass

IGA on XeHPG has no `cvt`/`f16to32`/`bf16to32` (all `invalid mnemonic`); a conversion is a typed move:
`mov (8|M0) r40.0<1>:bf r38.0<1;1,0>:f` (f32->bf16), `...:hf` (f32->f16), `mov ...:f r41.0<1;1,0>:hf`
(f16->f32) — exactly what IGC emits for `sycl::bfloat16`/`sycl::half`.

Cost, measured as real throughput kernels (read N f32, write N 16-bit, 8 elements per work-item,
device-time median of 15): bf16-mov vs integer-truncate control differ by at most
1.042 us (0.23%) across n = 524288..16777216, sign flipping with n, i.e. the conversion itself is
free and hidden under memory traffic (226-227 GB/s at n=16777216).

Where the money goes is the extra pass, not the instruction: packing A (1024x512) + B (512x1536) costs
22.396 + 29.167 = 51.563 us against the live tuned bf16 GEMM records (96.93-105.21 us, four
pinned runs, `docs/53` §2) i.e. 0.49-0.53x (0.343x if the same bytes move at the best measured
pack rate); the older 101.1 us figure is withdrawn as unreproducible. The tuned
ESIMD GEMM itself has zero input-side conversion (A/B are bf16 in memory) and only
16 epilogue `mov ...:bf` instructions out of 564 static instruction lines
(2.8%), with 5 dpas carrying `:bf` operands.
Rule: keep operands bf16 (or convert inside the load path, which is free), never run a per-call f32->bf16 pass.

## cr0 FP-mode bits: bit 7 is the denorm switch, and denorms are not slow

The OpenCL precision flags collapse into one prologue instruction,
`or (1|M0) cr0.0<1>:ud cr0.0<0;1,0>:ud 0x4C0:uw`; patching only that immediate
(`probes/069-cr0-fp-mode`) gives a clean A/B:

| cr0 mask | denorm input | denorm result | negative denorm |
|---|---|---|---|
| `0x4C0` (6,7,10) / `0x480` (7,10) | `003accec` kept | `001399a4` kept | `801399a4` (sign kept) |
| `0x040` (6) / `0x440` (6,10) | `00000000` | `00000000` | `00000000` (sign lost) |

So bit 7 (`0x80`) is the denorm enable; bits 6 and 10 do not affect denorm or inf in these tests
(do not generalise their meaning without a dedicated probe). Throughput is identical with and without
denorm handling (4096-op dependent loop, 108.1-108.2 us across all masks and inputs), i.e. flushing
denorms is a **numeric** choice, not a performance one. `-cl-denorms-are-zero` writes `0x40` into the AOT
image and the runtime image still shows `0x40` after the driver's recompile from IR, so the flag
survives; rounding is an instruction-level modifier, not part of this cr0 write.
