# Instruction Encoding (128-bit word) on DG2

**Scope**: A770 (DG2), IGA `1.1.0-640a706f`. Derived purely by differential assembly: assemble a
base `mad`, change one syntactic element at a time, diff the 16-byte encodings
(`probes/045-encoding/`), then verified by patching raw bytes in a driver kernel and running it.
Write-up: `docs/19-instruction-encoding.md`.

## Field map `[MEASURED]`

| Field | Bits | Notes |
|---|---|---|
| **Exec size** | `[18:16]` | **log2 of the width**: 0→1, 1→2, 2→4, 3→8, 4→16, 5→32 |
| **Quarter control** | `[21:20]` | 0=M0, 1=M8, 2=M16, 3=M24 |
| Destination register number | `[63:56]` | r10→r11 flips bit 56; r10→r40 (+0x30) flips bits 57 and 61 |
| Destination subregister | `[55:54]` | `.0`→`.4` flips bit 55 |
| Destination region width | `[49:48]` | `<1>`→`<2>` flips bit 48 |
| src0 subregister | `[71:70]` | `.0`→`.4` flips bit 71 |
| src0 horizontal stride | `[65:64]` | `<1;0>`→`<1;1>` flips bit 64 |
| src0 negate | `[45]` | `-r20` flips bit 45 |
| src0 absolute | `[44]` | `|r20|` flips bit 44 |
| src1 subregister | `[103:102]` | `.0`→`.4` flips bit 103 |
| src1 negate | `[87]` | `-r30` flips bit 87 |
| src2 subregister | `[119:118]` | `.0`→`.4` flips bit 119 |
| src2 negate | `[85]` | `-r40` flips bit 85 |
| Saturate | `[34]` | written as a `(sat)` **prefix on the destination**, not a `.sat` suffix |

## Compaction changes the instruction length, not just a flag `[MEASURED]`

`{Compacted}` turns the instruction into **8 bytes** instead of 16, and the compressed format does not
put the fields at the same offsets (the same `mad` no longer has its exec size at `[18:16]`). Any
bit-level patch must therefore stay within one format, or decompact the kernel first.

## Bit-level verification `[MEASURED]`

Rebuilt the driver's axpy kernel without compaction, located the first `mad`, flipped only `[18:16]`
(16→8, i.e. `0b100`→`0b011`), and ran it:

| version | device | errors |
|---|---|---|
| uncompacted original | 4.1 us | 0 / 4096 |
| first `mad` exec size 16→8 | 4.2 us | **1024 / 4096** |

Exactly 1/4 of the elements are wrong, which is the predicted consequence: each 32-lane work-group is
covered by two `(16|M0)`/`(16|M16)` mads, so halving the first one leaves 8 of every 32 lanes
uncomputed. Disassembly confirms `mad (8|M0)` while the second mad stays `(16|M16)`.

## Two traps when deriving or using this map

1. **Bit diffs are XOR, not OR.** New value = base value XOR the flipped-bit set. Reading the diff as
   an OR mask produced `0b111` for exec size 8, which IGA rejected with
   `GED reports invalid value for field ExecSize` — the disassembler is what caught the error.
2. **Always re-disassemble after patching bytes.** A wrong field silently changes results otherwise
   (or, as here, gets flagged immediately by the decoder).

## Syntax rejections worth knowing

| Attempt | Message |
|---|---|
| `mad.sat` | saturation flag prefixes the destination operand: `(sat)r10.0` |
| ternary src0 with `<2;0>` | `GED_SetSrc0VertStride: invalid value` (vstride only 0/1/4/8 for ternary src0) |
| `{NoMask}` as an instruction option | `NoMask goes precedes predication as (W)`: write `(W) op` |
| hand-written `{A@1,$2}` | `invalid SWSB`: tokens must reference a real producer |

## Divergence depth and calls: flattened, so neither stack is reachable from source

Ladder of 1/2/4/8/16/32/64/128/256/512 levels; every rung is compiled, run and checked, and the top two fail to compile
(`probes/065-divergence-stack/divergence.csv`). The if/else ladder is genuinely nested (each level's
then-branch contains the next, every level has a distinct predicate) and the call ladder is a chain of
`__attribute__((noinline))` functions.

| observation | value |
|---|---|
| first failing rung | **256**, and the failure is a *compiler* limit: `bracket nesting level exceeded maximum of 256` |
| rungs that compile, run and are numerically correct | 1..128 |
| `join` instructions in the ISA, at every depth that compiles (1..128) | **1** |
| `call` instructions in the ISA, any depth | **0** |
| device time, if/else form | 4.2-5.9 us |
| device time, call form (all inlined) | 2.9-3.0 us |

So the source-level nesting depth does not become ISA join depth (IGC reduces the whole nest to one
decision) and `noinline` does not survive (the chain is inlined). Neither the divergence stack nor the
call stack can be exercised from OpenCL source; doing so requires injecting `join`/`call` into a
runtime-dumped kernel. IGA rules, for when that is done: `join` takes an immediate label defined in the
same kernel (`src0 must be an immediate label` otherwise); `call` takes an immediate target inside the
kernel, a register target, and SIMD8/16/32; `ret` is legal. Source: `docs/39-divergence-stack.md`.

## Dual issue: not measurable with an injected straight-line block (C4)

The injected-block vehicle is fetch-bound, so it cannot answer the dual-issue question
(`probes/063-dual-issue`, blocks of 64 and 1024 instructions = 1/16 KiB, 15 repeats, median):

| check | measurement |
|---|---|
| `nop` block (occupies no execution pipe) | 11.46 ns per instruction (~27 cycles) |
| `mul` (FPU) / `add` (ALU) blocks | 12.60 / 15.42 ns per instruction |
| dependent chain vs independent chain at low occupancy (512 work-items) | both 23.83 ns per instruction |

`nop` costing the same as real work, and a dependent chain not being slower than an independent one,
both say the bottleneck is instruction fetch/dispatch of the 16-byte straight-line block, not issue
rate. The mixed/single ratio (0.903) is therefore uninterpretable either way. Do not schedule on
an assumption of two instructions per slot; the loop-resident vehicle was tried in E38 and failed
its controls (`docs/41`), so EU issue hardware counters are the remaining instrument. Source: `docs/37-dual-issue.md`.

## E38: a loop-resident vehicle still cannot price issue vs latency

The straight-line vehicle (`probes/063`) was fetch-bound; a loop-resident one is not enough either.
`probes/067-loop-vehicle` puts a K-instruction block inside a compiled loop (`goto.b` + `join`) and
takes the slope over trip 64..512, with an **empty-body variant (k0)** as
the loop-overhead baseline so block cost is `(slope - k0) / K`:

| check (block cost, baseline subtracted) | low occupancy (512) | full occupancy (262144) |
|---|---|---|
| dependent chain / independent block | 1.00 | 0.98 |
| `nop` block / `mul` block | 1.00 | 1.11 |
| ALU `add` block / FPU `mul` block | — | 1.99 |
| mixed (16 mul + 16 add) / mul block | — | 1.15 |

The pre-registered positive controls fail (a dependent chain is not slower than an independent one;
`nop` is not cheaper than `mul`), so this vehicle cannot price issue rate or dependency latency. At full
occupancy it is type-sensitive (ALU adds cost ~2.0x FPU muls) and the mixed block costs only
+15% over the pure-mul block — that is C4-relevant but **not attributable** to dual issue
while the controls fail. Do not schedule on either reading. Likewise: a hand-written conditional branch
before the EOT gateway hangs the GPU; loops must use `goto.b` plus a matching `join`
(`docs/41-loop-vehicle.md`).

## Toolchain 2026.2 pilot: AOT still carries IR, the driver decides the ISA

Tested side-by-side (`probes/068-toolchain-2026-2`), same kernel, same driver
`32.0.101.9150`:

| check | ocloc 2026.1 (32.0.101.8857 (26.24)) | ocloc 2026.2 (32.0.101.9028 (26.35)) |
|---|---|---|
| AOT zebin has `.spv` | True | True |
| driver logs `recompiled from IR` | True | True |
| runtime dump sha256 | 2ca730d02c34418c | 2ca730d02c34418c |

The two AOT binaries differ, but the driver-dumped runtime binary is byte-identical: **the executed ISA
is set by the graphics driver, not by ocloc**. The asm-patch pipeline still works under 2026.2 (disasm →
`illegal`→`nop` → inject → `ocloc asm` → run, no recompile warning), so upgrading does not break the
vehicles — it just changes nothing about what runs. SYCLBIN keeps an ELF image *and* a SPIR-V module in
the same bundle (1 ELF + 1 SPIR-V in the AOT executable), i.e. JIT fallback is by
design; it does not remove the recompile path. `-ftime-trace` works (host + device JSON). ESIMD
`prefetch` exists in the headers but the new SPIR-V prefetch extension was not exercised in this pilot.

## AOT vs driver JIT on real operators: identical for the dpas GEMM, 10-29% shorter for GEMV/RMSNorm

The AOT image you can disassemble is not necessarily what runs: SYCL AOT images still carry `.spv`, and the
driver recompiles it with its own IGC. Driving that path offline is possible without a debugger — take the
`.spv` out of the image and feed it to `clCreateProgramWithIL`, then dump `CL_PROGRAM_BINARIES`
(`genrun --il <spv> --dumpbin`), which yields the driver-JIT zebin plus its `.misc.buildOptions`.

Measured on three real operators (`probes/071-aot-vs-jit`, AOT = ocloc 9028, JIT = driver 9150):

| operator | AOT instrs | JIT instrs | delta | seq / bag similarity | only AOT / only JIT | sends | dpas | width AOT / JIT | bytes AOT -> JIT |
|---|---|---|---|---|---|---|---|---|---|
| gemv | 184 | 164 | -10.9% | 0.379 / 0.370 | 118 / 98 | 4 / 4 | 0 / 0 | 8 / 8 | 17408 → 10504 |
| rmsnorm | 429 | 306 | -28.7% | 0.163 / 0.142 | 374 / 251 | 50 / 50 | 0 / 0 | 16 / 16 | 34816 → 25040 |
| bf16_gemm | 543 | 543 | +0.0% | 1.000 / 1.000 | 0 / 0 | 42 / 42 | 5 / 5 | 1 / 1 | 34816 → 23472 |

So the bf16 dpas GEMM is instruction-for-instruction identical (inspecting the AOT ISA is trustworthy there),
while GEMV and RMSNorm come out 10.9% / 28.7% shorter under the driver's JIT — but their send-message counts and
dominant SIMD widths are unchanged, i.e. the difference sits in ordinary instruction selection/scheduling, not in
message width or occupancy shape. Practical rule: trust AOT dumps for message/occupancy analysis, verify per kernel
before trusting them for instruction counts, and never assume one ocloc version's ISA is what executes.

## {Compacted}, work-group dispatch order, and the source-side in-flight window

**{Compacted} is an 8-byte encoding, not a flag.** IGA accepts the token only on forms whose operands fit 8 bytes:
three-register ALU ops (`add`, `mul`) encode in 8 B, while `add` with a 32-bit immediate, `mov` with a 64-bit
immediate, `math.*`, `goto`/`join` (label operands) and typed `mov ...:hf` are all rejected with
`GED unable to compact instruction`. You can neither force nor forbid compaction from source.

**Work-group dispatch is not a linear walk.** Giving each work-group an atomic ticket (stored back through `y`,
read with `genrun --dumpy`) and correlating ticket with `get_group_id` gives Spearman rho -0.009…0.062 at 16 WGs,
0.161–0.170 at 64 and 0.261–0.298 at 256, with only 80%/68%/57% monotonic adjacent pairs. Dispatch is spread across
subslices; the sequence is stable across repeated launches, so it is structure, not noise. Do not assume wg i starts
before wg i+1.

**Source-side load window ~8.** A kernel with 16 independent `#pragma unroll`ed global loads compiles to 8 `send.ugm`
load messages (d32x4) carrying SBS tokens `$3..$10` (8 distinct tokens) — that is the load in-flight window. The whole
kernel has 14 send-class instructions and 13 distinct tokens, but 3 are `send.dc0` argument reads, 2 are stores and 1 is
`gtwy {EOT}`; that total is scoreboard occupancy, not the load window. Timing cannot resolve the window
(bandwidth/launch dominated), so latency questions still need EU counters.
