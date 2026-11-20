# ISA → decision: what the measured ISA facts tell you to do

Each line is a decision rule backed by a measurement in this skill repo. Numbers are the committed ones; the
linked page carries the CSV.

## Compute

| ISA fact | Decision |
|---|---|
| `dpas` is encodable for bf16/f16/f32-acc and 4/2-bit ints, and legal only as a ternary-regionll format (`docs/34`, `docs/35`) | Use `dpas` for any matmul-shaped work; accumulate in f32. Low-bit (u4/u2) inputs must accumulate in integers and dequantise afterwards. |
| `dpasw` exists only as an encoding (no source entry, `docs/46`) | Only reach for it in hand-written asm; there is no SYCL/ESIMD path. |
| DPAS latency hints are 21/22/32 cycles for repeat 1/2/8 (`docs/31`, vISA LatencyTable) | Size the accumulator set so the dependency distance is at least that; with 8 accumulator tiles the scheduler can hide one DPAS behind another. |
| `mad` is not a fused op: contraction only happens when the compiler contracts (`docs/40`) | For bit-exact results use explicit `fma`; for speed let the compiler contract, but record `{Compacted}`/contraction state when comparing outputs. |
| A dependent FP chain costs the full latency; independent chains overlap (`docs/31` 2.85 ns issue interval under real kernels) | Keep ≥4 independent accumulator chains per thread. |

## Memory / messages

| ISA fact | Decision |
|---|---|
| `load.ugm` is ~0.91 ns/B; `send.dc0` constant fetches cost 197-227 ns per message (`docs/29`) | Never put per-work-item data in `__constant`; pass kernel args (≤16 B ride free in the payload) and use `load.ugm` for bulk. |
| LSC message width is chosen by the compiler from the access pattern; 32 B/128 B per message (`docs/29`) | Vectorise to at least 128 bits per work-item so each message carries a full cache line. |
| Denormal handling is a cr0 bit with **no throughput cost** (`docs/43`) | Choose denorm semantics by numeric requirement only; never as a speed knob. |
| bf16/f16 conversion is one typed `mov` and is free inside a bandwidth-bound kernel, but a separate f32→bf16 pass costs 0.34-0.51x of a bf16 GEMM (`docs/44`) | Keep operands bf16 in memory (or convert in the load path); never run a per-call pack pass. |
| No prefetch instruction is encodable and the source path emits none (`docs/47`) | Get locality from blocking/sequential access and reuse; do not design around prefetch. |

## Scheduling / occupancy

| ISA fact | Decision |
|---|---|
| SWSB tokens are the scoreboard: 16 architectural token IDs, three instruction classes accept different spellings (`docs/26`) | In asm, annotate sends with `{$n}`/`{A@n}` per class; in source, trust IGC — 16 independent loads become 8 load messages carrying 8 tokens (`docs/48`). |
| Work-group dispatch is not linear (`docs/48`: rho −0.009…0.062 / 0.161–0.170 / 0.261–0.298 at 16/64/256 WGs) | Never assume wg i runs before wg i+1; make work-groups independent, and do not rely on launch order for correctness. |
| Thread-exit/EOT floor: ~0.47 ns per work-group, intercept ~1.5 us; a work-group of 32 threads already saturates the exit path (`docs/36`) | Keep work-groups big enough (≥64 threads) and the grid small enough that the 1.5 us intercept is amortised; small grids pay a fixed cost. |
| Occupancy curve saturates around 2048 threads/subslice; 256-GRF is unreachable (`docs/27`, `docs/22`) | Budget 128 GRF/thread and 2048 threads/subslice; going wider only adds pressure. |
| `{Compacted}` halves the encoding to 8 B but only for forms without immediates/labels/math/typed moves (`docs/48`) | Estimate code size as a mix; do not assume full compaction, and do not try to force it. |

## Which programming model

| Need | Reach for | Why |
|---|---|---|
| dpas / XMX, block loads, manual register control | ESIMD | `dpas`, `lsc` loads and typed moves all come out of the compiler (`docs/46`). |
| plain elementwise / reduction with good vectorisation | SYCL or OpenCL C | IGC already emits `load.ugm`, `mov ...:bf`, `r[a0.0]` and `call` from plain source (`docs/46`). |
| dpasw, SWSB tokens, exotic operand forms | hand-written asm on a runtime dump | No source entry; must patch the JIT image and confirm no `recompiled from IR` (`docs/46`, `docs/42`). |
| anything needing prefetch, 256-GRF or bindless | stop and check `references/workflow/instruments.md` first | Those three are unreachable on this stack. |

## Verification rules

- Compare numbers only against a record produced with the same protocol; `bench/README.md` holds the current baselines.
- If a kernel loses, attribute it to instructions, messages, sync or occupancy before changing anything.
- Before trusting an AOT disassembly for instruction counts, diff it against the JIT image (`docs/45`).
