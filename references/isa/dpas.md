# DPAS / XMX on DG2 (Xe-HPG) -- ISA Facts and 0-Dummy Scheduling

> Scope: Intel Arc A770 (DG2 / Xe-HPG), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`. Verified via IGA legality tests, `ocloc disasm`, and executed ESIMD kernels (`errors=0`).

## 1. Legal Operand Surface on DG2 `[ISA]` `[ARCH]`

| Item | DG2 Rule |
|---|---|
| Execution size | **`SIMD8` only** for both `dpas` and `dpasw` on DG2. `ExecutionSize=16` compiles/assembles but produces wrong results (`78/128` errors `[BUG]`). |
| Systolic depth (`SD`) & repeat count (`RC`) | `SD` fixed at `8`; `RC in {1, 2, 4, 8}`. `16x*` and `32x*` are illegal. |
| `K` per DPAS (`SD=8`) | `fp16`/`bf16`: `OPS_PER_CHAN=2 -> K=16`; `i8`/`u8` (or mixed `i8`+`u4`): `OPS_PER_CHAN=4 -> K=32`; both inputs `<= 4-bit` (`u4`/`s4`/`u2`/`s2`): `OPS_PER_CHAN=8 -> K=64`. |
| Legal precision pairs | `f <- bf, bf`; `f <- hf, hf`; `d/ud <- b/ub/s4/u4/s2/u2` (including mixed low-bit integer pairs). |
| Illegal on DG2 | `tf32` (XeHPC+ only), `fp8` (absent), `dst:hf`/`dst:bf` accumulators (IGA accepts `hf`, but DG2 VISA/codegen rejects `dst:hf`; always accumulate in `f32`), mixed `f16 + u4` (ESIMD public API rejects `APrecision != BPrecision`). |
| Operands, regioning, compaction | No operand regions (`<1;1,0>` rejected); `src1` (B) must be GRF-aligned subreg `0`; `{Compacted}` rejected; predication `(f0.0)` is silently dropped `[BUG]`. |
| Layout | `A` (`src2`) is row-major; `B` (`src1`) is VNNI-packed (`word = (k/2)*N + n` for 16-bit, 4-way K-packed for 8-bit). |
| Low-bit `INT2` note | `INT2` shares `OPS_PER_CHAN=8` (`K=64`) with `INT4`; it does **not** double `INT4` DPAS arithmetic throughput (value is storage/bandwidth reduction only). |

## 2. Measured Throughput and Latency `[MEASURED]`

Measured with straight-line `dpas.8x8` chains at `4,096` HW threads:

- **Per-EU issue ceiling**: **`~82 Gdpas/s`** aggregate (`~6.3 ns` = `~12.5–13 cycles` per `dpas.8x8` per EU), equivalent to `~168 TFLOPS` pure instruction limit (`137.63 TFLOPS` theoretical at `~2.1 GHz`; `100.6 TFLOPS` verified microkernel peak; **`92.5 TFLOPS`** end-to-end `4096^3` bf16 GEMM).
- **Single-chain latency**: `~10.7 ns` (`>= 21 cycles`) at low occupancy.
- **ILP and precision invariance**: `8` independent accumulator chains cost the same per instruction as `1` chain at saturation; `bf16` and `i8` chains run at identical per-instruction latency (`9.2 us` vs `9.2 us` at `512` ops). Throughput scales by filling XVEs and eliminating stalls, not by stacking redundant DPAS chains.

## 3. `dpasw<8,8>`: The Wide Sibling (Production Default) `[ISA]` `[MEASURED]`

`dpasw` (opcode `0x5A`, vs `0x59` for `dpas`) exploits DG2's Fused-EU architecture:

1. **Halves per-thread `A` operand size**: Paired even/odd threads `(2k, 2k+1)` share the fused EU pair and read only half of `A` (`256 B` instead of `512 B`), cutting exact `-32 Send/thread` on `bf16` and `-16 Send/thread` on `int8`.
2. **Source accessibility**: Unreachable from OpenCL C, but **natively reachable in SYCL/ESIMD** via `esimd::xmx::dpasw<8, 8, float>(acc, b, a)` (oneAPI 2026.2+). Pair with bit-swapped SLM `A` cooperative writes (`slm_tid = (tid & ~6) | ((tid & 2) << 1) | ((tid & 4) >> 1)`) so each thread issues a single coalesced `slm_block_load`.
3. **Never revert to `dpas<8,8>` on large GEMM**: Switching back to `dpas` doubles `A` SLM read bandwidth (hitting the `3.3–3.8 TB/s` SLM read wall) and still incurs IGC dummy insertions, regressing `2048x4096x2048` by **`-12.4%` vs `dpasw`** `[MEASURED]`.
4. **Do not mix `dpas` and `dpasw` in the same loop**: IGA forbids them from sharing a systolic macro, fragmenting scheduler slots.

## 4. IGC `dummy_dpasw (null:f)` Injection and 0-Dummy Grouping Theorem `[ISA]` `[MEASURED]`

In multi-accumulator ESIMD GEMM loops, `ocloc disasm` frequently reveals IGC inserting idle 8-cycle **`dpasw.8x8 (8|M0) rX:f null:f rX:bf rX.0:bf`** instructions (`15%–18%` XMX throughput loss `[MEASURED]`).

### Why IGC Inserts `null:f`

- DG2 lacks src2 read suppression (`hasDpasSrc2ReadSupression() == false`), and its `src1` (B) cache holds `512 B` (2 B operands).
- Whenever any `dpasw` `src1` in the kernel was written by a non-`send` instruction (e.g., packing `mov` or register reuse), IGC enables the systolic read-suppression workaround (`mayNeedRSWA = true`).
- Whenever an emitted `dpasw` macro has **`macro.len > 1` AND `first->src1 == second->src1`** (the first two instructions of a macro read the same `B` register), IGC prepends an 8-cycle `dummy_dpasw (null:f)`. Only the first macro of a basic block requires `1` unavoidable entry dummy; all remaining dummies can be eliminated.
- Naive alternating order (`b0, b1, b0, b1...`) still fails because IGC's local scheduler clusters `size == 1` same-`src1` nodes together and splits pairs arriving on different SLM load SBIDs (`a01` vs `a23`).

### 0-Dummy Grouping Theorem (Cuts `dummy_dpasw` from `19/16/11` Down to `1`) `[MEASURED]`

Satisfy all four conditions simultaneously:

1. **Single lead + grouped followers**: Inside each macro, allow at most **one** `size == 1` `src1` node (using distinct `B` registers across macro heads); pack every remaining same-`src1` use into a contiguous **`size >= 3`** block so the heuristic skips it.
2. **Macro-head alternation**: Ensure `first->src1 != second->src1` at the head of every macro.
3. **Tail-SBID anchoring**: Start each macro with the `A` slice from the **last-issued SLM load (`a23` / `a3`)**, so IGC waits once with `sync.allwr` and keeps the entire macro in exact source order.
4. **Respect DG2 macro cap `<= 8`**:
   - **`4xB` (`2x8` macros -- `128x256 cab4` & `SwiGLU`)**:
     - Macro 1 (8 ops): `[b0(a3)](1) -> [b1(a0,a1,a2,a3)](4) -> [b0(a0,a1,a2)](3)`
     - Macro 2 (8 ops): `[b2(a3)](1) -> [b3(a0,a1,a2,a3)](4) -> [b2(a0,a1,a2)](3)`
   - **`3xB` (`3x4` macros -- `Btile192`)**:
     - Macro 1 (4 ops): `[b0(a3)](1) -> [b1(a0,a1,a2)](3)`
     - Macro 2 (4 ops): `[b2(a3)](1) -> [b0(a0,a1,a2)](3)`
     - Macro 3 (4 ops): `[b1(a3)](1) -> [b2(a0,a1,a2)](3)`
   - **Never force a `1x16` macro**: IGC forcibly splits it via `sync.allwr` after the second `src1` switch, spiking `dummy` from `1 -> 17` and regressing `2048x4096x2048` from `385 -> 433 us` `[MEASURED]`.
