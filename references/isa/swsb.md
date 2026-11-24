# SWSB (Scoreboard), Barriers, Fences, and GRF Banks on DG2

> Scope: Intel Arc A770 (DG2 / Xe-HPG), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`.

## 1. SWSB Hardware Model and Budget `[ISA]` `[MEASURED]`

DG2 uses the `ThreeDistPipe` software scoreboard encoded in byte 1 of every instruction:

- **16 architectural SBID tokens (`$0..$15`)** per HW thread, shared across all out-of-order `send.ugm`, `send.slm`, `send.gtwy`, and `dpas`/`dpasw` instructions.
- **In-order register distance `0..7`** (`{@1}..{@7}` or per-pipe `A`/`F`/`I`/`L`).
- **Class-matching combined encodings**: normal ALU uses `{@n,$m.dst}`, math/DPAS uses `{@n,$m}`, send uses `{A@n,$m}`.
- **Silent IGA traps `[BUG]`**: `{$n.inc}` assembles silently to `0x00` (no-op) and `{$16}` folds silently to `0x50` (invalid token). Stripping dependencies hangs kernels (`CL_OUT_OF_RESOURCES`, no TDR).
- **Token starvation cost**: Collapsing distinct producer tokens onto one ID regresses even simple memory kernels by **`+33%`** vs distinct-token baseline `[MEASURED]`.

## 2. 16-SBID Saturation, Short UGM Waits, and SLM Address Chains `[MEASURED]`

In deep ESIMD GEMM loops (`cab4` / `GRF=256`), the 16-token pool is a primary performance wall:

1. **`Short UGM waits (<120 ins)` correlate strictly monotonically with runtime**:
   - When simultaneous in-flight `send.ugm + send.slm` exceeds `~14`, IGC's token allocator steals the oldest in-flight UGM SBID, forcing premature `sync.nop` / `sync.allwr` stalls before DRAM loads return.
   - Measured across `4096^3` `bf16` GEMM schedules: **`1 -> 5 -> 7 -> 9 -> 11` Short UGM waits (<120 ins) correspond directly to `1486 -> 1552 -> 1600 -> 1703 us`** (`1` wait is `14.6%` faster vs `11` waits `[MEASURED]`).
   - **2-iteration wrap-around audit**: Because UGM loads issued near the end of an unrolled loop iteration are consumed at the top of the next iteration, statically scan **two wrapped iterations across the backward `goto`/`jmpi`** and target **`Short UGM waits <= 1`** and **`sync.allrd == 0`**.
2. **Sparsely interleave UGM loads**: Merge `B` loads (`128 B` `d64x16t` halves SBIDs), coalesce SLM writes (`4x64 B`), and avoid bursting global loads back-to-back. Never pre-issue next-step `slm_block_load` into a tail already crowded with global UGM loads (`Btile192` regressed to `337 us` from 16-SBID saturation `[MEASURED]`).
3. **SLM address GRF reuse chains (`SLM addr chains`)**: Deep loops with ~60 SLM offsets exhaust free address GRFs, causing IGC to reuse one address register across consecutive `load.slm`/`store.slm` ops (`{$X.src}` + `{A@1}`). Interleaving `load_slm_ops` with `store_buf` cuts `SLM ld addr chains` from **`45/48` to `8/48`** (`2048x4096x2048` `382 -> 377.4 us` `[MEASURED]`).
4. **`-Xfinalizer` SWSB leverage**: Passing `-Xfinalizer -SBIDBundle -Xfinalizer -SWSBDepReduction -Xfinalizer -nopresched -Xfinalizer -twoSrcBundleBCR` to ESIMD VC lets non-overlapping `dpasw` macros share SBIDs and prunes redundant waits (`sync.allrd 5 -> 0`, `Short UGM waits 5 -> 1`, `sync.nop 34 -> 23` `[MEASURED]`).

## 3. Calibrated Hardware Latencies vs Compiler Model `[MEASURED]`

vISA's built-in latency model underestimates actual DG2 dependent-chain hardware latency by `2x–5x`:

| Instruction Class | Measured Hardware Latency | Compiler Model | Recommended SWSB / ILP Guidance |
|---|---:|---:|---|
| `mad` (FMA) | `~9.2 ns` (`~22 cycles`) | `10 cycles` | Knee at **`8` independent chains** (`16` adds only `7%`) |
| `math.exp` | `~32.4 ns` (`~78 cycles`) | `17 cycles` | Hide with `~12` independent lanes; throughput is cheap (`~3.3 cycles/EU`) |
| `1 / (v + b)` | `~21.1 ns` (`~51 cycles`) | none (`10`) | Budget `~50 cycles` |
| `dpas.8x8` | `>= 21 cycles` (`~12.5–13 cyc/EU` issue) | `21/22/32` | Fill XVEs; per-EU issue rate is the ceiling |

## 4. Barriers, Split-Barriers, Fences, and EOT `[ISA]` `[MEASURED]`

DG2 has no single barrier opcode -- a work-group barrier compiles to `fence.<slm|ugm>` + gateway payload `mov`s + `send.gtwy (signal barrier)` + `sync.bar`:

| Synchronization Primitive | Measured Cost (4,096 threads) | Emitted ISA / Meaning |
|---|---:|---|
| `group_barrier(local)` | **`~84 ns`** | `fence.slm` + `signal barrier` + `sync.bar` |
| `group_barrier(device)` | **`~109 ns`** | `fence.ugm` + `signal barrier` + `sync.bar` (`+25 ns` device-scope premium) |
| Single `fence.slm` / `fence.ugm` | **`~19 ns`** | Memory ordering only; does **not** synchronize threads |
| `barrier(0)` | Execution-only | `signal barrier` + `sync.bar` with **no memory fence** |
| `atomic_fence(acq_rel, device)` | `~98 ns` | Costlier than a full local barrier; avoid inside inner loops |
| `send.gtwy {EOT}` (thread exit) | `~0.47 ns/WG` + `~1.54 us` floor | Launch/retire scales with WG count; **`local >= 32`** avoids `20x` tiny-WG penalty |

**Split-barrier decoupling (ESIMD best practice)**: Pair `esimd::fence<local, none, group>()` + `exp_esimd::split_barrier<signal>()` + `split_barrier<wait>()` with `cab4` 4-buffer circular SLM so uninterrupted `dpasw` compute runs cleanly separated from barrier rendezvous. Never hand-inject raw gateway barriers in asm (`8/8` deadlock `[BUG]`).

## 5. GRF Register Banks and Operand Regions `[ISA]` `[MEASURED]`

- **Zero measured GRF bank-conflict cost on DG2**: Compute-bound 3-source `mad` loops (`K = 128..2048`) placing all three operands in one bank (`0,0,0`) vs three banks (`0,1,2`) measure byte-identical times (`4.4 / 9.2 / 28.2 us`), matching IGC's deliberate exclusion of DG2 from default BCR (`hasBankCollision()` excludes `Xe_DG2`).
- **Rule**: Do **not** manually scramble registers or sacrifice live ranges to avoid GRF bank conflicts (note: `-twoSrcBundleBCR` is kept solely as part of the empirically verified `-Xfinalizer` 4-flag VC bundle). SLM banking (`64 B`-multiple strides) is a completely separate structure -- see [hardware.md](../hardware/hardware.md#4-slm-banking-alignment-and-bandwidth-wall-measured).
- **Region legality (asm reference)**: 3-source (`mad`) uses `<V;H>` / `<H>` with `V in {0,1,4,8}` and `H in {0,1,2,4}`; 2-source uses `<V;W,H>`; `H=8` and `V=2` are unencodable on DG2.
