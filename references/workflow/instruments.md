# Instruments: what can and cannot be measured on this box

A770 / DG2, Windows, driver 32.0.101.9150, oneAPI 2026.1 + 2026.2 side by side. This page is the "can I measure X?"
index; each item names the working instrument and the fallback if it does not exist.

## Works

| What you want | Instrument | Where it is demonstrated |
|---|---|---|
| ISA that actually runs | runtime dump (`genrun --dumpbin` after loading an AOT image) then `ocloc disasm`; the driver recompiles from the embedded `.spv`, so this is the executed ISA | `docs/19`, `docs/42`, E39 |
| AOT ISA (what ocloc produced) | carve the ELF out of the SYCLBIN/exe (`probes/071-aot-vs-jit/extract.py`), `ocloc disasm` | `docs/45` |
| Driver JIT ISA (IGC inside the driver) | take `.spv` out of the image, `genrun --il <spv> --dumpbin out.bin` (`clCreateProgramWithIL` + `CL_PROGRAM_BINARIES`) | `docs/45` |
| Instruction encodings / legal operand forms | IGA text assembler round-trip (`build/iga/IGAExe/iga64.exe -a -p XeHPG`, then `-d`) | `docs/34`, `docs/40`, `docs/48` |
| Kernel device time, medians | `genrun --repeat/--warmup` (CL profiling queue) or SYCL event profiling in `bench/` | `docs/27`, `docs/36` |
| Bytes/instruction counts of a compiled kernel | parse the `.asm` from `ocloc disasm -dump`; count the emitted lines, never the source pattern | `docs/37`, `docs/44` |
| Work-group dispatch order | atomic ticket per WG written into `y`, read back with `genrun --dumpy` (single launch: `--warmup 0 --repeat 1`) | `docs/48` §2 |
| In-flight window of loads/sends | SBSB token count on the emitted `send`/`load` lines | `docs/48` §3 |
| Numeric behaviour of a flag bit | patch the prologue immediate on the runtime dump (`or cr0.0, ..., 0xNNN`) and read back `y` bits | `docs/43` |
| GPU hardware counters (instructions, occupancy, XVE stalls) | VTune 2026.6 `gpu-hotspots` with `characterization-mode=instruction-count` or `overview`; launch the workload directly (no `cmd /c` wrapper) | `docs/50`, `probes/076-eu-metrics` |
| Per-class GT-PIN dynamic instruction counts (CLI) | `vtune -report hotspots -r <rdir> -group-by computing-task -format csv` on a `characterization-mode=instruction-count` result (6 exhaustive classes summing to `GPU Instructions Executed` `[TOOLCHAIN]`) | `docs/86` §1.1, `probes/110-gemm-swizzle-counters/work/gtpin_ic.csv` |
| Short-kernel (~100-200 us) OA counters without cross-launch dilution | `characterization-mode=overview -knob gpu-sampling-interval=0.05` (50 us) instead of the 1 ms default `[MEASURED]` | `docs/86` §1.2, `probes/110-gemm-swizzle-counters/work/vtune_overview.csv` |
| Multi-lambda / prepack `computing-task` parsing and OA calibration gating | Match timed lambda by `Instance Count` closest to `samples * batch` and normalize per-row; apply duty-cycle calibration only when `< 200` us (`>= 335` us already captures `~97%`) `[TOOLCHAIN]` | `docs/92`, `docs/94`, `probes/116-sdpa-counters/`, `probes/118-operator-anatomy/` |

## Does not work (do not retry without a new instrument)

| What you want | Why it fails here | Required instrument |
|---|---|---|
| Per-instruction latency / dual issue | both injection vehicles failed their controls (fetch floor; nop == mul) | EU hardware issue/active counters, or an SWSB-aware simulator |
| Cost of `r[a0.0]` vs direct addressing | same vehicle; only the encoding difference is provable | EU counters |
| join/call stack depth cost | depth is encodable but not priceable | EU counters |
| Explicit prefetch | IGA has no prefetch mnemonic; CL `__builtin_prefetch` fails to compile; ESIMD `prefetch` emits 0 instructions | a newer IGA with prefetch, or IGC pass logs + counters |
| 256-GRF mode | driver never compiles with it (I-002) | a driver/IGC build that exposes it |
| Bindless (ExBSO) from OpenCL | no bindless heap in the OpenCL path | Level Zero |
| L1/L2 associativity | no cache-configuration counters | hardware counters; keep the `[HEURISTIC] ~192 KB/subslice` figure |

## Rules of thumb

- Anything about **messages and occupancy** can be read from AOT dumps; anything about **instruction counts/scheduling**
  must be checked against the JIT dump first (`docs/45`: the bf16 dpas GEMM is identical, GEMV/RMSNorm differ by 10-29%).
- Patching is only allowed on runtime dumps: replace `illegal` with `nop`, delete stale `.bin` files first, and confirm no
  `recompiled from IR` warning after reassembly.
- Timings are medians of at least 15 device-time samples; quote the CSV, not the console.

## VTune GPU hardware counters (A770)

VTune 2026.6 `gpu-hotspots` does collect real GPU hardware counters on this A770 (driver 32.0.101.9150) — this is the
EU/XVE-level instrument that `docs/33` recorded as missing. Launch the workload **directly** (`-- app.exe`); wrapping it
in `cmd /c "... && app.exe"` makes the GPU plugin silently disable itself (`plugin [gpu] was disabled because it does not
support the given collection type`) and the result contains no GPU data at all. `vtune` returns a non-zero exit code even
on success, so judge from the parsed report, never from the exit status.

| Preset | Gives | Caveat |
|---|---|---|
| `characterization-mode=instruction-count` | `GPU Instructions Executed` per kernel | counts **32-lane execution groups**, not lanes: two kernels differing by 64 source ops differ by 128 per group (= the static delta), while per-lane would be 64 x threads and per-SIMD16 32 x threads; the absolute value is static - 52 per group, so only deltas or same-kernel versions are comparable (`docs/50` §3) |
| `characterization-mode=overview` | GPU Time, XVE Array Stalled/Idle %, Occupancy (aggregate + per kernel), SIMD Width, SIMD Utilization %, GPU L3 Bandwidth Bound % | Occupancy is sampled and can spike (31-33 % typical for the ESIMD GEMM, one collection reported 71 %); quote medians, never a single run |
| `characterization-mode=lsc-slm` | LSC/SLM event set | needs `-allow-multiple-runs`; its counters do **not** reach `-report summary -format csv`, they live in the result's `sqlite-db/dicer.db` (`dd_gpu_exec_*`, `gpu_gtpin_data_agg_*`) |

Measured example (three runs, `docs/50`): ESIMD bf16 GEMM occupancy 31.4/32.2/33.1 %, XVE stalled 69.1-70.1 %,
L3 bandwidth bound 0.7 % of peak, SIMD width 16, SIMD utilisation 100 % — the kernel is stalled/occupancy-bound, not
DRAM-bound. The analytic "9 % occupancy" in `docs/22` (C0048) is a static GRF/SLM upper bound, a different metric; do
not mix the two when attributing the gap to oneDNN.

### Two P11 VTune/GT-PIN breakthroughs (`docs/86`, `probes/110-gemm-swizzle-counters`)

- **Per-class GT-PIN dynamic instruction counts via CLI (`docs/86` §1.1, `probes/110-gemm-swizzle-counters/work/gtpin_ic.csv`)**: While `sqlite-db/dicer.db` leaves `dd_gpu_exec_instruction_info` empty (`probes/084-gtpin-types`), running `vtune -report hotspots -r <rdir> -group-by computing-task -format csv` on a `characterization-mode=instruction-count` result resolves the zebin symbols and emits exact, deterministic counts across 6 exhaustive instruction classes (`Control Flow`, `Int16 & HP Float`, `Int32 & SP Float`, `Send`, `Synchronization`, `Other`, whose sum equals `GPU Instructions Executed` `[TOOLCHAIN]`). Note: per-ordinal static-instruction mapping is still absent, so C4/D3/E3 remain blocked on per-instruction cycle counters.
- **High-frequency OA sampling (`-knob gpu-sampling-interval=0.05` ms = 50 us) eliminates cross-launch dilution on ~100-200 us kernels (`docs/86` §1.2, `probes/110-gemm-swizzle-counters/work/vtune_overview.csv`)**: The default `gpu-sampling-interval=1` ms (1000 us) spans the host `ev.wait()` gap between synchronous batch launches, inflating `XVE Array:Idle(%)` to 86.30 % and capturing only 4.987 MB (11.32 %) of `S2_control`'s 44.04 MB compulsory L3 reads (`25.917 GB/s` reported `[MEASURED]`). Tightening `-knob gpu-sampling-interval=0.05` places ~3.86 OA samples inside each 193 us launch and recovers **42.621 MB (96.78 % capture, 95.50 % mean across 4 collections, `221.200 GB/s` raw / `268.927 GB/s` duty-cycle calibrated `[MEASURED]`)** without any external calibration multiplier.

### P12 calibration boundaries and multi-lambda parsing (`docs/92`, `docs/94`, `probes/116`, `probes/118`)

- **SDPA `4.8x` under-report fix vs `1` ms baseline and short-vs-long kernel calibration boundary (`docs/92`, `probes/116-sdpa-counters/work/vtune_memory_calibrated.csv`)**: At `gpu-sampling-interval=1` ms (`1000` us), `s2048_ctrl` (`403` us) and `s2048_order` (`336` us) cross `ev.wait()` idle buckets (`XVE Array:Idle(%)` `60.40%` / `69.00%`), capturing only `23.44%` / `19.92%` (`overview`) and `22.56%` / `20.05%` (`global-memory-accesses`, reproducing `probes/096-sdpa-tail`'s `4.8x` under-report vs `134.218` MB compulsory KV bytes `[MEASURED]`). Tightening to `0.05` ms (`50` us) restores raw DRAM read capture to **`96.85%` / `96.94%`** (`overview`, `4.13x` / `4.87x` recovery vs `1` ms baseline) and **`96.57%` / `97.10%`** (`global-memory-accesses` `[TOOLCHAIN]` `[MEASURED]`), with `HDC` reads at `0.0` GB/s (LSC->L3->DRAM path). Boundary rule:
  - **Sub-`200` us short kernels** (e.g., `s512_ctrl` `112` us at `68.03%` raw capture, or `bf16` S1 `gemm_prepacked` `82.5` us at `56.16%` raw capture): `50` us bucket edges cross `ev.wait()` gaps (external dilution), so duty-cycle / capture-rate calibration is required (`s512_ctrl` recovers `33.368` MB = `99.44%` of `33.554` MB compulsory `[TOOLCHAIN]` `[MEASURED]` `docs/92`).
  - **Medium/long kernels (`>= 335` us)** (e.g., `s2048` at `6.7`–`8.1` buckets and `~97%` raw capture): the remaining `13.15%`–`13.65%` `XVE Array:Idle(%)` stays nearly constant across `ctrl` / `order` / `slm_tree` and represents **intra-kernel barrier-drain / tail idle**, NOT `ev.wait()` dilution; dividing by `duty_cycle` over-estimates DRAM reads by `~12%` (`150.190` / `149.802` MB vs `134.218` MB theoretical, pushing resident bandwidth past the `403.1` GB/s roof `[TOOLCHAIN]` `docs/92`).
- **Multi-lambda and prepack kernel parsing in `vtune -report hotspots -group-by computing-task` (`docs/94`, `probes/118-operator-anatomy/accepted.json`)**: Executables with separate warmup, setup, or verification launches (e.g., `bench/f32_rmsnorm.cpp` `rmsnorm_row_tile` warmup `Instance Count=2/20` vs timed loop lambda `10/1000`, `bench/int8_gemm_esimd.cpp -DGEMM_PREPACK=1` `prepack()` `Instance Count=1`, or `bench/f32_gemv.cpp` verification `launch()` `1021` vs `1020`) emit multiple `computing-task` rows; `parse_hotspots_csv` selects the lambda row whose `Instance Count` is closest to `samples * batch` and normalizes per-launch instruction counts by that row's actual reported `Instance Count` (`[TOOLCHAIN]` `docs/94`).
- **Verifying `256-GRF` (`General Register File Size = 256`) and SWSB scoreboard `Synchronization` counts on Level Zero / ESIMD kernels (`docs/96`, `docs/97`, `probes/120`, `probes/121`)**: While `docs/32-large-grf.md` found OpenCL `.spv` runtime recompilation locked at `128` GRFs, `vtune -report hotspots -group-by computing-task` on `0.05` ms OA collections of `oneDNN` (`ONEDNN_VERBOSE=all` `grf256`, Level Zero `binary_kernels:enabled`) directly reports `Computing Task:General Register File Size = 256` (`Spill Memory Size = 0`) and `SIMD Width = 8` (`[TOOLCHAIN]` `[MEASURED]` `docs/96`). In GT-PIN `instruction-count`, Xe-HPG software scoreboard instructions (`sync.allrd`/`sync.nop`) are counted inside the `Synchronization` class alongside work-group barriers (`bf16_btile_dpasw` reports `96` `Synchronization`/thread vs `33` from `16` barriers alone `[TOOLCHAIN]` `[MEASURED]` `docs/97`).
