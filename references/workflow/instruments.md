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
