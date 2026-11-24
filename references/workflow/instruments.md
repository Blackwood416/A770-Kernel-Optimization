# Measurement, Disassembly Audit, Methodology, and TDR Safety Protocol

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`, VTune `2026.6`. Always diagnose with hardware counters and disassembly before restructuring a kernel.

## 1. Optimization Methodology and Mental Model

1. **Specify exact semantics and layout first**: `A*x` vs `x*W`, `[M,K]` row-major vs VNNI-packed, NCHW vs NHWC, causal vs full attention. Classify into **Domain I (DRAM stream-through, `403.1 GB/s` roof)** or **Domain II (L3-resident XMX, `GRF=256 dpasw`)**.
2. **Build an exact CPU reference & verified library baseline first**: Require `errors: 0/<total>` before any timing. Verify library baseline outputs against the CPU reference (`oneDNN 1xK` computes `x*A`, not `A*x`; `oneDNN bf16 u4` fails tolerance).
3. **Change one structural variable at a time and keep negative results**: Separate structural changes (bottleneck shifts) from run-to-run noise (on A770 independent medians can vary `~3%–10%`; use paired interleaved A/B/B/A deltas + MAD for `< 5%` decisions, and treat `<= 3%` margins as a simplification tie to pick the simpler kernel).
4. **Separate timing bases**: Report `device_time` (SYCL event or L0 timestamp), `wall_time` (host submit -> sync), and `pipeline_time` (including any per-call pre/post-processing). For `< 100 us` kernels, WDDM host overhead (`~60–140 us`) dominates single-shot wall time.

## 2. Working Instruments vs Unavailable Counters `[TOOLCHAIN]` `[MEASURED]`

| Goal | Working Instrument on A770 (Windows) | Key Caveat |
|---|---|---|
| Executed GPU ISA & register spills | Export native binary via **`zeModuleGetNativeBinary`** -> **`ocloc disasm -file <k>.zebin -device dg2 -dump <dir>`** | Walk `<dir>` with `os.walk` matching `.endswith(".asm")` (dot-prefixed `.text.*.asm` files are skipped by shell `*.asm` glob) |
| Dynamic instruction mix (6 classes) | `vtune -collect gpu-hotspots -knob characterization-mode=instruction-count` + `-report hotspots -group-by computing-task -format csv` | Launch exe **directly** (no `cmd /c` wrapper); counts 32-lane execution groups (compare deltas only); wall time is inflated `50x–60x` during collection |
| High-frequency OA counters (`100–400 us` kernels) | `vtune -collect gpu-hotspots -knob characterization-mode=overview -knob gpu-sampling-interval=0.05` (`50 us`) | Default `1 ms` interval dilutes short launches across `ev.wait()` gaps (`4.8x` DRAM under-report); `0.05 ms` recovers `~97%` raw capture for `>= 335 us` kernels (only apply duty-cycle calibration for `< 200 us` kernels) |
| Multi-lambda / prepack executables | Match timed kernel row in `computing-task` CSV by `Instance Count` closest to `samples * batch` | Normalize per-launch counts by that row's actual `Instance Count` |
| Full-compute / SLM bandwidth | `vtune -collect gpu-hotspots -allow-multiple-runs -knob characterization-mode=full-compute` (or `lsc-slm`) | Requires admin + `-allow-multiple-runs`; `lsc-slm` counters live in `sqlite-db/dicer.db` |
| `oneDNN` kernel selection | `ONEDNN_VERBOSE=profile,dispatch` (or `debuginfo=10`) | Captures `jit:gemm:any`, `grf256`, tile geometry, or `ocl:ref:any` fallback |

**Not supported on this box (do not retry without new hardware/driver)**:
- VTune `eu-stall-sampling` (unsupported on DG2 Windows `[TOOLCHAIN]`).
- Per-instruction cycle / dual-issue counters and L1/L2 per-set associativity counters (unexposed by driver).
- Legacy OpenCL `.spv` `256-GRF` or bindless ExBSO (use SYCL ESIMD `grf_size<256>` and native Level Zero instead).
- Device text LLVM IR (`-S -emit-llvm` fails; use `llvm-spirv --to-text` or `ocloc disasm`).

## 3. Mandatory Native-Binary Disassembly Audit Pipeline `[MEASURED]` `[BUG]`

For production ESIMD `GRF=256` kernels, **`KernelDiag.spillMemSize == 0` is a known false negative**:

1. Query `zeKernelGetProperties` and assert **`spillMemSize == 0 && privateMemSize == 0`** (catches LLVM SROA `priv=576–1088 B` spills).
2. Dump native binary via `zeModuleGetNativeBinary` and disassemble with `ocloc disasm -device dg2`.
3. Locate the hot loop basic block (maximum `dpasw` ops ending in a backward `goto`/`jmpi`) and run a **2-iteration wrap-around static scan**:
   - **Main-loop `send.dc0` count**: hard requirement **`== 0`** (any `send.dc0` inside the loop is a hidden register scratch spill).
   - **`dummy_dpasw` (`null:f`) count**: target **`1`** (basic-block entry only; `> 1` means `0-Dummy` grouping is broken).
   - **`sync.allrd` count**: target **`0`** with `-Xfinalizer -SWSBDepReduction`.
   - **`Short UGM waits (<120 ins)` / UGM SBID clobbers**: target **`<= 1`** (strictly correlated with `4096^3` latency: `1 -> 11` waits = `1486 -> 1703 us`).
   - **`SLM ld/st addr chains`**: check consecutive `load.slm`/`store.slm` reusing one address GRF (target low reuse via `store_buf`/`load_slm` interleaving).

## 4. Robustness and Windows TDR Safety Protocol `[BUG]`

1. **Isolate risky ESIMD shapes**: Run untested geometries (high `RPT`, new `WG` sizes, near-`64 KB` SLM, GPU-polled doorbells) in standalone one-shot processes under a `10–20 s` watchdog; never stack multiple GPU-stress processes simultaneously.
2. **Pre-launch gates**:
   - Never launch ESIMD `WG >= 256` (`WG=512` triggers Windows `VIDEO_TDR` LiveKernelEvent 141).
   - Never launch `load_2d`/`prefetch_2d` on DG2 (permanent hang requiring watchdog kill).
   - Require zero compiler spill (`spill == 0`, `send.dc0 == 0`) before launching DG2 Prefill SDPA with `RPT > 4` (any spill causes `UR_RESULT_ERROR_DEVICE_LOST`).
   - Calibrate `UncachedHost` doorbell spin caps against `~1.12 us/spin` (`<= 40 ms` = `~35,615` spins) to stay far below Windows' 2 s WDDM TDR limit.
3. **On hang / TDR / `DEVICE_LOST`**: Stop stacking risk probes immediately, verify system stability (check Windows Event Log / `<Windows>\Minidump`), record the exact driver/oneAPI version and geometry as a negative result, and revert to the last known-good baseline.
