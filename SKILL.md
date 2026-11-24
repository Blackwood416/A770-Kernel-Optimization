---
name: A770-Kernel-Optimization
description: "Methodology and production code guidance for optimizing SYCL, ESIMD, and native Level Zero compute kernels on Intel Arc A770 (Xe-HPG / DG2, Windows). Covers Domain I streaming kernels (GEMV, RMSNorm, Softmax, Decode SDPA, DAG fusion), Domain II XMX matrix kernels (GRF=256 cab4 dpasw BF16/INT8/4-bit GEMM, SwiGLU, Prefill SDPA), ISA/SWSB/SLM/cache rules, -Xfinalizer compiler knobs, Level Zero queues/Copy Engine/graphs, VTune & disassembly spill auditing, and verified oneDNN baselines."
---

# Arc A770 (DG2) Kernel Optimization

Use this skill as a **methodology and selective code-guidance layer** for optimizing SYCL, ESIMD, and Level Zero kernels on Intel Arc A770 (`Xe-HPG / DG2`, Windows 11, oneAPI `2026.2`, driver `32.0.101.9150`, Level Zero SDK `1.33.1`, `oneDNN 3.13.2`).

**Transfer discipline**: Measured numbers and `[DISPATCH]` rules apply only inside their stated A770 validity domain. For any other Intel GPU, driver, or compiler stack, transfer the **methodology only** and re-measure all hardware-specific assumptions before shipping.

## 1-Minute A770 Engineering Quick Sheet `[MEASURED]` `[ARCH]`

1. **Classify the kernel first** ([bandwidth.md](references/hardware/bandwidth.md)):
   - **Domain I (DRAM stream-through: GEMV, RMSNorm, Softmax, Decode SDPA, elementwise DAGs)**: `XMX = 0%`, `DRAM/L3` pass-through `79%–99.5%`. Target the **`403.1 GB/s`** `float4` read roof (`464.6 GB/s` write) using `vec16` coalesced loads, single-pass SLM staging for multi-pass row ops, interleaved streams (`SDPA_ORDER=1`, `386–391 GB/s`), and Tier-3 `.zebin` DAG fusion (`399.6 GB/s` = `99.1%` roof, `2.73x` vs unfused).
   - **Domain II (L3-resident XMX matrix: BF16/INT8/4-bit GEMM, SwiGLU, Prefill SDPA)**: `DRAM/L3` `10%–19%` (`81%–89%` L3 hit). Unlock **`GRF=256` (`experimental::grf_size<256>`) + `WG=32` + `cab4` (`4 x 12 KB = 48 KB` SLM -> `2 WGs/Xe-core = 64/64` HW threads)** + `split_barrier` + `0-Dummy` `dpasw<8,8>` (`2x8`/`3x4` -> `dummy=1`) + `bp_vec` (`-94 ins`) + ESIMD VC **`-Xfinalizer -SBIDBundle -SWSBDepReduction -nopresched -twoSrcBundleBCR`**: reaches **`92.5 TFLOPS`** (`4096^3`, `1.10x` vs `oneDNN 3.13.2`) and **`33–35 WIN | 4–6 TIE | 0 LOSE`** across 39 shapes (incl. `SwiGLU` `79.6 TFLOPS`, `INT8` `129.6 TOPS`, `4-bit BK=gs=64` online dequant `22.5–24.6 TFLOPS`).
2. **Cache-hint & swizzle rules** ([cache-geometry.md](references/hardware/cache-geometry.md)):
   - DG2 dispatches round-robin across 32 Xe-cores (`gid` & `gid+32` share L1). Use `swizzle=1` + B `ca.ca` for single-wave grids (`total_wgs <= 64`); use `2D 8x8 N-inner` snake + `L1StreamB` (`st.ca`) + `swizzle <= 4` (`K >= 4096`) for multi-wave grids.
   - Output `C` store: **`wb.wb`** for standard / `Bias` / `SiLU` (`-42 us` on LM-Head vs `st.uc`); **`st.uc`** for `ResidualAdd` / `BiasResidualAdd`.
3. **Runtime, Graph, and Copy Engine rules on Windows WDDM** ([execution.md](references/workflow/execution.md)):
   - Never enable `UR_LOADER_USE_LEVEL_ZERO_V2=1` globally on uncaptured loops (`7.93x` slowdown vs v1; dual-SYCL Copy overlap collapses `99.95% -> <= 0%`).
   - Best zero-tax runtime: **Native Level Zero Regular Command List** (`zeCommandListAppendBarrier` + fixed-address `GraphControl`, `0.484 us/edge`, `1`-replay break-even) + explicit **`ordinal 1` (`COPY`-only)** Copy Engine (`91.6%–99.95%` overlap) + pre-allocated Device Arena (avoids UMF `>= 16 MB` cliff) + static `zeEventPool` (`zeEventHostReset` `0.10–0.20 us`).
   - Single-CCS (`numQueues=1` on `ordinal 0`) means `0.00%` multi-queue compute concurrency -- horizontally fuse small compute ops (`SwiGLU`, DAGs) instead.
   - GPU-polled host doorbells require `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` (`10/10`, `6.00 us` RTT) with TDR-calibrated spin caps (`~1.12 us/spin`).
4. **Production spill & SBID iron gate** ([instruments.md](references/workflow/instruments.md)):
   - `KernelDiag.spillMemSize == 0` has false negatives. Always verify **`spill == 0 && priv == 0`** AND **`ocloc disasm` main-loop `send.dc0 == 0`**, `dummy_dpasw == 1`, `sync.allrd == 0`, and `Short UGM waits (<120 ins) <= 1`.

## Scope Gate

Before writing or tuning a kernel:

1. Confirm the target is A770 (DG2); otherwise apply workflow only and re-measure on target hardware.
2. State exact operator semantics, layout (`A*x` vs `x*W`, row-major vs VNNI, NCHW vs NHWC), dtype, shape, and target metric (`device_time`, `wall_time`, or `pipeline_time`).
3. Build an exact CPU reference (`errors: 0/<total>`) and verify any library baseline against it before timing.
4. Load only the relevant reference pages from the routers below; label any unmeasured proposal `[HYPOTHESIS]`.

## Route the Task

### Operator Router

| Task | Primary Reference | Next Reference |
|---|---|---|
| Dense GEMM (`bf16`/`int8`/`f32`), `SwiGLU`, GEMV-N1, RMSNorm, Softmax | [techniques.md](references/techniques/techniques.md) | [code-snippets.md](references/api/code-snippets.md) |
| Weight-only `INT4` / `NF4` GEMM & Decode GEMV (`x*W`) | [weight-only-gemv.md](references/weight-only-gemv.md) | [api-usage.md](references/api/api-usage.md#4-onednn-baseline-recipes-correctness) |
| Attention / SDPA (Decode `Q=1` & Prefill `Q>1`) | [sdpa-a770.md](references/techniques/sdpa-a770.md) | [code-snippets.md](references/api/code-snippets.md#8-stable-online-softmax-rescale-core) |
| Reduction or work-group scan | [reductions-scan.md](references/techniques/reductions-scan.md) | [numerics.md](references/techniques/numerics.md) |
| Odd/unaligned shapes, sparse GEMM, or direct convolution | [irregular-shapes.md](references/techniques/irregular-shapes.md) | [techniques.md](references/techniques/techniques.md) |
| Precision, accumulators, math ops, or verification tolerances | [numerics.md](references/techniques/numerics.md) | [pitfalls.md](references/workflow/pitfalls.md) |

### Architecture, ISA, Runtime, and Diagnostic Router

| Situation | Open |
|---|---|
| Hardware topology, `128-GRF vs 256-GRF` residency matrix, XMX saturation, SLM banking | [hardware.md](references/hardware/hardware.md) |
| Empirical bandwidth roofs (`403.1 GB/s`), Domain I/II taxonomy, stride/width penalties | [bandwidth.md](references/hardware/bandwidth.md) |
| Cache latency ladder, `gid & gid+32` co-residency, swizzle rules, LSC cache hints | [cache-geometry.md](references/hardware/cache-geometry.md) |
| `dpas` / `dpasw<8,8>` legality, `0-Dummy` grouping theorem (`2x8`/`3x4`) | [dpas.md](references/isa/dpas.md) |
| SWSB 16-SBID saturation, `Short UGM waits`, `split_barrier`, fences, GRF banks | [swsb.md](references/isa/swsb.md) |
| LSC `256 B` cap, A64 `bp_vec`, `load_2d`/prefetch bans, `send.dc0`, atomics | [lsc-messages.md](references/isa/lsc-messages.md) |
| SYCL/ESIMD `-Xfinalizer` 4-flag optimum, blacklist, `-O2`/unroll, OpenCL flags | [knobs.md](references/codegen/knobs.md) |
| Level Zero queues, `ordinal 1` Copy Engine, graphs, doorbells, UMF pools, Tier-3 `.zebin` | [execution.md](references/workflow/execution.md) |
| Symptom -> ISA -> action playbook & DG2 ISA capability matrix | [performance-playbook.md](references/workflow/performance-playbook.md) |
| Known API/hardware gates, codegen spills, WDDM/L0 traps, and negative results | [pitfalls.md](references/workflow/pitfalls.md) |
| Profiling (VTune `0.05 ms` OA), `ocloc disasm` spill audit, methodology, TDR safety | [instruments.md](references/workflow/instruments.md) |
| Modern SYCL/ESIMD/L0 APIs, `oneDNN` baselines, and build/VTune commands | [api-usage.md](references/api/api-usage.md) |
| Curated production kernel & runtime code snippets | [code-snippets.md](references/api/code-snippets.md) |

Complete directory map: [references/index.md](references/index.md).

## Evidence Labels and Measurement Contracts

| Label | Meaning |
|---|---|
| `[ARCH]` | Xe-HPG / DG2 architectural constraint |
| `[ISA]` | Instruction encoding or hardware legality fact |
| `[MEASURED]` | Verified on A770 (`errors == 0`) inside stated validity domain |
| `[DISPATCH]` | Validated production selection rule within stated shape/stack domain |
| `[HEURISTIC]` | Starting point derived from measurements; re-measure on new shapes |
| `[TOOLCHAIN]` | Compiler/runtime/API capability or gate; re-probe after upgrades |
| `[BUG]` | Version-specific driver/compiler defect or silent hazard |
| `[CORRECTNESS]` | Layout, alignment, precision, or synchronization correctness requirement |
| `[HYPOTHESIS]` | Unverified proposal; test before recommending |

- **Timing contract**: Warm up to avoid `15–20 us` cold-GPU downclocking; take `>= 20` samples, report median + p10/p90 (or MAD), flag high CV, and separate `device_time`, `wall_time`, and `pipeline_time`. Use paired interleaved A/B/B/A testing for `< 5%` margins.
- **`oneDNN` baseline contract**: Run with `ONEDNN_VERBOSE=profile,dispatch`, record implementation string (`jit:gemm:any` vs `ocl:ref:any`), tags, dtypes, `fpmath_mode`, and verify against the CPU reference. Never report a speedup against a baseline that fails tolerance (`fastest_only` / `invalid`).
- **Safety gate**: Run risky ESIMD geometries (high `RPT`, new `WG`, `UncachedHost` polling) one process at a time under a watchdog, require zero spills before launch, and never use `WG >= 256`, `load_2d`, or `named_barrier` on DG2.
