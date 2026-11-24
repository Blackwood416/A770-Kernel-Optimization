# A770 Pitfalls, Hard Gates, and Verified Negative Results

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`. Every entry is deduplicated and reconciled against current best practices. Re-probe `[BUG]` and `[TOOLCHAIN]` entries after any driver or compiler upgrade.

## 1. Hard API and Hardware Gates `[ARCH]` `[TOOLCHAIN]` `[BUG]`

| Feature / Pattern | A770 Status | Consequence & Rule |
|---|---|---|
| ESIMD `WG >= 256` (incl. `WG=512` attention) | Never schedules / TDR `[BUG]` | `WG=256` hangs silently; `WG=512` triggers Windows `VIDEO_TDR` even on a trivial kernel. Keep ESIMD `WG <= 128` (`128-GRF`) or `WG = 32` (`256-GRF` / SDPA) |
| `load_2d` / `store_2d` / `prefetch_2d` | PVC/Xe2+ only `[BUG]` | Compiles/assembles on DG2, then hangs forever (`TIMEOUT_KILLED`). Use `<= 256 B` `block_load` |
| `named_barrier` | PVC-only `[TOOLCHAIN]` | Passes host compile, rejected at device JIT on Xe-HPG. Use standard `barrier()` or `split_barrier` |
| `block_load` `512 B` | PVC-only `[ARCH]` | DG2 maximum LSC block width is `256 B` |
| DPAS `ExecutionSize=16` or `dst:hf` | Broken / rejected `[BUG]` | `ExecutionSize=16` compiles with wrong outputs (`78/128` errors); DG2 requires `SIMD8` and `f32`/`i32` accumulators |
| `fp8` / `tf32` / mixed `f16+u4` public DPAS | Unsupported `[TOOLCHAIN]` | Absent on DG2; unpack `4-bit -> bf16` via `BK=gs=64` + SLM LUT before `dpasw<8,8>` |
| Global `-ze-opt-large-register-file` | Negative / fails `[TOOLCHAIN]` | Use per-kernel `experimental::grf_size<256>` (`WG=32`, `SLM <= 48 KB`) instead |
| `esimd::prefetch` / `__builtin_prefetch` | Unusable / harmful `[MEASURED]` | OpenCL fails compile; in `K=4096` ESIMD GEMM loops `esimd::prefetch` adds `~50` ALU ops + `4–6` SBID conflicts for net regression |
| `64 KB` SYCL `local_accessor` | Launch failure `[MEASURED]` | Standard SYCL `65,536 B` `local_accessor` exits without diagnostic; keep standard SYCL tiles `<= 32–48 KB` |

## 2. ESIMD Codegen, Register Spill, and Microarchitecture Traps `[MEASURED]` `[BUG]`

- **`KernelDiag.spillMemSize == 0` false negative (`send.dc0` hidden spills) `[BUG]`**: `zeKernelGetProperties` often reports `spillMemSize == 0 B` even when the main loop emits `35–67` `send.dc0` scratch spills (e.g., `acc[4][6]` consuming `192 GRFs` out of `256`). Always require **`spill == 0 && priv == 0`** AND verify **`ocloc disasm` main-loop `send.dc0 == 0`**.
- **Runtime-indexed `simd[]` breaks LLVM SROA (`privateMemSize = 576–1088 B`)**: Indexing a `simd` array with a runtime loop variable prevents register promotion (`spillMemSize` stays `0`, `priv` spikes, slowing `512x4096` GEMM `~2x` from `240 -> 661 us` vs named SSA). Use compile-time 2-way unroll + named SSA variables.
- **`.select<16,1>() = block_load(...)` emits blocking `mov {$dst}`**: Load directly into standalone 1-GRF `simd` variables first. Also note `.bit_cast_view<uint32_t>()` rejects `const simd&`; pass non-`const` references.
- **`esimd::slm_block_store` `16 B` silent overwrite `[BUG]` `[CORRECTNESS]`**: An `8 B`-aligned offset (`slm_block_store<uint32_t, 2>`) silently truncates down to `16 B` and corrupts neighbor elements. Always use `16 B`-aligned `slm_block_store<uint32_t, 4>` or wider.
- **`CAB3` dynamic 3-buffer index under 4-way unroll (`4 % 3 != 0`)**: Produces `~394K` cross-check errors; match circular buffer count statically (`cab4` with 4-step compile-time unroll).
- **SLM / WG mismatch at `256-GRF`**: `WG=48 / 56 KB SLM` (`452 us`, `1 WG/Xe-core` barrier stalls) and `WG=24 / 40 KB` (`362 us`) both lose heavily to `WG=32 / 48 KB` (`290–300 us`, `2 WGs/Xe-core`). Similarly `WG=64` loses on RowMajor GEMM.
- **Wrong workarounds for `dummy_dpasw (null:f)`**: Reverting from `dpasw<8,8>` to `dpas<8,8>` still emits `null:f` while doubling `A` SLM read traffic (`-12.4%` regression vs `dpasw` on `2048x4096x2048`); forcing a `1x16` macro splits via `sync.allwr` (`dummy 1 -> 17`, `385 -> 433 us`). Use `0-Dummy` `2x8` / `3x4` grouping instead.
- **`slm_scatter` `256 B` stride & non-transposed `lsc_gather`**: `slm_scatter` at `256 B` stride hits an 8-way SLM write bank conflict (`+79 us` on `4096^3`); non-transposed `SIMD8` `lsc_gather` regresses `LMHd` by `-20%` vs contiguous `block_load` (`406 -> 506 us`).
- **`K >= 4096` with `swizzle >= 8`, or `wt.wb` C stores**: `swizzle=8` overflows the `16 MB` L3 on `4096^3` (`+71 us` slower vs `swizzle=4`); `wt.wb` C stores regress `wide` by `+17 us` vs `wb.wb`.
- **CFG / register-coloring sensitivity & `-Xfinalizer` blacklist**: Minor control-flow or scope edits outside the inner loop can alter IGC coloring (`SLM addr chains` `18/40` vs `40/40`) -- always re-audit disassembly. Never pass ICE flags (`-dpasScheduleBarrier`, `-presched-ctrl 1/2`, `-IGASWSB`, `-raBlockLocalScan`, `-enableDpasFwd 1`, `-dpasMacroBySrc1RS`) or regressors (`-globalTokenAllocation`, `-distPropTokenAllocation`, `-SBIDL3Hit`, `-noschedule`, `-SWSBMakeLocalWAR`, `-SWSBReplaceARWithAW`).
- **SDPA zero-spill & barrier hoisting traps**: On DG2 Prefill SDPA, any compiler spill with `RPT > 4` triggers `UR_RESULT_ERROR_DEVICE_LOST`. On Decode SDPA, hoisting next KV loads across the reduction barrier spills registers and slows execution `2.8x–3.2x` (`556 -> 1567–1763 us`) vs zero-spill baseline.

## 3. Level Zero, WDDM, and Runtime Traps `[MEASURED]` `[ARCH]`

- **`UR_LOADER_USE_LEVEL_ZERO_V2=1` dual regression**: Slows uncaptured `L=256` SYCL `in_order` chains **`7.93x`** (`1237 -> 9811 us`) and collapses dual-SYCL Compute+Copy overlap from `99.95%` to `<= 0%`. Pin async DMA to native Level Zero `ordinal 1` and use Regular Command Lists for hot chains.
- **Single-CCS compute serialization**: A770 exposes `numQueues=1` on `ordinal 0` (`0.00%` multi-queue compute timestamp overlap); fuse small compute ops horizontally instead of splitting across queues.
- **Same-queue pageable D2H & subgraph splitting**: Mid-chain `4 KB` pageable D2H closes UR v1 dynamic batching (`+80.5%`); splitting `1x256` graphs into `32x8` subgraphs inflates replay `9.47x` (`v1`) / `5.32x` (`v2`) vs monolithic graph.
- **`zeEvent` vs `AppendBarrier` inversion**: Per-node `zeEvent`s inside a Regular CL double dispatch time (`249–295 us` vs `131 us` `AppendBarrier`), while `AppendBarrier` inside an Immediate CL triggers per-step WDDM flushes (`4823 vs 2566 us`).
- **Cached doorbells & uncalibrated spins**: `malloc_host` / `malloc_shared` stall `0/10` on GPU doorbell polling; only `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` works (`10/10`), and each spin costs `~1.12 us` (cap spins to avoid 2 s TDR).
- **UMF `>= 16 MB` cliff & `zeEventPoolCreate`**: Hot `>= 16 MB` `malloc_device` spikes to `637–1863 us`, and `zeEventPoolCreate` costs `103–232 us`; pre-allocate a Device Arena and static `zeEventPool` (`zeEventHostReset` = `0.10–0.20 us`).

## 4. Correctness, Library Baseline, and Tooling Traps `[CORRECTNESS]` `[BUG]`

- **Silent wrong results in reductions & sub-groups**:
  - `esimd::reduce(v, std::plus<float>{})` compiles and silently returns `0`; always pass `std::plus<>{}` / `std::multiplies<>{}`.
  - `sycl::reduce_over_group` on 32 lanes silently halves sums if register pressure narrows IGC to `SIMD16` (no cross-pass SLM merge); stage in per-slice SLM or audit `simd_size: 32`.
  - Never hardcode `SG=16` trip counts (compiler may pick `32`); never assume 2D local dim 0 is the sub-group axis (prefer 1D `nd_range`).
  - Always initialize local reduction accumulators before `atomic_ref::fetch_add`.
- **Control flow & layout traps**:
  - Runtime `if/else` in ESIMD can hang; use `if constexpr` (never `#if FUSED` on template params).
  - Dead fields in kernel argument structs break native L0 `zeKernelSetArgumentValue` ABI counts.
  - Packed 4-bit K-block byte offset is `bk / 2`, not `bk * (BK/2)`; VNNI B reference math must use unpacked B; online-softmax rescale must stay `<= 1.0` to avoid fp32 `inf`/`NaN`.
- **Library baseline traps**:
  - `oneDNN 1xK` computes `x*A` (`A^T*x`), not `A*x`; `oneMKL` row-major requires a column-major call `gemm(N, M, K, B, N, A, K, C, N)`.
  - `oneDNN` RMSNorm `scale` (`gamma`) must be `f32` even for `f16`/`bf16` inputs; `oneDNN u4` requires `ba` weights + explicit `fpmath_mode(..., true)` (and `bf16`-src `u4` fails tolerance -- use `f16`-src).
- **Benchmarking & Windows shell traps**:
  - Cold-GPU first launch runs `15–20 us` slower from boost downclocking -- always warm up before sampling.
  - PowerShell `cmd /c 'set VAR=val && prog.exe'` appends a trailing space to `VAR`; set `$env:VAR = "val"` in PowerShell.
  - `ocloc disasm` emits dot-prefixed `.text.*.asm` files (use `os.walk`, not `*.asm` glob); VTune `eu-stall-sampling` is unsupported on DG2 Windows, and wrapping VTune targets in `cmd /c` silently disables the GPU plugin.
