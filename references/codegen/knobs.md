# Compiler Flags, Codegen Behaviour, and `-Xfinalizer` Knobs on DG2

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.2 (`icx` / `icx-cl` / `ocloc`), driver `32.0.101.9150`. Always verify whether a flag changed the disassembly (`ocloc disasm`), never just the binary hash.

## 1. SYCL/ESIMD VC Backend: `-Xfinalizer` Reachable Knobs `[MEASURED]` `[TOOLCHAIN]`

Unlike legacy OpenCL `ocloc`, the **SYCL/ESIMD VC backend** delivers `-Xfinalizer <opt>` directly to the IGC vISA finalizer and substantially rewrites SWSB token allocation, scheduler order, and synchronization.

Two verified delivery channels:

1. **Static AOT link embedding (zero runtime env vars)**:
   `-fsycl "SHELL:-Xs \"-Xfinalizer -SBIDBundle -Xfinalizer -SWSBDepReduction -Xfinalizer -nopresched -Xfinalizer -twoSrcBundleBCR\""`
2. **Rapid runtime ablation sweep (no relink)**:
   Set `SYCL_CACHE_PERSISTENT=0` (prevents IGC persistent cache from silently serving stale binaries) and `SYCL_PROGRAM_APPEND_COMPILE_OPTIONS="-Xfinalizer <flag> ..."`.

Across a ~45-flag systematic ablation on `GRF=256 dpasw` BF16 GEMM:

| Category | Flag(s) | Measured Effect on DG2 ESIMD |
|---|---|---|
| **Optimal 4-flag set** | `-Xfinalizer -SBIDBundle` `-Xfinalizer -SWSBDepReduction` `-Xfinalizer -nopresched` `-Xfinalizer -twoSrcBundleBCR` | `sync.allrd 5 -> 0`, `Short UGM waits 5 -> 1`, `sync.nop 34 -> 23`, `4096^3` bf16 GEMM `1555 -> 1488 us` (`1.10x` faster vs `oneDNN 3.13.2` `[MEASURED]`) |
| `-SBIDBundle` | Part of optimal set | Lets non-overlapping RAW/WAW `dpasw` macros share SBIDs, preventing 16-SBID thrashing |
| `-SWSBDepReduction` | Part of optimal set | Global token liveness analysis removes redundant `sync.allrd` and cuts `sync.nop` |
| `-nopresched` | Part of optimal set | Disables pre-RA scheduler so hand-ordered ESIMD load/SLM/dpasw interleaving survives |
| `-twoSrcBundleBCR` | Part of optimal set | Dual-source bundle optimization; empirically beneficial as part of the 4-flag combo |
| **Crash / ICE blacklist `[BUG]`** | `-dpasScheduleBarrier`, `-presched-ctrl 1`, `-presched-ctrl 2`, `-IGASWSB`, `-raBlockLocalScan`, `-enableDpasFwd 1`, `-dpasMacroBySrc1RS` | IGC Access Violation or compile failure (`rc=1`); never pass on DG2 |
| **Regression blacklist** | `-globalTokenAllocation`, `-distPropTokenAllocation`, `-SBIDL3Hit`, `-noschedule`, `-SWSBMakeLocalWAR`, `-SWSBReplaceARWithAW` | Regresses large GEMM (`1024x4096x4096` `~408 -> 423 us`, `4096^3` `~1555 -> 1602–1640 us` vs optimal set `[MEASURED]`) |

## 2. Standard SYCL Optimization Flags and Unrolling `[MEASURED]`

Measured on non-XMX `bf16` standard-SYCL kernels (GEMM `1024x1536x512` and RMSNorm `1024x4096`):

| Flag / Transform | Instruction & Wall Effect | Verdict |
|---|---|---|
| `-O0` | `SIMD8`, `15x` slower on GEMM | Use only for ISA forensics (preserves calls/CFG) |
| `-O2` (default) | Auto-vectorizes to `SIMD32` at `100%` util | **Recommended baseline** |
| `-O3`, `-ffp-contract=off`, `-ffast-math` | Byte-identical JIT instructions, `+0.3% to +9.3%` wall jitter | Neutral on simple loops; do not rely on `-O3` alone |
| Explicit `unroll=2` | `-16.6%` instr / **`-23.0%` wall** (GEMM); `-17.1%` wall (RMSNorm) | **Primary source-level win** on simple loops |
| Explicit `unroll=4` alone | Worse alone than `unroll=2` (`+7.7%` vs `u2`) | Only pays when combined with AOT + `-O3` + fast-math |
| `-fsycl-device-code-split=per_kernel` | Isolates per-kernel register/module compilation | **Always enable** in multi-kernel ESIMD libraries |

**Large GRF flag discipline**:

- **Never pass global `-ze-opt-large-register-file`** on XMX/ESIMD builds (negative on `joint_matrix` and crashes/fails on legacy ESIMD).
- **Use per-kernel `sycl::ext::intel::experimental::grf_size<256>`** in ESIMD (`WG=32 + SLM <= 48 KB`) to enable `256-GRF` cleanly without affecting other kernels.
- For AOT ESIMD sidecars where explicit AOT target flags are required, `-fsycl-targets=spir64_gen -Xs "-device dg2 -options -doubleGRF"` enables 256-GRF AOT binaries.

## 3. OpenCL `ocloc` Path: Dead vISA Surface vs Reachable CL Flags `[TOOLCHAIN]` `[MEASURED]`

On the **OpenCL C (`ocloc`) path only**, all 487 vISA `-options` (`-igc_opts`, `IGC_VISAOptions`, `IGC_Options`, 22 pass switches) are unreachable no-ops. Only CL-level options change OpenCL codegen (baseline = `140` instructions):

| OpenCL Flag | Instructions | What Actually Changes |
|---|---:|---|
| `-cl-opt-disable` | `603` (`+463`) | Disables inlining (`call`/`goto`/`ret`) and emits coarse `sync.allrd`/`sync.allwr` |
| `-cl-fp32-correctly-rounded-divide-sqrt` | `447` (`+307`) | Replaces `math.sqt` with `math.rsqt` + Newton sequence (`~3x` instructions) |
| `-cl-fast-relaxed-math` / `-cl-unsafe-math-optimizations` | `110` / `112` | Folds comparisons into predication (drops `not`) |
| `-cl-denorms-are-zero`, `-cl-finite-math-only`, `-cl-no-signed-zeros` | `140` | Rewrites `cr0.0` prologue bitmask only (`0x4C0 -> 0x40`); zero throughput change |
| `-cl-intel-256-GRF-per-thread` | `141` (AOT only) | Changes AOT RA, but OpenCL runtime recompiles `.spv` back to `128-GRF` |

## 4. IR and Assembly Dump Contract `[TOOLCHAIN]`

- Device text LLVM IR (`-S -emit-llvm`) is unsupported on oneAPI Windows; preserve LLVM bitcode, SPIR-V text (`llvm-spirv --to-text`), and GEN assembly (`ocloc disasm -file <kernel>.zebin -device dg2 -dump <dir>`).
- For A770 AOT targets, use `-device dg2` (or `dg2-g10-a0`).
- **CFG / graph-coloring sensitivity**: Even minor edits outside a hot loop (e.g., prologue/epilogue structure or `if constexpr` branches) can perturb IGC register coloring and SBID assignment. Re-run disassembly audit (`send.dc0`, `dummy_dpasw`, `Short UGM waits`) after every structural edit.
