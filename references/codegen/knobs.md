# Compiler Knobs That Actually Change the ISA on DG2

**Scope**: A770 (DG2), ocloc IGC `32.0.101.8857`, icpx 2026.1. Established by compiling one fixed
kernel with each flag, disassembling, and diffing line by line (`probes/048-knobs/knob_atlas.ps1`,
`docs/23-compiler-knobs.md`). Kernel: mul-add, divide, `sqrt`, int conversion, SLM + barrier, atomic add.

## 1. The vISA flag surface is unreachable from OpenCL `ocloc` `[TOOLCHAIN]`

`VISAOptionsDefs.h` defines **487** `-options` (416 of them not dump/debug). None of them can be
delivered via the legacy OpenCL `ocloc` path:

| attempt | result |
|---|---|
| `ocloc ... -igc_opts '...'` | `Invalid option: -igc_opts` |
| `ocloc ... -options "-igc_opts '<flag>'"` | builds, but the disassembly is byte-identical to the baseline |
| env `IGC_VISAOptions` / `IGC_Options` | no effect |
| `-dumpVisaOptionsAll` (a vISA dump flag) as a probe | no output at all - the flag never reaches vISA |

22 pass switches (SendFusion, WriteFusion, AtomicFusion, MACOpt, MsgHeaderOpt, RegRenaming, CSEL,
SCF, MergeScalar, CSE4Send, InstCombine, CopyProp, FlagOpt, DefHoist, FoldAddrImm, DCE,
forceGRFModeUp, PrefetchL1, IfCvt, AccSub family) were all no-ops in the disassembly.
Treat vISA knobs mentioned in IGC documentation as dead configuration on this toolchain.

## 2. Reachable knobs that do change codegen `[MEASURED]`

Baseline is 140 instructions.

| flag | instructions | what changed |
|---|---|---|
| `-cl-opt-disable` (= O0) | **603** (+463) | introduces `call`/`goto`/`ret` (no inlining), `sync.allrd`/`sync.allwr`, `asr`/`bfn`/`shr` |
| `-cl-fp32-correctly-rounded-divide-sqrt` | **447** (+307) | `math.sqt` disappears, replaced by `math.rsqt` + `bfn`/`asr`/`shr`/`xor` Newton iterations |
| `-cl-fast-relaxed-math` | 110 (-30) | drops `not` (comparisons folded into predication) |
| `-cl-unsafe-math-optimizations` | 112 | same, drops `not` |
| `-cl-finite-math-only` | 140 | only two `mov null<1>:ud <imm>` immediates differ |
| `-cl-denorms-are-zero` | 140 | **only `or cr0.0 ..., 0x4C0` becomes `0x40`** - the denormal bit is cleared in cr0 |
| `-cl-no-signed-zeros` | 140 | cr0 / immediates only, like finite-math |
| `-cl-intel-256-GRF-per-thread` | 141 (fixed test kernel; 158 on an earlier kernel, `knob_atlas.csv`) | register allocation changes, one `mad` becomes `csel` |
| `-cl-std=CL3.0` | 140 | bytes change, instruction count does not |

No change at all: `-cl-mad-enable` (already using `mad`), `-cl-single-precision-constant`,
`-cl-intel-enable-auto-large-GRF-mode`, `-cl-no-subgroup-ifp`.
Rejected by ocloc: `-options -O2` / `-O3`, `-cl-intel-has-bindless-style-buffers`.

## 3. How to use this

1. **Instruction-level levers available without touching source**: O0 (for ISA forensics - it keeps
   calls and control flow, so the disassembly is far easier to map to source), `-cl-fast-relaxed-math`
   (removes `not`), and `-cl-fp32-correctly-rounded-divide-sqrt` (trades `math.sqt` for an rsqrt +
   Newton sequence at ~3x the instruction count).
2. **Precision knobs mostly rewrite `cr0`, not opcodes.** denorm / signed-zero / finite-math all live
   in cr0 bits, so they leave the instruction count untouched while changing numerical behaviour.
   Always record them in the environment fingerprint of a numerical comparison.
3. **`-cl-intel-256-GRF-per-thread` is the only reachable large-GRF switch.** It changes register
   allocation wholesale and is the entry point for measuring large-GRF behaviour.
4. A flag that "builds fine" has not necessarily done anything: always diff the disassembly, not just
   the binary hash (metadata alone makes hashes differ).

## 4. SYCL/ESIMD VC `-Xfinalizer` reachable knobs `[MEASURED]` `[TOOLCHAIN]`

Unlike OpenCL `ocloc`, the **SYCL/ESIMD VC backend** (verified on A770 / oneAPI 2026.2 / driver `32.0.101.9150`)
delivers `-Xfinalizer <flag>` straight into the IGC vISA finalizer and substantially rewrites SWSB tokens,
scheduler ordering, and sync placement.

Two verified delivery channels:

1. **Static AOT (CMake `target_link_options`, zero runtime env vars)**:
   `-fsycl "SHELL:-Xs \"-Xfinalizer -flag1 -Xfinalizer -flag2 ...\""`
2. **Runtime rapid sweep without rebuild**:
   Set `SYCL_CACHE_PERSISTENT=0` (prevents IGC persistent cache from silently serving stale binaries) and
   `SYCL_PROGRAM_APPEND_COMPILE_OPTIONS="-Xfinalizer <opt> ..."`.

Across a ~45-flag systematic ablation on `GRF=256 dpasw` BF16 GEMM (`bf81332` baseline vs `7ec1b93`):

| Status | Flag(s) | Measured effect on DG2 ESIMD `dpasw` |
|---|---|---|
| **Optimal 4-flag set** | `-Xfinalizer -SBIDBundle -Xfinalizer -SWSBDepReduction -Xfinalizer -nopresched -Xfinalizer -twoSrcBundleBCR` | `[MEASURED]` `sync.allrd 5 -> 0`, `Short UGM waits 5 -> 1`, `sync.nop 34 -> 23`, `4096x4096x4096` bf16 GEMM `1555 -> 1488 us` (`1.10x` vs oneDNN) |
| **`-SBIDBundle` (role)** | Part of optimal set | Allows non-overlapping RAW/WAW `dpasw` macros to share SBIDs (e.g., `$9` / `$12`), preventing 16-SBID thrashing |
| **`-SWSBDepReduction` (role)** | Part of optimal set | Global token liveness pruning removes redundant `sync.allrd` and cuts `sync.nop` |
| **`-nopresched` (role)** | Part of optimal set | Disables pre-RA scheduler so hand-ordered ESIMD load/SLM/dpasw interleaving stays intact |
| **`-twoSrcBundleBCR` (role)** | Part of optimal set | Reduces GRF bank conflicts across dual-source instruction bundles |
| **Crash / ICE blacklist** | `-dpasScheduleBarrier`, `-presched-ctrl 1`, `-presched-ctrl 2`, `-IGASWSB`, `-raBlockLocalScan`, `-enableDpasFwd 1`, `-dpasMacroBySrc1RS` | `[BUG]` IGC Access Violation or compile failure (`rc=1`); do not pass on DG2 |
| **Regression blacklist** | `-globalTokenAllocation`, `-distPropTokenAllocation`, `-SBIDL3Hit`, `-noschedule`, `-SWSBMakeLocalWAR`, `-SWSBReplaceARWithAW` | `[MEASURED]` Regresses `LLM` (`~408 -> 423 us`), `XL` (`~1555 -> 1602–1640 us`), or expands sync stalls vs optimal set |
