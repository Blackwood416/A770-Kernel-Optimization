# Compiler Knobs That Actually Change the ISA on DG2

**Scope**: A770 (DG2), ocloc IGC `32.0.101.8857`, icpx 2026.1. Established by compiling one fixed
kernel with each flag, disassembling, and diffing line by line (`probes/048-knobs/knob_atlas.ps1`,
`docs/23-compiler-knobs.md`). Kernel: mul-add, divide, `sqrt`, int conversion, SLM + barrier, atomic add.

## 1. The vISA flag surface is unreachable from ocloc `[TOOLCHAIN]`

`VISAOptionsDefs.h` defines **487** `-options` (416 of them not dump/debug). None of them can be
delivered by ocloc:

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
