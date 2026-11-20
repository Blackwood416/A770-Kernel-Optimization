# Microbenchmarking GEN Assembly on A770 (method)

**Scope**: Intel Arc A770 (DG2), driver IGC `32.0.101.9150`, OpenCL host `tools/genrun`,
watchdog `scripts/run_variant.ps1`. This page is about how to make an instruction-level
measurement trustworthy, not about any single instruction.

## 1. Amplify device time with straight-line code

`[MEASURED]` At n=4096 one axpy launch costs ~0.19 ms of **wall** time but only ~0.004 ms of
**device** time, so host launch overhead swamps any instruction-level difference. Two rules follow:

1. Amplify inside the kernel (duplicate the body), never by timing more launches.
2. Report **both** device and wall time for any kernel under ~100 µs device
   (`bench/README.md` in the research repo shows ~66–103 µs of launch/queue overhead on this box).

`[MEASURED][TOOLCHAIN]` Duplicating an asm block up to 2048 instructions (~32 KB of code) still
runs normally and stays linear in instruction count, so amplification is a valid instrument as long
as it is **straight-line**. A hand-injected backward branch with a predicate hangs the GPU instead
(no TDR event); see `workflow/robustness.md`.

Tool: `scripts/inject_repeat.py --dump <dir> --out <dir> --match <regex> --times N --assemble`.

## 2. Separate timing from correctness

`[MEASURED]` An accumulating kernel (`y += a*x`) must be timed with `--expect none`, because 200
repetitions legitimately produce 200× the expected value. Run correctness as a **separate**
`--repeat 1 --warmup 0` invocation and merge the two results. Mixing them turns a correct kernel
into "4096 wrong elements" and invites a false conclusion.

## 3. Distinguish "not measured" from "zero"

`[MEASURED]` A dependency-broken kernel does not hang the machine: `clFinish` returns
`CL_OUT_OF_RESOURCES`, the process exits, and **no bench line is printed**. A parser that defaults
a missing measurement to 0 will report a spectacular speedup. Treat a missing bench line as a
failure of the run.

Supporting measurements:

- `[MEASURED]` Stripping all SWSB annotations, or pinning every register distance to 7, makes the
  kernel fail this way (`CL_OUT_OF_RESOURCES`), while the timing harness sees only silence.
- `[MEASURED]` Deleting syncs can also produce a *fast and wrong* run (axpy: 4.2 µs with 3936/4096
  elements wrong), which is why every timing variant needs its own correctness run.

## 4. Prove the baseline is compute-bound before pricing an instruction

`[MEASURED]` Six SWSB variants and seven register-bank variants of axpy all measured 4.2 µs with
overlapping p10–p90 — the kernel is memory-bound at n=4096 and cannot resolve instruction-level
differences. The quick test: vary the number of injected instructions and check that device time
grows linearly (axpy: flat; dpas chain: 4.2 → 9.2 → 26.1 µs for 0 → 512 → 2048 instructions).

## 5. Environment discipline

- Every record carries the fingerprints: oneAPI release, ocloc/IGC version, driver version, GPU.
- Conclusions about *encodings* transfer across drivers; conclusions about *timings* do not.
- Interleave the control variant between test variants: the control reproduces byte-identically,
  which proves the pipeline itself did not perturb the measurement.
