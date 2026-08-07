# A770 Robustness and TDR Protocol

Measured on Intel Arc A770, oneAPI 2026.1, driver `32.0.101.8724`.

> Evidence: `[BUG]` / `[MEASURED]` behavior is tied to this driver/oneAPI
> combination; re-run the probes after any toolchain upgrade.

## Environment Baseline

- Device: Intel Arc A770, confirmed at runtime.
- Windows graphics driver: `32.0.101.8724`.
- oneAPI: `2026.1.0`.
- Level-Zero: `1.15.37669`.

The watchdog harness pins and checks the version baseline at runtime and
flags mismatches in its report; the exact file layout is
implementation-specific.

## Verified Baselines

| Baseline | Shape | Time (3 runs) | Correctness |
|---|---|---|---|
| bf16 DPAS GEMM (ESIMD `dpas<8,8,float>`, VNNI B) | 64x64x64 | 0.0081 / 0.0078 / 0.0065 ms | `errors: 0/4096` |
| f32 GEMV (sub-group per row, direct L2) | 4096x4096 | 0.2295 / 0.2287 / 0.2282 ms | `errors: 0/4096` |

Both baselines passed `stress10` (10 independent launches each).

## Controlled Probe Results

Each probe runs in its own process with a 15 s watchdog. This round produced
no `VIDEO_TDR_FAILURE` and no new system minidump.

| Probe | Compile | Runtime behavior | Verdict |
|---|---|---|---|
| `load_2d<uint32_t,8,8>` | passes | kernel hangs; no stdout | `TIMEOUT_KILLED` (15.2 s) |
| `named_barrier` | passes | device JIT rejects: "Named barriers are not supported by XeHPG" | `EXIT_2` |
| `__esimd_dpas2<u4, fp16>` (N1=32, N2=64) | passes | no hang, but layout unverified | PASS only as no-hang smoke |
| 600 batch ESIMD submissions (out-of-order) | passes | no crash; historical 100+ crash not reproduced | PASS (stress10 10/10) |

## Risk API Table

| API / combination | A770 status | Typical symptom | Isolation |
|---|---|---|---|
| ESIMD `load_2d` / `prefetch_2d` | PVC-only | kernel hang, no output | one-shot process + 15 s watchdog |
| `load_2d` + `Transposed` + bf16 | compile rejection | u32/u64 only gate | compile gate first |
| ESIMD `named_barrier_init/signal/wait` | XeHPG rejection | JIT error, exit 2 | one-shot process, capture logs |
| mixed `__esimd_dpas2<u4, fp16>` | compiles, layout unverified | possible wrong results | smoke only |
| DPAS `ExecutionSize=16` | compiles, wrong results | 78/128 errors | keep as failed variant |
| batch ESIMD submissions (100+, no wait) | historical driver crash | `q.wait()` crash, possible 0x116 | one-shot stress, collect dump |
| runtime `if/else` inside ESIMD | hang | no output | use `if constexpr` |

## Rules

1. Always run risky ESIMD shapes in one-shot processes with a watchdog; never
   run multiple GPU pressure processes at the same time.
2. Verify stable baselines before probing risks, and snapshot Event Log and
   minidump state before each round.
3. Treat any hang, non-zero exit, new minidump, or TDR as evidence and stop
   stacking more risk probes.
4. Keep failed variants as standalone source files and record the negative
   result with driver/oneAPI versions.

## Safety Checklist

Before running:

1. Save all work and confirm the pinned version baseline matches the machine.
2. Close other GPU load: browser acceleration, recording, remote desktop,
   other inference/render programs.
3. Run baselines first: `--suite baseline --mode smoke`.
4. Run each risk probe with `--allow-risk` in its own process; never merge
   multiple risk targets.
5. Use short timeouts (10-20 s) for known hang probes; smoke before stress.
6. Snapshot Event Log and minidump state before probing.

During/after:

7. Treat `TIMEOUT_KILLED`, non-zero exit, new minidump, or TDR as evidence and
   stop stacking more probes.
8. On bugcheck `0x116`, copy `C:\Windows\Minidump\*.dmp` after reboot; the
   watchdog also tries to copy them into the run directory.
9. Record driver, oneAPI, submission count, kernel shape, and reproduction
   probability.
10. Keep failed variants as standalone source files and record negative
    results; never delete "bad" code.

## Negative Results

- `load_2d<uint32_t,8,8>` hangs the A770 kernel and must be killed by the
  watchdog (`TIMEOUT_KILLED`).
- `named_barrier` is rejected at device JIT with "Named barriers are not
  supported by XeHPG", not at host compile time.
- The historical 100+ batch ESIMD crash was not reproduced in a controlled
  600-submission stress run; treat it as a non-deterministic risk.

## Commands

```powershell
python <watchdog-script> --suite baseline --mode smoke --timeout 120
python <watchdog-script> --suite probes --mode smoke --allow-risk --timeout 15
python <watchdog-script> --target risk_batch_esimd --mode stress10 --allow-risk --timeout 30
```

Artifacts: build status, per-run stdout/stderr/summary, minidump copies, and
a machine-readable report.
