# Reusable Experiment Harness

The core A770 measurement scripts are bundled under `scripts/` in this skill,
with minimal example kernels under `examples/`. All commands below are
relative to the skill root. If a deployment strips the scripts, verify script
existence before invoking them; in that case this page is an interface
contract only and no command may be reported as executed.

These helpers are intentionally thin and optional. They are not a mandatory
evidence pipeline: a campaign can use a simple script or manual record when
that is clearer. Full per-experiment metadata is not required for every case
study.

## Pipeline

```text
probe_environment
  -> build
  -> compare_outputs
  -> benchmark
  -> probe_onednn
  -> parse_vtune
  -> watchdog
  -> record_experiment
```

`record_experiment.py` is the normal entry point for a finished experiment.
`e2e.py` runs the whole GEMV smoke end to end:

```powershell
python scripts\e2e.py --warmup 20 --samples 20 --batch 100
```

It builds the example kernel, verifies it against the CPU reference, runs the
unified benchmark, probes the oneDNN baseline, and writes a JSON record plus a
Markdown evidence file.

## Scripts

### Environment probe

```powershell
python scripts\probe_environment.py --out artifacts\environment.json
```

Emits device name, Windows graphics driver, oneAPI version, Level-Zero version
from `sycl-ls`, oneDNN version, and a timestamp. Use it as the version gate
before a campaign.

### Build

```powershell
python scripts\build.py
```

Builds the DPC++ examples under `build/` and writes
`artifacts/build_status.json`. The oneDNN target adds the active oneAPI
include/library paths automatically.

### Correctness compare

```powershell
python scripts\compare_outputs.py --stdout artifacts\benchmark_stdout.txt
python scripts\compare_outputs.py --actual out.bin --expected ref.bin --dtype f32
```

Parses `errors: N/TOTAL max_abs: X` or compares flat binary/CSV arrays with
relative + absolute tolerance.

### Benchmark

```powershell
python scripts\benchmark.py --exe build\f32_gemv.exe `
    --warmup 20 --samples 20 --batch 100 --out artifacts\benchmark.json
```

The kernel executable must accept `--warmup N --samples N --batch N --json`
and print one JSON object per sample:

```json
{"sample":0,"device_us":214.0,"wall_us":220.0,"pipeline_us":220.0}
```

The script reports median, p10/p90, MAD, CV percent, and a CV flag. Default
flag threshold is CV > 10%.

### Correctness contract

Each benchmark executable must emit one JSON correctness object after its
samples:

```json
{
  "errors": 0,
  "total": 262144,
  "max_abs_err": 0.0012,
  "max_rel_err": 0.0031,
  "rel_tol": 0.05,
  "abs_tol": 0.01,
  "reference": "cpu_f16_precast",
  "semantics_id": "matmul_mkn_rowmajor",
  "accuracy_mode": "strict",
  "relaxed_accuracy": false
}
```

The requested `--rel-tol` / `--abs-tol` must equal the executable-reported
values. A mismatch produces `CORRECTNESS_CONTRACT_MISMATCH`; a missing
contract produces `CORRECTNESS_CONTRACT_MISSING`. Neither is emitted as
`[MEASURED]`. `FAIL` is `invalid` unless `semantics_id` is present and
`relaxed_accuracy=true`, in which case it is `fastest_only`.

### oneDNN baseline probe

```powershell
python scripts\probe_onednn.py --exe build\f32_gemv_onednn.exe `
    --warmup 20 --samples 20 --batch 100 --out artifacts\onednn.json
```

Runs with `ONEDNN_VERBOSE=profile,dispatch`, saves verbose lines, extracts the
implementation string, and keeps benchmark plus correctness data. If the
installed oneDNN build does not emit verbose lines, the script falls back to
`primitive_desc.impl_info_str()` so the implementation string is still
recorded. The artifact also records `accuracy_class`, `reference_tolerance`,
`baseline_correctness_status`, and `comparable_for_speedup`; a baseline that
fails the required tolerance is `fastest_only` only when operator semantics
are confirmed and relaxed math is declared; otherwise it is `invalid`. Such a
baseline must not be used for speedup ratios. The executable must report the
tolerance it actually used; if it differs from the requested tolerance, the
result is
`CORRECTNESS_CONTRACT_MISMATCH` and no `[MEASURED]` record is emitted.

### VTune parse

```powershell
python scripts\parse_vtune.py --collect --exe build\f32_gemv.exe `
    --characterization instruction-count
python scripts\parse_vtune.py --csv artifacts\vtune_report\hotspots.csv `
    --out artifacts\vtune.json --markdown artifacts\vtune.md
```

`--collect` runs VTune `gpu-hotspots`; parsing also works on an existing CSV
export. The Markdown output includes the `[MEASURED]` evidence label.

### Watchdog

```powershell
python scripts\watchdog.py --exe build\f32_gemv.exe `
    --iterations 10 --timeout 30 --label f32_gemv --out artifacts\watchdog
```

Launches one executable at a time, enforces a timeout, kills the process tree,
captures the exit code, snapshots/copies new minidumps from `CrashDumps` and
the Windows Minidump folder, and queries System Event Log entries including
`VIDEO_TDR_FAILURE` / bugcheck `0x116`. Risk probes must run one at a time.

### Experiment record

```powershell
python scripts\record_experiment.py --operator gemv --shape 4096x4096 `
    --dtype f32 --variant sycl_subgroup_direct_l2 --exe build\f32_gemv.exe `
    --probe-onednn --vtune-json artifacts\vtune.json `
    --out artifacts\records\f32_gemv.json
```

Core JSON fields when using `record_experiment.py`:

```json
{
  "operator": "gemv",
  "shape": "4096x4096",
  "dtype": "f32",
  "variant": "sycl_subgroup_direct_l2",
  "driver": "32.0.101.8724",
  "oneapi": "2026.1.0",
  "device_median_us": 229.27875,
  "wall_median_us": 331.864,
  "pipeline_median_us": 331.864,
  "max_abs_err": 1.06112e-05,
  "max_rel_err": 1.2e-04,
  "errors": 0,
  "total": 4096,
  "reference": "cpu_f64",
  "semantics_id": "gemv_n1_mkn_rowmajor",
  "accuracy_mode": "default",
  "relaxed_accuracy": false,
  "accuracy_class": "matched",
  "reference_tolerance": "rel=0.0001, abs=0.0001",
  "requested_rel_tol": 0.0001,
  "requested_abs_tol": 0.0001,
  "correctness_status": "PASS",
  "baseline_correctness_status": "PASS",
  "comparable_for_speedup": true,
  "vtune": null,
  "status": "PASS"
}
```

It also writes a Markdown record containing the `[MEASURED]` validity domain.
Full artifacts add `benchmark` statistics (median, p10/p90, MAD, CV flag),
`baseline`, and `environment` fields.

## Measured Smoke Record

`[MEASURED]` f32 GEMV `4096x4096`, `sycl_subgroup_direct_l2`, A770 / oneAPI
2026.1 / driver `32.0.101.8724`:

| Field | Value |
|---|---:|
| device_median_us | 229.28 |
| wall_median_us | 331.86 |
| errors | 0/4096 |
| oneDNN implementation | `jit:gemm:any` |
| oneDNN device_median_us | 467.69 |
| oneDNN wall_median_us | 632.67 |
| oneDNN accuracy_class | matched |
| oneDNN comparable_for_speedup | true |
| status | PASS |

The harness validates the full pipeline, not a new kernel record. Use it to
turn future campaigns into machine-readable JSON instead of Markdown-only
results.

Provenance: this is the automation harness smoke record
(standard-SYCL `sycl_subgroup_direct_l2`, warmup 20 + samples 20 + batch 100,
run 2026-08-08). It is a separate campaign from the dense-operator GEMV
ladder in `techniques.md`; retain the harness metadata when comparing values.

On 2026-08-09 the same e2e path was re-run as a contract test on the current
host (driver `32.0.101.8860`, Level Zero `1.15.38308`) with 2 samples to
verify the correctness JSON and tolerance binding. That run was noisy (CV
flagged) and is not a dispatch conclusion; historical campaign numbers keep
their original `32.0.101.8724` provenance.

## Rules

- Keep generated reports free of machine-specific absolute paths; command
  examples above are relative to the skill root.
- Before invoking any harness command, verify that the script exists. If it
  is unavailable, treat `automation.md` as an interface contract only and do
  not claim the script was executed.
- Do not force every experiment through the full record schema; use the
  helpers for correctness and timing, and keep case studies readable.
- The correctness contract is machine-readable: the executable reports
  `rel_tol`, `abs_tol`, `reference`, and `semantics_id`. Requested tolerance
  must equal executable-reported tolerance; `FAIL` is `invalid` unless
  semantics are confirmed and relaxed accuracy is declared.
- Before converting discrete benchmark points into an inequality dispatch,
  sample points around every decision boundary; unswept values are
  `[HEURISTIC]`.
- Keep failed variants as standalone source files and record them as negative
  results; never delete a failing kernel merely because it failed.
- Convert `record_experiment.py` output into skill documents with the
  evidence labels defined in `SKILL.md`.
- For oneDNN baselines, keep the implementation string and verbose lines so a
  JIT win can be distinguished from a reference fallback.
