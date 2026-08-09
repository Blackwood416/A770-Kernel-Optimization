# Reusable Experiment Harness

The A770 measurement scripts live in a separate harness checkout; this page
stores the CLI contract, record schema, and workflow so any campaign can reuse
them without embedding machine-specific paths. All commands below are
relative to the harness repo root.

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

### oneDNN baseline probe

```powershell
python scripts\probe_onednn.py --exe build\f32_gemv_onednn.exe `
    --warmup 20 --samples 20 --batch 100 --out artifacts\onednn.json
```

Runs with `ONEDNN_VERBOSE=profile,dispatch`, saves verbose lines, extracts the
implementation string, and keeps benchmark plus correctness data. If the
installed oneDNN build does not emit verbose lines, the script falls back to
`primitive_desc.impl_info_str()` so the implementation string is still
recorded.

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

Required JSON fields:

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
  "errors": 0,
  "total": 4096,
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
| status | PASS |

The harness validates the full pipeline, not a new kernel record. Use it to
turn future campaigns into machine-readable JSON instead of Markdown-only
results.

Provenance: this is the automation harness smoke record
(standard-SYCL `sycl_subgroup_direct_l2`, warmup 20 + samples 20 + batch 100,
run 2026-08-08). It is a separate campaign from the dense-operator GEMV
ladder in `techniques.md`; retain the harness metadata when comparing values.

## Rules

- Keep generated reports free of machine-specific absolute paths; command
  examples above are relative to the harness repo root.
- Keep failed variants as standalone source files and record them as negative
  results; never delete a failing kernel merely because it failed.
- Convert `record_experiment.py` output into skill documents with the
  evidence labels defined in `SKILL.md`.
- For oneDNN baselines, keep the implementation string and verbose lines so a
  JIT win can be distinguished from a reference fallback.
