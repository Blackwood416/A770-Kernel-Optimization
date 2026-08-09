#!/usr/bin/env python3
"""Run a kernel executable and aggregate samples into median/p10/p90/CV.

The target executable should accept ``--warmup N --samples N --batch N --json``
and print one JSON object per sample with the keys ``sample``, ``device_us``,
``wall_us`` and ``pipeline_us``. The script turns those samples into the
unified benchmark protocol statistics.

Example:
    python scripts/benchmark.py --exe build/f32_gemv.exe \
        --warmup 20 --samples 20 --batch 100 --out artifacts/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from harness_common import ARTIFACTS_DIR, BUILD_DIR, ROOT, run_in_oneapi, stats, write_json


def parse_sample_json(stdout: str) -> list[dict[str, Any]]:
    samples = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "sample" in obj and "device_us" in obj:
            samples.append(obj)
    return samples


def parse_legacy_avg(stdout: str) -> list[dict[str, Any]]:
    samples = []
    for line in stdout.splitlines():
        m = re.search(r"run\[(\d+)\]:\s*([0-9.]+)\s*ms avg", line)
        if m:
            ms = float(m.group(2))
            samples.append(
                {
                    "sample": int(m.group(1)),
                    "device_us": ms * 1000.0,
                    "wall_us": ms * 1000.0,
                    "pipeline_us": ms * 1000.0,
                }
            )
    return samples


def benchmark_executable(
    exe: pathlib.Path,
    *,
    warmup: int = 20,
    samples: int = 20,
    batch: int = 100,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-4,
    timeout: float = 600,
    legacy: bool = False,
) -> tuple[dict[str, Any], str]:
    exe = pathlib.Path(exe)
    if not exe.is_absolute():
        if exe.parts and exe.parts[0] == "build":
            exe = ROOT / exe
        else:
            exe = BUILD_DIR / exe
    command = (
        f'"{exe}" --warmup {warmup} --samples {samples} --batch {batch} '
        f"--rel-tol {rel_tol:g} --abs-tol {abs_tol:g} --json"
    )
    cp = run_in_oneapi(command, cwd=BUILD_DIR, timeout=timeout)
    stdout = (cp.stdout or "") + (cp.stderr or "")
    parsed = parse_sample_json(cp.stdout or "")
    if not parsed and legacy:
        parsed = parse_legacy_avg(cp.stdout or "")

    device = [float(p["device_us"]) for p in parsed]
    wall = [float(p["wall_us"]) for p in parsed]
    pipeline = [float(p["pipeline_us"]) for p in parsed]
    result = {
        "exe": str(exe),
        "warmup": warmup,
        "samples": samples,
        "batch": batch,
        "device_us": stats(device),
        "wall_us": stats(wall),
        "pipeline_us": stats(pipeline),
        "sample_values": parsed,
        "returncode": cp.returncode,
        "status": "OK" if parsed else "NO_SAMPLES",
    }
    return result, stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=pathlib.Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--legacy", action="store_true", help="parse 'run[N]: X ms avg'")
    parser.add_argument("--out", type=pathlib.Path, default=ARTIFACTS_DIR / "benchmark.json")
    args = parser.parse_args()

    result, _stdout = benchmark_executable(
        args.exe,
        warmup=args.warmup,
        samples=args.samples,
        batch=args.batch,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        timeout=args.timeout,
        legacy=args.legacy,
    )
    write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
