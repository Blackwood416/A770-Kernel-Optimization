#!/usr/bin/env python3
"""Run a oneDNN baseline with ONEDNN_VERBOSE=profile,dispatch and parse it.

The target executable should be a standalone oneDNN primitive that also
verifies against a CPU reference and emits one JSON benchmark sample per line.
This script keeps the implementation string, per-mode verbose lines, timings
and correctness in one JSON artifact.

Example:
    python scripts/probe_onednn.py --exe build/f32_gemv_onednn.exe \
        --warmup 20 --samples 20 --batch 100 --out artifacts/onednn.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmark import parse_sample_json
from compare_outputs import parse_stdout
from harness_common import (
    ARTIFACTS_DIR,
    BUILD_DIR,
    ROOT,
    run_in_oneapi,
    stats,
    write_json,
)


def parse_implementation(verbose_text: str) -> tuple[str | None, list[str]]:
    impl = None
    lines = []
    from_verbose = False
    for line in verbose_text.splitlines():
        if "dnnl_verbose" not in line:
            continue
        lines.append(line.strip())
        if impl is not None:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0] == "dnnl_verbose" and parts[1] in {
            "dispatch",
            "profile",
        }:
            impl = parts[3]
            from_verbose = True
    if impl is None:
        m = re.search(r"implementation:\s*(\S+)", verbose_text)
        if m:
            impl = m.group(1)
    if impl is not None and not lines:
        lines = [f"implementation: {impl}"]
    return impl, lines, from_verbose


def probe_onednn(
    exe: pathlib.Path,
    *,
    warmup: int = 20,
    samples: int = 20,
    batch: int = 100,
    timeout: float = 600,
    operator: str = "gemv",
    primitive: str = "matmul",
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-4,
) -> dict[str, Any]:
    exe = pathlib.Path(exe)
    if not exe.is_absolute():
        if exe.parts and exe.parts[0] == "build":
            exe = ROOT / exe
        else:
            exe = BUILD_DIR / exe
    env = os.environ.copy()
    env["ONEDNN_VERBOSE"] = "profile,dispatch"
    env["DNNL_VERBOSE"] = "profile,dispatch"
    command = f'"{exe}" --warmup {warmup} --samples {samples} --batch {batch} --json'
    cp = run_in_oneapi(command, cwd=BUILD_DIR, env=env, timeout=timeout)
    combined = (cp.stdout or "") + (cp.stderr or "")
    implementation, verbose_lines, from_verbose = parse_implementation(combined)
    samples_data = parse_sample_json(cp.stdout or "")
    compare = parse_stdout(combined)
    baseline_ok = compare.get("status") == "PASS"
    if compare.get("status") == "NO_VERIFY":
        accuracy_class = "unknown"
    elif baseline_ok:
        accuracy_class = "matched"
    else:
        accuracy_class = "fastest"
    device = [float(s["device_us"]) for s in samples_data]
    wall = [float(s["wall_us"]) for s in samples_data]
    pipeline = [float(s["pipeline_us"]) for s in samples_data]
    status = "PASS"
    if not samples_data:
        status = "NO_SAMPLES"
    elif not implementation:
        status = "NO_IMPL"
    elif compare.get("status") != "PASS":
        status = "VERIFY_FAIL"
    return {
        "operator": operator,
        "primitive": primitive,
        "implementation": implementation,
        "implementation_source": "ONEDNN_VERBOSE" if from_verbose else "primitive_desc.impl_info_str()",
        "verbose": "profile,dispatch",
        "verbose_lines": verbose_lines,
        "device_us": stats(device),
        "wall_us": stats(wall),
        "pipeline_us": stats(pipeline),
        "sample_values": samples_data,
        "max_abs_err": compare.get("max_abs_err"),
        "errors": compare.get("errors"),
        "total": compare.get("total"),
        "accuracy_class": accuracy_class,
        "reference_tolerance": f"rel={rel_tol:g}, abs={abs_tol:g}",
        "baseline_correctness_status": compare.get("status"),
        "comparable_for_speedup": baseline_ok,
        "returncode": cp.returncode,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=pathlib.Path, default=BUILD_DIR / "f32_gemv_onednn.exe")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--operator", default="gemv")
    parser.add_argument("--primitive", default="matmul")
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--out", type=pathlib.Path, default=ARTIFACTS_DIR / "onednn.json")
    args = parser.parse_args()

    result = probe_onednn(
        args.exe,
        warmup=args.warmup,
        samples=args.samples,
        batch=args.batch,
        timeout=args.timeout,
        operator=args.operator,
        primitive=args.primitive,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )
    write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
