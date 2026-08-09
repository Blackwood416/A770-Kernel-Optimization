#!/usr/bin/env python3
"""Record one experiment as a machine-readable JSON + evidence Markdown.

The script fingerprints the environment, runs the kernel benchmark, parses the
CPU-reference verification, optionally runs the oneDNN baseline, and writes the
unified record fields requested by the skill.

Example:
    python scripts/record_experiment.py --operator gemv --shape 4096x4096 \
        --dtype f32 --variant sycl_subgroup_direct_l2 --exe build/f32_gemv.exe \
        --probe-onednn --out artifacts/records/f32_gemv.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmark import benchmark_executable
from compare_outputs import parse_stdout
from harness_common import (
    ARTIFACTS_DIR,
    BUILD_DIR,
    detect_environment,
    read_json,
    write_json,
)
from probe_onednn import probe_onednn


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def write_evidence_markdown(record: dict[str, Any], out: pathlib.Path) -> None:
    env = record["environment"]
    lines = [
        "# Experiment Record",
        "",
        f"> `[MEASURED]` validity domain: operator=`{record['operator']}`, "
        f"shape=`{record['shape']}`, dtype=`{record['dtype']}`, "
        f"variant=`{record['variant']}`, device=`{env.get('device')}`, "
        f"driver=`{record['driver']}`, oneAPI=`{record['oneapi']}`.",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| device_median_us | {record['device_median_us']:.4f} |",
        f"| wall_median_us | {record['wall_median_us']:.4f} |",
        f"| pipeline_median_us | {record['pipeline_median_us']:.4f} |",
        f"| max_abs_err | {fmt(record['max_abs_err'])} |",
        f"| errors | {fmt(record['errors'])}/{fmt(record['total'])} |",
        f"| status | {record['status']} |",
        f"| accuracy_class | {record.get('accuracy_class', '-')} |",
        f"| reference_tolerance | {record.get('reference_tolerance', '-')} |",
        f"| comparable_for_speedup | {record.get('comparable_for_speedup', '-')} |",
    ]
    baseline = record.get("baseline")
    if baseline:
        lines += [
            "",
            "## oneDNN Baseline",
            "",
            f"- implementation: `{baseline.get('implementation')}`",
            f"- verbose: `{baseline.get('verbose')}`",
            f"- device_median_us: {fmt(baseline.get('device_us', {}).get('median'))}",
            f"- wall_median_us: {fmt(baseline.get('wall_us', {}).get('median'))}",
            f"- status: {baseline.get('status')}",
            f"- accuracy_class: {baseline.get('accuracy_class')}",
            f"- baseline_correctness_status: {baseline.get('baseline_correctness_status')}",
            f"- comparable_for_speedup: {baseline.get('comparable_for_speedup')}",
        ]
    vtune = record.get("vtune")
    if vtune:
        lines += ["", "## VTune", ""]
        for key, value in vtune.get("summary", {}).items():
            if key != "rows":
                lines.append(f"- {key}: {value}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--exe", type=pathlib.Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--probe-onednn", action="store_true")
    parser.add_argument("--onednn-exe", type=pathlib.Path, default=BUILD_DIR / "f32_gemv_onednn.exe")
    parser.add_argument("--vtune-json", type=pathlib.Path)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ARTIFACTS_DIR / "records" / "experiment.json",
    )
    args = parser.parse_args()

    env = detect_environment()
    benchmark, stdout = benchmark_executable(
        args.exe,
        warmup=args.warmup,
        samples=args.samples,
        batch=args.batch,
    )
    compare = parse_stdout(stdout)
    compare["rel_tol"] = args.rel_tol
    compare["abs_tol"] = args.abs_tol

    baseline = None
    if args.probe_onednn:
        baseline = probe_onednn(
            args.onednn_exe,
            warmup=args.warmup,
            samples=args.samples,
            batch=args.batch,
            operator=args.operator,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
        )

    vtune = None
    if args.vtune_json:
        vtune = read_json(args.vtune_json)

    status = "PASS"
    if benchmark.get("status") != "OK":
        status = "BENCH_FAIL"
    elif compare.get("status") != "PASS":
        status = "VERIFY_FAIL"

    record: dict[str, Any] = {
        "operator": args.operator,
        "shape": args.shape,
        "dtype": args.dtype,
        "variant": args.variant,
        "driver": env["driver_version"],
        "oneapi": env["oneapi_version"],
        "device_median_us": benchmark["device_us"]["median"],
        "wall_median_us": benchmark["wall_us"]["median"],
        "pipeline_median_us": benchmark["pipeline_us"]["median"],
        "max_abs_err": compare.get("max_abs_err"),
        "errors": compare.get("errors"),
        "total": compare.get("total"),
        "accuracy_class": "matched" if status == "PASS" else "failed",
        "reference_tolerance": f"rel={args.rel_tol:g}, abs={args.abs_tol:g}",
        "baseline_correctness_status": (
            baseline.get("baseline_correctness_status") if baseline else None
        ),
        "comparable_for_speedup": (
            bool(baseline.get("comparable_for_speedup")) if baseline else None
        ),
        "vtune": vtune,
        "baseline": baseline,
        "status": status,
        "environment": {
            "device": env["device"],
            "level_zero": env["level_zero_version"],
            "dnnl": env["dnnl_version"],
            "captured_at": env["captured_at"],
        },
        "benchmark": {
            "device_us": benchmark["device_us"],
            "wall_us": benchmark["wall_us"],
            "pipeline_us": benchmark["pipeline_us"],
            "warmup": args.warmup,
            "samples": args.samples,
            "batch": args.batch,
        },
        "recorded_at": dt.datetime.now().astimezone().isoformat(),
    }
    write_json(args.out, record)
    md_path = args.out.with_suffix(".md")
    write_evidence_markdown(record, md_path)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"written: {args.out}")
    print(f"markdown: {md_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
