#!/usr/bin/env python3
"""Run/parse a VTune GPU report and keep the instruction-count/full metrics.

The parser accepts either a VTune result directory (and runs ``vtune -report``)
or an already exported CSV. It stores rows as JSON and can emit a compact
Markdown table for the skill's evidence files.

Example:
    python scripts/parse_vtune.py --result-dir artifacts/vtune \
        --out artifacts/vtune.json --markdown artifacts/vtune.md
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from harness_common import ARTIFACTS_DIR, find_oneapi_root, run_in_oneapi, write_json


def find_vtune() -> pathlib.Path:
    root = find_oneapi_root()
    versions = sorted((root / "vtune").glob("*/bin64/vtune.exe"), reverse=True)
    if versions:
        return versions[0]
    raise FileNotFoundError("vtune.exe not found under oneAPI root")


def collect(
    exe: pathlib.Path,
    result_dir: pathlib.Path,
    characterization: str = "instruction-count",
    timeout: float = 1800,
) -> subprocess.CompletedProcess[str]:
    result_dir.mkdir(parents=True, exist_ok=True)
    vtune = find_vtune()
    return run_in_oneapi(
        f'"{vtune}" -collect gpu-hotspots -knob characterization-mode={characterization} '
        f'-result-dir "{result_dir}" -- "{exe}"',
        timeout=timeout,
    )


def report(
    result_dir: pathlib.Path,
    report_dir: pathlib.Path,
    group_by: str = "computing-task",
    timeout: float = 900,
) -> pathlib.Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "hotspots.csv"
    vtune = find_vtune()
    cp = run_in_oneapi(
        f'"{vtune}" -report hotspots -r "{result_dir}" -format csv '
        f'-report-output "{out}" -group-by {group_by}',
        timeout=timeout,
    )
    if cp.returncode != 0:
        raise RuntimeError((cp.stdout or "") + (cp.stderr or ""))
    return out


def parse_csv(csv_path: pathlib.Path) -> dict[str, Any]:
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    clean = []
    for row in rows:
        item = {k.strip(): v.strip() for k, v in row.items() if k is not None}
        clean.append(item)

    summary: dict[str, Any] = {"rows": len(clean)}
    for row in clean:
        for key in (
            "Total Time",
            "GPU Time",
            "ALU0 Instructions",
            "ALU1 Instructions",
            "Send Instructions",
            "XMX Instructions",
            "GPU Barriers",
            "XMX Pipeline Active",
            "XVE Active",
            "Occupancy",
            "L3 Bandwidth Bound",
        ):
            if key in row and key not in summary:
                try:
                    summary[key] = float(row[key].rstrip("%"))
                except ValueError:
                    summary[key] = row[key]
    return {"summary": summary, "rows": clean}


def write_markdown(payload: dict[str, Any], out: pathlib.Path) -> None:
    lines = [
        "# VTune metrics",
        "",
        "> `[MEASURED]` on A770 / oneAPI 2026.1 / driver `32.0.101.8724`; validity domain:",
        "> the exact kernel, shape, submission count and characterization mode in the attached run.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in payload.get("summary", {}).items():
        if key != "rows":
            lines.append(f"| {key} | {value} |")
    lines += ["", "## Rows", "", "```json", json.dumps(payload.get("rows", []), indent=2), "```"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="run vtune collection first")
    parser.add_argument("--exe", type=pathlib.Path)
    parser.add_argument("--characterization", default="instruction-count")
    parser.add_argument("--result-dir", type=pathlib.Path, default=ARTIFACTS_DIR / "vtune")
    parser.add_argument("--report-dir", type=pathlib.Path, default=ARTIFACTS_DIR / "vtune_report")
    parser.add_argument("--csv", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=ARTIFACTS_DIR / "vtune.json")
    parser.add_argument("--markdown", type=pathlib.Path, default=ARTIFACTS_DIR / "vtune.md")
    args = parser.parse_args()

    if args.collect:
        if not args.exe:
            parser.error("--collect requires --exe")
        cp = collect(args.exe, args.result_dir, args.characterization)
        if cp.returncode != 0:
            print((cp.stdout or "") + (cp.stderr or ""))
            return cp.returncode

    csv_path = args.csv
    if csv_path is None:
        csv_path = report(args.result_dir, args.report_dir)
    payload = parse_csv(csv_path)
    payload["source_csv"] = str(csv_path)
    write_json(args.out, payload)
    if args.markdown:
        write_markdown(payload, args.markdown)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
