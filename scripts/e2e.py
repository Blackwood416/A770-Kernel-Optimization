#!/usr/bin/env python3
"""End-to-end smoke: build -> verify -> benchmark -> oneDNN probe -> record.

Example:
    python scripts/e2e.py --samples 20 --batch 100
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "artifacts" / "records" / "f32_gemv_e2e.json")
    args = parser.parse_args()

    subprocess.run([sys.executable, "scripts/build.py"], cwd=ROOT, check=True)
    cmd = [
        sys.executable,
        "scripts/record_experiment.py",
        "--operator",
        "gemv",
        "--shape",
        "4096x4096",
        "--dtype",
        "f32",
        "--variant",
        "sycl_subgroup_direct_l2",
        "--exe",
        "build/f32_gemv.exe",
        "--probe-onednn",
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--batch",
        str(args.batch),
        "--out",
        str(args.out),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
