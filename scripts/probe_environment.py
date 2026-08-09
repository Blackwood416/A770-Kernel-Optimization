#!/usr/bin/env python3
"""Fingerprint the A770 environment (driver, oneAPI, Level-Zero, oneDNN).

Example:
    python scripts/probe_environment.py --out artifacts/environment.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from harness_common import ARTIFACTS_DIR, detect_environment, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=ARTIFACTS_DIR / "environment.json")
    parser.add_argument("--print", action="store_true", help="also print JSON to stdout")
    args = parser.parse_args()

    env = detect_environment()
    env["status"] = "OK"
    write_json(args.out, env)
    print(json.dumps(env, indent=2, ensure_ascii=False))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
