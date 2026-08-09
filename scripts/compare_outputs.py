#!/usr/bin/env python3
"""Compare device output against a CPU reference (or parse the kernel's report).

Two modes are supported:

* ``--stdout FILE`` parses the ``errors: N/TOTAL max_abs: X`` line emitted by
  the kernel harness after its CPU-reference check.
* ``--actual FILE --expected FILE`` compares two flat f32 binary files or CSV
  files directly.

Example:
    python scripts/compare_outputs.py --stdout build/run_stdout.txt \
        --rel-tol 1e-4 --abs-tol 1e-4 --out artifacts/compare.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from harness_common import ARTIFACTS_DIR, write_json


def parse_stdout(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "errors" in obj and "total" in obj:
            return {
                "errors": int(obj["errors"]),
                "total": int(obj["total"]),
                "max_abs_err": float(obj.get("max_abs_err", 0.0)),
                "max_rel_err": (
                    float(obj["max_rel_err"]) if "max_rel_err" in obj else None
                ),
                "rel_tol": float(obj["rel_tol"]) if "rel_tol" in obj else None,
                "abs_tol": float(obj["abs_tol"]) if "abs_tol" in obj else None,
                "reference": obj.get("reference"),
                "semantics_id": obj.get("semantics_id"),
                "accuracy_mode": obj.get("accuracy_mode"),
                "relaxed_accuracy": bool(obj.get("relaxed_accuracy", False)),
                "status": "PASS" if int(obj["errors"]) == 0 else "FAIL",
            }
    m = re.search(r"errors:\s*(\d+)/(\d+)\s+max_abs:\s*([0-9.eE+-]+)", text)
    if m:
        errors = int(m.group(1))
        total = int(m.group(2))
        return {
            "errors": errors,
            "total": total,
            "max_abs_err": float(m.group(3)),
            "max_rel_err": None,
            "rel_tol": None,
            "abs_tol": None,
            "reference": None,
            "semantics_id": None,
            "accuracy_mode": None,
            "relaxed_accuracy": False,
            "status": "PASS" if errors == 0 else "FAIL",
        }
    return {
        "status": "NO_VERIFY",
        "errors": None,
        "total": None,
        "max_abs_err": None,
        "max_rel_err": None,
        "rel_tol": None,
        "abs_tol": None,
        "reference": None,
        "semantics_id": None,
        "accuracy_mode": None,
        "relaxed_accuracy": False,
    }


def read_numbers(path: pathlib.Path, dtype: str, count: int | None) -> list[float]:
    if path.suffix.lower() in {".csv", ".txt"}:
        text = path.read_text(encoding="utf-8")
        values = []
        for raw in text.split():
            for token in raw.replace(",", " ").split():
                values.append(float(token))
        return values[:count] if count else values
    size = struct.calcsize("f" if dtype == "f32" else "d")
    data = path.read_bytes()
    if count is None:
        count = len(data) // size
    else:
        count = min(count, len(data) // size)
    fmt = "<" + ("f" if dtype == "f32" else "d") * count
    return list(struct.unpack(fmt, data[: count * size]))


def compare_files(
    actual: pathlib.Path,
    expected: pathlib.Path,
    *,
    dtype: str = "f32",
    count: int | None = None,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-4,
) -> dict[str, Any]:
    got = read_numbers(actual, dtype, count)
    want = read_numbers(expected, dtype, count)
    if count is not None:
        if len(got) < count or len(want) < count:
            return {
                "status": "SHAPE_MISMATCH",
                "errors": None,
                "total": None,
                "max_abs_err": None,
                "max_rel_err": None,
                "actual_count": len(got),
                "expected_count": len(want),
                "prefix": count,
                "dtype": dtype,
                "rel_tol": rel_tol,
                "abs_tol": abs_tol,
            }
        got = got[:count]
        want = want[:count]
    if len(got) != len(want):
        return {
            "status": "SHAPE_MISMATCH",
            "errors": None,
            "total": None,
            "max_abs_err": None,
            "max_rel_err": None,
            "actual_count": len(got),
            "expected_count": len(want),
            "dtype": dtype,
            "rel_tol": rel_tol,
            "abs_tol": abs_tol,
        }
    total = len(got)
    errors = 0
    max_abs = 0.0
    max_rel = 0.0
    for i in range(total):
        d = abs(got[i] - want[i])
        max_abs = max(max_abs, d)
        denom = max(abs(got[i]), abs(want[i]), 1e-30)
        max_rel = max(max_rel, d / denom)
        bound = rel_tol * max(abs(got[i]), abs(want[i])) + abs_tol
        if not (d <= bound):
            errors += 1
    return {
        "errors": errors,
        "total": total,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "reference": "cpu_file",
        "semantics_id": None,
        "accuracy_mode": None,
        "relaxed_accuracy": False,
        "status": "PASS" if total and errors == 0 else "FAIL",
        "dtype": dtype,
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
    }


def validate_correctness_contract(
    compare: dict[str, Any],
    requested_rel: float,
    requested_abs: float,
    *,
    expected_semantics_id: str | None = None,
) -> tuple[str, str]:
    """Return (correctness_status, accuracy_class).

    Accuracy classes:
      matched       - PASS with executable-reported tolerance equal to request
      relaxed_matched - PASS but executable used a looser or different tolerance
      fastest_only  - FAIL, operator semantics confirmed, relaxed accuracy mode
      invalid       - FAIL without confirmed semantics / relaxed-accuracy marker
      unknown       - no verification or no contract fields
    """
    status = compare.get("status")
    if status == "NO_VERIFY":
        return "NO_VERIFY", "unknown"

    reported_rel = compare.get("rel_tol")
    reported_abs = compare.get("abs_tol")
    if reported_rel is None or reported_abs is None:
        return "CORRECTNESS_CONTRACT_MISSING", "unknown"

    tol_ok = (
        abs(float(reported_rel) - requested_rel) <= 1e-12
        and abs(float(reported_abs) - requested_abs) <= 1e-12
    )
    if not tol_ok:
        return "CORRECTNESS_CONTRACT_MISMATCH", "unknown"

    semantics_id = compare.get("semantics_id")
    if expected_semantics_id is not None and semantics_id != expected_semantics_id:
        return "SEMANTICS_MISMATCH", "invalid"

    if status == "PASS":
        if compare.get("relaxed_accuracy"):
            return "PASS_RELAXED", "relaxed_matched"
        return "PASS", "matched"

    if status == "FAIL":
        if semantics_id and compare.get("relaxed_accuracy"):
            return "FAIL_RELAXED", "fastest_only"
        return "FAIL", "invalid"

    return str(status), "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", type=pathlib.Path)
    parser.add_argument("--actual", type=pathlib.Path)
    parser.add_argument("--expected", type=pathlib.Path)
    parser.add_argument("--dtype", default="f32", choices=["f32", "f64"])
    parser.add_argument("--count", type=int)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--out", type=pathlib.Path, default=ARTIFACTS_DIR / "compare.json")
    args = parser.parse_args()

    if args.stdout:
        result = parse_stdout(args.stdout.read_text(encoding="utf-8", errors="ignore"))
    elif args.actual and args.expected:
        result = compare_files(
            args.actual,
            args.expected,
            dtype=args.dtype,
            count=args.count,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
        )
    else:
        parser.error("provide --stdout or both --actual and --expected")

    if result.get("reference") is None:
        result["reference"] = "cpu"
    result["requested_rel_tol"] = args.rel_tol
    result["requested_abs_tol"] = args.abs_tol
    result["correctness_status"], result["accuracy_class"] = validate_correctness_contract(
        result,
        args.rel_tol,
        args.abs_tol,
    )
    write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
