#!/usr/bin/env python3
"""Build the example A770 kernels with oneAPI DPC++.

Example:
    python scripts/build.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from harness_common import (
    ARTIFACTS_DIR,
    BUILD_DIR,
    ROOT,
    find_oneapi_root,
    find_setvars,
    run_in_oneapi,
    write_json,
)


def target_table() -> list[dict[str, Any]]:
    return [
        {
            "name": "f32_gemv",
            "src": ROOT / "examples" / "f32_gemv.cpp",
            "extra": "",
        },
        {
            "name": "f32_gemv_onednn",
            "src": ROOT / "examples" / "f32_gemv_onednn.cpp",
            "extra": None,  # filled from the oneAPI root
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["f32_gemv", "f32_gemv_onednn"])
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    root = find_oneapi_root()
    dnnl_include = root / "dnnl" / "latest" / "include"
    dnnl_lib = root / "dnnl" / "latest" / "lib" / "dnnl.lib"
    setvars = find_setvars()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {}
    targets = [t for t in target_table() if not args.target or t["name"] == args.target]
    for target in targets:
        exe = BUILD_DIR / f"{target['name']}.exe"
        log = BUILD_DIR / f"{target['name']}.build.log"
        extra = target["extra"]
        if extra is None:
            extra = f'/I "{dnnl_include}" "{dnnl_lib}"'
        command = (
            f'icx-cl /fsycl /O2 /EHsc "{target["src"]}" {extra} /Fe:"{exe}"'
        )
        print(f"building {target['name']} ...")
        cp = run_in_oneapi(command, cwd=ROOT, timeout=args.timeout)
        log.write_text((cp.stdout or "") + (cp.stderr or ""), encoding="utf-8")
        status[target["name"]] = {
            "ok": cp.returncode == 0,
            "exit_code": cp.returncode,
            "exe": str(exe),
            "log": str(log),
            "source": str(target["src"]),
        }
        print(f"  {'OK' if cp.returncode == 0 else 'FAIL'} -> {exe}")
        if cp.returncode != 0:
            print(log.read_text(encoding="utf-8", errors="ignore"))

    write_json(ARTIFACTS_DIR / "build_status.json", status)
    print(f"build status: {ARTIFACTS_DIR / 'build_status.json'}")
    return 0 if all(v["ok"] for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
