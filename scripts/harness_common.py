"""Shared helpers for the A770 experiment harness.

The scripts in this directory deliberately stay small. They speak to the
oneAPI command line through ``setvars`` so an executable built with DPC++ finds
its runtime DLLs, and they keep machine-specific paths out of generated
reports.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
ARTIFACTS_DIR = ROOT / "artifacts"

_ONEAPI_CANDIDATES = [
    r"C:\Program Files (x86)\Intel\oneAPI",
    r"C:\Program Files\Intel\oneAPI",
]


def find_oneapi_root() -> pathlib.Path:
    env_root = os.environ.get("ONEAPI_ROOT")
    if env_root:
        p = pathlib.Path(env_root)
        if p.exists():
            return p
    for candidate in _ONEAPI_CANDIDATES:
        p = pathlib.Path(candidate)
        if p.exists():
            return p
    raise FileNotFoundError("oneAPI installation not found")


def find_setvars() -> pathlib.Path:
    root = find_oneapi_root()
    for candidate in (
        root / "setvars.bat",
        root.parent / "setvars.bat",
        pathlib.Path(r"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("setvars.bat not found")


def run_in_oneapi(
    command: str,
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    setvars = find_setvars()
    full = f'"{setvars}" >nul 2>&1 && {command}'
    return subprocess.run(
        full,
        shell=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_powershell(script: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def video_controller() -> dict[str, Any]:
    script = """
$v = @(Get-CimInstance Win32_VideoController)
$arc = @($v | Where-Object { $_.Name -like '*Arc*' })
if ($arc.Count -gt 0) { $v = $arc }
$item = $v[0]
if ($null -eq $item) { exit 0 }
$item | Select-Object Name,DriverVersion,PNPDeviceID,VideoProcessor | ConvertTo-Json -Compress
"""
    cp = run_powershell(script)
    text = (cp.stdout or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    return parsed


def parse_oneapi_version(icx_output: str) -> str:
    m = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", icx_output)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return "unknown"


def parse_level_zero_version(sycl_ls_output: str) -> str:
    for line in sycl_ls_output.splitlines():
        if "level_zero" not in line.lower():
            continue
        m = re.search(r"\[(\d+\.\d+\.\d+)\]", line)
        if m:
            return m.group(1)
    return "unknown"


def dnnl_version() -> str:
    root = find_oneapi_root()
    header = root / "dnnl" / "latest" / "include" / "oneapi" / "dnnl" / "dnnl_version.h"
    if not header.exists():
        return "unknown"
    text = header.read_text(encoding="utf-8", errors="ignore")
    major = re.search(r"#define DNNL_VERSION_MAJOR\s+(\d+)", text)
    minor = re.search(r"#define DNNL_VERSION_MINOR\s+(\d+)", text)
    patch = re.search(r"#define DNNL_VERSION_PATCH\s+(\d+)", text)
    if major and minor and patch:
        return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
    return "unknown"


def detect_environment() -> dict[str, Any]:
    video = video_controller()
    icx_cp = run_in_oneapi("icx --version")
    icx_output = (icx_cp.stdout or "") + (icx_cp.stderr or "")
    sycl_cp = run_in_oneapi("sycl-ls")
    sycl_output = (sycl_cp.stdout or "") + (sycl_cp.stderr or "")
    return {
        "device": video.get("Name", "unknown"),
        "driver_version": video.get("DriverVersion", "unknown"),
        "pnp_device_id": video.get("PNPDeviceID", "unknown"),
        "video_processor": video.get("VideoProcessor", "unknown"),
        "oneapi_version": parse_oneapi_version(icx_output),
        "icx_version": icx_output.strip(),
        "level_zero_version": parse_level_zero_version(sycl_output),
        "sycl_ls": sycl_output.strip(),
        "dnnl_version": dnnl_version(),
        "captured_at": __import__("datetime").datetime.now().astimezone().isoformat(),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def stats(values: list[float], cv_threshold_pct: float = 10.0) -> dict[str, Any]:
    if not values:
        return {
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "mad": 0.0,
            "cv_pct": 0.0,
            "flag": False,
            "count": 0,
        }
    med = float(statistics.median(values))
    mad = statistics.median([abs(v - med) for v in values])
    mean = statistics.fmean(values)
    cv = (statistics.pstdev(values) / mean * 100.0) if mean else 0.0
    return {
        "median": med,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "mad": mad,
        "cv_pct": cv,
        "flag": bool(cv > cv_threshold_pct),
        "count": len(values),
    }


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: pathlib.Path) -> Any:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
