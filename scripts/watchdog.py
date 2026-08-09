#!/usr/bin/env python3
"""Single-process watchdog: timeout kill, exit code, minidump, Event Log/TDR.

Launch each target executable in its own process, enforce a timeout, kill the
process tree on timeout, snapshot new minidumps, and query the System Event Log
for Kernel-Power/Display entries including ``VIDEO_TDR_FAILURE`` bugcheck
``0x116``.

Example:
    python scripts/watchdog.py --exe build/f32_gemv.exe --iterations 10 \
        --timeout 30 --label f32_gemv --out artifacts/watchdog
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from harness_common import (
    ARTIFACTS_DIR,
    BUILD_DIR,
    ROOT,
    detect_environment,
    run_powershell,
    write_json,
)

CREATE_NEW_PROCESS_GROUP = 0x00000200
TDR_PATTERNS = ["0x00000116", "0x116", "VIDEO_TDR_FAILURE"]
WATCH_EVENT_IDS = {41, 1001, 4101, 6008}


def dump_dirs() -> list[pathlib.Path]:
    dirs = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(pathlib.Path(local) / "CrashDumps")
    dirs.append(pathlib.Path(r"C:\Windows\Minidump"))
    return [d for d in dirs if d.is_dir()]


def dump_snapshot() -> dict[str, float]:
    snap = {}
    for d in dump_dirs():
        try:
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in {".dmp", ".mdmp"}:
                    snap[str(p)] = p.stat().st_mtime
        except OSError:
            continue
    return snap


def copy_new_dumps(
    before: dict[str, float], after: dict[str, float], dest: pathlib.Path
) -> list[str]:
    copied = []
    for path_str, mtime in after.items():
        if before.get(path_str) == mtime:
            continue
        src = pathlib.Path(path_str)
        try:
            dest.mkdir(parents=True, exist_ok=True)
            out = dest / src.name
            shutil.copy2(src, out)
            copied.append(str(out))
        except OSError as exc:
            copied.append(f"<copy failed: {src}: {exc}>")
    return copied


def query_events(since: dt.datetime) -> list[dict]:
    since_str = since.astimezone().isoformat()
    script = f"""
$start = [datetime]'{since_str}'
Get-WinEvent -FilterHashtable @{{LogName='System'; StartTime=$start}} -ErrorAction SilentlyContinue |
  Where-Object {{ $_.Id -in 41,1001,4101,6008 -or $_.ProviderName -eq 'Display' }} |
  ForEach-Object {{
    [pscustomobject]@{{
      Time = $_.TimeCreated.ToString('o')
      Id = $_.Id
      Provider = $_.ProviderName
      Level = $_.LevelDisplayName
      Message = ($_.Message -replace "`r`n", ' ')
    }}
  }} |
  ConvertTo-Json -Compress
"""
    cp = run_powershell(script)
    text = (cp.stdout or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [{"raw": text}]
    if isinstance(parsed, dict):
        parsed = [parsed]
    return parsed


def is_tdr(event: dict) -> bool:
    message = str(event.get("Message", ""))
    eid = event.get("Id")
    if any(p.lower() in message.lower() for p in TDR_PATTERNS):
        return True
    if eid == 4101:
        return True
    if eid == 1001 and "116" in str(event):
        return True
    return False


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            timeout=20,
        )
    except Exception:
        pass


def run_once(
    exe: pathlib.Path,
    run_dir: pathlib.Path,
    timeout: float,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().astimezone()
    before = dump_snapshot()
    out_path = run_dir / "stdout.txt"
    err_path = run_dir / "stderr.txt"
    timed_out = False
    from harness_common import find_setvars

    setvars = find_setvars()
    with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
        launch = f'"{setvars}" >nul 2>&1 && "{exe}"'
        proc = subprocess.Popen(
            launch,
            cwd=str(exe.parent),
            stdout=fo,
            stderr=fe,
            creationflags=CREATE_NEW_PROCESS_GROUP,
            shell=True,
        )
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_tree(proc)
            try:
                proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                pass

    elapsed_ms = (dt.datetime.now().astimezone() - started).total_seconds() * 1000.0
    after = dump_snapshot()
    minidumps = copy_new_dumps(before, after, run_dir / "minidumps")
    events = query_events(started)
    tdr_events = [e for e in events if is_tdr(e)]
    returncode = proc.returncode if proc.returncode is not None else -1

    if timed_out:
        verdict = "TIMEOUT_KILLED"
    elif returncode == 0:
        verdict = "PASS"
    else:
        verdict = f"EXIT_{returncode}"
    if tdr_events:
        verdict += "+TDR"
    if minidumps:
        verdict += "+MINIDUMP"

    summary = {
        "started": started.isoformat(),
        "elapsed_ms": round(elapsed_ms, 3),
        "timed_out": timed_out,
        "returncode": returncode,
        "verdict": verdict,
        "minidumps": minidumps,
        "events": events,
        "tdr_events": tdr_events,
        "stdout": str(out_path),
        "stderr": str(err_path),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=pathlib.Path, default=ARTIFACTS_DIR / "watchdog")
    args = parser.parse_args()

    env = detect_environment()
    runs = []
    labels = args.label or [p.stem for p in args.exe]
    if len(labels) != len(args.exe):
        parser.error("--label count must match --exe count")

    for exe, label in zip(args.exe, labels):
        exe = pathlib.Path(exe)
        if not exe.is_absolute():
            if exe.parts and exe.parts[0] == "build":
                exe = ROOT / exe
            else:
                exe = BUILD_DIR / exe
        for iteration in range(1, args.iterations + 1):
            run_dir = args.out / "runs" / label / f"it{iteration:03d}"
            summary = run_once(exe, run_dir, args.timeout)
            summary["target"] = label
            summary["iteration"] = iteration
            runs.append(summary)
            print(
                f"[{label:>24} it={iteration:>3}] {summary['verdict']:<28} "
                f"rc={summary['returncode']:>4} {summary['elapsed_ms']:>10.1f} ms"
            )
            for tdr in summary.get("tdr_events", []):
                print(f"  TDR: {tdr.get('Time')} id={tdr.get('Id')} msg={tdr.get('Message')}")
            for dump in summary.get("minidumps", []):
                print(f"  minidump: {dump}")

    report = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "iterations": args.iterations,
        "timeout_seconds": args.timeout,
        "environment": env,
        "runs": runs,
    }
    write_json(args.out / "report.json", report)
    lines = [
        "# Watchdog report",
        "",
        f"- device: {env.get('device')}",
        f"- driver: {env.get('driver_version')}",
        f"- oneAPI: {env.get('oneapi_version')}",
        "",
        "| target | iteration | verdict | returncode | elapsed_ms | dumps | tdr |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        lines.append(
            f"| {run['target']} | {run['iteration']} | {run['verdict']} | "
            f"{run['returncode']} | {run['elapsed_ms']} | {len(run['minidumps'])} | "
            f"{len(run['tdr_events'])} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
