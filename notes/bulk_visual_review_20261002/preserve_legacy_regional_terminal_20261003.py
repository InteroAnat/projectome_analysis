"""Preserve the live legacy batch's terminal ledger before a successor replaces it."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--max-seconds", type=float, default=21600)
args = parser.parse_args()
ledger_path = ROOT / "group_analysis/visual_review_20261002/manifest/regional_context_run_20261003.json"
observer_path = HERE / f"legacy_terminal_watch_{args.pid}_20261003.json"
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.WaitForSingleObject.restype = wintypes.DWORD
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
handle = kernel.OpenProcess(0x00100000, False, args.pid)  # SYNCHRONIZE only
if not handle:
    raise ctypes.WinError(ctypes.get_last_error())
started = time.monotonic()
seen_mtime = None
last_ledger = None
def atomic_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
def record(state, **extra):
    atomic_json(observer_path, {
        "observed_utc": datetime.now(timezone.utc).isoformat(), "observer_state": state,
        "target_pid": args.pid, "ownership": "OpenProcess SYNCHRONIZE handle; no process control",
        "max_seconds": args.max_seconds, "ledger_modified": False, **extra})
record("watching")
print(json.dumps({"observer_state": "watching", "target_pid": args.pid}), flush=True)
try:
    while time.monotonic() - started < args.max_seconds:
        stamp = ledger_path.stat().st_mtime_ns
        if stamp != seen_mtime:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger["pid"] != args.pid:
                if last_ledger is not None:
                    fallback = HERE / f"legacy_last_observation_{args.pid}_20261003.json"
                    atomic_json(fallback, last_ledger)
                record("terminal_window_missed", observed_successor_pid=ledger["pid"])
                raise SystemExit("Successor observed before target terminal ledger; no terminal state inferred")
            seen_mtime, last_ledger = stamp, ledger
            if ledger["state"].startswith("terminal_"):
                stamp_name = datetime.fromisoformat(ledger["started_utc"]).strftime("%Y%m%dT%H%M%S%fZ")
                target = HERE / f"terminal_regional_run_{stamp_name}_{args.pid}.json"
                if target.exists() and json.loads(target.read_text(encoding="utf-8")) != ledger:
                    raise RuntimeError("Saved terminal ledger differs; preserve both for inspection")
                atomic_json(target, ledger)
                record("captured_terminal", terminal_state=ledger["state"],
                       terminal_record=str(target.relative_to(ROOT)),
                       sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                       selected_identities=len(ledger["identities"]))
                print(json.dumps({"observer_state": "captured_terminal", "terminal_record": str(target)}), flush=True)
                break
        process_state = kernel.WaitForSingleObject(handle, 0)
        if process_state == 0:
            # Re-read once on exit, even if a timestamp read raced with the final write.
            seen_mtime = None
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger.get("pid") == args.pid and ledger["state"].startswith("terminal_"):
                continue
            record("process_exited_without_terminal", observed_ledger_pid=ledger.get("pid"),
                   observed_ledger_state=ledger.get("state"))
            raise SystemExit("Target exited without a captured terminal record")
        if process_state != 258:
            raise ctypes.WinError(ctypes.get_last_error())
        time.sleep(0.5)
    else:
        record("observation_timeout", target_still_live=kernel.WaitForSingleObject(handle, 0) == 258)
        print(json.dumps({"observer_state": "observation_timeout", "target_pid": args.pid}), flush=True)
finally:
    kernel.CloseHandle(handle)
