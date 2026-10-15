"""Verify incremental outputs from one existing regional producer; no downloads."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER = ROOT / "group_analysis/visual_review_20261002/manifest/regional_context_run_20261003.json"
MANIFEST = LEDGER.with_name("identity_manifest.csv")
AUDIT = HERE / "successor_regional_readback_20261003.json"
STATUS = HERE / "regional_checkpoint_watcher_20261003.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interval-seconds", type=float, default=120)
    parser.add_argument("--baseline-audit", action="append", type=Path)
    args = parser.parse_args()
    baseline_audits = args.baseline_audit or [HERE / "terminal_first_regional_readback_20261003.json"]
    for path in baseline_audits:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        if baseline.get("failures") or baseline.get("manifest_uid_set_matches") is not True:
            raise ValueError("Baseline regional audit is not passing: " + str(path))
    if args.interval_seconds < 30:
        parser.error("Checkpoint interval must be at least 30 seconds")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    lock_path = HERE / "regional_checkpoint_watcher_20261003.lock"
    with lock_path.open("a+b") as lock:
        if lock_path.stat().st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        initial = json.loads(LEDGER.read_text(encoding="utf-8"))
        if initial["run_id"] != args.run_id:
            raise ValueError("Producer run identity changed before watcher startup")
        manifest_sha = initial["manifest_sha256"]
        handle = kernel.OpenProcess(0x00100000 | 0x1000, False, initial["pid"])
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        state = {
            "watcher_pid": os.getpid(), "run_id": args.run_id,
            "producer_pid": initial["pid"], "manifest_sha256": manifest_sha,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "watcher_sha256": digest(Path(__file__)), "downloads_started": 0,
            "checkpoints": [], "state": "watching",
            "baseline_audits": [{"path": str(path), "sha256": digest(path)} for path in baseline_audits],
        }

        def save():
            state["observed_utc"] = datetime.now(timezone.utc).isoformat()
            temporary = STATUS.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(STATUS)

        def read_bound():
            ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
            if ledger["run_id"] != args.run_id:
                raise ValueError("Producer run identity changed; watcher will not follow a new run")
            if ledger["manifest_sha256"] != manifest_sha or digest(MANIFEST) != manifest_sha:
                raise ValueError("Producer input manifest changed")
            return ledger

        def command(script, arguments):
            read_bound()
            with (HERE / "regional_checkpoint_watcher_20261003.log").open("ab") as log:
                subprocess.run([sys.executable, "-B", str(HERE / script), *arguments],
                               cwd=ROOT, stdout=log, stderr=log, check=True, timeout=900)
            read_bound()

        try:
            previous_attempts = None
            previous_check = float("-inf")
            save()
            while True:
                ledger = read_bound()
                wait = kernel.WaitForSingleObject(handle, 0)
                if wait not in (0, 258):
                    raise ctypes.WinError(ctypes.get_last_error())
                live = wait == 258
                terminal = ledger["state"].startswith("terminal_")
                attempts = sum(row["state"] in {"attempt_finished", "reused_existing_derived"}
                               for row in ledger["identities"].values())
                state.update(producer_live=live, producer_state=ledger["state"],
                             finished_attempts=attempts, selected_identities=len(ledger["identities"]))
                due = attempts != previous_attempts and (
                    previous_attempts is None or attempts - previous_attempts >= 5
                    or time.monotonic() - previous_check >= args.interval_seconds)
                if due or terminal or not live:
                    command("verify_completed_regional_20261003.py", ["--summary-path", str(AUDIT)])
                    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
                    if audit.get("failures") or audit.get("manifest_uid_set_matches") is not True:
                        raise ValueError("Independent regional verification failed")
                    command("refresh_review_artifacts_20261003.py", [
                        *[arg for path in baseline_audits for arg in ("--regional-audit", str(path))],
                        "--regional-audit", str(AUDIT)])
                    refresh_path = HERE / "review_artifact_refresh_20261003.json"
                    refresh = json.loads(refresh_path.read_text(encoding="utf-8"))
                    if refresh["status"] != "passed":
                        raise ValueError("Gallery/workbook readback did not pass")
                    state["checkpoints"].append({
                        "observed_utc": datetime.now(timezone.utc).isoformat(),
                        "finished_attempts": attempts,
                        "verified_current_run": len(audit["already_checked"]) + len(audit["newly_checked"]),
                        "verified_across_reports": refresh["independently_audited_fields_exposed"],
                        "audit_sha256": digest(AUDIT), "artifact_refresh_sha256": digest(refresh_path),
                        "human_fields_preserved": refresh["human_fields_preserved"],
                        "canonical_inputs_unchanged": refresh["canonical_inputs_unchanged"],
                    })
                    previous_attempts, previous_check = attempts, time.monotonic()
                if terminal or not live:
                    state["state"] = "terminal_verified" if terminal else "stopped_producer_exited_without_terminal"
                    save()
                    break
                save()
                time.sleep(15)
        except Exception as error:
            state.update(state="stopped_error", error=str(error))
            save()
            raise
        finally:
            kernel.CloseHandle(handle)


if __name__ == "__main__":
    main()
