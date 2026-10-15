"""Wait for one bound regional run, then launch its copied-source audit once."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from verify_all_soma_pixels_20261003 import save, sha
from verify_copied_context_source_pixels_20261003 import HERE, OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--producer-pid", type=int, required=True)
    parser.add_argument("--producer-created", type=float, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    path = HERE / "copied_source_audit_queue_20261003.json"
    assert not path.exists(), "Queue already exists; inspect it before creating another worker"
    verifier = HERE / "verify_copied_context_source_pixels_20261003.py"
    code_hash = sha(verifier)
    state = {"state": "waiting_regional", "queue_pid": os.getpid(), "regional_run_id": args.run_id,
             "producer_pid": args.producer_pid, "producer_created": args.producer_created,
             "manifest_sha256": args.manifest_sha256, "verifier_sha256": code_hash,
             "source_audit_pid": None, "error": None,
             "started_utc": datetime.now(timezone.utc).isoformat()}
    save(path, state)
    start = time.monotonic()
    exited_at = None
    try:
        while True:
            assert time.monotonic() - start < 18 * 3600, "Bound regional wait exceeded 18 hours"
            ledger = json.loads((OUT / "manifest/regional_context_run_20261003.json").read_text())
            assert ledger["run_id"] == args.run_id and ledger["pid"] == args.producer_pid, "Bound regional run changed"
            assert sha(OUT / "manifest/identity_manifest.csv") == args.manifest_sha256, "Manifest changed"
            try:
                process = psutil.Process(args.producer_pid)
                assert abs(process.create_time() - args.producer_created) < .01, "Producer PID was reused"
                live = process.is_running()
            except psutil.NoSuchProcess:
                live = False
            if not live:
                exited_at = exited_at or time.monotonic()
                assert ledger["state"].startswith("terminal_"), "Producer exited without a terminal ledger"
                watcher = json.loads((HERE / "regional_checkpoint_watcher_20261003.json").read_text())
                assert watcher["run_id"] == args.run_id and watcher["producer_pid"] == args.producer_pid
                assert watcher["manifest_sha256"] == args.manifest_sha256
                if watcher["state"] == "terminal_verified":
                    break
                assert watcher["state"] not in {"failed", "stopped_producer_exited_without_terminal"}, "Regional verification failed"
                assert time.monotonic() - exited_at < 1800, "Final regional verification exceeded 30 minutes"
                state["state"] = "waiting_final_regional_verification"
            state["observed_utc"] = datetime.now(timezone.utc).isoformat()
            save(path, state)
            time.sleep(15)
        assert sha(verifier) == code_hash, "Queued verifier code changed"
        stdout_path = HERE / "copied_context_source_pixel_stdout_20261003.log"
        stderr_path = HERE / "copied_context_source_pixel_stderr_20261003.log"
        command = [sys.executable, "-B", str(verifier)]
        state.update(state="launching_source_audit", command=command,
                     stdout=str(stdout_path), stderr=str(stderr_path),
                     observed_utc=datetime.now(timezone.utc).isoformat())
        save(path, state)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=OUT.parents[1], stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(state="source_audit_running", source_audit_pid=child.pid,
                         source_audit_started_utc=datetime.now(timezone.utc).isoformat())
            save(path, state)
            exit_code = child.wait()
        report_path = HERE / "copied_context_source_pixel_readback_20261003.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        assert exit_code == 0 and report.get("status") == "passed", "Copied-source audit did not pass"
        assert report["regional_run_id"] == args.run_id and report["manifest_sha256"] == args.manifest_sha256
        assert report["selected_identities"] == report["checked_identities"] == 301
        state.update(state="source_audit_passed", exit_code=exit_code,
                     report=str(report_path), report_sha256=sha(report_path))
    except Exception as exc:
        state.update(state="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        state["observed_utc"] = datetime.now(timezone.utc).isoformat()
        save(path, state)


if __name__ == "__main__":
    main()
