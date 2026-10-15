"""Bind full terminal regional scope to fresh per-sample artifact readbacks."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone

import psutil

from verify_sample_regional_milestone_20261004 import HERE, OUT, digest, read_json


def main():
    manifest = OUT / "manifest/identity_manifest.csv"
    ledger_path = OUT / "manifest/regional_context_run_20261003.json"
    watcher_path = HERE / "regional_checkpoint_watcher_20261003.json"
    summary_path = HERE / "successor_regional_readback_20261003.json"
    inputs = [manifest, ledger_path, watcher_path, summary_path,
              HERE / "verify_sample_regional_milestone_20261004.py"]
    hashes = {str(path): digest(path) for path in inputs}
    ledger, watcher, summary = map(read_json, (ledger_path, watcher_path, summary_path))
    assert ledger["state"] == "terminal_scope_attempted" and not ledger.get("stop_reason")
    assert not psutil.pid_exists(ledger["pid"])
    assert watcher["state"] == "terminal_verified" and not watcher.get("error")
    assert not psutil.pid_exists(watcher["watcher_pid"])
    assert watcher["run_id"] == ledger["run_id"] and watcher["producer_pid"] == ledger["pid"]
    assert not summary["failures"] and summary["manifest_uid_set_matches"]
    assert summary["producer_pid"] == ledger["pid"]
    assert ledger["manifest_sha256"] == watcher["manifest_sha256"] == summary["manifest_sha256"] == hashes[str(manifest)]
    expected = {"250432": 108, "251730": 117, "252383": 104, "252718": 89, "252790": 113}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["SampleID"] in expected]
    uids = {row["UID"] for row in rows}
    assert len(rows) == len(uids) == len(ledger["identities"]) == 531
    assert uids == set(ledger["identities"])
    records, samples = {}, {}
    for sample, selected in expected.items():
        path = HERE / f"terminal_sample_{sample}_regional_readback_20261004.json"
        hashes[str(path)] = digest(path)
        report = read_json(path)
        assert report["status"] == "passed" and report["run_id"] == ledger["run_id"]
        assert report["manifest_sha256"] == hashes[str(manifest)]
        assert report["independent_audit_sha256"] == hashes[str(summary_path)]
        assert report["selected"] == len(report["records"]) == selected
        assert report["network_requests"] == 0 and report["anatomical_acceptance"] is False
        sample_uids = {row["UID"] for row in rows if row["SampleID"] == sample}
        assert sample_uids == {record["uid"] for record in report["records"]}
        for record in report["records"]:
            uid = record["uid"]
            assert uid not in records and ledger["identities"][uid]["status"] == record["status"]
            assert ledger["identities"][uid]["state"] in {"attempt_finished", "reused_existing_derived"}
            records[uid] = record
        samples[sample] = {key: report[key] for key in ("selected", "usable_fields_verified", "outcome_counts", "current_artifact_hashes_checked")}
    assert set(records) == uids
    usable = {uid for uid, record in records.items() if record["root_acquired"]}
    unavailable = uids - usable
    assert len(usable) == 528 and unavailable == {"252718|145.swc", "252718|169.swc", "252718|161.swc"}
    verified = set(summary["already_checked"]) | {record["uid"] for record in summary["newly_checked"]}
    assert verified == usable
    assert all(record["unattempted_tiles"] == 0 for uid, record in records.items() if uid in usable)
    assert all(records[uid]["image_validation_passed"] is False for uid in unavailable)
    assert all(digest(path) == expected_sha for path, expected_sha in hashes.items())
    result = {"status": "passed", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": ledger["run_id"], "selected": 531, "usable_fields_verified": 528,
              "explicit_unavailable_roots": sorted(unavailable),
              "outcome_counts": dict(Counter(record["status"] for record in records.values())),
              "samples": samples, "current_artifact_hashes_checked": sum(sample["current_artifact_hashes_checked"] for sample in samples.values()),
              "producer_exited": True, "watcher_exited": True, "network_requests": 0,
              "source_pixels_recomputed_in_this_readback": False, "anatomical_acceptance": False,
              "input_hashes": hashes}
    target = HERE / "terminal_regional_scope_readback_20261004.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: value for key, value in result.items() if key != "input_hashes"}))


if __name__ == "__main__":
    main()
