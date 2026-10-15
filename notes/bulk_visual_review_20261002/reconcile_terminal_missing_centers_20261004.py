"""Reconcile every earlier missing soma center after both source jobs finish."""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import psutil

from verify_sample_regional_milestone_20261004 import HERE, OUT, check_unavailable_root, digest, read_json


def main():
    ledger_path = OUT / "manifest/regional_context_run_20261003.json"
    ledger = read_json(ledger_path)
    assert ledger["state"] == "terminal_scope_attempted", "Full regional scope is not terminal"
    assert not psutil.pid_exists(ledger["pid"]), "Regional producer still exists"
    watcher_path = HERE / "regional_checkpoint_watcher_20261003.json"
    watcher = read_json(watcher_path)
    assert watcher["state"] == "terminal_verified"
    assert watcher["run_id"] == ledger["run_id"] and watcher["producer_pid"] == ledger["pid"]
    queue_path = HERE / "copied_source_audit_queue_20261003.json"
    queue = read_json(queue_path)
    assert queue["state"] == "source_audit_passed", "Copied TIFF source verification is not complete"
    assert queue["regional_run_id"] == ledger["run_id"] and queue["producer_pid"] == ledger["pid"]
    assert queue["exit_code"] == 0
    assert not psutil.pid_exists(queue["source_audit_pid"]), "Copied source verifier still exists"
    manifest = OUT / "manifest/identity_manifest.csv"
    manifest_sha = digest(manifest)
    assert ledger["manifest_sha256"] == manifest_sha
    assert watcher["manifest_sha256"] == queue["manifest_sha256"] == manifest_sha
    assert len(ledger["identities"]) == 531
    assert all(record["state"] in {"attempt_finished", "reused_existing_derived"}
               for record in ledger["identities"].values())
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest_uids = {row["UID"] for row in rows}
    assert len(rows) == len(manifest_uids) == 832
    inputs = [manifest, ledger_path, watcher_path, queue_path]
    audits = {}
    for name in ("all_soma_source_pixel_readback_20261003.json",
                 "copied_context_package_readback_20261003.json",
                 "copied_context_source_pixel_readback_20261003.json"):
        path = HERE / name
        audit = read_json(path)
        assert audit["status"] == "passed" and not audit["failures"]
        assert audit["manifest_sha256"] == manifest_sha
        assert len({record["uid"] for record in audit["records"]}) == len(audit["records"])
        audits[name] = audit
        inputs.append(path)
    soma = audits["all_soma_source_pixel_readback_20261003.json"]
    copied = audits["copied_context_package_readback_20261003.json"]
    source = audits["copied_context_source_pixel_readback_20261003.json"]
    assert source["regional_run_id"] == ledger["run_id"] and source["selected_identities"] == 301
    assert source["local_package_audit_sha256"] == digest(HERE / "copied_context_package_readback_20261003.json")
    assert {record["uid"] for record in soma["records"]} == manifest_uids
    copied_records = {record["uid"]: record for record in copied["records"]}
    source_records = {record["uid"]: record for record in source["records"]}
    assert len(copied_records) == 301 and copied_records.keys() == source_records.keys()
    assert digest(HERE / "copied_context_source_pixel_readback_20261003.json") == queue["report_sha256"]
    earlier = read_json(HERE / "center_unavailable_reconciliation_20261004.json")
    inputs.append(HERE / "center_unavailable_reconciliation_20261004.json")
    missing = [record for record in soma["records"] if record["outcome"] == "center_unavailable"]
    assert len(missing) == 20
    assert {record["uid"] for record in missing} == {record["uid"] for record in earlier["records"]}
    hashes = {str(path): digest(path) for path in inputs}
    records = []
    for old in missing:
        uid = old["uid"]
        sample, filename = uid.split("|", 1)
        raw = OUT / "swc_raw" / sample / filename
        assert digest(raw) == old["raw_sha256"], uid
        provenance_path = OUT / sample / Path(filename).stem / "provenance.json"
        provenance = read_json(provenance_path)
        metadata = provenance["soma_metadata"]
        assert provenance["coverage_status"] == "center_unavailable", "Soma changed; rerun full Soma audit"
        assert metadata["loaded"] == [] and metadata["missing"] == [old["root_cube"]], uid
        native = [provenance[f"native_soma_{axis}_um"] for axis in "xyz"]
        cube = [math.floor(value / (spacing * size))
                for value, spacing, size in zip(native, (0.65, 0.65, 3.0), (360, 360, 90))]
        assert cube == old["root_cube"] and metadata["center_xyz_um"] == native, uid
        hashes[str(raw)] = digest(raw)
        hashes[str(provenance_path)] = digest(provenance_path)
        record = {"uid": uid, "earlier_soma_root_cube": cube, "soma_center_still_unavailable": True}
        if uid in ledger["identities"]:
            state = ledger["identities"][uid]
            assert state["state"] == "attempt_finished" and state["status"] == "context_unavailable", \
                "Regional root recovered or changed; reassess/regenerate Soma before completion"
            report_path = OUT / "manifest" / f"derived_context_pilot_{sample}_{Path(filename).stem}.json"
            report = read_json(report_path)
            assert report["UID"] == uid and report["swc_sha256"] == old["raw_sha256"]
            assert report["swc_url"] == f"http://10.10.31.31/swc/newswc/Monkey/{sample}/swc_raw/{filename}"
            assert not report["nmt_used"] and not report["borrowed_sample"]
            assert check_unavailable_root(report, raw) == cube
            hashes[str(report_path)] = digest(report_path)
            record.update(regional_root_recovered=False, regional_root_failure_checked=True,
                          recorded_source_errors=report["source_errors"], usable_copied_context=False)
        else:
            local, pixel = copied_records[uid], source_records[uid]
            assert local["usable_context"] and pixel["outcome"] == "source_pixels_equal", uid
            assert pixel["stable_during_read"] and pixel["voxels_checked"] > 0, uid
            assert pixel["artifact_hashes"] == local["artifact_hashes"], uid
            for path, expected in pixel["artifact_hashes"].items():
                hashes.setdefault(path, digest(path))
                assert hashes[path] == expected, uid
            record.update(regional_root_recovered=None, regional_root_failure_checked=False,
                          usable_copied_context=True, copied_source_voxels_checked=pixel["voxels_checked"])
        records.append(record)
    assert sum(record["usable_copied_context"] for record in records) == 17
    assert sum(record["regional_root_failure_checked"] for record in records) == 3
    assert all(digest(path) == expected for path, expected in hashes.items()), "Inputs changed during reconciliation"
    result = {"status": "passed", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": ledger["run_id"], "manifest_sha256": manifest_sha,
              "earlier_missing_centers_reconciled": 20, "source_verified_copied_contexts": 17,
              "dated_regional_root_failures": 3, "regional_root_recoveries": 0,
              "requires_soma_regeneration": False, "requires_terminal_reassessment": False,
              "server_requests_in_this_reconciliation": 0,
              "global_source_availability_reprobed": False, "anatomical_acceptance": False,
              "current_input_hashes": hashes, "records": records}
    target = HERE / "terminal_missing_center_reconciliation_20261004.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: value for key, value in result.items() if key not in {"records", "current_input_hashes"}}))


if __name__ == "__main__":
    main()
