"""Read back one completed sample against current local phase-audit hashes."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "group_analysis/visual_review_20261002"


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_unavailable_root(report, raw_path):
    """Validate only the recorded own-root 404 gate; never accept an image."""
    roots = []
    with raw_path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            values = text.split()
            if values[6] == "-1":
                roots.append([float(value) for value in values[2:5]])
    assert len(roots) == 1 and all(math.isfinite(value) for value in roots[0])
    assert roots[0] == report["native_soma_xyz_um"]
    root_cube = [math.floor(value / (spacing * size))
                 for value, spacing, size in zip(roots[0], (0.65, 0.65, 3.0), (360, 360, 90))]
    assert root_cube == report["central_cube_xyz"]
    assert report["status"] == "context_unavailable"
    assert report["central_cube_loaded"] is False and report["root_pixel_covered"] is False
    assert report["coverage_fraction"] == 0 and not report["loaded"] and not report["source_paths"]
    assert report["missing"] == [root_cube] and not report["invalid"]
    requested = {tuple(cube) for cube in report["requested"]}
    unattempted = {tuple(cube) for cube in report["unattempted"]}
    assert len(requested) == len(report["requested"]) == report["cube_count"]
    assert len(unattempted) == len(report["unattempted"])
    assert tuple(root_cube) in requested and tuple(root_cube) not in unattempted
    assert requested == unattempted | {tuple(root_cube)}
    assert report["source_errors"] == [{"cube_xyz": root_cube, "reason": "HTTP Error 404: Not Found"}]
    return root_cube


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, help="Separate report path for a fresh terminal readback")
    parser.add_argument("--allow-unavailable-root", action="append", default=[], metavar="UID",
                        help="Explicit UID whose own-root 404 must be validated as unavailable")
    args = parser.parse_args()
    manifest = OUT / "manifest/identity_manifest.csv"
    manifest_sha = digest(manifest)
    ledger_path = OUT / "manifest/regional_context_run_20261003.json"
    ledger = read_json(ledger_path)
    assert ledger["manifest_sha256"] == manifest_sha
    summary_path = HERE / "successor_regional_readback_20261003.json"
    summary_sha = digest(summary_path)
    summary = read_json(summary_path)
    assert not summary["failures"] and summary["manifest_uid_set_matches"] is True
    assert summary["manifest_sha256"] == manifest_sha
    assert summary["producer_pid"] == ledger["pid"]
    verified = set(summary["already_checked"])
    verified.update(record["uid"] for record in summary["newly_checked"])
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["SampleID"] == args.sample]
    uids = [row["UID"] for row in rows]
    assert uids and len(uids) == len(set(uids))
    allowed_unavailable = set(args.allow_unavailable_root)
    assert len(allowed_unavailable) == len(args.allow_unavailable_root)
    assert allowed_unavailable <= set(uids)
    assert set(uids) == {uid for uid in ledger["identities"] if uid.startswith(args.sample + "|")}
    hashes, records = {}, []
    for uid in uids:
        state = ledger["identities"][uid]
        assert state["state"] in {"attempt_finished", "reused_existing_derived"}, uid
        if uid in allowed_unavailable:
            assert state["state"] == "attempt_finished" and state["status"] == "context_unavailable", uid
            assert uid not in verified, uid
        else:
            assert state["status"] in {"derived_highres", "derived_partial"}, uid
            assert uid in verified, uid
        filename = uid.split("|", 1)[1]
        stem = Path(filename).stem
        report_path = OUT / "manifest" / f"derived_context_pilot_{args.sample}_{stem}.json"
        report_sha = digest(report_path)
        report = read_json(report_path)
        assert report["UID"] == uid and report["status"] == state["status"], uid
        assert report["requested_field_um_xyz"] == [4000.0, 4000.0, 30.0], uid
        assert not report["nmt_used"] and not report["borrowed_sample"], uid
        raw_path = OUT / "swc_raw" / args.sample / filename
        raw_sha = digest(raw_path)
        assert raw_sha == report["swc_sha256"], uid
        assert report["swc_url"] == f"http://10.10.31.31/swc/newswc/Monkey/{args.sample}/swc_raw/{filename}", uid
        hashes[str(raw_path)] = raw_sha
        if uid in allowed_unavailable:
            root_cube = check_unavailable_root(report, raw_path)
            assert state["unattempted_tiles"] == len(report["unattempted"]), uid
            hashes[str(report_path)] = report_sha
            assert digest(report_path) == report_sha and digest(raw_path) == raw_sha, uid
            records.append({"uid": uid, "status": report["status"], "root_acquired": False,
                            "root_cube_xyz": root_cube, "own_raw_sha256": raw_sha,
                            "report_sha256": report_sha, "missing_tiles": 1,
                            "unattempted_nonroot_tiles": len(report["unattempted"]),
                            "source_errors": report["source_errors"],
                            "image_validation_passed": False,
                            "source_availability_reprobed": False})
            continue
        assert report["central_cube_loaded"] and report["root_pixel_covered"], uid
        assert not report["unattempted"], uid
        phase_hashes = {}
        for phase in ("pilot", "mask", "panel"):
            path = HERE / f"derived_{phase}_readback_{args.sample}_{stem}_20261003.json"
            phase_sha = digest(path)
            audit = read_json(path)
            assert audit["status"] == "passed" and audit["uid"] == uid and audit["stable_during_read"], uid
            assert audit["artifact_hashes"].get(str(report_path)) == report_sha, uid
            for name, expected in audit["artifact_hashes"].items():
                if name not in hashes:
                    hashes[name] = digest(name)
                assert hashes[name] == expected, (uid, phase, name)
            if phase == "mask":
                assert audit["root_acquired"] and audit["unavailable_intensity_zero"], uid
                assert audit["coverage_fraction"] == report["coverage_fraction"], uid
            if phase == "panel":
                assert audit["pngs_fully_decoded"] == 6 and audit["unmarked_pixels_checked"] > 0, uid
                assert audit["maximum_display_error_8bit"] <= 1, uid
            assert digest(path) == phase_sha, uid
            phase_hashes[phase] = phase_sha
        assert digest(report_path) == report_sha and digest(raw_path) == raw_sha, uid
        records.append({"uid": uid, "status": report["status"],
                        "coverage_fraction": report["coverage_fraction"],
                        "requested_field_um_xyz": report["requested_field_um_xyz"],
                        "root_acquired": True, "missing_tiles": len(report["missing"]),
                        "invalid_tiles": len(report["invalid"]), "unattempted_tiles": 0,
                        "own_raw_sha256": raw_sha, "report_sha256": report_sha,
                        "phase_audit_sha256": phase_hashes})
    assert digest(manifest) == manifest_sha
    assert digest(summary_path) == summary_sha, "Summary changed during readback; retry against next checkpoint"
    fresh_ledger = read_json(ledger_path)
    assert fresh_ledger["run_id"] == ledger["run_id"] and fresh_ledger["manifest_sha256"] == manifest_sha
    for uid in uids:
        assert fresh_ledger["identities"][uid] == ledger["identities"][uid], uid
    result = {"observed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "scope": f"All {len(uids)} selected {args.sample} regional fields",
              "sample": args.sample, "selected": len(uids), "run_id": ledger["run_id"],
              "manifest_sha256": manifest_sha, "independent_audit_sha256": summary_sha,
              "outcome_counts": dict(Counter(record["status"] for record in records)),
              "usable_fields_verified": len(records) - len(allowed_unavailable),
              "explicit_unavailable_roots": sorted(allowed_unavailable),
              "current_artifact_hashes_checked": len(hashes), "network_requests": 0,
              "source_pixels_recomputed_in_this_readback": False,
              "anatomical_acceptance": False, "records": records}
    target = args.output or HERE / f"all_{args.sample}_regional_milestone_20261004.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
