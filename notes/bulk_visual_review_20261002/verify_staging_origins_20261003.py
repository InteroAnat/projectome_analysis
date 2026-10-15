"""Compare current review artifacts with the independent160-UID source ledger."""

import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTES = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def row_reference_matches(actual, source_path, expected, uid):
    pieces = actual.split("|")
    if len(pieces) != 4:
        return False
    path = Path(pieces[0])
    if not path.is_absolute():
        path = ROOT / path
    sample, neuron = uid.split("|", 1)
    return (
        path.resolve() == Path(source_path).resolve()
        and pieces[1:3] == [sample, neuron]
        and pieces[3] == "row" + str(expected)
    )


def main():
    ledger = json.loads((NOTES / "expected_staging_origins_20261003.json").read_text())
    problems = []
    unchanged_sources = 0
    for source_id, source in ledger["sources"].items():
        path = Path(source["path"])
        if digest(path) != source["sha256"]:
            problems.append(f"source hash changed: {source_id}")
        else:
            unchanged_sources += 1
    tables = {
        "manifest": ROOT / "group_analysis/visual_review_20261002/manifest/identity_manifest.csv",
        "correction_table": ROOT / "group_analysis/visual_review_20261002/tables/correction_table.csv",
    }
    checked = {}
    for name, path in tables.items():
        before = digest(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if frame["UID"].duplicated().any():
            problems.append(f"{name}: duplicate UID")
        rows = frame.set_index("UID").to_dict("index")
        count = 0
        for expected in ledger["records"]:
            uid = expected["UID"]
            row = rows.get(uid)
            if row is None:
                problems.append(f"{name}: missing {uid}")
                continue
            count += 1
            fields = {
                "staging_refined_label": expected["expected_staging_refined_label"],
                "staging_harmonized_region_source": expected["expected_staging_label_origin"],
                "refined_region_source": expected["earlier_recovery_label_origin"],
            }
            for field, value in fields.items():
                if row.get(field) != value:
                    problems.append(f"{name}: {uid} {field} actual={row.get(field)!r} expected={value!r}")
            for reference in expected["source_rows"]:
                source_id = reference["source_id"]
                source = ledger["sources"][source_id]
                if source_id == "staging_harmonized":
                    if row.get("staging_harmonized_neuron_uid") != reference["row_key"]:
                        problems.append(f"{name}: {uid} harmonized NeuronUID mismatch")
                    field = "staging_harmonized_row_uid"
                    ordinal = reference["excel_row_1based"]
                elif source_id == "recovery_csv":
                    field = "refined_row_uid"
                    ordinal = reference["record_ordinal_with_header_1based"]
                else:
                    continue
                if not row_reference_matches(row.get(field, ""), source["path"], ordinal, uid):
                    problems.append(f"{name}: {uid} {field} mismatch")
        after = digest(path)
        if before != after:
            problems.append(f"{name}: changed during read; rerun after writer finishes")
        checked[name] = {"rows": len(frame), "expected_uids_checked": count,
                         "sha256": after, "stable_during_read": before == after}
    report = {
        "agent": "Codex", "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Read-only source/manifest/CSV comparison; only this verification JSON written",
        "expected_uids": len(ledger["records"]), "unchanged_source_files": unchanged_sources,
        "tables": checked, "problems": problems,
        "scientific_acceptance": False,
    }
    (NOTES / "staging_origin_verification_20261003.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "problems"}, indent=2))
    print("problems", len(problems), problems[:10])
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
