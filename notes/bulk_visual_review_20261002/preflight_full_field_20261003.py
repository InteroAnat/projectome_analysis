"""Read-only full-field admission from cached native inputs; no image requests."""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
import regional_context_batch_20261003 as batch
from fmost_derived_context import context_plan


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


output = ROOT / "group_analysis/visual_review_20261002"
manifest = output / "manifest/identity_manifest.csv"
before = digest(manifest)
with manifest.open(encoding="utf-8-sig", newline="") as stream:
    rows = [row for row in csv.DictReader(stream) if row["SampleID"] in batch.review.PENDING_SAMPLES]
records, problems, smaller = [], [], []
for row in rows:
    sample, neuron, uid = row["SampleID"], row["NeuronID"], row["UID"]
    try:
        swc = output / "swc_raw" / sample / neuron
        swc_hash = digest(swc)
        provenance = json.loads((output / sample / Path(neuron).stem / "provenance.json").read_text(encoding="utf-8"))
        if provenance["swc_sha256"] != swc_hash or provenance["UID"] != uid:
            raise ValueError("Cached raw SWC/provenance identity changed")
        native = [provenance[f"native_soma_{axis}_um"] for axis in "xyz"]
        field, plan = batch.requested_field(native)
        _, old_plan = context_plan(native)
        records.append({"uid": uid, "native_root_xyz_um": native, "raw_swc_sha256": swc_hash,
                        "field_um": field, "plan": plan,
                        "preceding_400_cube_plan_status": old_plan.get("status", "admitted")})
        existing = output / "manifest" / f"derived_context_pilot_{sample}_{Path(neuron).stem}.json"
        if existing.exists():
            saved = json.loads(existing.read_text(encoding="utf-8"))
            if any(value < 4000 for value in saved["requested_field_um_xyz"][:2]):
                smaller.append({"uid": uid, "current_field_um_xyz": saved["requested_field_um_xyz"],
                                "required_action": "upgrade to admitted 4 mm field after current producer is terminal"})
    except (OSError, ValueError, KeyError) as error:
        problems.append({"uid": uid, "error": str(error)})
if digest(manifest) != before:
    problems.append({"error": "Manifest changed during readback"})
summary = {"captured_utc": datetime.now(timezone.utc).isoformat(), "selected_identities": len(rows),
           "planned": len(records), "manifest_sha256": before,
           "planned_by_sample": dict(Counter(item["uid"].split("|", 1)[0] for item in records)),
           "full_4mm_admitted": sum(item["field_um"] == 4000 and item["plan"].get("status") != "not_run" for item in records),
           "preceding_bound_rejections": sum(item["preceding_400_cube_plan_status"] == "not_run" for item in records),
           "maximum_working_estimate_bytes": max((item["plan"]["working_bytes_estimate"] for item in records), default=0),
           "maximum_cubes": max((item["plan"]["cube_count"] for item in records), default=0),
           "smaller_fields_to_upgrade": smaller, "problems": problems,
           "image_requests": 0, "anatomical_acceptance": False,
           "scope": "Planning from declared native roots; not acquisition success or physical calibration",
           "records": records}
(HERE / "full_field_preflight_20261003.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in summary.items() if key != "records"}))
if problems:
    raise SystemExit(1)
