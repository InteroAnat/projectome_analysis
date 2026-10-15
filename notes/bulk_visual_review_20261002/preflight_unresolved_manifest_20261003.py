"""Build an isolated opt-in candidate preview; never replace the live manifest."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
import pandas as pd
import visual_review_20261002 as review

live = review.OUT_ROOT / "manifest/identity_manifest.csv"
before = review.sha256_file(live)
original = pd.read_csv(live, dtype=str).fillna("").set_index("UID")
report = HERE / "unresolved_full_source_validation_20261003.json"
preview = HERE / "unresolved_candidate_preview"
review.build_manifest(preview, unresolved_report=report)
preview_manifest = preview / "manifest/identity_manifest.csv"
first_preview_sha = review.sha256_file(preview_manifest)
review.build_manifest(preview)
if review.sha256_file(preview_manifest) != first_preview_sha:
    raise ValueError("Default regeneration changed the activated extended preview")
candidate = pd.read_csv(preview / "manifest/identity_manifest.csv", dtype=str).fillna("").set_index("UID")
if not candidate.index.is_unique or not set(original.index) <= set(candidate.index):
    raise ValueError("Candidate extension duplicates or drops existing identities")
pd.testing.assert_frame_equal(original.sort_index(), candidate.loc[original.index, original.columns].sort_index())
added = candidate.loc[candidate.index.difference(original.index)]
validation = json.loads(report.read_text(encoding="utf-8"))
expected = {r["uid"] for r in validation["records"] if r.get("graph_validated")}
if set(added.index) != expected:
    raise ValueError("Added identities differ from fully validated unresolved source records")
if not (added["review_category"] == "metadata_unresolved").all():
    raise ValueError("Blank metadata acquired an anatomical category")
if not (added["portal_region_base"] == "").all():
    raise ValueError("Blank atlas metadata changed")
if not (added["selection_reasons"] == "metadata_unresolved_raw_source").all():
    raise ValueError("Unexpected anatomical or cohort reason for unresolved candidate")
if review.sha256_file(live) != before:
    raise ValueError("Live manifest changed during preflight")
snapshot = {
    "observed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
    "original_identities": len(original), "preview_identities": len(candidate),
    "added_metadata_unresolved": len(added), "added_by_sample": added.groupby("SampleID").size().to_dict(),
    "all_original_fields_unchanged": True, "live_manifest_unchanged": True,
    "default_regeneration_retains_extension": True,
    "live_manifest_sha256": before, "preview_manifest_sha256": review.sha256_file(preview / "manifest/identity_manifest.csv"),
    "report_sha256": review.sha256_file(report), "anatomical_acceptance": False,
    "activation": "Only after current producer and Cursor return terminal results; preserve terminal ledger first",
    "code_sha256": {str(path.relative_to(ROOT)): review.sha256_file(path) for path in (
        ROOT / "group_analysis/scripts/visual_review_20261002.py",
        ROOT / "group_analysis/scripts/regional_context_batch_20261003.py",
        ROOT / "main_scripts/fmost_review_lock.py",
        ROOT / "tests/test_visual_review_manifest.py")},
}
(HERE / "unresolved_manifest_preflight_20261003.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
print(json.dumps(snapshot))
