"""Refresh current gallery/table and independently read back the review artifacts."""
import argparse
import json
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
import pandas as pd
import numpy as np
import visual_review_20261002 as review

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--regional-audit", action="append", type=Path,
                    help="Independent regional readback report; repeat for separate run scopes")
args = parser.parse_args()
audit_paths = args.regional_audit or [HERE / "incremental_regional_readback_20261003.json"]
audited = set()
audit_scopes = []
for audit_path in audit_paths:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("failures") or audit.get("manifest_uid_set_matches") is not True:
        raise ValueError(f"Regional readback is not passing: {audit_path}")
    scope = set(audit["already_checked"]) | {item["uid"] for item in audit["newly_checked"]}
    audited.update(scope)
    audit_scopes.append({"path": str(audit_path), "sha256": review.sha256_file(audit_path),
                         "verified_identities": len(scope), "producer_state": audit["producer_state"],
                         "observed_utc": audit["observed_utc"]})

output_root = review.OUT_ROOT
tables = output_root / "tables"
csv_path, xlsx_path = tables / "correction_table.csv", tables / "correction_table.xlsx"
manifest_path = output_root / "manifest/identity_manifest.csv"
manifest_sha = review.sha256_file(manifest_path)
canonical = review.guard_canonical_baseline(output_root)
annotations = review.merge_human_annotations(review.load_human_annotations(csv_path), review.load_human_annotations(xlsx_path))
archive = tables / "orphaned_annotations.csv"
if archive.is_file():
    annotations = review.merge_human_annotations(annotations, review.load_human_annotations(archive))
before = {"csv": review.sha256_file(csv_path), "xlsx": review.sha256_file(xlsx_path)}
review.write_gallery(output_root)
csv_frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False).set_index("UID")
xlsx_frame = pd.read_excel(xlsx_path, dtype=str, keep_default_na=False).set_index("UID")
manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False).set_index("UID")
numeric = {
    "category_rank", "sample_insula_count", "nmt_nii_x", "nmt_nii_y", "nmt_nii_z",
    "nmt_portal_soma_x_um", "nmt_portal_soma_y_um", "nmt_portal_soma_z_um",
    "context_coverage_fraction", "native_soma_x_um", "native_soma_y_um", "native_soma_z_um",
    "swc_node_count", "context_nonzero_fraction", "soma_nonzero_fraction",
    "context_trace_segments", "soma_trace_segments",
}
numeric_readback = {}
def compare(left, right, columns, label):
    left, right = left.sort_index(), right.sort_index()
    for field in columns:
        if field not in numeric:
            pd.testing.assert_series_equal(left[field], right[field])
            continue
        if not (left[field].eq("") == right[field].eq("")).all():
            raise ValueError(f"{label}: missing numeric values differ in {field}")
        present = left[field].ne("")
        a, b = left.loc[present, field].astype(float), right.loc[present, field].astype(float)
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-12, err_msg=f"{label}: {field}")
        numeric_readback[f"{label}:{field}"] = float(np.abs(a - b).max()) if len(a) else 0.0
for name, frame in (("CSV", csv_frame), ("XLSX", xlsx_frame)):
    if not frame.index.is_unique or set(frame.index) != set(manifest.index):
        raise ValueError(f"{name} identities differ from live manifest")
    compare(manifest, frame, manifest.columns, name)
    for uid, fields in annotations.items():
        if uid not in frame.index:
            continue
        for field, value in fields.items():
            if frame.at[uid, field] != value:
                raise ValueError(f"{name} changed human field {uid} {field}")
compare(csv_frame, xlsx_frame, csv_frame.columns, "CSV_vs_XLSX")
diagnostic_sidecars = []
for uid, row in csv_frame.iterrows():
    diagnostics = json.loads(row["context_source_errors"])
    if not isinstance(diagnostics, dict):
        continue
    target = (output_root / diagnostics["detail_file"]).resolve()
    if output_root.resolve() not in target.parents or not target.is_file():
        raise ValueError(f"Diagnostic sidecar missing or outside review root: {uid}")
    if review.sha256_file(target) != diagnostics["sha256"]:
        raise ValueError(f"Diagnostic sidecar hash differs: {uid}")
    errors = json.loads(target.read_text(encoding="utf-8"))
    sample, filename = uid.split("|", 1)
    provenance = json.loads((output_root / sample / Path(filename).stem / "provenance.json").read_text(encoding="utf-8"))
    expected = (provenance.get("derived_context") or {}).get("source_errors") or []
    if errors != expected or diagnostics["items"] != len(errors):
        raise ValueError(f"Diagnostic sidecar differs from full source provenance: {uid}")
    diagnostic_sidecars.append({"uid": uid, **diagnostics})

class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.sources.append(attrs["src"])
        if tag == "a":
            self.links.append(attrs["href"])

pages = {}
for sample in review.NEW_SAMPLES:
    page = output_root / "gallery" / f"{sample}.html"
    markup = page.read_text(encoding="utf-8")
    parser = Images()
    parser.feed(markup)
    for src in parser.sources:
        target = (page.parent / src).resolve()
        if not target.is_file() or output_root.resolve() not in target.parents:
            raise ValueError(f"Gallery image target missing or outside review root: {src}")
        if target.relative_to(output_root).parts[0] != sample:
            raise ValueError(f"Gallery links another sample's image: {src}")
    for uid in manifest[manifest["SampleID"] == sample].index:
        if uid not in markup:
            raise ValueError(f"Gallery omits UID {uid}")
    pages[sample] = {"sha256": review.sha256_file(page), "image_links_checked": len(parser.sources)}
index = output_root / "gallery/index.html"
parser = Images()
parser.feed(index.read_text(encoding="utf-8"))
expected_links = {f"{sample}.html" for sample in review.NEW_SAMPLES}
if not expected_links <= set(parser.links):
    raise ValueError("Index omits a selected monkey")
if not audited <= set(manifest.index):
    raise ValueError("Regional readback contains identities absent from current manifest")
for uid in audited:
    sample, filename = uid.split("|", 1)
    report = json.loads((output_root / "manifest" / f"derived_context_pilot_{sample}_{Path(filename).stem}.json").read_text(encoding="utf-8"))
    row = csv_frame.loc[uid]
    if row["context_coverage_status"] != report["status"]:
        raise ValueError(f"Table context status differs from audited field: {uid}")
    if abs(float(row["context_coverage_fraction"]) - float(report["coverage_fraction"])) > 1e-12:
        raise ValueError(f"Table coverage fraction differs from audited source: {uid}")
    if json.loads(row["context_requested_field_xyz_um"]) != report["requested_field_um_xyz"]:
        raise ValueError(f"Table requested field differs from audited source: {uid}")
    for field in ("context_unmarked.png", "context_trace.png", "context_coverage.png", "context.nii.gz"):
        if not row[field] or not (output_root / row[field]).is_file():
            raise ValueError(f"Table lacks audited image {uid} {field}")
if review.sha256_file(manifest_path) != manifest_sha or review.canonical_hashes() != canonical:
    raise ValueError("Manifest or canonical inputs changed during refresh")
snapshot = {"observed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
            "manifest_sha256": manifest_sha, "rows": len(manifest), "all_nine_pages_checked": pages,
            "csv_xlsx_equal_with_numeric_readback_tolerance": True,
            "all_manifest_fields_equal_with_numeric_readback_tolerance": True,
            "numeric_tolerance": {"relative": 1e-14, "absolute": 1e-12},
            "numeric_maximum_absolute_errors": numeric_readback,
            "human_fields_preserved": True, "canonical_inputs_unchanged": True,
            "lossless_diagnostic_sidecars_checked": diagnostic_sidecars,
            "independently_audited_fields_exposed": len(audited),
            "regional_readback_scopes": audit_scopes,
            "derived_context_rows_at_refresh": int(csv_frame["context_coverage_status"].isin(review.DERIVED_CONTEXT_STATES).sum()),
            "before_sha256": before, "after_sha256": {"csv": review.sha256_file(csv_path), "xlsx": review.sha256_file(xlsx_path)},
            "regional_worker_may_still_be_running": True, "anatomical_acceptance": False}
(HERE / "review_artifact_refresh_20261003.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in snapshot.items() if key != "all_nine_pages_checked"}))
