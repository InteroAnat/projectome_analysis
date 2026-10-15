"""Refresh marked/trace captions from saved native volumes without source downloads."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
import visual_review_20261002 as review

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--uid", action="append", required=True)
parser.add_argument("--summary-path", type=Path, default=HERE / "derived_caption_refresh_20261003.json")
args = parser.parse_args()
records = []
for uid in dict.fromkeys(args.uid):
    sample, neuron = uid.split("|", 1)
    if sample not in review.NEW_SAMPLES or neuron != review.canon_neuron_id(neuron):
        raise ValueError("Invalid own-sample identity")
    folder = review.OUT_ROOT / sample / Path(neuron).stem
    report_path = review.OUT_ROOT / "manifest" / f"derived_context_pilot_{sample}_{Path(neuron).stem}.json"
    report = json.loads(report_path.read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    if report["status"] not in review.DERIVED_CONTEXT_STATES:
        raise ValueError("No usable derived field to refresh")
    protected = [report_path, folder / "provenance.json", folder / "context.nii.gz",
                 Path(str(folder / "context.nii.gz") + ".json"), folder / "context_unmarked.png",
                 folder / "context_coverage.nii.gz", folder / "context_coverage.png"]
    before = {str(path): review.sha256_file(path) for path in protected}
    volume = np.asanyarray(nib.load(str(folder / "context.nii.gz")).dataobj).transpose(2, 1, 0)
    native, origin, spacing = report["native_soma_xyz_um"], report["origin_xyz_um"], report["derived_spacing_xyz_um"]
    local = review.native_to_index(native, origin, spacing)
    raw_path = review.OUT_ROOT / "swc_raw" / sample / neuron
    if review.sha256_file(raw_path) != provenance["swc_sha256"]:
        raise ValueError("Raw trace source differs from saved provenance")
    rows = review.parse_swc(raw_path.read_text(encoding="utf-8-sig"), source=str(raw_path))
    title = review.derived_context_title(sample, neuron, report, volume.shape, spacing)
    review._draw_marked(volume[int(np.floor(local[2]))], folder / "context_marked.png",
                        local[:2], spacing[:2], title, scale_um=500)
    trace = review._save_trace_panel(volume, local, spacing, rows, origin,
                                    folder / "context_trace.png", title, 500, report["source_plane_ids"])
    if trace != provenance["context_trace_slab"]:
        raise ValueError("Trace slab semantics changed during caption refresh")
    if before != {str(path): review.sha256_file(path) for path in protected}:
        raise ValueError("Caption refresh changed protected data or source metadata")
    records.append({"uid": uid, "title": title, "protected_artifacts_unchanged": True,
                    "trace_slab_unchanged": True,
                    "refreshed_panels": {name: review.sha256_file(folder / name)
                                         for name in ("context_marked.png", "context_trace.png")}})
    del volume, rows
result = {"observed_utc": datetime.now(timezone.utc).isoformat(), "network_requests": 0,
          "anatomical_acceptance": False, "refresher_sha256": review.sha256_file(Path(__file__)),
          "records": records}
args.summary_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"refreshed_fields": len(records), "network_requests": 0,
                  "protected_artifacts_unchanged": all(row["protected_artifacts_unchanged"] for row in records),
                  "summary_path": str(args.summary_path)}))
