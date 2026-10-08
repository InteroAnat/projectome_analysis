"""Refresh the ancillary MSTIM provenance without altering external sources."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
from export_arm_projection_tables import sha256


def main():
    output = Path(__file__).resolve().parent / "ancillary_source_readback.json"
    if output.exists():
        raise FileExistsError("Existing source readback is immutable")
    readiness = ROOT / "group_analysis/evolution_20261008/crossmodal/input_readiness.json"
    old = json.loads(readiness.read_text(encoding="utf-8"))
    selected = {key: old["sources"][key] for key in
                ("science_project_note", "linked_pipeline_project_note", "registry",
                 "cm032_stimulation_ontology", "cm033_stimulation_ontology", "pipeline_template_config")}
    selected["cm032_normalization_moving_image"] = old["models"]["cm032"]["moving_image_current_file"]
    selected["cm033_staged_moving_image"] = old["models"]["cm033"]["normalization_staged_moving_image"]
    selected["fmri_reference"] = old["fmri_normalization_template"]
    selected["projectome_reference"] = old["projectome_reference"]
    rows = []
    for kind, record in selected.items():
        path = Path(record["path"])
        actual = sha256(path)
        rows.append({"kind": kind, "path": str(path), "sha256": actual,
                     "matches_readiness_capture": actual == record["sha256"]})
    a, b = [nib.load(selected[k]["path"]) for k in ("fmri_reference", "projectome_reference")]
    same = a.shape == b.shape and np.array_equal(a.affine, b.affine) and np.array_equal(a.get_fdata(), b.get_fdata())
    result = {"status": "passed" if same and all(row["matches_readiness_capture"] for row in rows) else "failed",
              "checked_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(__file__),
              "readiness_sha256": sha256(readiness), "sources": rows,
              "reference_grid_and_scaled_values_identical": same,
              "transform_correction": "Actual CM032 chain uses identity from EPI to its already EPI-aligned moving image, followed by CMT and NMT affine/warp pairs. Exact saved contrast reproduction supersedes the older generic inverse-affine chain description.",
              "registration_acceptance": False,
              "stimulation_coordinate_acceptance": False,
              "T_image_transform_independently_reproduced": False,
              "visual_readback": "Readable QC text fits; anatomy display contour follows the brain boundary at three declared cuts. This limited visual check does not establish regional registration accuracy or laterality.",
              "source_writes": False}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "sources": len(rows), "identical_reference_grid_values": same}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
