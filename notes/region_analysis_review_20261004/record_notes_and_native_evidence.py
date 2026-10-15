"""Bind the historical-decision audit and actual native-image observations to sources."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def source(path):
    path = ROOT / path
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    notes = [f".cursor/agent-journal/{day}.md" for day in
             ("2026-04-10", "2026-04-16", "2026-04-20", "2026-04-22", "2026-04-23",
              "2026-04-24", "2026-04-27", "2026-07-22", "2026-08-04", "2026-08-05")]
    notes += [".cursor/agent-journal/QC_neuron_table_diff_old_vs_new.md",
              "notes/qc_neuron_table_diff_old_vs_new.md", "notes/sq3d_mixed_method_note.md",
              "notes/sq4_sq5_distance_digest.md", "notes/whole_insula_lr_continuation_plan_v2.md",
              "notes/projectome_latest_update_progress.md", "notes/LR_insula_analysis_review.md",
              "notes/gou_stats_methods_index.md"]
    patch_path = "notes/qc_neuron_type_rule_validation_new_table.csv"
    rows = pd.read_csv(ROOT / patch_path)
    patched = rows[rows.assigned_type.eq("Unclassified") & rows.predicted_type.eq("ITi")]
    expected = {"112.swc", "114.swc", "422.swc", "423.swc", "439.swc", "440.swc",
                "444.swc", "458.swc", "459.swc", "460.swc", "461.swc", "469.swc"}
    assert set(patched.NeuronID) == expected
    stamp = datetime.now().astimezone().isoformat()
    report = {"status": "historical_decisions_cross_referenced", "recorded_at": stamp,
              "sources": [source(p) for p in notes + [patch_path]],
              "manual_type_patch_ids": sorted(expected),
              "manual_type_patch_rows": patched.to_dict(orient="records"),
              "interpretation": "Historical choices preserved; historical counts and statistical conclusions are not fresh acceptance."}
    (OUT / "historical_notes_readback.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    examples = [("252384", "048", "CL_Ial", "viewer-right"),
                ("252384", "077", "CR_Ial", "viewer-left"),
                ("252385", "025", "CL_Ia/Id", "viewer-right"),
                ("252385", "050", "CR_PrCO", "viewer-left"),
                ("252527", "088", "CL_Ial", "viewer-right"),
                ("252527", "048", "CR_Iai", "viewer-left"),
                ("252714", "090", "CL_Ia/Id", "viewer-right"),
                ("252714", "032", "CR_Ia/Id", "viewer-left")]
    observations = []
    for sample, neuron, label, side in examples:
        folder = f"group_analysis/visual_review_20261002/{sample}/{neuron}"
        provenance = json.loads((ROOT / folder / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["UID"] == f"{sample}|{neuron}.swc"
        observations.append({"UID": provenance["UID"], "atlas_label": label,
            "atlas_label_side": "L" if label.startswith("CL_") else "R",
            "observed_locator_roi_side": side, "inspection": "actual image pixels inspected with view_image",
            "native_soma_xyz_um": [provenance[f"native_soma_{a}_um"] for a in "xyz"],
            "native_frame": provenance["native_frame"], "raw_swc_sha256": provenance["swc_sha256"],
            "locator": source(folder + "/section_locator.png"),
            "provenance": source(folder + "/provenance.json"),
            "interpretation": "Global relative hemisphere visible; locator is a native grayscale full section with a crop box, without anatomical L/R labels."})
    extra = [source("group_analysis/visual_review_20261002/252385/050/" + name)
             for name in ("context_trace.png", "soma_trace.png")]
    native = {"status": "eight_native_full_section_locators_inspected", "recorded_at": stamp,
              "observations": observations, "additional_actual_trace_panels_inspected": extra,
              "resolution_roles": {"copied_samples": ["252384", "252385", "252527", "252714"],
                                   "copied_context": "5x5x3 um wide field plus full-section locator",
                                   "other_five_context": "wide local field derived from high-resolution blocks; no full-section locator",
                                   "soma": "native high-resolution block/blocks"},
              "absolute_anatomical_laterality": "not independently established by screen position",
              "bulk_artifacts_modified": False}
    (OUT / "native_bulk_laterality_readback.json").write_text(json.dumps(native, indent=2), encoding="utf-8")
    print(json.dumps({"historical_note_sources": len(report["sources"]), "manual_type_ids": len(expected),
                      "native_locator_observations": len(observations), "extra_trace_panels": len(extra)}))


if __name__ == "__main__":
    main()
