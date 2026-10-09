"""Add an immutable rejected-fit review; never rerun or alter the fit."""
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_coordinate_initialization import binding

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "rejected_candidate_review.json"
    if output.exists():
        raise FileExistsError(output)
    readback = json.loads((HERE / "candidate_saved_readback.json").read_text())
    assert readback["rotation_gate_pass"] is False
    paths = [
        "run_projectome_cm033_bridge.py", "bridge_candidate_provenance.json",
        "candidate_saved_readback.json", "finalize_bridge_candidate_qc.py",
        "fit_execution_log.txt", "afni_commands.log", "afni_3dAllineate_help.txt",
        "fit/desc-refmeanInAnat.nii", "fit/desc-epiToAnatAffine.aff12.1D",
        "fit/desc-taskAToCachedEpiApply.aff12.1D",
        "coarse_qc/sub-cm033_desc-taskAInCachedEpi_brain.nii.gz",
        "coarse_qc/sub-cm033_desc-taskAPositiveSupportInCachedEpi_mask.nii.gz",
        "coarse_qc/readable/cm033_cachedSPM_mean_on_taskA_candidate.png",
        "coarse_qc/readable/cm033_cachedSPM_mean_on_taskA_candidate.json",
        "coarse_qc/readable/cm033_taskA_support_on_cachedSPM_mean_candidate.png",
        "coarse_qc/readable/cm033_taskA_support_on_cachedSPM_mean_candidate.json",
        "coordinate_initialization_evidence.json", "audit_coordinate_initialization.py",
        "README.md", "freeze_rejected_candidate.py",
    ]
    result = {
        "status": "rejected_direct_bridge_candidate_not_for_statistic_or_mask_propagation",
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "reviewers": ["Codex projection_methods", "Codex root (reported image inspection)"],
        "actual_readable_pngs_inspected": True,
        "visual_observation": "Broad contour/slab mismatch in forward and inverse views; readable titles and legends fit.",
        "rotation_degrees": readback["rotation_degrees_independent_SVD"],
        "unchanged_rotation_gate_degrees": 30,
        "gate_pass": False,
        "fit_execution_count": 1,
        "finite_and_grid_software_checks_pass": True,
        "statistical_images_warped": False,
        "second_fit_started": False,
        "anatomical_or_physical_laterality_acceptance": False,
        "artifacts": [binding(HERE / path) for path in paths],
        "scope": "Additive review only; original fit/producer provenance and external files unchanged.",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"receipt": str(output), "sha256": binding(output)["sha256"], "bindings": len(paths)}))


if __name__ == "__main__":
    main()
