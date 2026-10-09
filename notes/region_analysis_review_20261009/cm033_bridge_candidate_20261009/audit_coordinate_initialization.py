"""Read-only geometry/source check; writes a proposal, never fits or changes inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FOLLOWUP = ROOT / "notes/region_analysis_review_20261009/cm033_integration_followup_20261009"
DEB = Path("C:/Users/laika_yan/Documents/ChatGPT/deb_fmri")
EXT = DEB / "alignment_review/cm033_Dsource_20261009"


def binding(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def image_record(path):
    image = nib.load(str(path))
    qform, qcode = image.header.get_qform(coded=True)
    sform, scode = image.header.get_sform(coded=True)
    assert image.header.get_xyzt_units()[0] == "mm"
    assert qcode or scode
    assert not (qcode and scode) or np.allclose(qform, sform, atol=1e-4, rtol=0)
    return dict(binding(path), shape=list(image.shape), affine_mm=image.affine.tolist(),
                header_axcodes=list(nib.aff2axcodes(image.affine)),
                qform_code=int(qcode), sform_code=int(scode))


def main():
    output = HERE / "coordinate_initialization_evidence.json"
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((FOLLOWUP / "spm_bridge_input_audit.json").read_text())
    cached = np.array(original["cached_zero_based_affine"])
    mean = image_record(original["prepared_reference"]["path"])
    assert mean["sha256"] == original["prepared_reference"]["sha256"]
    assert np.array_equal(mean["affine_mm"], cached)
    runs = []
    for row in original["runs"]:
        current = image_record(row["path"])
        assert current["sha256"] == row["source_binding"]["sha256"]
        assert np.array_equal(current["affine_mm"], row["current_affine"])
        recoding = np.array(current["affine_mm"]) @ np.linalg.inv(cached)
        assert np.array_equal(recoding, row["cached_to_current_world_coordinate_change"])
        assert np.array_equal(recoding @ cached, current["affine_mm"])
        runs.append(dict(current, scanner_id=row["scanner_id"], cached_to_current_world_mm=recoding.tolist()))
    assert all(np.array_equal(runs[0]["cached_to_current_world_mm"], r["cached_to_current_world_mm"]) for r in runs)
    reference = EXT / "bids/sub-cm033/ses-20160120/derivatives/preproc/anat/sub-cm033_ses-20160120_desc-refmean_bold.nii"
    current_reference = image_record(reference)
    current_reference["finite_values"] = bool(np.isfinite(nib.load(str(reference)).get_fdata()).all())
    assert current_reference["finite_values"]
    external_paths = [
        EXT / "spatial_rebuild_completed.json",
        EXT / "full_series_qc/full_series_checks.json",
        EXT / "functional_normalization_qc/chain_checks.json",
        EXT / "bids/code/config/analysis_cm033_20160120.json",
        EXT / "bids/sub-cm033/ses-20160120/derivatives/preproc/qc/epi_sform_manifest.json",
        EXT / "bids/sub-cm033/ses-20160120/derivatives/preproc/qc/epi_sform_spm.json",
        EXT / "source_manifest.json",
        EXT / "resume_spatial_after_windows_spm.py",
        DEB / "fmri_pipeline/preproc/orientation_policy.py",
        DEB / "fmri_pipeline/preproc/nii_reorientation.py",
        DEB / "fmri_pipeline/preproc/run_anat_epi_bids.py",
        DEB / "fmri_pipeline/workflows/nipype_anat_epi.py",
    ]
    sources = [binding(p) for p in external_paths]
    completion = json.loads(external_paths[0].read_text())
    config = json.loads(external_paths[3].read_text())
    spm = json.loads(external_paths[5].read_text())
    # SPM matrices use one-based voxel coordinates, nibabel zero-based ones.
    one_to_zero = np.eye(4)
    one_to_zero[:3, 3] = 1
    spm_zero = np.array(spm["spm_matrix"]) @ one_to_zero
    assert np.allclose(spm_zero, current_reference["affine_mm"], atol=1e-6, rtol=0)
    result = {
        "status": "read_only_declared_coordinate_recoding_supported_second_fit_not_started",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "code": binding(__file__),
        "prior_input_audit": binding(FOLLOWUP / "spm_bridge_input_audit.json"),
        "prepared_mean": mean,
        "all_five_current_inputs_fresh_hash_and_affine_match": True,
        "current_inputs": runs,
        "B_cached_world_to_current_world_mm": runs[0]["cached_to_current_world_mm"],
        "B_meaning": "Same array index expressed in two declared world-coordinate descriptions; no array flip or interpolation and no proof of physical orientation.",
        "actual_new_reference": current_reference,
        "new_reference_equals_current_input_world_grid": bool(np.array_equal(current_reference["affine_mm"], runs[0]["affine_mm"])),
        "external_source_bindings": sources,
        "external_spatial_completion_snapshot": completion,
        "configured_reference_scan": config["afni_ref_scan"],
        "configured_orientation_policy": config["preproc"]["sform"],
        "spm_post_write_gate": spm["gate_passed"],
        "spm_one_based_matrix_agrees_with_actual_zero_based_reference": True,
        "proposal": "After root reviews the rejected direct fit and actual new reference, a fresh explicitly B-recoded work-copy mean could be fitted to the actual new same-contrast reference using verified public rigid LPA workflow. This is a separate candidate fit, not header substitution or established historical transform.",
        "public_method": {"warp_type": "shift_rotate", "cost": "lpa", "center_of_mass": True, "final_interpolation": "wsinc5", "rotation_review_gate_degrees": 30},
        "later_statistics_composition_requirement": "Retain B as a separate named coordinate step, followed by the candidate fitted transform; use appropriate inverse/pullback composition. Never use the rejected direct matrix or attach the new reference affine to old statistics.",
        "physical_orientation_authoritatively_resolved": False,
        "original_estimation_payload_byte_identity_independently_proven": False,
        "second_fit_started": False,
        "statistics_or_masks_warped": False,
        "external_mutations": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "B": result["B_cached_world_to_current_world_mm"], "new_reference": current_reference, "external_completion": completion}, indent=2))


if __name__ == "__main__":
    main()
