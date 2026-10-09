"""Add readable displays and saved-output checks; never refit the candidate."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DEB = Path("/mnt/c/Users/laika_yan/Documents/ChatGPT/deb_fmri")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = HERE / "coarse_qc/readable"
    receipt = HERE / "candidate_saved_readback.json"
    if out.exists() or receipt.exists():
        raise FileExistsError("Preserve existing QC variants and readback")
    os.environ["MPLCONFIGDIR"] = str(HERE / "mpl_config")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["NIPYPE_CONFIG_DIR"] = str(HERE / "nipype_config")
    sys.path.insert(0, str(DEB))
    import nibabel as nib
    import numpy as np
    from nipype.utils.filemanip import loadpkl
    from fmri_pipeline.qc.overlay import write_anat_epi_overlay
    provenance_path = HERE / "bridge_candidate_provenance.json"
    source = json.loads(provenance_path.read_text())
    if source["status"] != "software_completed_spatial_bridge_candidate_awaiting_image_review":
        raise ValueError("Fit must already be completed; this helper never refits")
    moving, fixed = Path(source["moving"]["path"]), Path(source["fixed"]["path"])
    aligned = Path(source["aligned_mean"]["path"])
    inverse_brain = Path(source["inverse_anatomy"]["path"])
    inverse_support = Path(source["inverse_support"]["path"])
    support = HERE / "coarse_qc/sub-cm033_desc-taskAPositiveSupport_mask.nii.gz"
    for key in ["moving", "fixed", "aligned_mean", "inverse_anatomy", "inverse_support", "matrix", "inverse_matrix"]:
        if sha(Path(source[key]["path"])) != source[key]["sha256"]:
            raise ValueError(f"Source/artifact changed: {key}")
    runtime_path = HERE / "nipype_work/cm033_cachedSPM_to_taskA_candidate/allineate_refmean_to_brain/result_allineate_refmean_to_brain.pklz"
    result = loadpkl(runtime_path)
    runtime = result.runtime
    with (HERE / "fit_execution_log.txt").open("x") as stream:
        stream.write("COMMAND\n"+runtime.cmdline+"\nSTDOUT\n"+runtime.stdout+"\nSTDERR\n"+runtime.stderr)
    out.mkdir()
    forward = out / "cm033_cachedSPM_mean_on_taskA_candidate.png"
    inverse = out / "cm033_taskA_support_on_cachedSPM_mean_candidate.png"
    title = "CM033 bridge candidate: rotation review gate failed\nPhysical LR and historical payload remain unaccepted"
    write_anat_epi_overlay(aligned, fixed, forward, mask=support, n_slices=6, title=title,
                           underlay_label="registered EPI mean", contour_label="taskA positive support (QC)")
    write_anat_epi_overlay(moving, inverse_brain, inverse, mask=inverse_support, n_slices=6, title=title,
                           underlay_label="original EPI mean", contour_label="inverse taskA support (QC)")
    matrix = np.array(source["matrix"]["values"])
    inverse_matrix = np.array(source["inverse_matrix"]["values"])
    u, singular_values, vh = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vh
    degrees = float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))
    pairs = [(aligned, fixed), (inverse_brain, moving), (inverse_support, moving)]
    checks = []
    for path, master in pairs:
        image, target = nib.load(path), nib.load(master)
        if image.shape != target.shape or not np.array_equal(image.affine, target.affine):
            raise ValueError(f"Saved master grid does not match exactly: {path}")
        if not np.isfinite(np.asanyarray(image.dataobj)).all():
            raise ValueError(f"Nonfinite saved image: {path}")
        checks.append({"path": str(path), "sha256": sha(path), "master": str(master),
                       "exact_affine_and_shape": True, "finite": True})
    record = {
        "status": "software_saved_readback_pass_candidate_rotation_gate_failed",
        "checked_utc": datetime.now(timezone.utc).isoformat(), "fit_rerun": False,
        "fit_runtime_returncode": int(runtime.returncode), "fit_runtime_seconds": float(runtime.duration),
        "rotation_degrees_independent_SVD": degrees, "public_rotation_review_gate_degrees": 30.,
        "rotation_gate_pass": degrees <= 30,
        "linear_determinant": float(np.linalg.det(matrix[:3, :3])), "linear_singular_values": singular_values.tolist(),
        "matrix_pair_max_abs_identity_error": float(np.max(np.abs(matrix @ inverse_matrix - np.eye(4)))),
        "saved_grid_checks": checks,
        "inverse_support_binary": bool(set(np.unique(np.asanyarray(nib.load(inverse_support).dataobj)).tolist()) <= {0., 1.}),
        "anatomical_acceptance": False, "statistical_maps_warped": False,
        "driver_provenance_sha256": sha(provenance_path),
        "display_only_variants_preserve_originals": True,
        "display_code": {"path": inspect.getsourcefile(write_anat_epi_overlay), "sha256": sha(Path(inspect.getsourcefile(write_anat_epi_overlay)))},
        "finalizer_sha256": sha(Path(__file__)),
        "display_artifacts": [{"path": str(path), "sha256": sha(path)} for path in sorted(out.iterdir())],
        "fit_execution_log_sha256": sha(HERE / "fit_execution_log.txt"),
    }
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({key: record[key] for key in ["status", "fit_runtime_returncode", "fit_runtime_seconds", "rotation_degrees_independent_SVD", "rotation_gate_pass", "matrix_pair_max_abs_identity_error"]}, indent=2))


if __name__ == "__main__":
    main()
