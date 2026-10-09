"""Reproduce the existing CM032 T warp; sources stay read-only, outputs fresh."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
MSTIM = ROOT / "notes/region_analysis_review_20261009/mstim_integration_20261009"
MODEL = Path("D:/multimodal_fmri/derivatives/glm/sub-cm032/ses-20160309/glm-mstim-pos1")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wsl(path):
    path = Path(path).resolve()
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def geometry(image):
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    return {
        "shape": list(image.shape),
        "zooms": list(map(float, image.header.get_zooms())),
        "units": list(image.header.get_xyzt_units()),
        "dtype": str(image.get_data_dtype()),
        "affine": image.affine.tolist(),
        "qform_code": int(qcode),
        "sform_code": int(scode),
        "qform": None if qform is None else qform.tolist(),
        "sform": None if sform is None else sform.tolist(),
    }


def same_grid(left, right):
    return left.shape == right.shape and np.allclose(
        left.affine, right.affine, rtol=0, atol=1e-6
    )


def main():
    destinations = [OUTPUT / name for name in [
        "reproduced_T_space-NMTv2p1.nii.gz",
        "reproduced_coverage_space-NMTv2p1.nii.gz",
        "T.log", "coverage.log", "t_warp_recheck.json",
    ]]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Fresh T-recheck output files are required")
    contrast_receipt = MSTIM / "cm032_warp_reproduction/warp_reproduction.json"
    regional_receipt = MSTIM / "cm032_regional_comparison/integration_provenance.json"
    prior = read_json(contrast_receipt)
    regional = read_json(regional_receipt)
    if prior["status"] != "software_reproduction_passed_registration_unaccepted":
        raise ValueError("Existing signed-contrast chain is not verified")
    native = MODEL / "maps/spmT_0001.nii"
    saved = MODEL / "maps-NMT/spmT_0001_in_NMT.nii.gz"
    native_mask = MODEL / "maps/mask.nii"
    reference = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz"
    prior_mask = Path(prior["outputs"]["analysis_coverage"]["path"])
    bindings = dict(prior["source_sha256"])
    if sha(saved) != regional["source_sha256"][str(saved)]:
        raise ValueError("Saved T no longer matches delivered regional input")
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError(f"Recorded signed-contrast source changed: {path}")
    for path in [native, saved, prior_mask, contrast_receipt, regional_receipt, Path(__file__)]:
        bindings[str(path)] = sha(path)
    if sha(prior_mask) != prior["outputs"]["analysis_coverage"]["sha256"]:
        raise ValueError("Previously reproduced coverage changed")
    if not same_grid(nib.load(native), nib.load(native_mask)):
        raise ValueError("Native T and SPM mask grids differ")
    reference_image = nib.load(reference)
    commands = []
    # Keep the recorded transform arguments; only replace input and destination.
    for original, source, destination, log in zip(
        prior["commands"], [native, native_mask], destinations[:2], destinations[2:4]
    ):
        command = list(original)
        command[command.index("-i") + 1] = wsl(source)
        command[command.index("-o") + 1] = wsl(destination)
        commands.append(command)
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"ANTs failed; see {log}")
    version = subprocess.check_output([
        "wsl.exe", "-d", "Ubuntu-24.04", "--",
        "/home/binbin/ants-2.5.1/bin/antsRegistration", "--version",
    ], text=True).strip()
    images = {
        "saved_T": nib.load(saved),
        "reproduced_T": nib.load(destinations[0]),
        "prior_coverage": nib.load(prior_mask),
        "reproduced_coverage": nib.load(destinations[1]),
    }
    grids_match = all(same_grid(image, reference_image) for image in images.values())
    coded_mm = all(
        image.header.get_xyzt_units()[0] == "mm"
        and int(image.header["qform_code"]) > 0
        and int(image.header["sform_code"]) > 0
        for image in images.values()
    )
    left = images["reproduced_T"].get_fdata()
    right = images["saved_T"].get_fdata()
    mask = images["reproduced_coverage"].get_fdata()
    old_mask = images["prior_coverage"].get_fdata()
    comparison = {
        "all_reference_grids_match_atol_1e-6": bool(grids_match),
        "coded_mm": bool(coded_mm),
        "all_T_voxels_finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
        "T_voxels_checked": int(left.size),
        "exact_T_values_equal": bool(np.array_equal(left, right)),
        "maximum_absolute_T_difference": float(np.max(np.abs(left - right))),
        "T_values_match_rtol_1e-7_atol_1e-9": bool(np.allclose(left, right, rtol=1e-7, atol=1e-9)),
        "T_compressed_file_hash_equal": sha(destinations[0]) == sha(saved),
        "coverage_binary": bool(np.isin(mask, [0, 1]).all() and np.isin(old_mask, [0, 1]).all()),
        "exact_coverage_values_equal": bool(np.array_equal(mask, old_mask)),
        "coverage_voxels": int(np.count_nonzero(mask)),
        "exact_T_values_equal_within_coverage": bool(np.array_equal(left[mask == 1], right[mask == 1])),
        "coverage_compressed_file_hash_equal": sha(destinations[1]) == sha(prior_mask),
    }
    unchanged = all(sha(path) == expected for path, expected in bindings.items())
    passed = all(comparison[key] for key in [
        "all_reference_grids_match_atol_1e-6", "coded_mm", "all_T_voxels_finite",
        "T_values_match_rtol_1e-7_atol_1e-9", "coverage_binary", "exact_coverage_values_equal",
    ]) and unchanged
    report = {
        "status": "software_T_warp_reproduction_passed_registration_unaccepted" if passed else "failed_T_warp_reproduction",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "ants_version": version,
        "script_sha256": sha(__file__), "source_sha256": bindings,
        "source_hashes_unchanged_after_reproduction": unchanged,
        "lineage": {
            "contrast_receipt": str(contrast_receipt), "native_T": str(native),
            "saved_T": str(saved), "reference": str(reference),
            "statistic_interpolation": "Linear", "coverage_interpolation": "NearestNeighbor",
            "original_statistic_interpolation_source": "D:/multimodal_fmri/code/tools/norm/run_nmt_warp.py:233-238",
            "transform_application_order": prior["transform_application_order"],
        },
        "commands": commands, "comparison": comparison,
        "geometry": {name: geometry(image) for name, image in images.items()},
        "outputs": {path.name: sha(path) for path in destinations[:4]},
        "source_writes": False, "new_registration_fit": False,
        "anatomical_acceptance": False, "statistical_acceptance": False,
        "interpretation": "Existing voxelwise T image only; regional summaries are not new t statistics. Reproducibility does not accept registration or stimulation coordinates.",
    }
    destinations[4].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparison": comparison, "sources_unchanged": unchanged}))
    if not passed:
        raise RuntimeError("T reproduction failed; sources and bounded failure receipt preserved")


if __name__ == "__main__":
    main()
