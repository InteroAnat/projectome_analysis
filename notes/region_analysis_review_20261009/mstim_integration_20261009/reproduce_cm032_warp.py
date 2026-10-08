"""Reproduce a saved CM032 contrast and its actual SPM analysis coverage.

Existing transforms are reused without fitting or accepting registration.
External sources remain read-only. Outputs require a fresh projectome folder.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

import h5py
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FMRI = Path("D:/multimodal_fmri")
MODEL = FMRI / "derivatives/glm/sub-cm032/ses-20160309/glm-mstim-pos1"
NORM = FMRI / "derivatives/norm/sub-cm032/ses-20160309_ana"
ANTS = "/home/binbin/ants-2.5.1/bin"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wsl(path):
    path = Path(path).resolve()
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def spm_metadata(path):
    with h5py.File(path, "r") as source:
        def value(dataset):
            data = dataset[()]
            if h5py.check_dtype(ref=dataset.dtype):
                return [value(source[reference]) for reference in data.ravel() if reference]
            if dataset.attrs.get("MATLAB_class", b"") == b"char":
                return "".join(chr(int(character)) for character in data.ravel())
            return data
        names = value(source["SPM/xX/name"])
        weights = np.asarray(value(source["SPM/xCon/c"])[0]).ravel()
        contrast = value(source["SPM/xCon/name"])[0]
        coefficients = [{"column": names[index], "weight": float(weight)}
                        for index, weight in enumerate(weights) if weight]
        if (contrast != "stimulus > implicit baseline - All Sessions" or len(coefficients) != 4
                or any(item["weight"] != 0.25 or not item["column"].endswith(" stimulus") for item in coefficients)):
            raise ValueError("Actual SPM contrast differs from the four-session stimulus contrast")
        scaling = np.asarray(value(source["SPM/xGX/gSF"])).ravel()
        return {"contrast_name": contrast, "nonzero_coefficients": coefficients,
                "residual_df": float(np.asarray(value(source["SPM/xX/erdf"])).item()),
                "global_mean_target": float(np.asarray(value(source["SPM/xGX/GM"])).item()),
                "global_scaling_policy": value(source["SPM/xGX/sGMsca"]),
                "global_scale_factor_range": [float(scaling.min()), float(scaling.max())],
                "contrast_units": "SPM globally scaled fitted contrast; percent signal change not established"}


def main(output):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError("Use a fresh output directory inside the projectome checkout")
    reference = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz"
    native, mask, saved = MODEL / "maps/con_0001.nii", MODEL / "maps/mask.nii", MODEL / "maps-NMT/con_0001_in_NMT.nii.gz"
    transforms = [NORM / ("cm032_20160309_ana_" + name) for name in
                  ("NMT_1Warp.nii.gz", "NMT_0GenericAffine.mat", "CMT_1Warp.nii.gz", "CMT_0GenericAffine.mat")]
    sources = [reference, native, mask, saved, MODEL / "SPM.mat", MODEL / "contrasts.json",
               MODEL / "glm_config_summary.txt", NORM / "norm_run.log", FMRI / "code/tools/norm/run_nmt_warp.py", *transforms]
    bindings = {str(path): sha(path) for path in sources}
    native_image, mask_image = nib.load(native), nib.load(mask)
    if native_image.shape != mask_image.shape or not np.allclose(native_image.affine, mask_image.affine, rtol=0, atol=1e-5):
        raise ValueError("SPM estimability mask differs from the native contrast grid")
    mask_values = np.unique(np.asanyarray(mask_image.dataobj))
    if not np.isin(mask_values, [0, 1]).all() or not np.any(mask_values == 1):
        raise ValueError("SPM analysis mask must have binary support")
    metadata = spm_metadata(MODEL / "SPM.mat")
    output.mkdir(parents=True)
    version = subprocess.check_output(["wsl.exe", "-d", "Ubuntu-24.04", "--", ANTS + "/antsRegistration", "--version"], text=True).strip()
    commands, outputs = [], {}
    for name, source, interpolation in (("reproduced_contrast", native, "Linear"), ("analysis_coverage", mask, "NearestNeighbor")):
        destination = output / (name + "_space-NMTv2p1.nii.gz")
        command = ["wsl.exe", "-d", "Ubuntu-24.04", "--", ANTS + "/antsApplyTransforms",
                   "-d", "3", "-i", wsl(source), "-r", wsl(reference), "-n", interpolation]
        for transform in transforms:
            command += ["-t", "[" + wsl(transform) + ",0]" if transform.suffix == ".mat" else wsl(transform)]
        command += ["-o", wsl(destination)]
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        (output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"ANTs failed {name}; no comparison accepted")
        commands.append(command)
        outputs[name] = {"path": str(destination), "sha256": sha(destination), "interpolation": interpolation}
    reproduced, previous = nib.load(outputs["reproduced_contrast"]["path"]), nib.load(saved)
    left, right = reproduced.get_fdata(), previous.get_fdata()
    if (reproduced.shape != previous.shape or not np.allclose(reproduced.affine, previous.affine, rtol=0, atol=1e-6)
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("Reproduced/saved contrast geometry or finite values failed")
    comparison = {"exact_values_equal": bool(np.array_equal(left, right)),
                  "maximum_absolute_difference": float(np.max(np.abs(left - right))),
                  "all_values_match_rtol_1e-7_atol_1e-9": bool(np.allclose(left, right, rtol=1e-7, atol=1e-9))}
    if not comparison["all_values_match_rtol_1e-7_atol_1e-9"]:
        raise ValueError("Identity EPI-to-moving chain does not reproduce the saved contrast")
    warped_mask = nib.load(outputs["analysis_coverage"]["path"])
    if (warped_mask.shape != previous.shape or not np.allclose(warped_mask.affine, previous.affine, rtol=0, atol=1e-6)
            or not np.isin(np.unique(np.asanyarray(warped_mask.dataobj)), [0, 1]).all()):
        raise ValueError("Analysis coverage failed binary/grid readback")
    if any(sha(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("External source changed during read-only reproduction")
    report = {"status": "software_reproduction_passed_registration_unaccepted", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha(__file__), "python": sys.version, "ants_version": version,
              "source_sha256": bindings, "spm_contrast": metadata, "outputs": outputs, "commands": commands,
              "comparison": comparison, "transform_application_order": ["identity: EPI to already EPI-aligned moving image", "CMT affine", "CMT warp", "NMT affine", "NMT warp"],
              "affine_caveat": "The existing converted inverse anat-to-EPI file is not used; its presence does not establish application",
              "coverage": "Actual SPM estimation mask transformed with nearest-neighbor interpolation; no support inferred from zero statistics",
              "source_writes": False, "new_registration_fit": False, "anatomical_acceptance": False,
              "paired_fMOST_animal": "unknown", "CM033": "Not integrated: rejected orientation and no accepted NMT model"}
    (output / "warp_reproduction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparison": comparison, "ants_version": version}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "cm032_warp_reproduction")
    main(parser.parse_args().output)
