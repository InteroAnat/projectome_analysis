"""Independently read back newly finished contexts; never produce image data."""
import importlib.util
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "derived_readback", HERE / "verify_derived_pilot_20261003.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def current_audit(path, required_fields=()):
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("status") != "passed" or not saved.get("artifact_hashes"):
        return False
    if any(name not in saved for name in required_fields):
        return False
    return all(Path(name).exists() and module.digest(Path(name)) == sha
               for name, sha in saved["artifact_hashes"].items())


def verify_mask(sample, neuron):
    """Reconstruct acquired-voxel membership from physical coordinates only."""
    output = ROOT / "group_analysis/visual_review_20261002"
    folder = output / sample / neuron
    paths = [folder / "context.nii.gz", folder / "context_coverage.nii.gz",
             folder / "context_coverage.nii.gz.json",
             output / "manifest" / f"derived_context_pilot_{sample}_{neuron}.json"]
    before = {str(path): module.digest(path) for path in paths}
    report = json.loads(paths[-1].read_text(encoding="utf-8"))
    image, mask_image = (nib.load(str(path)) for path in paths[:2])
    mask = np.asanyarray(mask_image.dataobj)
    assert mask.dtype == np.uint8 and mask.shape == image.shape
    assert mask_image.header.get_xyzt_units()[0] == "micron"
    np.testing.assert_allclose(mask_image.affine, image.affine, atol=1e-6)
    origin, spacing = np.array(report["origin_xyz_um"]), np.array(report["derived_spacing_xyz_um"])
    source_spacing = [.65, .65, 3.0]
    cube_size = [360, 360, 90]
    axes = [np.rint((origin[i] + np.arange(image.shape[i]) * spacing[i]) /
                    source_spacing[i]).astype(np.int64) // cube_size[i]
            for i in range(3)]
    expected = np.zeros(image.shape, dtype=np.uint8)
    for cube in report["loaded"]:
        indices = [np.flatnonzero(axes[i] == cube[i]) for i in range(3)]
        expected[np.ix_(*indices)] = 1
    np.testing.assert_array_equal(mask, expected)
    assert abs(float(mask.mean()) - report["coverage_fraction"]) < 1e-12
    intensity = np.asanyarray(image.dataobj)
    assert np.all(intensity[mask == 0] == 0)
    root_index = np.rint((np.array(report["native_soma_xyz_um"]) - origin) / spacing).astype(int)
    assert mask[tuple(root_index)] == 1
    assert before == {str(path): module.digest(path) for path in paths}
    result = {"uid": f"{sample}|{neuron}.swc", "status": "passed",
              "all_mask_voxels_checked": int(mask.size), "coverage_fraction": float(mask.mean()),
              "root_acquired": True, "unavailable_intensity_zero": True,
              "artifact_hashes": before, "stable_during_read": True}
    (HERE / f"derived_mask_readback_{sample}_{neuron}_20261003.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_panels(sample, neuron):
    folder = ROOT / "group_analysis/visual_review_20261002" / sample / neuron
    names = ["unmarked", "marked", "trace", "stack", "ortho", "coverage"]
    panels = [folder / f"context_{name}.png" for name in names]
    nii = folder / "context.nii.gz"
    report_path = folder.parents[1] / "manifest" / f"derived_context_pilot_{sample}_{neuron}.json"
    paths = panels + [nii, Path(str(nii) + ".json"), folder / "provenance.json", report_path]
    before = {str(path): module.digest(path) for path in paths}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    local_z = ((report["native_soma_xyz_um"][2] - report["origin_xyz_um"][2]) /
               report["derived_spacing_xyz_um"][2])
    z = int(np.floor(local_z))
    source_plane = report["source_plane_ids"][z]
    assert provenance["context_central_z_index"] == source_plane
    assert provenance["context_plane_stats"]["display"] == (
        "percentile 0.5-99.5 then gamma 0.5; raw values are in the NIfTI")
    # Reconstruct the documented display from the persisted XYZ volume, without
    # calling the renderer or its normalization/coordinate helpers.
    raw = np.asanyarray(nib.load(str(nii)).dataobj)[:, :, z].T.astype(np.float64)
    lo, hi = np.quantile(raw, [.005, .995])
    normalized = np.zeros(raw.shape) if hi <= lo else np.clip((raw - lo) / (hi - lo), 0, 1)
    expected_display = (np.sqrt(normalized) * 255).astype(np.uint8)
    dimensions, unmarked_extrema = {}, None
    for path in panels:
        with Image.open(path) as picture:
            assert picture.format == "PNG"
            picture.verify()
        with Image.open(path) as picture:
            picture.load()
            assert min(picture.size) > 0
            dimensions[path.name] = {"width": picture.width, "height": picture.height,
                                     "mode": picture.mode}
            if path.name == "context_unmarked.png":
                assert picture.size == tuple(nib.load(str(nii)).shape[:2])
                unmarked_extrema = picture.convert("L").getextrema()
                actual_display = np.asarray(picture.convert("L"))
                difference = np.abs(actual_display.astype(np.int16) - expected_display.astype(np.int16))
                maximum_error = int(difference.max())
                assert maximum_error <= 1, "Unmarked PNG does not match its declared native plane/display"
    assert before == {str(path): module.digest(path) for path in paths}
    result = {"uid": f"{sample}|{neuron}.swc", "status": "passed",
              "pngs_fully_decoded": len(panels), "dimensions": dimensions,
              "unmarked_display_extrema": list(unmarked_extrema),
              "constant_unmarked_display": unmarked_extrema[0] == unmarked_extrema[1],
              "unmarked_pixels_checked": int(raw.size), "source_plane_id": source_plane,
              "central_depth_index": z, "display_quantiles": [float(lo), float(hi)],
              "maximum_display_error_8bit": maximum_error,
              "display_tolerance_8bit": 1,
              "artifact_hashes": before, "stable_during_read": True,
              "anatomical_acceptance": False}
    (HERE / f"derived_panel_readback_{sample}_{neuron}_20261003.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ledger", type=Path, default=ROOT / "group_analysis/visual_review_20261002/manifest/regional_context_run_20261003.json")
parser.add_argument("--manifest", type=Path, default=ROOT / "group_analysis/visual_review_20261002/manifest/identity_manifest.csv")
parser.add_argument("--summary-path", type=Path, default=HERE / "incremental_regional_readback_20261003.json")
args = parser.parse_args()
ledger_path = args.ledger
ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
manifest_path = args.manifest
manifest_sha = module.digest(manifest_path)
if manifest_sha != ledger["manifest_sha256"]:
    raise ValueError("Current manifest differs from the producer's selected-input hash")
samples_in_run = sorted({uid.split("|", 1)[0] for uid in ledger["identities"]})
with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
    expected_uids = {row["UID"] for row in csv.DictReader(stream)
                     if row["SampleID"] in samples_in_run}
if expected_uids != set(ledger["identities"]):
    raise ValueError("Run ledger omits or adds identities for its represented samples")
if module.digest(manifest_path) != manifest_sha:
    raise ValueError("Manifest changed during scope readback")
terminal_record = None
if ledger["state"].startswith("terminal_"):
    stamp = datetime.fromisoformat(ledger["started_utc"]).strftime("%Y%m%dT%H%M%S%fZ")
    terminal_record = HERE / f"terminal_regional_run_{stamp}_{ledger['pid']}.json"
    terminal_record.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
checked, existing, masks, panels, smaller_fields, failures = [], [], [], [], [], []
for uid, row in ledger["identities"].items():
    if row["state"] not in {"attempt_finished", "reused_existing_derived"}:
        continue
    if row.get("status") not in {"derived_highres", "derived_partial"}:
        continue
    sample, filename = uid.split("|", 1)
    neuron = Path(filename).stem
    target = HERE / f"derived_pilot_readback_{sample}_{neuron}_20261003.json"
    try:
        if current_audit(target):
            existing.append(uid)
        else:
            audit = module.verify(sample, neuron)
            checked.append({"uid": uid, "sources": audit["sources_hash_verified"],
                            "pixel_checks": len(audit["sampled_pixel_checks"])})
        mask_target = HERE / f"derived_mask_readback_{sample}_{neuron}_20261003.json"
        if not current_audit(mask_target):
            result = verify_mask(sample, neuron)
            masks.append({"uid": uid, "mask_voxels": result["all_mask_voxels_checked"]})
        panel_target = HERE / f"derived_panel_readback_{sample}_{neuron}_20261003.json"
        if not current_audit(panel_target, required_fields=("unmarked_pixels_checked",)):
            result = verify_panels(sample, neuron)
            panels.append({"uid": uid, "pngs_decoded": result["pngs_fully_decoded"],
                           "unmarked_pixels_checked": result["unmarked_pixels_checked"],
                           "maximum_display_error_8bit": result["maximum_display_error_8bit"],
                           "constant_unmarked_display": result["constant_unmarked_display"]})
        saved_report = ROOT / "group_analysis/visual_review_20261002/manifest" / f"derived_context_pilot_{sample}_{neuron}.json"
        requested = json.loads(saved_report.read_text(encoding="utf-8"))["requested_field_um_xyz"]
        if any(float(value) < 4000 for value in requested[:2]):
            smaller_fields.append({"uid": uid, "verified_current_field_um_xyz": requested,
                                   "preferred_field_um": 4000,
                                   "required_action": "upgrade after current producer is terminal"})
    except Exception as error:
        failures.append({"uid": uid, "error": str(error)})
summary = {"observed_utc": datetime.now(timezone.utc).isoformat(),
           "producer_ledger_path": str(ledger_path), "input_manifest_path": str(manifest_path),
           "artifact_scope": "Current output files for this ledger's selected identities; producer history retained separately",
           "producer_pid": ledger["pid"], "producer_state": ledger["state"],
           "terminal_record": str(terminal_record) if terminal_record else None,
           "selected_identities": len(ledger["identities"]),
           "represented_samples": samples_in_run,
           "manifest_uid_set_matches": True, "manifest_sha256": manifest_sha,
           "already_checked": existing, "newly_checked": checked,
           "new_masks_checked": masks,
           "new_panels_checked": panels,
           "smaller_verified_fields_pending_upgrade": smaller_fields,
           "failures": failures, "anatomical_acceptance": False}
args.summary_path.write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary))
if failures:
    raise SystemExit(1)
