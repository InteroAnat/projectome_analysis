"""Audit all copied-overview packages locally; no server pixel reads.

This checks saved-crop/display agreement and recorded own-source geometry.
It does not establish server TIFF pixel equivalence or anatomical acceptance.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path, PureWindowsPath

import nibabel as nib
import numpy as np
from PIL import Image

from verify_all_soma_pixels_20261003 import root_from_raw, save, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "group_analysis/visual_review_20261002"
SAMPLES = {"252384", "252385", "252527", "252714"}
SPACING = np.array([5., 5., 3.])
PANELS = ("context_unmarked.png", "context_marked.png", "context_trace.png",
          "context_stack.png", "section_locator.png")


def check(row):
    uid, sample, neuron = row["UID"], row["SampleID"], row["NeuronID"]
    assert uid == f"{sample}|{neuron}" and sample in SAMPLES
    folder = OUT / sample / Path(neuron).stem
    prov_path = folder / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["UID"] == uid and not prov.get("derived_context")
    nii = folder / "context.nii.gz"
    raw = OUT / "swc_raw" / sample / neuron
    if prov["coverage_status"] == "swc_unavailable":
        assert not nii.exists() and not any((folder / name).exists() for name in PANELS)
        assert not raw.exists() and prov.get("nmt_fallback") is False
        own_url = f"http://10.10.31.31/swc/newswc/Monkey/{sample}/swc_raw/{neuron}"
        assert any(own_url in message and "after 3 requests" in message
                   for message in prov["errors"])
        return {"uid": uid, "outcome": "swc_unavailable", "usable_context": False,
                "artifact_hashes": {str(prov_path): sha(prov_path)}, "source_errors": prov["errors"]}
    meta_path = Path(str(nii) + ".json")
    paths = [prov_path, raw, nii, meta_path] + [folder / name for name in PANELS]
    before = {str(path): sha(path) for path in paths}
    root = root_from_raw(raw)
    assert before[str(raw)] == prov["swc_sha256"]
    assert prov["swc_url"] == f"http://10.10.31.31/swc/newswc/Monkey/{sample}/swc_raw/{neuron}"
    np.testing.assert_allclose(root, [prov["native_soma_" + a + "_um"] for a in "xyz"], atol=1e-9)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    acq = meta["acquisition"]
    assert meta["sample_id"] == acq["sample_id"] == sample
    assert meta["array_axis_order"] == meta["nifti_axis_order"] == "XYZ"
    assert meta["source_array_axis_order"] == acq["array_axis_order"] == "ZYX"
    np.testing.assert_allclose(meta["center_xyz_um"], root, atol=1e-9)
    np.testing.assert_allclose(acq["center_xyz_um"], root, atol=1e-9)
    np.testing.assert_allclose(meta["spacing_xyz_um"], SPACING, atol=1e-9)
    np.testing.assert_allclose(prov["declared_spacing_low_xyz_um"], SPACING, atol=1e-9)
    image = nib.load(str(nii))
    assert image.get_data_dtype() == np.dtype("uint16")
    assert image.header.get_xyzt_units()[0] == "micron"
    assert list(image.shape) == meta["shape_xyz"] == meta["shape_zyx"][::-1]
    assert meta["shape_zyx"] == acq["shape_zyx"] == prov["context_shape_zyx"]
    origin = np.array(meta["origin_xyz_um"])
    expected_affine = np.diag([*SPACING, 1.])
    expected_affine[:3, 3] = origin
    np.testing.assert_allclose(image.affine, expected_affine, atol=1e-3)
    np.testing.assert_allclose(origin, acq["origin_xyz_um"], atol=1e-9)
    np.testing.assert_allclose(origin, prov["context_origin_xyz_um"], atol=1e-9)
    local = (root - origin) / SPACING
    assert ((local >= 0) & (local < image.shape)).all()
    np.testing.assert_allclose(local, prov["context_local_xyz"], atol=1e-9)
    requested, loaded, missing = (set(acq[key]) for key in ("requested", "loaded", "missing"))
    assert loaded.isdisjoint(missing) and requested == loaded | missing
    first_z = int(round(origin[2] / SPACING[2]))
    assert requested == set(range(first_z, first_z + image.shape[2]))
    for key in ("requested", "loaded", "missing", "source_paths", "complete"):
        assert meta[key] == acq[key]
    assert set(prov["context_loaded_z"]) == loaded and set(prov["context_missing_z"]) == missing
    assert meta["complete"] == prov["context_complete"] == (not missing)
    source_paths = {entry["z_index"]: entry["path"] for entry in acq["source_paths"]}
    assert len(source_paths) == len(acq["source_paths"]) and set(source_paths) == requested
    assert prov["context_source_paths"] == acq["source_paths"]
    for z, path in source_paths.items():
        source = PureWindowsPath(path)
        assert source.parent == PureWindowsPath(row["low_res_dir"])
        assert source.name == f"{sample}_{z:05d}_CH1_resample.tif"
    central = int(np.floor(root[2] / SPACING[2]))
    z = int(np.floor(local[2]))
    assert central == first_z + z == prov["context_central_z_index"] and central in loaded
    assert prov["context_central_path"] == source_paths[central]
    slab = prov["context_trace_slab"]
    start, stop = max(0, z - 2), min(image.shape[2], z + 3)
    assert slab["z_start"] == start and slab["z_stop_exclusive"] == stop
    assert slab["sections"] == stop - start and slab["nominal_depth_um"] == (stop - start) * 3
    plane_ids = list(range(first_z + start, first_z + stop))
    assert slab["source_plane_ids"] == prov["context_stack_z"] == plane_ids
    assert slab["edges_in_slab"] == prov["context_trace_segments"]
    assert slab["edges_in_slab"] + slab["edges_outside_slab"] == slab["edges_in_full_crop"]
    assert slab["background"] == "maximum projection of this slab"
    data = np.asanyarray(image.dataobj)
    for plane_id in missing:
        assert np.count_nonzero(data[:, :, plane_id - first_z]) == 0
    plane = data[:, :, z].T.astype(np.float64)
    low, high = np.quantile(plane, [.005, .995])
    normalized = np.zeros(plane.shape) if high <= low else np.clip((plane - low) / (high - low), 0, 1)
    expected = (np.sqrt(normalized) * 255).astype(np.uint8)
    sizes = {}
    for name in PANELS:
        with Image.open(folder / name) as picture:
            picture.load()
            assert picture.format == "PNG" and min(picture.size) > 0
            sizes[name] = list(picture.size)
            if name == "context_unmarked.png":
                assert picture.mode == "L" and picture.size == tuple(image.shape[:2])
                error = int(np.abs(np.asarray(picture).astype(np.int16) - expected.astype(np.int16)).max())
                assert error <= 1, "Unmarked overview does not match its saved root plane"
    assert before == {str(path): sha(path) for path in paths}, "Copied context changed during readback"
    return {"uid": uid, "outcome": "partial_sections" if missing else "complete",
            "usable_context": True, "loaded_sections": len(loaded), "missing_sections": len(missing),
            "central_section": central, "trace_source_sections": plane_ids,
            "saved_shape_xyz": list(image.shape), "sampled_extent_xyz_um": (np.array(image.shape) * SPACING).tolist(),
            "clipped_to_image_bounds": acq["crop_clipped_to_image_bounds"],
            "unmarked_pixels_checked": int(plane.size), "maximum_display_error_8bit": error,
            "pngs_fully_decoded": len(PANELS), "panel_sizes": sizes,
            "artifact_hashes": before, "stable_during_read": True,
            "server_source_pixel_equivalence_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", action="append")
    parser.add_argument("--output", type=Path, default=HERE / "copied_context_package_readback_20261003.json")
    args = parser.parse_args()
    manifest = OUT / "manifest/identity_manifest.csv"
    manifest_hash = sha(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["SampleID"] in SAMPLES]
    assert len(rows) == len({row["UID"] for row in rows}) == 301
    assert all(row["overview_source_status"] == "copied_ch1" for row in rows)
    if args.uid:
        rows = [row for row in rows if row["UID"] in set(args.uid)]
        assert {row["UID"] for row in rows} == set(args.uid)
    result = {"status": "running", "manifest_sha256": manifest_hash, "selected_identities": len(rows),
              "scope": "All 301 copied-overview identities" if not args.uid else "Explicit diagnostic subset",
              "network_requests": 0, "server_source_pixel_equivalence_checked": False,
              "anatomical_acceptance": False, "records": [], "failures": []}
    save(args.output, result)
    for row in rows:
        try:
            record = check(row)
            result["records"].append(record)
            print(json.dumps({"uid": row["UID"], "outcome": record["outcome"], "passed": True}), flush=True)
        except Exception as exc:
            result["failures"].append({"uid": row["UID"], "type": type(exc).__name__, "error": str(exc)})
            print(json.dumps({"uid": row["UID"], "passed": False, "error": str(exc)}), flush=True)
        result["observed_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output, result)
    assert sha(manifest) == manifest_hash
    result["status"] = "failed" if result["failures"] else "passed"
    result["checked_identities"] = len(result["records"])
    result["outcome_counts"] = dict(Counter(row["outcome"] for row in result["records"]))
    result["pngs_fully_decoded"] = sum(row.get("pngs_fully_decoded", 0) for row in result["records"])
    result["unmarked_pixels_checked"] = sum(row.get("unmarked_pixels_checked", 0) for row in result["records"])
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "failures"}}), flush=True)
    raise SystemExit(1 if result["failures"] else 0)


if __name__ == "__main__":
    main()
