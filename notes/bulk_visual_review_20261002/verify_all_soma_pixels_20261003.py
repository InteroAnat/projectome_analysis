"""Read-only full-cohort soma/source-pixel audit; never access the network.

Acquisition axes and spacing are nominal. Passing this audit establishes saved
pixel lineage, not anatomical orientation, tracing validity, or region labels.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from PIL import Image
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "group_analysis/visual_review_20261002"
SPACING = np.array([.65, .65, 3.])
CUBE = np.array([360, 360, 90])
PANELS = ("soma_unmarked.png", "soma_marked.png", "soma_trace.png",
          "soma_stack.png", "soma_ortho.png")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.25)


def root_from_raw(path):
    roots = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if float(fields[6]) == -1:
            roots.append([float(value) for value in fields[2:5]])
    assert len(roots) == 1, "Raw source does not have exactly one root"
    root = np.array(roots[0])
    assert np.isfinite(root).all() and (root >= 0).all()
    return root


def check(row):
    uid = row["UID"]
    sample, neuron = uid.split("|")
    folder = OUT / sample / Path(neuron).stem
    prov = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    assert prov["UID"] == uid
    nii = folder / "soma.nii.gz"
    raw = OUT / "swc_raw" / sample / neuron
    if prov.get("coverage_status") == "swc_unavailable":
        assert not nii.exists() and not any((folder / name).exists() for name in PANELS)
        assert not prov.get("swc_sha256") and not raw.exists()
        errors = prov.get("errors") or []
        own_url = f"http://10.10.31.31/swc/newswc/Monkey/{sample}/swc_raw/{neuron}"
        assert any(own_url in error and "after 3 requests" in error and
                   "raw SWC unavailable" in error for error in errors), "Missing bounded own-source failure record"
        assert prov.get("nmt_fallback") is False, "Missing explicit no-NMT-fallback declaration"
        return {"uid": uid, "outcome": "swc_unavailable",
                "provenance_sha256": sha(folder / "provenance.json"),
                "source_errors": prov["errors"], "usable_soma": False}
    root = root_from_raw(raw)
    raw_hash = sha(raw)
    assert raw_hash == prov["swc_sha256"]
    np.testing.assert_allclose(root, [prov["native_soma_" + a + "_um"] for a in "xyz"], atol=1e-9)
    root_cube = tuple(int(value) for value in np.floor(root / SPACING / CUBE))
    if not nii.exists():
        assert prov["coverage_status"] == "center_unavailable"
        assert not any((folder / name).exists() for name in PANELS)
        acquisition = prov["soma_metadata"]
        assert acquisition["sample_id"] == sample
        assert root_cube in {tuple(c) for c in acquisition["missing"]}
        assert root_cube not in {tuple(c) for c in acquisition["loaded"]}
        np.testing.assert_allclose(root, acquisition["center_xyz_um"], atol=1e-9)
        assert acquisition["missing_errors"]
        return {"uid": uid, "outcome": "center_unavailable", "raw_sha256": raw_hash,
                "root_cube": list(root_cube), "usable_soma": False,
                "source_errors": acquisition["missing_errors"]}
    meta_path = Path(str(nii) + ".json")
    paths = [nii, meta_path, raw] + [folder / name for name in PANELS]
    before = {str(path): sha(path) for path in paths}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    acquisition = meta["acquisition"]
    assert meta["sample_id"] == acquisition["sample_id"] == sample
    assert meta["array_axis_order"] == meta["nifti_axis_order"] == "XYZ"
    assert meta["source_array_axis_order"] == acquisition["array_axis_order"] == "ZYX"
    np.testing.assert_allclose(meta["center_xyz_um"], root, atol=1e-9)
    np.testing.assert_allclose(acquisition["center_xyz_um"], root, atol=1e-9)
    np.testing.assert_allclose(meta["spacing_xyz_um"], SPACING, atol=1e-9)
    requested = {tuple(c) for c in acquisition["requested"]}
    loaded = {tuple(c) for c in acquisition["loaded"]}
    missing = {tuple(c) for c in acquisition["missing"]}
    assert len(requested) == len(acquisition["requested"])
    assert loaded.isdisjoint(missing) and requested == loaded | missing and root_cube in loaded
    expected_urls = {f"http://bap.cebsit.ac.cn/monkeydata/{sample}/cube/{z}/{x}_{y}_{z}.tif"
                     for x, y, z in requested}
    assert set(acquisition["source_paths"]) == expected_urls
    assert len(acquisition["source_paths"]) == len(requested)
    assert meta["complete"] == acquisition["complete"] == (not missing)
    if "soma_complete" in prov:
        assert prov["soma_complete"] == (not missing)
    assert {tuple(c) for c in prov["soma_loaded"]} == loaded
    assert {tuple(c) for c in prov["soma_missing"]} == missing
    assert set(prov["soma_source_urls"]) == expected_urls
    image = nib.load(str(nii))
    assert image.get_data_dtype() == np.dtype("uint16")
    assert image.header.get_xyzt_units()[0] == "micron"
    assert list(image.shape) == meta["shape_xyz"] == meta["shape_zyx"][::-1]
    origin = np.array(meta["origin_xyz_um"])
    np.testing.assert_allclose(image.affine, np.diag([*SPACING, 1.]) +
                               np.array([[0, 0, 0, origin[0]], [0, 0, 0, origin[1]],
                                         [0, 0, 0, origin[2]], [0, 0, 0, 0]]), atol=1e-3)
    first = np.rint(origin / SPACING / CUBE).astype(int)
    np.testing.assert_allclose(first * SPACING * CUBE, origin, atol=1e-6)
    assert all(shape % size == 0 for shape, size in zip(image.shape, CUBE))
    grid_shape = np.array(image.shape) // CUBE
    grid = {tuple(first + [x, y, z]) for x in range(grid_shape[0])
            for y in range(grid_shape[1]) for z in range(grid_shape[2])}
    assert requested == grid, "Acquisition does not partition the entire saved crop"
    data = np.asanyarray(image.dataobj)
    assert data.dtype == np.uint16
    source_hashes = {}
    source_voxels = gap_voxels = 0
    for cube in sorted(requested):
        start = (np.array(cube) - first) * CUBE
        slices = tuple(slice(int(a), int(a + size)) for a, size in zip(start, CUBE))
        saved = data[slices]
        if cube in missing:
            assert np.count_nonzero(saved) == 0, "Missing-source region contains intensities"
            gap_voxels += saved.size
            continue
        x, y, z = cube
        source = OUT / "cache/cubes" / sample / "high_res_http" / str(z) / f"{x}_{y}_{z}.tif"
        digest = sha(source)
        with tifffile.TiffFile(source) as tif:
            assert len(tif.series) == 1
            assert tif.series[0].shape == (90, 360, 360)
            assert tif.series[0].dtype == np.uint16
            pixels = tif.series[0].asarray()
        assert sha(source) == digest, "Source cube changed during readback"
        assert np.array_equal(saved, pixels.transpose(2, 1, 0)), f"Source-pixel mismatch: {cube}"
        source_hashes[str(source)] = digest
        source_voxels += saved.size
    assert source_voxels + gap_voxels == data.size
    local = (root - origin) / SPACING
    assert ((local >= 0) & (local < image.shape)).all()
    plane = data[:, :, int(np.floor(local[2]))].T.astype(np.float64)
    lo, hi = np.quantile(plane, [.005, .995])
    normalized = np.zeros(plane.shape) if hi <= lo else np.clip((plane - lo) / (hi - lo), 0, 1)
    expected = (np.sqrt(normalized) * 255).astype(np.uint8)
    sizes = {}
    for name in PANELS:
        with Image.open(folder / name) as panel:
            panel.load()
            assert panel.format == "PNG" and min(panel.size) > 0
            sizes[name] = list(panel.size)
            if name == "soma_unmarked.png":
                assert panel.mode == "L" and panel.size == tuple(image.shape[:2])
                actual = np.asarray(panel)
                error = int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max())
                assert error <= 1, "Soma unmarked PNG is not its declared root plane"
    assert before == {str(path): sha(path) for path in paths}, "Soma artifacts changed during audit"
    return {"uid": uid, "outcome": "partial_neighbor" if missing else "complete",
            "usable_soma": True, "source_voxels_checked": source_voxels,
            "gap_voxels_checked": gap_voxels, "all_saved_voxels_checked": int(data.size),
            "loaded_cubes": len(loaded), "missing_cubes": len(missing),
            "unmarked_pixels_checked": int(plane.size), "maximum_display_error_8bit": error,
            "pngs_fully_decoded": len(PANELS), "panel_sizes": sizes,
            "artifact_hashes": before, "cached_source_hashes": source_hashes,
            "stable_during_read": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", action="append", help="Diagnostic subset; default is all manifest identities")
    parser.add_argument("--output", type=Path, default=HERE / "all_soma_source_pixel_readback_20261003.json")
    parser.add_argument("--resume", action="store_true", help="Reuse passing voxel checks only after rehashing all recorded artifacts and source cubes")
    args = parser.parse_args()
    manifest = OUT / "manifest/identity_manifest.csv"
    manifest_hash = sha(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    assert len(all_rows) == len({row["UID"] for row in all_rows}) == 832
    rows = all_rows if not args.uid else [row for row in all_rows if row["UID"] in set(args.uid)]
    if args.uid:
        assert {row["UID"] for row in rows} == set(args.uid)
    previous = {}
    if args.resume:
        old = json.loads(args.output.read_text(encoding="utf-8"))
        assert old["manifest_sha256"] == manifest_hash
        assert old["selected_identities"] == len(rows)
        previous = {record["uid"]: record for record in old["records"]}
        assert len(previous) == len(old["records"])
        assert set(previous).issubset({row["UID"] for row in rows})
    result = {"observed_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "scope": "All manifest identities" if not args.uid else "Explicit diagnostic subset",
              "manifest_sha256": manifest_hash, "selected_identities": len(rows),
              "network_requests": 0, "anatomical_acceptance": False,
              "records": [], "failures": [], "hash_revalidated_reused": 0}
    save(args.output, result)
    for row in rows:
        try:
            saved = previous.get(row["UID"], {})
            hashes = {**saved.get("artifact_hashes", {}), **saved.get("cached_source_hashes", {})}
            reusable = bool(saved.get("usable_soma") and saved.get("stable_during_read") and hashes)
            if reusable:
                for name, digest in hashes.items():
                    path = Path(name)
                    if not path.is_file() or sha(path) != digest:
                        reusable = False
                        break
            if reusable:
                record = saved
                result["hash_revalidated_reused"] += 1
            else:
                record = check(row)
            result["records"].append(record)
            print(json.dumps({"uid": row["UID"], "outcome": record["outcome"], "passed": True}), flush=True)
        except Exception as exc:
            result["failures"].append({"uid": row["UID"], "error": str(exc), "type": type(exc).__name__})
            print(json.dumps({"uid": row["UID"], "passed": False, "error": str(exc)}), flush=True)
        result["observed_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output, result)
    assert sha(manifest) == manifest_hash, "Manifest changed during audit"
    result["status"] = "failed" if result["failures"] else "passed"
    result["outcome_counts"] = dict(Counter(record["outcome"] for record in result["records"]))
    result["checked_identities"] = len(result["records"])
    result["pngs_fully_decoded"] = sum(record.get("pngs_fully_decoded", 0) for record in result["records"])
    result["saved_voxels_checked"] = sum(record.get("all_saved_voxels_checked", 0) for record in result["records"])
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "failures"}}), flush=True)
    raise SystemExit(1 if result["failures"] else 0)


if __name__ == "__main__":
    main()
