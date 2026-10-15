"""Read-only native pixel/geometry/source readback; writes an audit JSON only."""
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main_scripts"))
from Visual_toolkit import _read_highres_cube
from swc_validation import parse_swc


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(sample="252790", neuron="014"):
    output = ROOT / "group_analysis" / "visual_review_20261002"
    folder = output / sample / neuron
    report_path = output / "manifest" / f"derived_context_pilot_{sample}_{neuron}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] not in {"derived_highres", "derived_partial"}:
        raise ValueError(f"No usable derived context: {report['status']}")
    nii = folder / "context.nii.gz"
    side = Path(str(nii) + ".json")
    before = {str(p): digest(p) for p in (nii, side, report_path, folder / "provenance.json")}
    meta = json.loads(side.read_text(encoding="utf-8"))
    image = nib.load(str(nii))
    values = np.asanyarray(image.dataobj)
    assert values.dtype == np.uint16
    assert image.header.get_xyzt_units()[0] == "micron"
    assert list(values.shape) == meta["shape_xyz"] == report["shape_zyx"][::-1]
    np.testing.assert_allclose(image.affine[:3, 3], report["origin_xyz_um"], atol=1e-3)
    np.testing.assert_allclose(np.diag(image.affine)[:3], report["derived_spacing_xyz_um"], atol=1e-6)
    source_spacing = np.array([.65, .65, 3.0])
    world_origin = np.asarray(report["origin_xyz_um"])
    out_spacing = np.asarray(report["derived_spacing_xyz_um"])
    np.testing.assert_allclose(world_origin / source_spacing,
                               np.rint(world_origin / source_spacing), atol=1e-7)
    swc = output / "swc_raw" / sample / (neuron + ".swc")
    assert digest(swc) == report["swc_sha256"]
    nodes = parse_swc(swc.read_text(encoding="utf-8"), source=str(swc))
    raw_root = next(row[2:5] for row in nodes if row[6] == -1)
    np.testing.assert_allclose(raw_root, report["native_soma_xyz_um"], atol=1e-6)
    sources = {tuple(row["cube_xyz"]): row for row in report["source_paths"]}
    assert len(sources) == len(report["loaded"])
    verified_sources = 0
    for row in sources.values():
        path = Path(row["path"])
        assert f"/{sample}/cube/" in row["url"]
        assert digest(path) == row["sha256"]
        assert path.stat().st_size == row["file_bytes"] <= 32 * 1024**2
        verified_sources += 1
    samples = []
    by_cube = {}
    # Use world coordinates from the reported affine; no sampler/helper reuse.
    shape = np.array(values.shape)
    for fractions in ((0, 0, 0), (1, 1, 1), (0, 1, 0), (1, 0, 1),
                      (.5, .5, .5), (.25, .75, .5), (.75, .25, .5),
                      (.1, .6, .2), (.9, .4, .8)):
        index = np.rint((shape - 1) * fractions).astype(int)
        fine = np.rint((world_origin + index * out_spacing) / source_spacing).astype(int)
        cube = tuple(int(v) for v in fine // [360, 360, 90])
        local = fine % [360, 360, 90]
        by_cube.setdefault(cube, []).append((index, fine, local))
    for cube, points in by_cube.items():
        if cube not in sources:
            for index, fine, local in points:
                assert values[tuple(index)] == 0
                samples.append({"index_xyz": index.tolist(), "cube_xyz": list(cube), "source_missing": True})
            continue
        data = _read_highres_cube(sources[cube]["path"])
        for index, fine, local in points:
            expected = int(data[tuple(local[::-1])])
            actual = int(values[tuple(index)])
            assert actual == expected, (index, fine, actual, expected)
            samples.append({"index_xyz": index.tolist(), "fine_index_xyz": fine.tolist(),
                            "cube_xyz": list(cube), "expected": expected, "actual": actual})
        del data
    assert any(not row.get("source_missing") for row in samples)
    planes = meta["acquisition"]["source_plane_ids"]
    expected_planes = [int(round((world_origin[2] + k * out_spacing[2]) / 3))
                       for k in range(values.shape[2])]
    assert planes == expected_planes
    after = {path: digest(Path(path)) for path in before}
    assert before == after, "Artifacts changed during readback"
    audit = {"uid": f"{sample}|{neuron}.swc", "status": "passed", "sources_hash_verified": verified_sources,
             "sampled_pixel_checks": samples, "artifact_hashes": before, "stable_during_read": True,
             "limitations": ["Nominal source coordinate convention only; absolute calibration and orientation unverified.",
                             "Full-section locator and fine anatomical acceptance remain unavailable."]}
    target = ROOT / "notes" / "bulk_visual_review_20261002" / f"derived_pilot_readback_{sample}_{neuron}_20261003.json"
    target.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(target), "sources": verified_sources, "pixel_checks": len(samples)}))
    return audit


if __name__ == "__main__":
    verify()
