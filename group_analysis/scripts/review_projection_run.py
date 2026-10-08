"""Independently read back complete projection outputs and optional MIP sheets.

Checks saved pixels/geometry, full manifest identities/source hashes, length
accounting, density units and equal-animal aggregation. This validates the
software artifact; it cannot establish anatomical/tracing/registration validity.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd

from build_projection_maps import sha256
from projection_maps import ReferenceGrid


REVIEWER_VERSION = "2.0"
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    """Fail closed even when Python's optimization disables bare assertions."""
    if not condition:
        raise ValueError(message)


def checked_hash(path, expected, description):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-fA-F]{64}", expected),
            f"Missing or invalid {description} SHA256")
    require(Path(path).is_file(), f"Missing {description}: {path}")
    actual = sha256(path)
    require(actual == expected.lower(), f"{description} hash mismatch: {path}")
    return actual


def run_file(run_dir, name):
    require(isinstance(name, str) and bool(name.strip()), "Missing saved artifact path")
    path = (run_dir / name).resolve()
    require(path.is_relative_to(run_dir.resolve()), f"Saved artifact escapes run directory: {name}")
    return path


def review(run_dir, output, render=False):
    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError("Use a new readback destination")
    record = json.loads((run_dir / "run_provenance.json").read_text(encoding="utf-8"))
    required = {"status", "preserved_manifest", "manifest_sha256", "reference", "reference_sha256",
                "brain_mask", "brain_mask_sha256", "source_code_sha256", "shape", "affine_mm",
                "voxel_volume_mm3", "selected_neurons", "animal_maps", "group_maps",
                "length_value_units", "density_value_units"}
    require(required <= record.keys(), f"Missing run provenance fields: {sorted(required - record.keys())}")
    require(record["status"] == "software_verified_descriptive", "Run is not terminal and software verified")
    has_mask = record["brain_mask"] is not None
    require(has_mask or record["brain_mask_sha256"] is None, "Mask hash present without a brain mask")
    require(not render or has_mask, "Rendering requires a brain mask; use readback without --render for a mask-free run")
    manifest_path = run_file(run_dir, record["preserved_manifest"])
    checked_hash(manifest_path, record["manifest_sha256"], "Preserved manifest")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    manifest_columns = {"SampleID", "NeuronID", "AnimalID", "Subregion", "SWCPath", "SWCSHA256",
                        "ReferenceSHA256", "CoordinateFrame", "IndexScaleUm", "AnatomyStatus", "RegistrationStatus"}
    require(manifest_columns <= set(manifest.columns) and not manifest.empty, "Missing required manifest columns or empty manifest")
    for name in manifest_columns - {"IndexScaleUm"}:
        require(manifest[name].str.strip().ne("").all() and manifest[name].eq(manifest[name].str.strip()).all(),
                f"Missing or whitespace-padded manifest {name}")
    require(not manifest.AnimalID.str.lower().isin(["unknown", "tbd", "na", "nan", "none"]).any(),
            "Animal identity must be established")
    require(manifest.groupby("SampleID").AnimalID.nunique().eq(1).all(), "Sample maps to multiple animals")
    metrics_path = run_dir / "per_neuron_measurements.csv"
    metrics_hash = sha256(metrics_path)
    metadata_columns = ["SampleID", "NeuronID", "AnimalID", "Subregion", "SWCSHA256", "AnatomyStatus", "RegistrationStatus"]
    metrics = pd.read_csv(metrics_path, dtype={name: str for name in metadata_columns}, keep_default_na=False)
    lengths = ["selected_axon_length_mm", "in_reference_length_mm", "outside_reference_length_mm"]
    coverage = ["in_brain_mask_length_mm", "outside_brain_mask_in_FOV_length_mm"]
    require(set(metadata_columns + lengths + ["coordinate_frame"]) <= set(metrics.columns),
            "Missing required per-neuron measurements/identity columns")
    if has_mask:
        require(set(coverage) <= set(metrics.columns), "Brain mask run is missing coverage measurements")
        lengths += coverage
    uid = lambda frame: list(zip(frame.SampleID, frame.NeuronID))
    require(len(manifest) == len(metrics) == record["selected_neurons"], "Selected neuron count mismatch")
    require(len(set(uid(manifest))) == len(manifest) and len(set(uid(metrics))) == len(metrics)
            and set(uid(manifest)) == set(uid(metrics)), "Duplicate or mismatched composite neuron identities")
    aligned = metrics.set_index(["SampleID", "NeuronID"]).reindex(pd.MultiIndex.from_tuples(uid(manifest)))
    for name in metadata_columns[2:] + ["coordinate_frame"]:
        source_name = "CoordinateFrame" if name == "coordinate_frame" else name
        require(aligned[name].tolist() == manifest[source_name].tolist(), f"Per-neuron {name} differs from manifest")
    for name in lengths:
        metrics[name] = pd.to_numeric(metrics[name], errors="raise")
        require(np.isfinite(metrics[name]).all() and (metrics[name] >= 0).all(), f"Invalid nonnegative length measurement: {name}")
    require(np.isfinite(metrics.select_dtypes(include="number")).all().all(), "Nonfinite numerical measurements")
    np.testing.assert_allclose(metrics.in_reference_length_mm + metrics.outside_reference_length_mm,
                               metrics.selected_axon_length_mm, rtol=1e-10, atol=1e-8)
    if has_mask:
        np.testing.assert_allclose(metrics.in_brain_mask_length_mm + metrics.outside_brain_mask_in_FOV_length_mm,
                                   metrics.in_reference_length_mm, rtol=1e-10, atol=1e-8)
    sources = set()
    for source in manifest.to_dict("records"):
        path = Path(source["SWCPath"])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.name == source["NeuronID"] and path not in sources, "SWC filename identity mismatch or source reused")
        sources.add(path)
        checked_hash(path, source["SWCSHA256"], "SWC source")
        require(source["ReferenceSHA256"].lower() == record["reference_sha256"].lower(), "Manifest reference hash mismatch")
        if source["CoordinateFrame"] == "atlas_index_um":
            scale = np.asarray([float(v) for v in source["IndexScaleUm"].split(";")])
            require(scale.shape == (3,) and np.isfinite(scale).all() and (scale > 0).all(), "Invalid index coordinate scales")
        else:
            require(source["CoordinateFrame"] == "nifti_world_mm" and not source["IndexScaleUm"], "Invalid coordinate frame or index scales")
    checked_hash(record["reference"], record["reference_sha256"], "Reference")
    if has_mask:
        checked_hash(record["brain_mask"], record["brain_mask_sha256"], "Brain mask")
    source_code = {"build_projection_maps.py": ROOT / "group_analysis/scripts/build_projection_maps.py",
                   "projection_maps.py": ROOT / "main_scripts/projection_maps.py",
                   "swc_validation.py": ROOT / "main_scripts/swc_validation.py"}
    require(set(record["source_code_sha256"]) == set(source_code), "Missing or unexpected source code provenance")
    for name, path in source_code.items():
        checked_hash(path, record["source_code_sha256"][name], name)
    shape, affine = tuple(record["shape"]), np.asarray(record["affine_mm"])
    grid = ReferenceGrid(shape, affine)
    reference_grid = ReferenceGrid.from_image(nib.load(record["reference"]))
    require(grid.shape == reference_grid.shape and np.allclose(grid.affine_mm, reference_grid.affine_mm), "Recorded reference geometry mismatch")
    volume = grid.voxel_volume_mm3
    require(np.isclose(volume, record["voxel_volume_mm3"]), "Recorded voxel volume mismatch")
    require(record["length_value_units"] == "mm per selected neuron"
            and record["density_value_units"] == "mm per mm^3 per selected neuron", "Map value units mismatch")
    if has_mask:
        mask_image = nib.load(record["brain_mask"])
        mask_grid = ReferenceGrid.from_image(mask_image)
        require(mask_grid.shape == shape and np.allclose(mask_grid.affine_mm, affine)
                and np.isfinite(mask_image.get_fdata()).all(), "Brain mask geometry or pixels invalid")
    checked = []
    map_paths = set()

    def load_pair(entry):
        arrays = []
        for kind in ("length", "density"):
            path = run_file(run_dir, entry[f"{kind}_path"])
            require(path not in map_paths, "Saved map path reused across entries")
            map_paths.add(path)
            actual_hash = checked_hash(path, entry[f"{kind}_sha256"], "Saved map")
            image = nib.load(path)
            map_grid = ReferenceGrid.from_image(image)
            require(map_grid.shape == shape and np.allclose(map_grid.affine_mm, affine), "Saved map geometry mismatch")
            require(image.header.get_xyzt_units()[0] == "mm", "Saved map spatial units must be mm")
            data = image.get_fdata()
            require(np.isfinite(data).all() and np.all(data >= 0), "Saved map contains nonfinite or negative pixels")
            arrays.append(data)
            checked.append({"file": str(path.relative_to(run_dir.resolve())), "sha256": actual_hash,
                            "voxels": int(data.size), "sum": float(data.sum()), "maximum": float(data.max())})
        np.testing.assert_allclose(arrays[1], arrays[0] / volume, rtol=1e-6, atol=1e-7)
        return arrays[0]

    expected_groups = set(zip(manifest.AnimalID, manifest.Subregion))
    actual_groups = [(entry["AnimalID"], entry["Subregion"]) for entry in record["animal_maps"]]
    require(len(actual_groups) == len(set(actual_groups)) and set(actual_groups) == expected_groups,
            "Duplicate, missing or extra animal/subregion maps")
    for entry in record["animal_maps"]:
        length = load_pair(entry)
        group = metrics[(metrics.AnimalID == entry["AnimalID"]) & (metrics.Subregion == entry["Subregion"])]
        require(len(group) == entry["selected_neurons"], "Animal map selected-neuron count mismatch")
        np.testing.assert_allclose(length.sum(), group.in_reference_length_mm.mean(), rtol=1e-6, atol=1e-7)
    regions = set(manifest.Subregion)
    group_regions = [entry["Subregion"] for entry in record["group_maps"]]
    require(len(group_regions) == len(set(group_regions)) and set(group_regions) == regions,
            "Duplicate, missing or extra group regions")
    for entry in record["group_maps"]:
        length = load_pair(entry)
        animals = [item for item in record["animal_maps"] if item["Subregion"] == entry["Subregion"]]
        require(entry["contributing_animals"] == [item["AnimalID"] for item in animals], "Contributing animal identities/order mismatch")
        require(entry["n_animals"] == len(animals), "Group animal count mismatch")
        require(entry["n_neurons"] == sum(item["selected_neurons"] for item in animals), "Group neuron count mismatch")
        expected = np.zeros(shape)
        for animal in animals:
            expected += nib.load(run_dir / animal["length_path"]).get_fdata() / len(animals)
        np.testing.assert_allclose(length, expected, rtol=1e-6, atol=1e-7)
    require(sha256(metrics_path) == metrics_hash, "Measurements changed during readback")
    output.mkdir(parents=True)
    report = {
        "status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": {"version": REVIEWER_VERSION, "source": str(Path(__file__).resolve()),
                     "sha256": sha256(__file__), "python": sys.version,
                     "numpy": np.__version__, "nibabel": nib.__version__, "pandas": pd.__version__},
        "run_provenance_sha256": sha256(run_dir / "run_provenance.json"),
        "manifest_sha256": sha256(manifest_path), "neurons": len(manifest),
        "per_neuron_measurements_sha256": metrics_hash,
        "animal_subregions": len(actual_groups), "animals": manifest.AnimalID.unique().tolist(),
        "all_map_files": checked, "full_saved_voxels_read": sum(item["voxels"] for item in checked),
        "total_template_axon_length_mm": float(metrics.selected_axon_length_mm.sum()),
        "outside_reference_length_mm": float(metrics.outside_reference_length_mm.sum()),
        "brain_mask_coverage_status": "assessed" if has_mask else "not assessed; no brain mask supplied",
        "outside_brain_mask_in_FOV_length_mm": float(metrics.outside_brain_mask_in_FOV_length_mm.sum()) if has_mask else None,
        "anatomical_acceptance": "not established", "statistical_inference": "none",
    }
    if render:
        report["figures"] = render_sheets(record, run_dir, output)
    with (output / "projection_run_readback.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def render_sheets(record, run_dir, output):
    require(record.get("brain_mask") is not None, "Rendering requires a brain mask")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image
    reference = nib.load(record["reference"]).get_fdata()
    mask = nib.load(record["brain_mask"]).get_fdata() > 0
    entries = record["animal_maps"]
    projections = []
    for entry in entries:
        data = nib.load(run_dir / entry["density_path"]).get_fdata()
        projections.append([np.log10(1 + data.max(axis=axis)).T for axis in (2, 1, 0)])
    values = np.concatenate([plane[plane > 0] for planes in projections for plane in planes])
    vmax = float(np.percentile(values, 99.5)) if len(values) else 1.0
    figures = []
    for start in range(0, len(entries), 4):
        count = min(4, len(entries) - start)
        figure, axes = plt.subplots(count, 3, figsize=(12, count*3.0), squeeze=False, layout="constrained")
        for offset in range(count):
            entry, planes = entries[start+offset], projections[start+offset]
            for column, axis in enumerate((2, 1, 0)):
                panel, data = axes[offset, column], planes[column]
                panel.imshow(reference.max(axis=axis).T, cmap="gray", origin="lower")
                panel.contour(mask.max(axis=axis).T, levels=[.5], colors="white", linewidths=.35)
                plotted = panel.imshow(np.ma.masked_where(data <= 0, data), cmap="inferno",
                                       norm=Normalize(0, vmax), origin="lower", interpolation="nearest")
                panel.set_title(f"{entry['Subregion']} | n={entry['selected_neurons']} | animal {entry['AnimalID']}", fontsize=9)
                panel.set_xlabel(["X voxel", "X voxel", "Y voxel"][column], fontsize=8)
                panel.set_ylabel(["Y voxel", "Z voxel", "Z voxel"][column], fontsize=8)
                panel.tick_params(labelsize=7)
        figure.colorbar(plotted, ax=axes, shrink=.7, label="log10(1 + mean template axon length density [mm/mm³/neuron])")
        figure.suptitle("Descriptive INS projection maps · unmirrored NMT v2.1 grid\n"
                        "Maximum-intensity projections; child type 2; anatomy/registration provisional; no t test", fontsize=11)
        path = output / f"space-{record['space_entity']}_desc-axonLengthDensity_mip_page-{start//4+1:02d}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        with Image.open(path) as image:
            image.load()
            size = image.size
        figures.append({"file": path.name, "sha256": sha256(path), "pixels": size,
                        "display_transform": "log10(1 + maximum density along declared voxel axis)",
                        "common_display_vmax": vmax, "upper_percentile_clipping": 99.5,
                        "data_maps_modified": False})
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    result = review(args.run, args.output, args.render)
    print(f"{result['status']}: {result['neurons']} neurons; {len(result['all_map_files'])} complete map files")
