"""Build descriptive maps of axon-labelled segment length in NMT reference space.

Within each animal and selected source group, each voxel shows length summed
over selected neurons divided by their number. Density additionally divides
by reference voxel volume. Across animals, average the contributing animal
maps equally; a missing source group is not an observed zero. These are
descriptive physical measurements. Axon labels come from SWC node types and
still require compartment and registration review.

The manifest records AnimalID, SampleID, NeuronID, Subregion (the declared
source group), SWCPath, SWCSHA256, ReferenceSHA256, CoordinateFrame,
IndexScaleUm, AnatomyStatus and RegistrationStatus. atlas_index_um uses
three semicolon-separated scales; relative SWC paths use --input-root.
Exact identities and coordinate provenance must be supplied. The output
directory must be new. Source data and previous results remain preserved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main_scripts"))
from projection_maps import ReferenceGrid, axon_length_map, save_map


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_manifest(path, input_root):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"AnimalID", "SampleID", "NeuronID", "Subregion", "SWCPath",
                "SWCSHA256", "ReferenceSHA256", "CoordinateFrame", "IndexScaleUm", "AnatomyStatus",
                "RegistrationStatus"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError(f"Nonempty manifest requires columns {sorted(required)}")
    for column in required - {"IndexScaleUm"}:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Manifest contains missing {column}")
        if not frame[column].eq(frame[column].str.strip()).all():
            raise ValueError(f"Manifest {column} contains surrounding whitespace")
    if frame.AnimalID.str.lower().isin(["unknown", "tbd", "na", "nan", "none"]).any():
        raise ValueError("Animal identity must be established before animal aggregation")
    if frame.duplicated(["SampleID", "NeuronID"]).any():
        raise ValueError("Duplicate composite neuron identity")
    for sample, group in frame.groupby("SampleID"):
        if group.AnimalID.nunique() != 1:
            raise ValueError(f"Sample {sample} maps to multiple animals")
    records = []
    source_paths = set()
    for row in frame.to_dict("records"):
        for column in ("AnimalID", "SampleID", "Subregion"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", row[column]):
                raise ValueError(f"{column} must be a safe explicit output label")
        if Path(row["NeuronID"]).name != row["NeuronID"] or not row["NeuronID"].endswith(".swc"):
            raise ValueError("NeuronID must preserve an exact SWC basename")
        source = Path(row["SWCPath"])
        if not source.is_absolute():
            source = Path(input_root) / source
        source = source.resolve(strict=True)
        if source in source_paths:
            raise ValueError("One SWC source path was assigned to multiple neuron identities")
        source_paths.add(source)
        if source.name != row["NeuronID"]:
            raise ValueError("SWC filename and explicit neuron identity differ")
        expected = row["SWCSHA256"].lower()
        if not re.fullmatch("[0-9a-f]{64}", expected) or sha256(source) != expected:
            raise ValueError(f"SWC hash mismatch: {source}")
        scale = row["IndexScaleUm"]
        if row["CoordinateFrame"] == "atlas_index_um":
            scale = [float(value) for value in scale.split(";")]
            if len(scale) != 3 or not np.isfinite(scale).all() or min(scale) <= 0:
                raise ValueError("Atlas index encoding requires three positive scales")
        elif row["CoordinateFrame"] == "nifti_world_mm" and not scale:
            scale = None
        else:
            raise ValueError("Unknown coordinate frame or incompatible index scale")
        row.update(source_path=source, expected_sha256=expected, index_scale_um=scale)
        records.append(row)
    return records


def build(manifest, reference, output, *, input_root=ROOT, brain_mask=None,
          space="NMTv2p1"):
    manifest, reference, output = Path(manifest), Path(reference), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing results are not replaced")
    if not re.fullmatch(r"[A-Za-z0-9]+", space):
        raise ValueError("Explicit reference-space entity must be alphanumeric")
    manifest_hash, reference_hash = sha256(manifest), sha256(reference)
    records = checked_manifest(manifest, input_root)
    if any(row["ReferenceSHA256"].lower() != reference_hash for row in records):
        raise ValueError("SWC manifest does not bind the selected reference hash")
    groups = defaultdict(list)
    for record in records:
        groups[(record["AnimalID"], record["Subregion"])].append(record)
    # Entity delimiters and Windows case folding can otherwise map distinct
    # labels to the same filename. Reject the entire plan before writing.
    planned = []
    for animal, region in groups:
        prefix = f"animal-{animal}_space-{space}_region-{region}"
        planned.extend(output / "animal_maps" / f"{prefix}_desc-{metric}_map.nii.gz"
                       for metric in ("meanAxonLength", "meanAxonLengthDensity"))
    for region in {key[1] for key in groups}:
        prefix = f"space-{space}_region-{region}"
        planned.extend(output / "group_maps" / f"{prefix}_desc-{metric}_map.nii.gz"
                       for metric in ("animalMeanAxonLength", "animalMeanAxonLengthDensity"))
    keys = [str(p.resolve()).replace("\\", "/").casefold() for p in planned]
    if len(keys) != len(set(keys)):
        raise ValueError("Distinct labels collide in output paths, including case-insensitive names")
    if any(len(str(p.resolve())) > 240 for p in planned):
        raise ValueError("Output path exceeds the portable 240-character limit; shorten labels/destination")
    image = nib.load(str(reference))
    grid = ReferenceGrid.from_image(image)
    mask = None
    mask_hash = sha256(brain_mask) if brain_mask else None
    if brain_mask:
        mask_image = nib.load(str(brain_mask))
        mask_grid = ReferenceGrid.from_image(mask_image)
        mask_data = mask_image.get_fdata()
        if (mask_grid.shape != grid.shape or not np.allclose(mask_grid.affine_mm, grid.affine_mm)
                or not np.isfinite(mask_data).all()):
            raise ValueError("Brain mask must match reference geometry and contain finite values")
        mask = mask_data > 0
    output.mkdir(parents=True)
    animal_dir, group_dir = output / "animal_maps", output / "group_maps"
    animal_dir.mkdir()
    group_dir.mkdir()
    run = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(manifest.resolve()), "manifest_sha256": manifest_hash,
        "reference": str(reference.resolve()), "reference_sha256": reference_hash,
        "brain_mask": str(Path(brain_mask).resolve()) if brain_mask else None,
        "brain_mask_sha256": mask_hash, "shape": grid.shape,
        "brain_mask_policy": "coverage accounting only; delivered maps retain all in-FOV edges",
        "affine_mm": grid.affine_mm.tolist(), "voxel_volume_mm3": grid.voxel_volume_mm3,
        "space_entity": space, "selected_neurons": len(records),
        "measurement": "template-space child-type-2 edge length split at voxel faces",
        "regional_aggregation": "mean length per selected neuron within animal and declared subregion",
        "group_aggregation": "equal weights among contributing animals; absent subregions not zero-filled",
        "hemisphere_policy": "unmirrored reference coordinates; manifest subregion labels retained",
        "statistical_status": "descriptive; no t map, p value, or multiplicity inference",
        "scientific_status": "exploratory; registration, compartment labeling and anatomy require review",
        "source_code_sha256": {p.name: sha256(p) for p in
                               [Path(__file__), ROOT / "main_scripts/projection_maps.py",
                                ROOT / "main_scripts/swc_validation.py"]},
        "versions": {"python": sys.version, "numpy": np.__version__, "nibabel": nib.__version__},
        "preserved_manifest": "input_manifest.csv",
        "length_value_units": "mm per selected neuron",
        "density_value_units": "mm per mm^3 per selected neuron",
        "animal_maps": [], "group_maps": [],
    }
    provenance_path = output / "run_provenance.json"

    def checkpoint():
        provenance_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    checkpoint()
    neuron_metrics = []
    try:
        copied_manifest = output / "input_manifest.csv"
        copied_manifest.write_bytes(manifest.read_bytes())
        if sha256(copied_manifest) != manifest_hash:
            raise RuntimeError("Input manifest changed before preserving the run snapshot")
        for (animal, region), selected in sorted(groups.items()):
            accumulated = np.zeros(grid.shape, dtype=np.float64)
            for record in selected:
                source = record["source_path"]
                data, metrics = axon_length_map(
                    source.read_text(encoding="utf-8"), grid,
                    coordinate_frame=record["CoordinateFrame"],
                    index_scale_um=record["index_scale_um"], source=str(source))
                if sha256(source) != record["expected_sha256"]:
                    raise RuntimeError(f"Source changed while computing: {source}")
                if mask is not None:
                    metrics["in_brain_mask_length_mm"] = float(data[mask].sum())
                    metrics["outside_brain_mask_in_FOV_length_mm"] = float(data[~mask].sum())
                accumulated += data
                metrics.update({key: record[key] for key in
                                ("AnimalID", "SampleID", "NeuronID", "Subregion",
                                 "AnatomyStatus", "RegistrationStatus", "SWCSHA256")})
                neuron_metrics.append(metrics)
            accumulated /= len(selected)
            prefix = f"animal-{animal}_space-{space}_region-{region}"
            length_path = animal_dir / f"{prefix}_desc-meanAxonLength_map.nii.gz"
            stored_sum = save_map(length_path, accumulated, grid)
            density_path = animal_dir / f"{prefix}_desc-meanAxonLengthDensity_map.nii.gz"
            save_map(density_path, accumulated / grid.voxel_volume_mm3, grid)
            run["animal_maps"].append({
                "AnimalID": animal, "Subregion": region, "selected_neurons": len(selected),
                "length_path": str(length_path.relative_to(output)), "length_sha256": sha256(length_path),
                "density_path": str(density_path.relative_to(output)), "density_sha256": sha256(density_path),
                "mean_in_reference_length_mm": float(accumulated.sum()),
                "stored_float32_sum_mm": stored_sum,
            })
            checkpoint()
            print(f"animal={animal} region={region} neurons={len(selected)}", flush=True)
        by_region = defaultdict(list)
        for entry in run["animal_maps"]:
            by_region[entry["Subregion"]].append(entry)
        for region, entries in sorted(by_region.items()):
            group = np.zeros(grid.shape, dtype=np.float64)
            for entry in entries:
                group += nib.load(str(output / entry["length_path"])).get_fdata() / len(entries)
            prefix = f"space-{space}_region-{region}"
            length_path = group_dir / f"{prefix}_desc-animalMeanAxonLength_map.nii.gz"
            density_path = group_dir / f"{prefix}_desc-animalMeanAxonLengthDensity_map.nii.gz"
            save_map(length_path, group, grid)
            save_map(density_path, group / grid.voxel_volume_mm3, grid)
            run["group_maps"].append({
                "Subregion": region, "contributing_animals": [entry["AnimalID"] for entry in entries],
                "n_animals": len(entries), "n_neurons": sum(entry["selected_neurons"] for entry in entries),
                "length_path": str(length_path.relative_to(output)), "length_sha256": sha256(length_path),
                "density_path": str(density_path.relative_to(output)), "density_sha256": sha256(density_path),
            })
        if (sha256(manifest) != manifest_hash or sha256(reference) != reference_hash
                or (brain_mask and sha256(brain_mask) != mask_hash)):
            raise RuntimeError("Manifest/reference/mask changed during mapping")
        pd.DataFrame(neuron_metrics).to_csv(output / "per_neuron_measurements.csv", index=False)
        run["status"] = "software_verified_descriptive"
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        checkpoint()
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        checkpoint()
        raise
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help='CSV of exact neuron identities, source groups, SWC paths/hashes and coordinate provenance.')
    parser.add_argument("--reference", type=Path, required=True, help='Pinned NMT reference image bound by every manifest row.')
    parser.add_argument("--brain-mask", type=Path, help='Optional matching mask for coverage accounting; it does not trim the map.')
    parser.add_argument("--input-root", type=Path, default=ROOT, help='Base directory for relative SWC paths in the manifest.')
    parser.add_argument("--space", default="NMTv2p1", help='Existing reference-space name used in output filenames; this does not register data.')
    parser.add_argument("--output", type=Path, required=True, help='New destination directory for descriptive length maps and provenance.')
    args = parser.parse_args()
    result = build(args.manifest, args.reference, args.output,
                   input_root=args.input_root, brain_mask=args.brain_mask, space=args.space)
    print(f"{result['status']}: {result['selected_neurons']} neurons; "
          f"{len(result['animal_maps'])} animal/subregion maps")
