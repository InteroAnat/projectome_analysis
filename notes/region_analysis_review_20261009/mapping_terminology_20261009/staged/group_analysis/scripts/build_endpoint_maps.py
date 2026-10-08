"""Build descriptive maps of candidate axon ends and actual ARM-level labels.

A candidate axon end is a non-root axon-labelled (SWC type 2) node with no
children in the complete stored reconstruction. This graph rule does not
verify an image-reviewed terminal arbor, synapse or termination site.
An endpoint-eligible neuron has at least one such end anywhere, including
outside the reference field of view. Neurons without one remain unassessed.

For each animal and declared source group, endpoint count/density is the
candidate count per eligible neuron (density also divides by reference voxel
volume). Occupancy is the fraction of eligible neurons with at least one end
in a voxel; multiple ends from one neuron count once. Across animals, average
contributing animal maps equally. All selected neurons remain auditable;
no image review, registration or statistical inference occurs here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_projection_maps import checked_manifest, sha256
from endpoint_atlas import EndpointAtlas, coded_mm_grid, matching_grid
from region_analysis.laterality import LateralityParser
from terminal_sites import terminal_site_summary

IDENTITY_COLUMNS = ["AnimalID", "SampleID", "NeuronID", "Subregion", "OriginalSourceLabel",
                    "AnatomyStatus", "RegistrationStatus", "SWCPath", "SWCSHA256",
                    "ReferenceSHA256", "CoordinateFrame", "IndexScaleUm", "source_metadata_json"]
QC_COLUMNS = IDENTITY_COLUMNS + [
    "node_count", "edge_count", "root_node_id", "type2_node_count", "leaf_count",
    "candidate_axon_endpoint_count", "in_reference_candidate_count", "outside_reference_candidate_count",
    "occupied_candidate_voxel_count", "zero_length_candidate_terminal_edges", "candidate_status",
    "map_proxy_status", "n_selected", "n_computable", "biological_innervation_state", "image_review_state",
    "soma_anchor_state", "soma_anchor_node_id", "node_type_counts_json", "leaf_type_counts_json",
    "leaf_class_counts_json", "type1_node_ids_json", "in_brain_mask_candidate_count",
    "outside_brain_mask_in_FOV_candidate_count", "unknown_target_counts_by_level_json",
    "hemisphere_conflict_counts_by_level_json",
]
TARGET_COLUMNS = ["level", "target_status", "target_index", "target_abbreviation",
                  "target_full_name", "target_volume_mm3"]
NEURON_TARGET_COLUMNS = IDENTITY_COLUMNS + TARGET_COLUMNS + ["endpoint_count", "neurons_with_target_evidence"]
REGION_COLUMNS = ["AnimalID", "Subregion"] + TARGET_COLUMNS + [
    "endpoint_count", "neurons_with_target_evidence", "n_selected", "n_computable",
    "frequency_per_computable_neuron", "mean_endpoints_per_computable_neuron",
    "map_available", "biological_innervation_state",
]
METRICS = {
    "count": ("meanCandidateEndpointCount", "candidate endpoints per computable neuron"),
    "density": ("meanCandidateEndpointDensity", "candidate endpoints per template mm3 per computable neuron"),
    "occupancy": ("candidateEndpointOccupancy", "fraction of computable neurons with candidate endpoint evidence"),
}


def _save_endpoint_map(path, values, grid, metric):
    """Validate every stored pixel, then publish atomically without replacement."""
    stored = np.asarray(values, dtype=np.float32)
    if stored.shape != grid.shape or not np.isfinite(stored).all() or np.any(stored < 0):
        raise ValueError("Endpoint map must match reference and contain finite nonnegative values")
    if metric == "occupancy" and np.any(stored > 1):
        raise ValueError("Endpoint occupancy must be between zero and one")
    image = nib.Nifti1Image(stored, grid.affine_mm)
    image.header.set_xyzt_units("mm")
    image.set_sform(grid.affine_mm, code=2)
    image.set_qform(None, code=0)
    image.header["descrip"] = "Candidate endpoint proxy; unresolved anatomy; not synapses or t statistic"
    descriptor, temporary = tempfile.mkstemp(prefix=".endpoint-map-", suffix=".nii.gz", dir=path.parent)
    os.close(descriptor)
    try:
        nib.save(image, temporary)
        check = nib.load(temporary)
        if (check.shape != grid.shape or not np.allclose(check.affine, grid.affine_mm, rtol=0, atol=1e-7)
                or check.header.get_xyzt_units()[0] != "mm"
                or not np.array_equal(np.asanyarray(check.dataobj), stored)):
            raise IOError("Endpoint map failed complete pixel/geometry readback")
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _paths(output, animal, region, space):
    folder = "animal_maps" if animal is not None else "group_maps"
    prefix = f"animal-{animal}_space-{space}_region-{region}" if animal is not None else f"space-{space}_region-{region}"
    return {metric: output / folder / f"{prefix}_desc-{descriptor}_map.nii.gz"
            for metric, (descriptor, _) in METRICS.items()}


def _identity(record):
    metadata = {key: value for key, value in record.items()
                if key not in ("source_path", "expected_sha256", "index_scale_um")}
    identity = {key: record.get(key, "") for key in IDENTITY_COLUMNS if key != "source_metadata_json"}
    identity["source_metadata_json"] = json.dumps(metadata, sort_keys=True, allow_nan=False)
    return identity


def _target_columns(target):
    return dict(level=target["level"], target_status=target["target_status"], target_index=target["index"],
                target_abbreviation=target["abbreviation"], target_full_name=target["full_name"],
                target_volume_mm3=target["target_volume_mm3"])


def build(manifest, reference, output, *, atlas_path, atlas_key, hemisphere_mask,
          brain_mask=None, input_root=ROOT, space="NMTv2p1"):
    """Create a new run directory; source files and previous runs are untouched."""
    manifest, reference, output = Path(manifest), Path(reference), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing results are not replaced")
    if not re.fullmatch(r"[A-Za-z0-9]+", space):
        raise ValueError("Explicit reference-space entity must be alphanumeric")
    inputs = {"manifest": manifest, "reference": reference, "atlas": Path(atlas_path),
              "atlas_key": Path(atlas_key), "hemisphere_mask": Path(hemisphere_mask)}
    if brain_mask is not None:
        inputs["brain_mask"] = Path(brain_mask)
    hashes = {name: sha256(path) for name, path in inputs.items()}
    records = checked_manifest(manifest, input_root)
    if any(record["ReferenceSHA256"].lower() != hashes["reference"] for record in records):
        raise ValueError("SWC manifest does not bind the selected reference hash")
    groups = defaultdict(list)
    for record in records:
        groups[(record["AnimalID"], record["Subregion"])].append(record)
    planned = [output / name for name in ("run_provenance.json", "input_manifest.csv", "leaf_records.jsonl",
                                        "per_neuron_qc.csv", "per_neuron_target_counts.csv", "animal_region_summaries.csv")]
    for animal, region in groups:
        planned.extend(_paths(output, animal, region, space).values())
    for region in {key[1] for key in groups}:
        planned.extend(_paths(output, None, region, space).values())
    keys = [str(path.resolve()).replace("\\", "/").casefold() for path in planned]
    if len(set(keys)) != len(keys):
        raise ValueError("Distinct labels collide in output paths, including case-insensitive names")
    if any(len(str(path.resolve())) > 240 for path in planned):
        raise ValueError("Output path exceeds portable 240-character limit")

    grid = coded_mm_grid(nib.load(str(reference)))
    atlas = EndpointAtlas(atlas_path, atlas_key, hemisphere_mask, grid)
    mask = None
    if brain_mask is not None:
        mask_image = nib.load(str(brain_mask))
        matching_grid(mask_image, grid)
        values = np.asanyarray(mask_image.dataobj)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Brain mask must contain finite nonnegative values")
        mask = values > 0
    # Catch inputs modified while validating, before creating a destination.
    if any(sha256(path) != hashes[name] for name, path in inputs.items()):
        raise RuntimeError("An input changed during validation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "animal_maps").mkdir()
    (output / "group_maps").mkdir()
    code_paths = [Path(__file__), ROOT / "main_scripts/endpoint_atlas.py", ROOT / "main_scripts/terminal_sites.py",
                  ROOT / "main_scripts/projection_maps.py", ROOT / "main_scripts/swc_validation.py",
                  Path(__file__).with_name("build_projection_maps.py"), ROOT / "main_scripts/region_analysis/laterality.py"]
    run = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: {"path": str(path.resolve()), "sha256": hashes[name]} for name, path in inputs.items()},
        "input_root": str(Path(input_root).resolve()),
        "reference": str(reference.resolve()), "reference_sha256": hashes["reference"],
        "space_entity": space, "shape": grid.shape, "affine_mm": grid.affine_mm.tolist(),
        "voxel_volume_mm3": grid.voxel_volume_mm3, "n_selected": len(records), "n_computable": 0,
        "measurement": "non-root type2 zero-child leaves in complete graph; candidate endpoints only",
        "map_denominator": "neurons with at least one full-graph candidate axonal leaf anywhere, including outside FOV",
        "group_aggregation": "equal contributing-animal means within each declared source subregion; missing groups not imputed",
        "brain_mask_policy": "coverage accounting only; in-FOV endpoints retained regardless of mask",
        "atlas_policy": "direct voxel lookup at each of six actual ARM volumes; no inferred hierarchy parents",
        "target_volume_policy": "mapped label voxel count at its actual atlas level across the full reference grid times template voxel volume; no brain-mask trimming",
        "hemisphere_policy": "unmirrored coordinates; mask semantics validated against finest atlas labels",
        "hemisphere_value_to_side": atlas.value_to_side, "hemisphere_label_contingency": atlas.contingency,
        "image_review_state": "unresolved", "biological_innervation_state": "unassessed",
        "scientific_status": "exploratory candidate proxy; compartment, tracing, anatomy and registration require review",
        "statistical_status": "descriptive; no smoothing, t/p values, or synapse claims",
        "map_value_units": {metric: definition[1] for metric, definition in METRICS.items()},
        "source_code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
        "versions": {"python": sys.version, "numpy": np.__version__, "nibabel": nib.__version__, "pandas": pd.__version__},
        "preserved_manifest": "input_manifest.csv", "animal_maps": [], "group_maps": [],
        "unassessed_neurons": [], "unresolved_compartment_neurons": [], "artifacts": {},
    }
    qc_rows, neuron_targets, region_rows = [], [], []
    group_counters = {}
    accounting = Counter()
    # Keep the originally created provenance descriptor open. Checkpointing
    # updates this owned file, never opens/truncates a concurrently added file.
    with (output / "run_provenance.json").open("x+", encoding="utf-8") as provenance:
        def checkpoint():
            provenance.seek(0)
            json.dump(run, provenance, indent=2, allow_nan=False)
            provenance.truncate()
            provenance.flush()

        checkpoint()
        try:
            copied = output / "input_manifest.csv"
            with copied.open("xb") as stream:
                stream.write(manifest.read_bytes())
            if sha256(copied) != hashes["manifest"]:
                raise RuntimeError("Input manifest changed before preservation")
            with (output / "leaf_records.jsonl").open("x", encoding="utf-8", newline="\n") as leaf_stream:
                for (animal, region), selected in sorted(groups.items()):
                    count_voxels, occupied_voxels = Counter(), Counter()
                    target_counts, target_neurons, target_metadata = Counter(), Counter(), {}
                    n_computable = 0
                    for record in selected:
                        source = record["source_path"]
                        payload = source.read_bytes()
                        if hashlib.sha256(payload).hexdigest() != record["expected_sha256"]:
                            raise RuntimeError(f"Source changed before computing: {source}")
                        result = terminal_site_summary(payload.decode("utf-8-sig"), grid,
                            coordinate_frame=record["CoordinateFrame"], index_scale_um=record["index_scale_um"], source=str(source))
                        if sha256(source) != record["expected_sha256"]:
                            raise RuntimeError(f"Source changed while computing: {source}")
                        identity = _identity(record)
                        qc = result["qc"]
                        computable = qc["candidate_axon_endpoint_count"] > 0
                        n_computable += int(computable)
                        if not computable:
                            run["unassessed_neurons"].append({**identity, "reason": qc["candidate_status"]})
                        unresolved = {kind: count for kind, count in qc["node_type_counts"].items()
                                      if int(kind) not in (1, 2, 3, 4)}
                        if unresolved:
                            run["unresolved_compartment_neurons"].append({**identity, "node_type_counts": unresolved})
                        local_targets, local_metadata = Counter(), {}
                        mask_counts, unknown_counts, conflict_counts = Counter(), Counter(), Counter()
                        for leaf in result["leaves"]:
                            side, targets = atlas.lookup(leaf["voxel"])
                            mask_status = "out_of_FOV" if leaf["voxel"] is None else (
                                "not_provided" if mask is None else ("inside" if mask[tuple(leaf["voxel"])] else "outside"))
                            saved = {**identity, **leaf, "source_metadata": json.loads(identity["source_metadata_json"]),
                                     "source_sha256": record["expected_sha256"], "source_path": str(source),
                                     "coordinate_frame": record["CoordinateFrame"], "index_scale_um": record["index_scale_um"],
                                     "hemisphere_side": side, "brain_mask_status": mask_status, "atlas_levels": targets}
                            saved.pop("source_metadata_json")
                            leaf_stream.write(json.dumps(saved, allow_nan=False) + "\n")
                            if not leaf["candidate_axon_endpoint"]:
                                continue
                            mask_counts[mask_status] += 1
                            for target in targets:
                                key = (target["level"], target["target_status"], target["index"])
                                local_targets[key] += 1
                                local_metadata[key] = _target_columns(target)
                                if target["target_status"] != "mapped":
                                    unknown_counts[str(target["level"])] += 1
                                if target["hemisphere_conflict"]:
                                    conflict_counts[str(target["level"])] += 1
                        for key in sorted(local_targets, key=lambda item: (item[0], item[1], -1 if item[2] is None else item[2])):
                            count = local_targets[key]
                            neuron_targets.append({**identity, **local_metadata[key], "endpoint_count": count,
                                                   "neurons_with_target_evidence": 1})
                            target_counts[key] += count
                            target_neurons[key] += 1
                            target_metadata[key] = local_metadata[key]
                        for item in result["endpoint_counts"]:
                            count_voxels[tuple(item["voxel"])] += item["count"]
                        occupied_voxels.update(tuple(voxel) for voxel in result["neuron_occupancy"])
                        flat_qc = {key: value for key, value in qc.items() if key in QC_COLUMNS}
                        for name in ("node_type_counts", "leaf_type_counts", "leaf_class_counts", "type1_node_ids"):
                            flat_qc[name + "_json"] = json.dumps(qc[name], sort_keys=True)
                        flat_qc.update(map_proxy_status="computable_candidate_proxy" if computable else "unassessed_no_eligible_candidate",
                            n_selected=1, n_computable=int(computable), image_review_state="unresolved",
                            in_brain_mask_candidate_count=mask_counts["inside"] if mask is not None else None,
                            outside_brain_mask_in_FOV_candidate_count=mask_counts["outside"] if mask is not None else None,
                            unknown_target_counts_by_level_json=json.dumps(unknown_counts, sort_keys=True),
                            hemisphere_conflict_counts_by_level_json=json.dumps(conflict_counts, sort_keys=True))
                        qc_rows.append({**identity, **flat_qc})
                        accounting.update(candidate_endpoints=qc["candidate_axon_endpoint_count"],
                                          in_reference_candidates=qc["in_reference_candidate_count"],
                                          outside_reference_candidates=qc["outside_reference_candidate_count"], all_leaves=qc["leaf_count"])
                        accounting.update({"candidate_brain_mask_" + key: value for key, value in mask_counts.items()})
                    group_counters[(animal, region)] = (count_voxels, occupied_voxels, len(selected), n_computable)
                    run["n_computable"] += n_computable
                    entry = dict(AnimalID=animal, Subregion=region, n_selected=len(selected), n_computable=n_computable,
                                 map_available=bool(n_computable))
                    if not n_computable:
                        entry["unavailable_reason"] = "no_computable_candidate_endpoint_neurons"
                        region_rows.append(dict(AnimalID=animal, Subregion=region,
                            target_status="unavailable_no_computable_neurons", endpoint_count=0, neurons_with_target_evidence=0,
                            n_selected=len(selected), n_computable=0, map_available=False, biological_innervation_state="unassessed"))
                    else:
                        for key in sorted(target_counts, key=lambda item: (item[0], item[1], -1 if item[2] is None else item[2])):
                            region_rows.append(dict(AnimalID=animal, Subregion=region, **target_metadata[key],
                                endpoint_count=target_counts[key], neurons_with_target_evidence=target_neurons[key],
                                n_selected=len(selected), n_computable=n_computable,
                                frequency_per_computable_neuron=target_neurons[key] / n_computable,
                                mean_endpoints_per_computable_neuron=target_counts[key] / n_computable,
                                map_available=True, biological_innervation_state="unassessed"))
                        _write_maps(output, entry, _paths(output, animal, region, space), grid,
                                    {voxel: count / n_computable for voxel, count in count_voxels.items()},
                                    {voxel: count / n_computable for voxel, count in occupied_voxels.items()})
                    run["animal_maps"].append(entry)
                    run["accounting"] = dict(accounting)
                    checkpoint()
                    print(f"animal={animal} subregion={region} selected={len(selected)} computable={n_computable}", flush=True)

            for region in sorted({key[1] for key in groups}):
                selected_entries = [entry for entry in run["animal_maps"] if entry["Subregion"] == region]
                available = [entry for entry in selected_entries if entry["map_available"]]
                entry = dict(Subregion=region, n_selected=sum(item["n_selected"] for item in selected_entries),
                             n_computable=sum(item["n_computable"] for item in available), n_animals=len(available),
                             contributing_animals=[item["AnimalID"] for item in available],
                             unavailable_animals=[item["AnimalID"] for item in selected_entries if not item["map_available"]],
                             map_available=bool(available))
                if available:
                    means, occupancies = Counter(), Counter()
                    for item in available:
                        counts, occupancy, _, denominator = group_counters[(item["AnimalID"], region)]
                        for voxel, count in counts.items():
                            means[voxel] += count / denominator / len(available)
                        for voxel, count in occupancy.items():
                            occupancies[voxel] += count / denominator / len(available)
                    _write_maps(output, entry, _paths(output, None, region, space), grid, means, occupancies)
                else:
                    entry["unavailable_reason"] = "no_contributing_animal_with_computable_neurons"
                run["group_maps"].append(entry)

            for name, rows, columns in [("per_neuron_qc.csv", qc_rows, QC_COLUMNS),
                                        ("per_neuron_target_counts.csv", neuron_targets, NEURON_TARGET_COLUMNS),
                                        ("animal_region_summaries.csv", region_rows, REGION_COLUMNS)]:
                with (output / name).open("x", encoding="utf-8", newline="") as stream:
                    pd.DataFrame(rows, columns=columns).to_csv(stream, index=False)
            if any(sha256(path) != hashes[name] for name, path in inputs.items()):
                raise RuntimeError("Manifest/reference/atlas/key/mask changed during mapping")
            if any(sha256(record["source_path"]) != record["expected_sha256"] for record in records):
                raise RuntimeError("SWC source changed before run completion")
            run["accounting"] = dict(accounting)
            run["csv_columns"] = {"per_neuron_qc.csv": QC_COLUMNS, "per_neuron_target_counts.csv": NEURON_TARGET_COLUMNS,
                                  "animal_region_summaries.csv": REGION_COLUMNS}
            for name in ("input_manifest.csv", "leaf_records.jsonl", "per_neuron_qc.csv", "per_neuron_target_counts.csv",
                         "animal_region_summaries.csv"):
                run["artifacts"][name] = {"sha256": sha256(output / name)}
            run.update(status="software_verified_candidate_endpoints", finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()
        except Exception as exc:
            run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            checkpoint()
            raise
    return run


def _write_maps(output, entry, paths, grid, count_values, occupancy_values):
    for metric, path in paths.items():
        values = np.zeros(grid.shape, dtype=np.float64)
        for voxel, value in (occupancy_values if metric == "occupancy" else count_values).items():
            values[voxel] = value / grid.voxel_volume_mm3 if metric == "density" else value
        _save_endpoint_map(path, values, grid, metric)
        entry[metric + "_path"] = str(path.relative_to(output))
        entry[metric + "_sha256"] = sha256(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "reference", "atlas-path", "atlas-key", "hemisphere-mask", "output"):
        parser.add_argument("--" + option, type=Path, required=True, help={
            "manifest": "CSV of exact neuron identities, source groups, hashes and coordinate provenance.",
            "reference": "Pinned reference image bound by the manifest.",
            "atlas-path": "Actual aligned ARM label volumes for direct voxel lookup.",
            "atlas-key": "Official ARM label names and indices.",
            "hemisphere-mask": "Matching mask with verified hemisphere label semantics.",
            "output": "New destination for candidate axon-end maps and audit records.",
        }[option])
    parser.add_argument("--brain-mask", type=Path, help='Optional coverage mask; endpoints inside the reference are retained even outside this mask.')
    parser.add_argument("--input-root", type=Path, default=ROOT, help='Base directory for relative SWC paths.')
    parser.add_argument("--space", default="NMTv2p1", help='Reference-space name for output filenames; no registration is performed.')
    args = parser.parse_args()
    result = build(args.manifest, args.reference, args.output, atlas_path=args.atlas_path, atlas_key=args.atlas_key,
                   hemisphere_mask=args.hemisphere_mask, brain_mask=args.brain_mask, input_root=args.input_root, space=args.space)
    print(f"{result['status']}: selected={result['n_selected']} computable={result['n_computable']}")
