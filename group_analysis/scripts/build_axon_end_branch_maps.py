"""Map reconstructed axon end-branches from one unchanged ARM neuron ledger.

Original full-graph axon endings define separate distal chains. Each chain
stops at the first original bifurcation, root or non-axon parent, including
its final child-axon transition edge. This graph measure is not a reviewed
terminal arbor, bouton or synapse. Length is in reference-template mm.

One pass produces sparse per-neuron ARM L1-L6 length matrices and one length
NIfTI per animal/source and equal-animal source group. Density is length
divided by reference voxel volume; no duplicate density volumes are needed.
Neurons without an eligible end remain NA; missing animals are not zeros.
Use a fresh destination. Inputs, source classifications and old maps survive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main_scripts"))
from projection_maps import ReferenceGrid, save_map
from terminal_sites import reconstructed_axon_end_branch_summary
from build_projection_maps import checked_manifest, sha256
from export_arm_projection_tables import catalog, validate_source_labels, _write_sheet

IDENTITY = ["SampleID", "NeuronID", "AnimalID", "Subregion", "ARMIndex", "ARMFullName", "Hemisphere"]
DESCRIPTION = "Reconstructed axon end-branch length; descriptive graph proxy"


def sparse_regional_lengths(sparse, outside, levels, targets, shape):
    """Integrate sparse voxel lengths within each actual hierarchy separately."""
    result = np.zeros(len(targets), dtype=float)
    lookup = {(int(row.Level), int(row.ARMIndex)): i
              for i, row in enumerate(targets.itertuples()) if row.TargetStatus != "out_of_FOV"}
    if sparse:
        xyz = np.asarray([entry["voxel"] for entry in sparse], dtype=int)
        occupied = np.ravel_multi_index(xyz.T, shape)
        weights = np.asarray([entry["length_mm"] for entry in sparse], dtype=float)
        for level, labels in enumerate(levels, 1):
            totals = np.bincount(labels[occupied].astype(np.int64), weights=weights)
            for label in np.flatnonzero(totals):
                result[lookup[level, int(label)]] = totals[label]
    for i, row in enumerate(targets.itertuples()):
        if row.TargetStatus == "out_of_FOV":
            result[i] = outside
    return result


def build(manifest, reference, atlas, atlas_key, output, *, input_root=ROOT):
    paths = {"manifest": Path(manifest), "reference": Path(reference),
             "atlas": Path(atlas), "atlas_key": Path(atlas_key)}
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh end-branch output directory")
    bindings = {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                for name, path in paths.items()}
    records = checked_manifest(paths["manifest"], input_root)
    frame = pd.read_csv(paths["manifest"], dtype=str, keep_default_na=False)
    if any(row["ReferenceSHA256"].lower() != bindings["reference"]["sha256"] for row in records):
        raise ValueError("Ledger does not bind this reference")
    grid = ReferenceGrid.from_image(nib.load(str(paths["reference"])))
    targets, levels = catalog(paths["atlas"], paths["atlas_key"], grid)
    validate_source_labels(frame, targets, bindings["atlas"]["sha256"], bindings["atlas_key"]["sha256"])
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row["AnimalID"], row["Subregion"]].append(i)
    planned = [f"animal-{a}_source-{s}_desc-axonEndBranchLength_map.nii.gz" for a, s in groups]
    planned += [f"source-{s}_desc-animalMeanAxonEndBranchLength_map.nii.gz" for _, s in groups]
    # Group names repeat across animals by design; filenames within each class must be unique.
    animal_names = planned[:len(groups)]
    group_names = sorted(set(planned[len(groups):]))
    if len({name.casefold() for name in animal_names}) != len(animal_names) or len({name.casefold() for name in group_names}) != len(group_names):
        raise ValueError("Distinct source groups collide in Windows filenames")
    if any(len(str(output.resolve() / "animal_maps" / name)) > 240 for name in animal_names + group_names):
        raise ValueError("Output path exceeds portable 240-character limit")
    output.mkdir(parents=True)
    (output / "animal_maps").mkdir()
    (output / "group_maps").mkdir()
    run = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": bindings, "selected_neurons": len(records), "shape": list(grid.shape),
        "affine_mm": grid.affine_mm.tolist(), "voxel_volume_mm3": grid.voxel_volume_mm3,
        "measurement": "reconstructed axon end-branch trajectory length",
        "chain_rule": "original non-root axon full-graph leaf to first original bifurcation/root/non-axon parent; final child-axon transition edge included; no non-axon chain bridging",
        "regional_assignment": "length split at reference voxel faces and integrated in each actual ARM hierarchy volume",
        "length_value_units": "reference-template mm per end-eligible neuron",
        "density_conversion": "divide length values by reference voxel volume in mm3",
        "animal_aggregation": "eligible neuron mean within animal/source group",
        "group_aggregation": "equal mean of animals with at least one eligible neuron; absent/unassessed groups not zero-filled",
        "hemisphere_policy": "unmirrored declared reference coordinates; official ARM source labels retained",
        "missing_policy": "no eligible original axon end => NA matrix row and no contribution to conditional map mean",
        "scientific_anatomical_acceptance": False,
        "limitations": ["unfinished unbranched shafts can qualify", "no accepted terminal arbor/bouton/synapse", "declared NMT export origin remains uncertain", "template length is not calibrated native tissue length"],
        "statistical_status": "descriptive; no t-map or independent-neuron inference",
        "source_code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
            [Path(__file__), ROOT / "main_scripts/terminal_sites.py", ROOT / "main_scripts/projection_maps.py",
             ROOT / "main_scripts/swc_validation.py", Path(__file__).with_name("export_arm_projection_tables.py"),
             Path(__file__).with_name("build_projection_maps.py")]},
        "versions": {"python": sys.version, "numpy": np.__version__, "nibabel": nib.__version__, "pandas": pd.__version__},
        "animal_maps": [], "group_maps": [], "artifacts": {},
    }
    provenance = output / "run_provenance.json"
    def checkpoint():
        provenance.write_text(json.dumps(run, indent=2), encoding="utf-8")
    checkpoint()
    matrix = np.full((len(records), len(targets)), np.nan)
    measurements = [None] * len(records)
    try:
        frame.to_csv(output / "input_ledger.csv", index=False)
        # This is a row-preserving snapshot, not another cohort or a source-label change.
        if not pd.read_csv(output / "input_ledger.csv", dtype=str, keep_default_na=False).equals(frame):
            raise IOError("Saved ledger changed cell values")
        for (animal, source_group), indices in sorted(groups.items()):
            accumulated = np.zeros(grid.shape, dtype=float)
            n_eligible = 0
            for i in indices:
                row = records[i]
                summary = reconstructed_axon_end_branch_summary(row["source_path"].read_text(encoding="utf-8"), grid,
                    coordinate_frame=row["CoordinateFrame"], index_scale_um=row["index_scale_um"], source=str(row["source_path"]))
                if sha256(row["source_path"]) != row["expected_sha256"]:
                    raise RuntimeError("SWC changed during end-branch mapping")
                qc = summary["qc"]
                child_ids = [edge["child_node_id"] for edge in summary["selected_edges"]]
                measurements[i] = {**{key: row[key] for key in IDENTITY}, "SWCSHA256": row["SWCSHA256"],
                    **qc, "selected_child_ids_sha256": hashlib.sha256(json.dumps(child_ids, separators=(",", ":")).encode()).hexdigest()}
                if qc["computable"]:
                    n_eligible += 1
                    sparse = summary["voxel_lengths_mm"]
                    matrix[i] = sparse_regional_lengths(sparse, qc["outside_reference_length_mm"], levels, targets, grid.shape)
                    for level in range(1, 7):
                        if not np.isclose(matrix[i, targets.Level.eq(level)].sum(), qc["total_length_mm"], rtol=1e-10, atol=1e-10):
                            raise ArithmeticError("ARM hierarchy does not conserve end-branch length")
                    if sparse:
                        xyz = np.asarray([entry["voxel"] for entry in sparse], dtype=int)
                        accumulated[tuple(xyz.T)] += [entry["length_mm"] for entry in sparse]
            entry = {"AnimalID": animal, "Subregion": source_group, "selected_neurons": len(indices), "eligible_neurons": n_eligible}
            if n_eligible:
                accumulated /= n_eligible
                path = output / "animal_maps" / f"animal-{animal}_source-{source_group}_desc-axonEndBranchLength_map.nii.gz"
                stored_sum = save_map(path, accumulated, grid, description=DESCRIPTION)
                entry.update(length_path=str(path.relative_to(output)), length_sha256=sha256(path),
                             mean_in_reference_length_mm=float(accumulated.sum()), stored_float32_sum_mm=stored_sum)
            run["animal_maps"].append(entry)
            checkpoint()
            print(f"animal={animal} source={source_group} eligible={n_eligible}/{len(indices)}", flush=True)
        for source_group in sorted({row["Subregion"] for row in records}):
            entries = [entry for entry in run["animal_maps"] if entry["Subregion"] == source_group]
            available = [entry for entry in entries if entry["eligible_neurons"]]
            entry = {"Subregion": source_group, "selected_neurons": sum(x["selected_neurons"] for x in entries),
                     "eligible_neurons": sum(x["eligible_neurons"] for x in entries),
                     "contributing_animals": [x["AnimalID"] for x in available], "n_animals": len(available)}
            if available:
                group = np.zeros(grid.shape, dtype=float)
                for animal_entry in available:
                    group += nib.load(str(output / animal_entry["length_path"])).get_fdata() / len(available)
                path = output / "group_maps" / f"source-{source_group}_desc-animalMeanAxonEndBranchLength_map.nii.gz"
                save_map(path, group, grid, description=DESCRIPTION)
                entry.update(length_path=str(path.relative_to(output)), length_sha256=sha256(path))
            run["group_maps"].append(entry)
        table = pd.DataFrame(measurements)
        table.drop(columns=["node_type_counts"]).to_csv(output / "per_neuron_measurements.csv", index=False)
        targets.to_csv(output / "targets.csv", index=False)
        book = Workbook(write_only=True)
        _write_sheet(book, "Neuron_QC", table.drop(columns=["node_type_counts"]))
        _write_sheet(book, "Official_ARM_Targets", targets)
        for level in range(1, 7):
            selected_targets = targets.Level.eq(level).to_numpy()
            values = pd.DataFrame(matrix[:, selected_targets], columns=targets.loc[selected_targets, "TargetID"], index=frame.index)
            sheet = pd.concat([frame[IDENTITY], values], axis=1)
            sheet.to_csv(output / f"ARM_L{level}_axon_end_branch_length_mm.csv", index=False)
            _write_sheet(book, f"L{level}_EndBranch_mm", sheet)
        book.save(output / "arm_axon_end_branch_hierarchy_tables.xlsx")
        for name, binding in bindings.items():
            if sha256(paths[name]) != binding["sha256"]:
                raise RuntimeError(f"Input changed: {name}")
        for relative, expected in run["source_code_sha256"].items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError("Producer code changed during mapping")
        run["eligible_neurons"] = int(table.computable.sum())
        run["original_axon_ends"] = int(table.candidate_axon_endpoint_count.sum())
        run["selected_original_axon_edges"] = int(table.selected_axon_edge_count.sum())
        run["total_end_branch_length_mm"] = float(table.total_length_mm.sum())
        run["artifacts"] = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file() and p != provenance}
        run.update(status="software_verified_descriptive_end_branches", finished_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        checkpoint()
        raise
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "reference", "atlas", "atlas-key", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.manifest, args.reference, args.atlas, args.atlas_key, args.output, input_root=args.input_root)
