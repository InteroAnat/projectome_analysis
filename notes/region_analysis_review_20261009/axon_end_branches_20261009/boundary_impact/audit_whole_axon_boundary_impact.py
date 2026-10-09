"""Exhaustive source-bound old/new whole-axon voxel allocation audit.

Only affected neurons require full map evaluation. No maps or sources are
written. Nonzero midpoint intervals are compared, not just total lengths.
"""
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "main_scripts"))
import projection_maps as corrected
from swc_validation import parse_swc

MANIFEST = ROOT / "group_analysis/evolution_20261008/projection_inputs/arm_labels_20261009/combined/projection_manifest.csv"
REFERENCE = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def intervals(start, end, shape, faces):
    """The unchanged clipped-face interval calculation from the pinned code."""
    delta = end - start
    lower, upper = 0., 1.
    for axis in range(3):
        if delta[axis] == 0:
            if not -.5 <= start[axis] < shape[axis] - .5:
                return
        else:
            times = ((-.5 - start[axis]) / delta[axis],
                     (shape[axis] - .5 - start[axis]) / delta[axis])
            lower = max(lower, min(times))
            upper = min(upper, max(times))
    if upper <= lower:
        return
    cuts = [lower, upper]
    for axis in range(3):
        if delta[axis] != 0:
            times = (faces[axis] - start[axis]) / delta[axis]
            cuts.extend(times[(times > lower) & (times < upper)])
    cuts = np.unique(cuts)
    for left, right in zip(cuts[:-1], cuts[1:]):
        point = start + (left + (right-left)/2) * delta
        base = np.floor(point)
        old = np.floor(point + .5).astype(int)
        new = (base + (point - base >= .5)).astype(int)
        yield old, new, float(right-left), point


def valid(voxel, shape):
    return bool(np.all(voxel >= 0) and np.all(voxel < shape))


def allocation(start, end, length, shape, same, voxel, module):
    if length == 0:
        return {}
    if same:
        return {tuple(int(x) for x in voxel): float(length)}
    values = Counter()
    for cell, fraction in module.segment_voxels(start, end, shape):
        values[tuple(int(x) for x in cell)] += float(length * fraction)
    return dict(values)


def main():
    destination = HERE / "all462_boundary_impact.json"
    if destination.exists():
        raise FileExistsError(destination)
    git = ["git", "-c", "safe.directory=D:/projectome_analysis"]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    old_bytes = subprocess.check_output(git + ["show", "HEAD:main_scripts/projection_maps.py"], cwd=ROOT)
    old_text = old_bytes.decode("utf-8")
    new_path = ROOT / "main_scripts/projection_maps.py"
    new_bytes = new_path.read_bytes()
    new_text = new_bytes.decode("utf-8-sig")
    old_tree, new_tree = ast.parse(old_text), ast.parse(new_text)
    old_defs = {node.name: ast.dump(node, include_attributes=False) for node in old_tree.body if hasattr(node, "name")}
    new_defs = {node.name: ast.dump(node, include_attributes=False) for node in new_tree.body if hasattr(node, "name")}
    changed = [name for name in old_defs if old_defs[name] != new_defs.get(name)]
    if set(changed) != {"segment_voxels", "axon_length_map", "save_map"}:
        raise AssertionError(f"Unexpected old/new semantic change scope: {changed}")
    expected_text = old_text.replace(
        "voxel = np.floor(start + (left + (right-left)/2) * delta + 0.5).astype(int)",
        "point = start + (left + (right-left)/2) * delta\n        base = np.floor(point)\n        voxel = (base + (point - base >= 0.5)).astype(int)")
    expected_text = expected_text.replace(
        "start_voxels = np.floor(starts + 0.5)\n    end_voxels = np.floor(ends + 0.5)",
        "start_base, end_base = np.floor(starts), np.floor(ends)\n    start_voxels = start_base + (starts - start_base >= 0.5)\n    end_voxels = end_base + (ends - end_base >= 0.5)")
    expected_defs = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(expected_text).body if hasattr(node, "name")}
    for name in ("segment_voxels", "axon_length_map"):
        if expected_defs[name] != new_defs[name]:
            raise AssertionError(f"Unexpected numerical changes beyond declared rounding fix: {name}")
    old = types.ModuleType("pinned_old_projection_maps")
    sys.modules[old.__name__] = old
    exec(compile(old_text, "HEAD:main_scripts/projection_maps.py", "exec"), old.__dict__)
    old_snapshot = HERE / "projection_maps_before_boundary_fix_HEAD.py"
    with old_snapshot.open("xb") as stream:
        stream.write(old_bytes)
    manifest_hash = sha(MANIFEST)
    ledger = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    ledger["NeuronUID"] = ledger.SampleID + "::" + ledger.NeuronID
    if len(ledger) != 462 or ledger.NeuronUID.duplicated().any():
        raise ValueError("Exact unique selected462 manifest required")
    grid = corrected.ReferenceGrid.from_image(nib.load(REFERENCE))
    reference_hash = sha(REFERENCE)
    shape = np.array(grid.shape)
    faces = [np.arange(n-1, dtype=float) + .5 for n in shape]
    records, edge_changes, full_map_checks = [], [], []
    source_bindings = []
    for count, meta in enumerate(ledger.to_dict("records"), 1):
        uid = meta["NeuronUID"]
        if meta["CoordinateFrame"] != "atlas_index_um" or meta["IndexScaleUm"] != "250;250;250" or meta["ReferenceSHA256"] != reference_hash:
            raise ValueError(f"Unexpected source coordinate declaration: {uid}")
        path = ROOT / meta["SWCPath"]
        payload = path.read_bytes()
        source_hash = hashlib.sha256(payload).hexdigest()
        if source_hash != meta["SWCSHA256"]:
            raise ValueError(f"Changed source: {uid}")
        text = payload.decode("utf-8-sig")
        rows = parse_swc(text, str(path))
        ids = {row[0]: i for i, row in enumerate(rows)}
        indices = np.asarray([row[2:5] for row in rows]) / 250.
        selected = [i for i, row in enumerate(rows) if row[1] == 2 and row[6] != -1]
        parents = [ids[rows[i][6]] for i in selected]
        starts, ends = indices[parents], indices[selected]
        lengths = np.linalg.norm((ends - starts) @ grid.affine_mm[:3, :3].T, axis=1)
        old_start, old_end = np.floor(starts + .5), np.floor(ends + .5)
        start_base, end_base = np.floor(starts), np.floor(ends)
        new_start = start_base + (starts - start_base >= .5)
        new_end = end_base + (ends - end_base >= .5)
        old_same = ((old_start == old_end).all(1) & (old_start >= 0).all(1) & (old_start < shape).all(1))
        new_same = ((new_start == new_end).all(1) & (new_start >= 0).all(1) & (new_start < shape).all(1))
        fast_candidates = (old_same != new_same) | (old_same & new_same & (old_start != new_start).any(1))
        candidate_edges = set(np.flatnonzero(fast_candidates).tolist())
        midpoint_count, changed_midpoints = 0, 0
        for i in np.flatnonzero((~old_same | ~new_same) & (lengths > 0)):
            for old_voxel, new_voxel, fraction, point in intervals(starts[i], ends[i], shape, faces):
                midpoint_count += 1
                if not np.array_equal(old_voxel, new_voxel):
                    changed_midpoints += 1
                    candidate_edges.add(int(i))
        neuron_changes = []
        for i in sorted(candidate_edges):
            old_values = allocation(starts[i], ends[i], lengths[i], shape, old_same[i], old_start[i], old)
            new_values = allocation(starts[i], ends[i], lengths[i], shape, new_same[i], new_start[i], corrected)
            delta = {voxel: new_values.get(voxel, 0.)-old_values.get(voxel, 0.) for voxel in old_values.keys() | new_values.keys()}
            delta = {voxel: value for voxel, value in delta.items() if value != 0}
            if delta:
                child = rows[selected[i]]
                change = {"NeuronUID": uid, "SWCSHA256": source_hash, "parent_node_id": child[6], "child_node_id": child[0],
                          "start_index_xyz": starts[i].tolist(), "end_index_xyz": ends[i].tolist(),
                          "edge_length_mm": float(lengths[i]), "old_fast_path": bool(old_same[i]), "new_fast_path": bool(new_same[i]),
                          "old_allocation": [{"voxel": list(voxel), "length_mm": value} for voxel, value in sorted(old_values.items())],
                          "new_allocation": [{"voxel": list(voxel), "length_mm": value} for voxel, value in sorted(new_values.items())],
                          "voxel_delta_mm": [{"voxel": list(voxel), "delta_mm": value} for voxel, value in sorted(delta.items())]}
                edge_changes.append(change)
                neuron_changes.append(change)
        if neuron_changes:
            old_map, _ = old.axon_length_map(text, grid, coordinate_frame="atlas_index_um", index_scale_um=[250]*3)
            new_map, _ = corrected.axon_length_map(text, grid, coordinate_frame="atlas_index_um", index_scale_um=[250]*3)
            delta = new_map-old_map
            full_map_checks.append({"NeuronUID": uid, "float64_changed_voxels": int(np.count_nonzero(delta)),
                                    "max_abs_delta_mm": float(np.max(np.abs(delta))), "L1_delta_mm": float(np.abs(delta).sum()),
                                    "float32_changed_voxels": int(np.count_nonzero(old_map.astype(np.float32) != new_map.astype(np.float32)))})
        records.append({"NeuronUID": uid, "SWCSHA256": source_hash, "selected_axon_edges": len(selected),
                        "zero_length_edges": int(np.count_nonzero(lengths == 0)),
                        "old_fast_edges": int(old_same.sum()), "new_fast_edges": int(new_same.sum()),
                        "endpoint_bin_different_edges": int(((old_start != new_start).any(1) | (old_end != new_end).any(1)).sum()),
                        "fast_allocation_candidate_edges": int(fast_candidates.sum()),
                        "clipped_midpoint_intervals_checked": midpoint_count, "midpoint_index_differences": changed_midpoints,
                        "allocation_changed_edges": len(neuron_changes)})
        source_bindings.append({"NeuronUID": uid, "path": relative(path), "sha256": source_hash})
        if count % 50 == 0 or count == 462:
            print(json.dumps({"processed": count, "selected": 462, "allocation_changed_edges": len(edge_changes)}), flush=True)
    if new_path.read_bytes() != new_bytes or sha(REFERENCE) != reference_hash or sha(MANIFEST) != manifest_hash:
        raise RuntimeError("Pinned input changed during audit")
    per_neuron = HERE / "per_neuron_boundary_impact.csv"
    with per_neuron.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    source_file = HERE / "source_bindings.json"
    with source_file.open("x") as stream:
        json.dump(source_bindings, stream, indent=2)
    totals = {key: sum(record[key] for record in records) for key in records[0] if key not in ("NeuronUID", "SWCSHA256")}
    report = {"status": "complete_source_allocation_no_impact" if not edge_changes else "complete_source_allocation_changes_found",
              "selected_neurons": 462, "totals": totals, "changed_neurons": len(full_map_checks), "changed_edges": edge_changes,
              "affected_neuron_full_map_checks": full_map_checks,
              "comparison": "every selected child-axon edge: both endpoint fast-bin rules; all unchanged clipped-face intervals for positive slow edges; exact positive voxel allocation comparison for every candidate difference",
              "no_maps_regenerated_or_written": True, "source_or_cohort_mutation": False,
              "head_commit": head, "old_code_sha256": hashlib.sha256(old_bytes).hexdigest(), "new_code_sha256": hashlib.sha256(new_bytes).hexdigest(),
              "changed_function_ASTs": changed, "numerical_functions_match_only_expected_rounding_fix": True,
              "save_map_difference_outside_audit": "description argument/validation; save_map never called",
              "shape": grid.shape, "affine_mm": grid.affine_mm.tolist(),
              "python_version": sys.version, "numpy_version": np.__version__,
              "inputs": [{"path": relative(MANIFEST), "sha256": manifest_hash}, {"path": relative(REFERENCE), "sha256": reference_hash},
                         {"path": relative(new_path), "sha256": hashlib.sha256(new_bytes).hexdigest()},
                         {"path": relative(ROOT / "main_scripts/swc_validation.py"), "sha256": sha(ROOT / "main_scripts/swc_validation.py")}],
              "artifacts": [{"path": relative(path), "sha256": sha(path)} for path in [Path(__file__), old_snapshot, per_neuron, source_file]]}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ["status", "selected_neurons", "changed_neurons", "totals"]}, indent=2))


if __name__ == "__main__":
    main()
