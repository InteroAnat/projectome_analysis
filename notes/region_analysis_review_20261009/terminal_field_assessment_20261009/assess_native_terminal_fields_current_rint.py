"""Local-only, descriptive SWC/image pilot; no automated terminal acceptance.

Run from the repository with the projectome Python environment. This script
never imports the source-image HTTP client or downloads imagery. Native XYZ
uses the client's nominal acquisition convention, not an anatomical transform.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from main_scripts.swc_validation import parse_swc
from main_scripts.fmost_image_geometry import clip_segment

HERE = Path(__file__).resolve().parent
SUMMARY = ROOT / "notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv"
RAW = ROOT / "group_analysis/visual_review_20261002/swc_raw"
CACHE = ROOT / "group_analysis/visual_review_20261002/cache/cubes"
ATLAS = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz"
KEY = ROOT / "atlas/ARM_key_all.txt"
SPACING = np.array([0.65, 0.65, 3.0])
SHAPE = np.array([360, 360, 90])
BLOCK = SPACING * SHAPE

# Purposive contrasts, not a random or prevalence-estimating sample. Each
# target has cached local endpoint imagery. Cube selection below is mechanical.
PILOT = [
    ("252790::014.swc", 41, "branched_field"),
    ("252790::032.swc", 229, "branched_field_candidate_source"),
    ("252383::018.swc", 729, "branched_field_second_animal"),
    ("252383::121.swc", 1556, "single_ending"),
    ("252384::047.swc", 1004, "limited_image_coverage"),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def cube_index(xyz):
    """Native block membership; exact upper edges belong to the next cube."""
    return tuple(np.floor(np.asarray(xyz) / BLOCK).astype(int).tolist())


def cube_path(sample, index):
    x, y, z = index
    return CACHE / sample / "high_res_http" / str(z) / f"{x}_{y}_{z}.tif"


def identity_signature(rows):
    return {int(row[0]): (int(row[1]), int(row[6])) for row in rows}


def graph_information(rows):
    by_id = {int(row[0]): row for row in rows}
    children = defaultdict(list)
    for row in rows:
        children[int(row[6])].append(int(row[0]))
    leaves = {int(row[0]) for row in rows
              if row[1] == 2 and row[6] != -1 and not children[int(row[0])]}
    return by_id, children, leaves


def atlas_labels(rows, atlas):
    indices = np.rint(rows[:, 2:5] / 250.0).astype(int)
    valid = ((indices >= 0) & (indices < np.array(atlas.shape))).all(axis=1)
    labels = np.full(len(rows), -1, dtype=int)
    labels[valid] = atlas[tuple(indices[valid].T)]
    return {int(row[0]): int(label) for row, label in zip(rows, labels)}


def target_components(rows, labels, target):
    """Components of original type-2 edges whose two nodes share a target.

    A cut at an atlas or compartment boundary never creates a graph endpoint.
    """
    by_id, children, leaves = graph_information(rows)
    selected = {int(row[0]) for row in rows if row[1] == 2 and labels[int(row[0])] == target}
    starts = sorted(node for node in selected if int(by_id[node][6]) not in selected)
    components = []
    for start in starts:
        pending, nodes = [start], []
        while pending:
            node = pending.pop()
            nodes.append(node)
            pending.extend(child for child in children[node] if child in selected)
        components.append(nodes)
    assert sum(map(len, components)) == len(selected)
    return components


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_csv(path, records):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def draw_image_views(volume, origin, edges, point, output, title, zoom=False, highlight_kind="selected leaf"):
    """Unmarked/marked orthogonal MIPs, with matching contrast in each pair."""
    if zoom:
        # Display extent only; it is never an arbor, terminal or bouton rule.
        half = np.array([50.0, 50.0, 27.0])
        lo = np.maximum(0, np.floor((point - half - origin) / SPACING).astype(int))
        hi = np.minimum(SHAPE, np.ceil((point + half - origin) / SPACING).astype(int) + 1)
        volume = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        origin = origin + lo * SPACING
    upper = origin + np.array(volume.shape) * SPACING
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for col, (x, y, depth) in enumerate([(0, 1, 2), (0, 2, 1), (1, 2, 0)]):
        projection = volume.max(axis=depth)
        remaining = [axis for axis in range(3) if axis != depth]
        if remaining != [x, y]:
            projection = projection.T
        lo, hi = np.percentile(projection, [1, 99.7])
        hi = max(float(hi), float(lo) + 1)
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(projection.T, origin="lower", cmap="gray", vmin=lo, vmax=hi,
                      extent=[origin[x]-SPACING[x]/2, upper[x]-SPACING[x]/2,
                              origin[y]-SPACING[y]/2, upper[y]-SPACING[y]/2], aspect="equal")
            if row:
                segments = []
                for a, b in edges:
                    clipped = clip_segment(a, b, origin, upper)
                    if clipped is not None:
                        segments.append(np.array(clipped)[:, [x, y]])
                ax.add_collection(LineCollection(segments, colors="#00cfff", linewidths=0.7))
                ax.scatter(point[x], point[y], s=40, facecolors="none", edgecolors="#ff654f", linewidths=1)
            ax.set_xlabel(f"Native {'XYZ'[x]} (nominal µm)")
            ax.set_ylabel(f"Native {'XYZ'[y]} (nominal µm)")
            ax.set_title(f"{'Unmarked image' if row == 0 else 'SWC overlay + ' + highlight_kind}; max over {'XYZ'[depth]}")
    fig.suptitle(title + "\nNominal sampling 0.65 × 0.65 × 3 µm; optical resolution unverified", fontsize=11)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def build(output=HERE / "native_cube_pilot_current_rint"):
    if output.exists():
        raise FileExistsError(output)
    table = pd.read_csv(SUMMARY, dtype=str, keep_default_na=False)
    if len(table) != 462 or table.NeuronUID.duplicated().any():
        raise ValueError("Expected the unchanged exact 462-neuron ledger")
    atlas = np.asarray(nib.load(ATLAS).dataobj)[..., 0, 5]
    key = pd.read_csv(KEY, sep="\t", dtype=str, keep_default_na=False)
    names = dict(zip(key.Index.astype(int), key.Full_Name))
    cache_counts = Counter(path.relative_to(CACHE).parts[0] for path in CACHE.rglob("*.tif"))
    output.mkdir()
    inventory, examples = [], []
    pilot_ids = {item[0] for item in PILOT}
    pairs = {}
    for row in table.to_dict("records"):
        raw_path = RAW / row["SampleID"] / row["NeuronID"]
        record = {"NeuronUID": row["NeuronUID"], "AnimalID": row["AnimalID"],
                  "SourceARMFullName": row["ARMFullName"], "native_swc_available": raw_path.exists(),
                  "native_swc_path": relative(raw_path) if raw_path.exists() else "",
                  "native_swc_sha256": "", "atlas_swc_path": row["SWCPath"],
                  "atlas_swc_sha256": row["SWCSHA256"], "node_identity_type_parent_match": "unassessed",
                  "candidate_axon_leaves": "", "leaves_with_cached_native_cube": ""}
        if raw_path.exists():
            native = np.loadtxt(raw_path, ndmin=2)
            atlas_path = ROOT / row["SWCPath"]
            if sha(atlas_path) != row["SWCSHA256"]:
                raise ValueError(f"Atlas source hash mismatch: {row['NeuronUID']}")
            warped = np.loadtxt(atlas_path, ndmin=2)
            if native.shape[1] < 7 or warped.shape[1] < 7:
                raise ValueError("Standard seven SWC columns required")
            matched = identity_signature(native) == identity_signature(warped)
            _, _, leaves = graph_information(native)
            covered = sum(cube_path(row["SampleID"], cube_index(node[2:5])).exists()
                          for node in native if int(node[0]) in leaves)
            record.update(native_swc_sha256=sha(raw_path), node_identity_type_parent_match=matched,
                          candidate_axon_leaves=len(leaves), leaves_with_cached_native_cube=covered)
            if row["NeuronUID"] in pilot_ids:
                if not matched:
                    raise ValueError("Pilot native/atlas node identities do not match")
                # Full parser validation for reviewed examples, not only numeric loading.
                native = np.array(parse_swc(raw_path.read_text(), str(raw_path)), dtype=float)
                warped = np.array(parse_swc(atlas_path.read_text(), str(atlas_path)), dtype=float)
                pairs[row["NeuronUID"]] = (row, native, warped, raw_path)
        inventory.append(record)
    write_csv(output / "selected462_native_image_coverage.csv", inventory)
    for uid, target, purpose in PILOT:
        row, native, warped, raw_path = pairs[uid]
        by_id, children, leaves = graph_information(native)
        labels = atlas_labels(warped, atlas)
        counts = Counter(cube_index(by_id[node][2:5]) for node in leaves
                         if labels[node] == target
                         and cube_path(row["SampleID"], cube_index(by_id[node][2:5])).exists())
        if not counts:
            raise ValueError(f"Missing target image coverage for {uid}")
        cube = sorted(counts, key=lambda index: (-counts[index], index))[0]
        path = cube_path(row["SampleID"], cube)
        cube_zyx = tifffile.imread(path)
        if cube_zyx.shape != (90, 360, 360) or cube_zyx.dtype != np.uint16:
            raise ValueError(f"Unexpected cached source cube shape/dtype: {path}")
        volume = cube_zyx.transpose(2, 1, 0)
        origin = np.array(cube) * BLOCK
        local_leaves = sorted(node for node in leaves if cube_index(by_id[node][2:5]) == cube and labels[node] == target)
        point_id = local_leaves[0]
        components = target_components(native, labels, target)
        field = next(nodes for nodes in components if point_id in nodes)
        field_set = set(field)
        field_leaves = sorted(field_set & leaves)
        branch_nodes = sorted(node for node in field if sum(child in field_set for child in children[node]) >= 2)
        edges = [(by_id[int(node[6])][2:5], node[2:5]) for node in native
                 if node[1] == 2 and node[6] != -1]
        field_edges = [(int(by_id[node][6]), node) for node in field if int(by_id[node][6]) in field_set]
        length = sum(np.linalg.norm(by_id[a][2:5] - by_id[b][2:5]) for a, b in field_edges)
        stem = uid.replace("::", "_").removesuffix(".swc")
        directory = output / stem
        directory.mkdir()
        title = f"{uid} | declared ARM level 6: {names[target]} (index {target})"
        draw_image_views(volume, origin, edges, by_id[point_id][2:5], directory / "cached_cube_image_and_swc.png", title)
        draw_image_views(volume, origin, edges, by_id[point_id][2:5], directory / "selected_ending_image_and_swc.png", title + f" | leaf {point_id}", zoom=True)
        # Full target-connected component, with original graph leaves preserved.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, (x, y) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            segments = [np.array([by_id[a][2:5], by_id[b][2:5]])[:, [x, y]] for a, b in field_edges]
            ax.add_collection(LineCollection(segments, colors="#536b8e", linewidths=0.45))
            p = np.array([by_id[n][2:5] for n in field])
            ax.scatter(p[:, x], p[:, y], s=0.15, color="#536b8e")
            if field_leaves:
                tips = np.array([by_id[n][2:5] for n in field_leaves])
                ax.scatter(tips[:, x], tips[:, y], s=8, color="#d54a35")
            ax.plot([origin[x], origin[x]+BLOCK[x], origin[x]+BLOCK[x], origin[x], origin[x]],
                    [origin[y], origin[y], origin[y]+BLOCK[y], origin[y]+BLOCK[y], origin[y]], color="#079d91", linewidth=1)
            ax.autoscale(); ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(f"Native {'XYZ'[x]} (nominal µm)"); ax.set_ylabel(f"Native {'XYZ'[y]} (nominal µm)")
        fig.suptitle(title + f"\nTarget-connected type-2 component: {len(field)} nodes, {len(field_leaves)} full-graph leaves, {len(branch_nodes)} branch nodes")
        fig.savefig(directory / "target_component_morphology.png", dpi=160)
        plt.close(fig)
        warped_by_id = {int(node[0]): node for node in warped}
        node_records = []
        for node_id in sorted(field_set):
            n = by_id[node_id]
            node_records.append({"NeuronUID": uid, "node_id": node_id, "node_type": int(n[1]), "parent_id": int(n[6]),
                                 "native_x_um": n[2], "native_y_um": n[3], "native_z_um": n[4],
                                 "atlas_x_declared_um": warped_by_id[node_id][2], "atlas_y_declared_um": warped_by_id[node_id][3],
                                 "atlas_z_declared_um": warped_by_id[node_id][4], "ARMLevel": 6, "ARMIndex": labels[node_id],
                                 "full_graph_axon_leaf": node_id in leaves,
                                 "cached_native_cube_available": cube_path(row["SampleID"], cube_index(n[2:5])).exists(),
                                 "inside_reviewed_cube": cube_index(n[2:5]) == cube})
        write_csv(directory / "target_component_nodes.csv", node_records)
        image = {"path": relative(path), "sha256": sha(path), "cube_index_xyz": cube,
                 "source_array_order": "ZYX", "source_shape": list(cube_zyx.shape), "dtype": str(cube_zyx.dtype),
                 "native_origin_um": origin.tolist(), "nominal_spacing_um": SPACING.tolist(),
                 "display_array_order": "XYZ", "optical_resolution": "unverified",
                 "origin_convention": "existing client nominal block index times block shape times spacing; no calibrated image/SWC origin proof",
                 "adjacent_cube_availability": {f"{x}_{y}_{z}": cube_path(row["SampleID"], (x, y, z)).exists()
                     for x in range(cube[0]-1, cube[0]+2) for y in range(cube[1]-1,cube[1]+2) for z in range(cube[2]-1,cube[2]+2)}}
        example = {"NeuronUID": uid, "AnimalID": row["AnimalID"], "SourceARMFullName": row["ARMFullName"],
                   "SourceARMStatus": row["SourceARMStatus"], "selection_purpose": purpose,
                   "selection_rule": "predeclared UID/target; cached cube with most target full-graph leaves, ties lexicographic cube XYZ; smallest leaf ID for detail",
                   "native_swc": {"path": relative(raw_path), "sha256": sha(raw_path)},
                   "atlas_swc": {"path": row["SWCPath"], "sha256": row["SWCSHA256"]},
                   "node_identity_type_parent_match": True, "full_graph_validated": True,
                   "declared_target": {"level": 6, "index": target, "full_name": names[target],
                                       "sampling": "np.rint(atlas_xyz/250), current ARM lookup policy with ties-to-even; anatomical/origin acceptance unresolved"},
                   "component_root_node_id": min(field, key=lambda node: int(by_id[node][6]) in field_set),
                   "component_node_count": len(field), "component_branch_node_ids": branch_nodes,
                   "component_full_graph_axon_leaf_ids": field_leaves, "nominal_native_component_length_um": float(length),
                   "selected_leaf_id": point_id, "selected_leaf_type": 2, "selected_leaf_parent_id": int(by_id[point_id][6]),
                   "selected_leaf_native_xyz_um": by_id[point_id][2:5].tolist(),
                   "selected_leaf_distance_to_cube_faces_um": np.concatenate([by_id[point_id][2:5]-origin, origin+BLOCK-by_id[point_id][2:5]]).tolist(),
                   "local_cube_target_leaf_ids": local_leaves, "image": image,
                   "reviewer": "Codex", "review_state": "awaiting_saved_figure_inspection",
                   "biological_terminal_state": "unassessed", "bouton_synapse_state": "not_assessed"}
        write_json(directory / "assessment_inputs.json", example)
        examples.append(example)
    write_json(output / "pilot_inputs.json", examples)
    bindings = [SUMMARY, ATLAS, KEY, Path(__file__), ROOT / "main_scripts/swc_validation.py",
                ROOT / "main_scripts/fmost_image_geometry.py", ROOT / "main_scripts/Visual_toolkit.py",
                ROOT / "group_analysis/evolution_20261008/references/fmost_full_methods_20261009.md",
                ROOT / "group_analysis/evolution_20261008/references/fmost_full_methods_evidence_20261009.json"]
    receipt = {"status": "software_generated_pending_Codex_image_review", "selected_neurons": len(table),
               "native_swcs_available": sum(row["native_swc_available"] for row in inventory),
               "exact_node_identity_type_parent_matches": sum(row["node_identity_type_parent_match"] is True for row in inventory),
               "available_native_candidate_leaves": sum(int(row["candidate_axon_leaves"] or 0) for row in inventory),
               "native_leaves_with_cached_cube": sum(int(row["leaves_with_cached_native_cube"] or 0) for row in inventory),
               "cache_cube_counts_by_sample": dict(cache_counts), "pilot_units": len(examples),
               "no_downloads": True, "no_source_changes": True,
               "inputs": [{"path": relative(path), "sha256": sha(path)} for path in bindings],
               "outputs": [{"path": relative(path), "sha256": sha(path)} for path in sorted(output.rglob("*")) if path.is_file()]}
    write_json(output / "generation_provenance.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("inputs", "outputs")}, indent=2))


if __name__ == "__main__":
    build()
