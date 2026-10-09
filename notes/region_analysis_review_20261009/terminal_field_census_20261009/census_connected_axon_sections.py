"""All-selected ARM6 graph morphology and local image availability census.

Sections are connected original axon-labelled nodes within one sampled label,
not segmented biological arbors. No source image or source graph is modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "main_scripts"))
from swc_validation import parse_swc
from projection_maps import ReferenceGrid

SUMMARY = ROOT / "notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv"
COVERAGE = ROOT / "notes/region_analysis_review_20261009/terminal_field_assessment_20261009/native_cube_pilot_current_rint/selected462_native_image_coverage.csv"
ATLAS = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz"
REFERENCE = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz"
KEY = ROOT / "atlas/ARM_key_all.txt"
CACHE = ROOT / "group_analysis/visual_review_20261002/cache/cubes"
BLOCK_UM = np.array([234., 234., 270.])


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_csv(path, records):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def identity(rows):
    return {row[0]: (row[1], row[6]) for row in rows}


def sample_labels(rows, atlas, scale):
    indices = np.rint(np.array([row[2:5] for row in rows]) / scale).astype(np.int64)
    valid = ((indices >= 0) & (indices < np.array(atlas.shape))).all(1)
    labels = np.full(len(rows), -1, dtype=np.int64)
    labels[valid] = atlas[tuple(indices[valid].T)]
    return dict(zip((row[0] for row in rows), labels.tolist()))


def analyze(rows, labels, linear_mm, *, native=None, covered_cube_indices=None):
    """One full validated tree; classify cuts without manufacturing endpoints.

    ``linear_mm`` converts source XYZ differences to reference-template mm.
    Native coordinates, when supplied, are nominal acquisition micrometres.
    Coverage means existence in a supplied local cube-name index, not review.
    """
    by_id = {row[0]: row for row in rows}
    children = defaultdict(list)
    root = next(row[0] for row in rows if row[6] == -1)
    for row in rows:
        if row[6] != -1:
            children[row[6]].append(row[0])
    ends = {row[0] for row in rows if row[1] == 2 and row[6] != -1 and not children[row[0]]}
    node_section, sections = {}, {}
    pending = [root]
    while pending:
        node_id = pending.pop()
        row = by_id[node_id]
        pending.extend(reversed(sorted(children[node_id])))
        if row[1] != 2:
            continue
        parent = row[6]
        section_root = node_section[parent] if parent in node_section and labels[parent] == labels[node_id] else node_id
        node_section[node_id] = section_root
        sections.setdefault(section_root, []).append(node_id)
    assert len(node_section) == sum(row[1] == 2 for row in rows)
    lengths = {}
    native_lengths = {}
    internal, boundary, compartment = Counter(), Counter(), Counter()
    internal_native = Counter()
    crossing_records = []
    for row in rows:
        node_id, kind, parent = row[0], row[1], row[6]
        if parent == -1 or kind != 2:
            continue
        delta = np.array(row[2:5]) - np.array(by_id[parent][2:5])
        length = float(np.linalg.norm(linear_mm @ delta))
        if not math.isfinite(length):
            raise ValueError("Nonfinite template edge length")
        lengths[node_id] = length
        if native is not None:
            native_lengths[node_id] = float(np.linalg.norm(np.array(native[node_id]) - native[parent]))
        section_root = node_section[node_id]
        if parent in node_section and node_section[parent] == section_root:
            internal[section_root] += length
            if native is not None:
                internal_native[section_root] += native_lengths[node_id]
            continue
        reason = "target_label_crossing" if by_id[parent][1] == 2 else "nonaxon_to_axon_type_boundary"
        if reason == "target_label_crossing":
            boundary[section_root] += length
        else:
            compartment[section_root] += length
        crossing_records.append({"parent_node_id": parent, "child_node_id": node_id,
                                 "parent_type": by_id[parent][1], "child_type": kind,
                                 "parent_ARM6_index": labels[parent], "child_ARM6_index": labels[node_id],
                                 "crossing_reason": reason, "child_section_root_node_id": section_root,
                                 "parent_section_root_node_id": node_section.get(parent),
                                 "edge_length_NMT_mm": length,
                                 "edge_length_native_nominal_um": native_lengths.get(node_id)})
    records, membership, coverage = [], [], []
    for section_root, nodes in sorted(sections.items()):
        node_set = set(nodes)
        section_ends = sorted(node_set & ends)
        local_branches = sorted(node for node in nodes if sum(child in node_set for child in children[node]) >= 2)
        full_branches = sorted(node for node in nodes if len(children[node]) >= 2)
        axon_branches = sorted(node for node in nodes if sum(by_id[child][1] == 2 for child in children[node]) >= 2)
        exits = [(node, child) for node in nodes for child in children[node] if child not in node_set]
        target_exits = [(a, b) for a, b in exits if by_id[b][1] == 2]
        type_exits = [(a, b) for a, b in exits if by_id[b][1] != 2]
        false_end_protected = sorted(node for node in nodes if children[node] and not any(by_id[child][1] == 2 for child in children[node]))
        parent = by_id[section_root][6]
        incoming = "original_root" if parent == -1 else (
            "target_label_crossing" if by_id[parent][1] == 2 else "nonaxon_to_axon_type_boundary")
        covered_ends = None
        if native is not None:
            covered_ends = []
            for node in section_ends:
                cube = tuple(np.floor(np.array(native[node]) / BLOCK_UM).astype(int).tolist())
                available = cube in covered_cube_indices
                if available:
                    covered_ends.append(node)
                coverage.append({"node_id": node, "section_root_node_id": section_root,
                                 "ARM6Index": labels[node], "cube_x": cube[0], "cube_y": cube[1], "cube_z": cube[2],
                                 "cached_cube_available": available})
        topology = ("branching_with_original_ends" if section_ends and local_branches else
                    "original_end_without_local_bifurcation" if section_ends else
                    "branching_without_original_ends" if local_branches else
                    "unbranched_without_original_ends")
        records.append({"section_root_node_id": section_root, "ARM6Index": labels[section_root],
                        "node_count": len(nodes), "internal_original_edge_count": len(nodes)-1,
                        "original_axon_end_count": len(section_ends), "local_branch_point_count": len(local_branches),
                        "full_graph_branch_point_count": len(full_branches), "axon_child_branch_point_count": len(axon_branches),
                        "internal_edge_length_NMT_mm": float(internal[section_root]),
                        "internal_edge_length_native_nominal_um": float(internal_native[section_root]) if native is not None else None,
                        "incoming_boundary": incoming, "incoming_parent_node_id": parent,
                        "incoming_target_crossing_length_NMT_mm": float(boundary[section_root]),
                        "incoming_type_boundary_length_NMT_mm": float(compartment[section_root]),
                        "outgoing_target_crossing_count": len(target_exits), "outgoing_nonaxon_type_boundary_count": len(type_exits),
                        "nonleaf_nodes_with_no_axon_children_count": len(false_end_protected),
                        "topology_description": topology, "native_pair_available": native is not None,
                        "original_axon_ends_with_cached_cube": len(covered_ends) if covered_ends is not None else None,
                        "ending_cube_coverage_fraction": len(covered_ends)/len(section_ends) if covered_ends is not None and section_ends else None,
                        "image_review_state": "not_assessed_by_census", "biological_terminal_state": "unassessed"})
        membership.append({"section_root_node_id": section_root, "ARM6Index": labels[section_root],
                           "original_node_ids": sorted(nodes), "original_axon_end_ids": section_ends,
                           "local_branch_point_ids": local_branches, "full_graph_branch_point_ids": full_branches,
                           "axon_child_branch_point_ids": axon_branches, "original_outgoing_target_links": target_exits,
                           "original_outgoing_nonaxon_links": type_exits,
                           "nonleaf_nodes_with_no_axon_children_ids": false_end_protected,
                           "original_axon_end_ids_with_cached_cube": covered_ends})
    total_length = math.fsum(lengths.values())
    partition = math.fsum(internal.values())+math.fsum(boundary.values())+math.fsum(compartment.values())
    if not math.isclose(total_length, partition, abs_tol=1e-8, rel_tol=1e-11):
        raise AssertionError("Edge length partition failed")
    assert sum(record["original_axon_end_count"] for record in records) == len(ends)
    summary = {"node_count": len(rows), "axon_node_count": len(node_section), "selected_child_type2_edge_count": len(lengths),
               "original_axon_end_count": len(ends), "connected_axon_section_count": len(records),
               "total_child_type2_edge_length_NMT_mm": total_length,
               "internal_section_edge_length_NMT_mm": math.fsum(internal.values()),
               "target_crossing_edge_length_NMT_mm": math.fsum(boundary.values()),
               "type_boundary_edge_length_NMT_mm": math.fsum(compartment.values()),
               "target_crossing_edge_count": sum(record["crossing_reason"] == "target_label_crossing" for record in crossing_records),
               "type_boundary_edge_count": sum(record["crossing_reason"] != "target_label_crossing" for record in crossing_records),
               "original_axon_ends_with_cached_cube": sum(item["cached_cube_available"] for item in coverage) if native is not None else None}
    return records, membership, crossing_records, coverage, summary


def build(output=HERE / "selected462_ARM6_sections"):
    if output.exists():
        raise FileExistsError(output)
    ledger = pd.read_csv(SUMMARY, dtype=str, keep_default_na=False)
    coverage_ledger = pd.read_csv(COVERAGE, dtype=str, keep_default_na=False).set_index("NeuronUID", verify_integrity=True)
    if len(ledger) != 462 or ledger.NeuronUID.duplicated().any() or set(ledger.NeuronUID) != set(coverage_ledger.index):
        raise ValueError("Exact selected 462 identity join required")
    reference = nib.load(REFERENCE)
    grid = ReferenceGrid.from_image(reference)
    atlas_image = nib.load(ATLAS)
    if atlas_image.shape != (*grid.shape, 1, 6) or atlas_image.header.get_xyzt_units()[0] != "mm" or not np.allclose(atlas_image.affine, reference.affine):
        raise ValueError("Pinned ARM reference geometry mismatch")
    atlas = np.asarray(atlas_image.dataobj)[..., 0, 5]
    key = pd.read_csv(KEY, sep="\t", dtype=str, keep_default_na=False)
    keys = key.set_index(key.Index.astype(int), verify_integrity=True).to_dict("index")
    if any(label not in keys or not int(keys[label]["First_Level"]) <= 6 <= int(keys[label]["Last_Level"])
           for label in np.unique(atlas) if label > 0):
        raise ValueError("Actual positive ARM6 labels require official level-compatible key entries")
    cube_indices = defaultdict(set)
    for path in CACHE.glob("*/high_res_http/*/*.tif"):
        sample = path.relative_to(CACHE).parts[0]
        xyz = tuple(int(value) for value in path.stem.split("_"))
        if len(xyz) != 3 or path.parent.name != str(xyz[2]):
            raise ValueError("Unexpected source cube filename geometry")
        cube_indices[sample].add(xyz)
    reference_hash = sha(REFERENCE)
    output.mkdir()
    sections, neurons, crossings, native_coverage = [], [], [], []
    source_bindings, reconciliations = [], []
    membership_path = output / "section_original_node_membership.jsonl"
    with membership_path.open("x", encoding="utf-8") as member_stream:
        for count, meta in enumerate(ledger.to_dict("records"), 1):
            uid = meta["NeuronUID"]
            old = coverage_ledger.loc[uid]
            if meta["CoordinateFrame"] != "atlas_index_um" or meta["ReferenceSHA256"] != reference_hash:
                raise ValueError(f"Coordinate/reference binding mismatch: {uid}")
            scale = np.array([float(value) for value in meta["IndexScaleUm"].split(";")])
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Invalid declared index scale")
            path = ROOT / meta["SWCPath"]
            source_bytes = path.read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != meta["SWCSHA256"] or old.atlas_swc_sha256 != meta["SWCSHA256"]:
                raise ValueError(f"SWC hash mismatch: {uid}")
            rows = parse_swc(source_bytes.decode("utf-8-sig"), str(path))
            labels = sample_labels(rows, atlas, scale)
            native = None
            source_bindings.append({"NeuronUID": uid, "role": "declared_NMT_SWC", "path": relative(path), "sha256": meta["SWCSHA256"]})
            if old.native_swc_available == "True":
                native_path = ROOT / old.native_swc_path
                raw_bytes = native_path.read_bytes()
                if hashlib.sha256(raw_bytes).hexdigest() != old.native_swc_sha256:
                    raise ValueError(f"Native hash mismatch: {uid}")
                native_rows = parse_swc(raw_bytes.decode("utf-8-sig"), str(native_path))
                if identity(rows) != identity(native_rows):
                    raise ValueError(f"Native ID/type/parent mismatch: {uid}")
                native = {row[0]: np.array(row[2:5]) for row in native_rows}
                source_bindings.append({"NeuronUID": uid, "role": "nominal_native_SWC", "path": relative(native_path), "sha256": old.native_swc_sha256})
            result = analyze(rows, labels, grid.affine_mm[:3, :3] @ np.diag(1/scale),
                             native=native, covered_cube_indices=cube_indices[meta["SampleID"]])
            sr, members, links, covered, metrics = result
            if metrics["original_axon_end_count"] != int(meta["Endpoint_candidate_axon_endpoint_count"]):
                raise ValueError(f"Saved original endpoint count mismatch: {uid}")
            saved_length = float(meta["Axon_selected_axon_length_mm"])
            if not math.isclose(metrics["total_child_type2_edge_length_NMT_mm"], saved_length, rel_tol=1e-10, abs_tol=1e-7):
                raise ValueError(f"Saved all-axon edge length mismatch: {uid}")
            if native is not None:
                if metrics["original_axon_end_count"] != int(old.candidate_axon_leaves) or metrics["original_axon_ends_with_cached_cube"] != int(old.leaves_with_cached_native_cube):
                    raise ValueError(f"Prior native/coverage census mismatch: {uid}")
            common = {"NeuronUID": uid, "SampleID": meta["SampleID"], "NeuronID": meta["NeuronID"], "AnimalID": meta["AnimalID"],
                      "SourceARMFullName": meta["ARMFullName"], "SourceARMStatus": meta["SourceARMStatus"],
                      "SourceARMIndex": meta["ARMIndex"], "SWCSHA256": meta["SWCSHA256"]}
            for record, member in zip(sr, members):
                label = record["ARM6Index"]
                name = keys[label]["Full_Name"] if label > 0 else "Atlas background" if label == 0 else "Outside reference field of view"
                status = "mapped" if label > 0 else "atlas_background" if label == 0 else "outside_reference"
                section_uid = f"{uid}|ARM6:{label}|root:{record['section_root_node_id']}"
                sections.append({**common, "SectionUID": section_uid, "ARM6FullName": name, "ARM6Status": status, **record})
                member_stream.write(json.dumps({**common, "SectionUID": section_uid, **member}, allow_nan=False)+"\n")
            crossings.extend({**common, **record} for record in links)
            native_coverage.extend({**common, **record} for record in covered)
            neurons.append({**meta, **{"Census_"+key: value for key, value in metrics.items()},
                            "Census_native_pair_available": native is not None,
                            "Census_original_axon_end_cube_coverage_fraction": metrics["original_axon_ends_with_cached_cube"]/metrics["original_axon_end_count"] if native is not None and metrics["original_axon_end_count"] else None})
            reconciliations.append({"NeuronUID": uid, "original_end_count_matches_saved": True, "native_coverage_matches_prior": True if native is not None else None,
                                    "all_axon_length_delta_NMT_mm": metrics["total_child_type2_edge_length_NMT_mm"]-saved_length})
            if count % 50 == 0 or count == 462:
                print(json.dumps({"processed": count, "selected": len(ledger), "sections": len(sections)}), flush=True)
    write_csv(output / "connected_axon_sections.csv", sections)
    write_csv(output / "neuron_morphology_and_coverage.csv", neurons)
    write_csv(output / "original_target_and_type_crossing_edges.csv", crossings)
    write_csv(output / "native_original_end_cube_coverage.csv", native_coverage)
    write_csv(output / "source_swc_bindings.csv", source_bindings)
    write_json(output / "per_neuron_reconciliation.json", reconciliations)
    native_count = sum(row["Census_native_pair_available"] for row in neurons)
    native_ends = sum(row["Census_original_axon_end_count"] for row in neurons if row["Census_native_pair_available"])
    covered_ends = sum(row["Census_original_axon_ends_with_cached_cube"] or 0 for row in neurons)
    with_coverage = sum(bool(row["Census_original_axon_ends_with_cached_cube"]) for row in neurons)
    if (native_count, native_ends, covered_ends, with_coverage) != (201, 48574, 9209, 184):
        raise AssertionError("Independent native census totals changed")
    bindings = [SUMMARY, COVERAGE, ATLAS, REFERENCE, KEY, Path(__file__), ROOT / "main_scripts/swc_validation.py",
                ROOT / "main_scripts/projection_maps.py", ROOT / "notes/region_analysis_review_20261009/literature_method_update_20261009/gao_local_methods.md",
                ROOT / "notes/region_analysis_review_20261009/literature_method_update_20261009/liu_local_methods.md"]
    receipt = {"status": "software_verified_graph_morphology_and_image_availability_census", "selected_neurons": len(neurons),
               "native_identity_pairs": native_count, "native_original_axon_ends": native_ends, "native_original_ends_with_cached_cube": covered_ends,
               "neurons_with_native_end_cube_coverage": with_coverage, "sections": len(sections),
               "all_selected_original_axon_ends": sum(row["Census_original_axon_end_count"] for row in neurons),
               "topology_descriptions": dict(Counter(row["topology_description"] for row in sections)),
               "target_lookup": "np.rint(XYZ / declared IndexScaleUm), ties-to-even; current declared policy, not registration acceptance",
               "length_definition": "Euclidean NMT reference-world edge length in mm; child type 2 selects edges. Internal section lengths require both original endpoint nodes in same sampled ARM6 label; not exact voxel-overlap regional lengths.",
               "native_length_definition": "Euclidean nominal acquisition µm when exact native node correspondence exists; not calibrated tissue length",
               "image_coverage": "existence in designated local high_res_http cube-name index, not image inspection, tracing completion or biological acceptance",
               "missing_native": "null/blank coverage and native length, not zero", "biological_terminal_state": "unassessed",
               "source_labels_preserved": True, "images_downloaded": False, "plots_generated": 0,
               "python_version": sys.version, "numpy_version": np.__version__, "nibabel_version": nib.__version__,
               "inputs": [{"path": relative(path), "sha256": sha(path)} for path in bindings],
               "outputs": [{"path": relative(path), "sha256": sha(path)} for path in sorted(output.iterdir()) if path.is_file()]}
    write_json(output / "census_provenance.json", receipt)
    print(json.dumps({key: receipt[key] for key in ["status", "selected_neurons", "sections", "all_selected_original_axon_ends", "topology_descriptions"]}, indent=2))


if __name__ == "__main__":
    build()
