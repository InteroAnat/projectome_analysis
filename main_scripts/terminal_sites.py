"""Candidate endpoint and axon end-branch descriptors from validated SWCs.

Graph leaves are reconstruction ends, not accepted biological terminals,
boutons or synapses. This pure kernel does no image review, registration,
source repair, anatomical exclusion, smoothing or statistical inference.
"""

from collections import Counter
import math

import numpy as np

from projection_maps import ReferenceGrid, segment_voxels
from swc_validation import parse_swc


def terminal_site_summary(swc_text, grid: ReferenceGrid, *, coordinate_frame,
                          index_scale_um=None, source="SWC"):
    """Return JSON-compatible leaf records and sparse candidate endpoint maps.

    ``atlas_index_um`` declares source XYZ / index_scale_um as reference voxel
    indices. ``nifti_world_mm`` declares literal reference-world millimetres.
    Coordinates must already be registered; neither declaration proves this.
    Voxel i owns [i-0.5, i+0.5), matching the existing length-map convention.

    Only non-root type-2 leaves of the FULL graph enter endpoint_counts and
    neuron_occupancy. Other leaves remain in ``leaves`` for audit. Counts can
    exceed one per voxel; occupancy contains each occupied voxel exactly once
    for this neuron. Empty maps never assert biological zero innervation.

    Distances and branch lengths describe the reference template. A unique
    type-1 node supplies the soma anchor; absent/ambiguous type-1 nodes yield
    null soma distances. Terminal branch length traverses from a leaf to the
    first upstream full-graph bifurcation or root, without compartment pruning.
    The child-type-2 portion is separately reported. Zero-length edges remain.
    """
    if not isinstance(grid, ReferenceGrid):
        raise TypeError("grid must be a ReferenceGrid")
    rows = parse_swc(swc_text, source)
    positions = {row[0]: i for i, row in enumerate(rows)}
    coordinates = np.asarray([row[2:5] for row in rows], dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        if coordinate_frame == "atlas_index_um":
            scale = np.asarray(index_scale_um, dtype=float)
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Index-coordinate SWCs require three positive um/voxel scales")
            indices = coordinates / scale
            world = indices @ grid.affine_mm[:3, :3].T + grid.affine_mm[:3, 3]
        elif coordinate_frame == "nifti_world_mm":
            if index_scale_um is not None:
                raise ValueError("Index scale does not apply to NIfTI world coordinates")
            world = coordinates.copy()
            inverse = np.linalg.inv(grid.affine_mm)
            indices = world @ inverse[:3, :3].T + inverse[:3, 3]
        else:
            raise ValueError("Declare atlas_index_um or nifti_world_mm coordinates explicitly")
    if not np.isfinite(indices).all() or not np.isfinite(world).all():
        raise ValueError("Mapped coordinates must be finite")

    children = {row[0]: [] for row in rows}
    root_id = next(row[0] for row in rows if row[6] == -1)
    edge_lengths = {}
    for row in rows:
        node_id, parent_id = row[0], row[6]
        if parent_id == -1:
            continue
        children[parent_id].append(node_id)
        # Subtract indices before applying the affine: translation must not
        # degrade short-edge length when the world origin is very large.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = indices[positions[node_id]] - indices[positions[parent_id]]
            length = float(np.linalg.norm(grid.affine_mm[:3, :3] @ delta))
        if not np.isfinite(length):
            raise ValueError("Template-space edge lengths must be finite")
        edge_lengths[node_id] = length

    soma_ids = [row[0] for row in rows if row[1] == 1]
    soma_id = soma_ids[0] if len(soma_ids) == 1 else None
    soma_state = "unique_type1_node" if soma_id is not None else (
        "no_type1_node" if not soma_ids else "ambiguous_multiple_type1_nodes")
    leaf_ids = [row[0] for row in rows if not children[row[0]]]
    coincident = Counter(tuple(indices[positions[node_id]]) for node_id in leaf_ids)
    endpoint_counts = Counter()
    leaves = []

    def distance(node_id, anchor_id):
        if anchor_id is None:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            delta = indices[positions[node_id]] - indices[positions[anchor_id]]
            value = float(np.linalg.norm(grid.affine_mm[:3, :3] @ delta))
        if not np.isfinite(value):
            raise ValueError("Template-space anchor distances must be finite")
        return value

    for node_id in leaf_ids:
        row = rows[positions[node_id]]
        node_type, parent_id = row[1], row[6]
        candidate = parent_id != -1 and node_type == 2
        leaf_class = "root_only" if parent_id == -1 else (
            "candidate_axon_endpoint" if candidate else (
                "dendritic_leaf" if node_type in (3, 4) else "unresolved_leaf"))
        point = indices[positions[node_id]]
        inside = bool(np.all(point >= -0.5) and np.all(point < np.asarray(grid.shape) - 0.5))
        # Avoid rounding a representable point just below +0.5 onto the face
        # when adding 0.5 in floating point. Compare the fractional part.
        base = np.floor(point)
        voxel = [int(value) for value in base + (point - base >= 0.5)] if inside else None
        if candidate and inside:
            endpoint_counts[tuple(voxel)] += 1

        current_id = node_id
        branch_length = 0.0
        branch_axon_length = 0.0
        branch_transitions = 0
        while rows[positions[current_id]][6] != -1:
            current_row = rows[positions[current_id]]
            upstream_id = current_row[6]
            branch_length += edge_lengths[current_id]
            if current_row[1] == 2:
                branch_axon_length += edge_lengths[current_id]
            branch_transitions += int(current_row[1] != rows[positions[upstream_id]][1])
            current_id = upstream_id
            if current_id == root_id or len(children[current_id]) != 1:
                break
        if not np.isfinite(branch_length) or not np.isfinite(branch_axon_length):
            raise ValueError("Template-space branch lengths must be finite")
        terminal_length = edge_lengths.get(node_id)
        leaves.append({
            "node_id": node_id, "node_type": node_type, "parent_id": parent_id,
            "source_xyz": list(row[2:5]), "index_xyz": point.tolist(),
            "world_mm": world[positions[node_id]].tolist(), "voxel": voxel,
            "leaf_class": leaf_class, "candidate_axon_endpoint": candidate,
            "in_reference": inside, "soma_distance_mm": distance(node_id, soma_id),
            "root_distance_mm": distance(node_id, root_id),
            "terminal_edge_length_mm": terminal_length,
            "terminal_branch_length_mm": branch_length,
            "terminal_branch_axon_length_mm": branch_axon_length,
            "branch_start_node_id": current_id,
            "terminal_branch_compartment_transitions": branch_transitions,
            "terminal_edge_type_transition": parent_id != -1 and node_type != rows[positions[parent_id]][1],
            "zero_length_terminal_edge": terminal_length == 0.0,
            "coincident_leaf_count": coincident[tuple(point)],
            "image_review_state": "unresolved",
        })

    candidates = [leaf for leaf in leaves if leaf["candidate_axon_endpoint"]]
    axon_nodes = sum(row[1] == 2 for row in rows)
    availability = "candidate_endpoints_present" if candidates else (
        "no_type2_nodes" if axon_nodes == 0 else "no_eligible_axon_leaf")
    voxels = sorted(endpoint_counts)
    return {
        "source": str(source),
        "measurement": "candidate full-graph axon endpoints; not accepted terminals or synapses",
        "coordinate_frame": coordinate_frame,
        "index_scale_um": np.asarray(index_scale_um, dtype=float).tolist() if coordinate_frame == "atlas_index_um" else None,
        "reference_shape": list(grid.shape), "reference_affine_mm": grid.affine_mm.tolist(),
        "voxel_convention": "voxel i owns [i-0.5, i+0.5); floor(index+0.5); half-open reference FOV",
        "image_review_state": "unresolved", "leaves": leaves,
        "endpoint_counts": [{"voxel": list(voxel), "count": endpoint_counts[voxel]} for voxel in voxels],
        "neuron_occupancy": [list(voxel) for voxel in voxels],
        "qc": {
            "node_count": len(rows), "edge_count": len(rows) - 1, "root_node_id": root_id,
            "node_type_counts": {str(kind): count for kind, count in sorted(Counter(row[1] for row in rows).items())},
            "leaf_type_counts": {str(kind): count for kind, count in sorted(Counter(leaf["node_type"] for leaf in leaves).items())},
            "type2_node_count": axon_nodes, "leaf_count": len(leaves),
            "leaf_class_counts": dict(Counter(leaf["leaf_class"] for leaf in leaves)),
            "candidate_axon_endpoint_count": len(candidates),
            "in_reference_candidate_count": sum(leaf["in_reference"] for leaf in candidates),
            "outside_reference_candidate_count": sum(not leaf["in_reference"] for leaf in candidates),
            "occupied_candidate_voxel_count": len(voxels),
            "zero_length_candidate_terminal_edges": sum(leaf["zero_length_terminal_edge"] for leaf in candidates),
            "candidate_availability": availability,
            "candidate_status": availability,
            "candidate_axon_endpoints": {
                "total": len(candidates),
                "in_reference": sum(leaf["in_reference"] for leaf in candidates),
                "outside_reference": sum(not leaf["in_reference"] for leaf in candidates),
            },
            "biological_innervation_state": "unassessed",
            "soma_anchor_state": soma_state, "soma_anchor_node_id": soma_id,
            "type1_node_ids": soma_ids,
            "length_space": "reference template; native tissue length not established",
        },
    }


def reconstructed_axon_end_branch_summary(swc_text, grid: ReferenceGrid, *,
                                         coordinate_frame, index_scale_um=None,
                                         source="SWC"):
    """Return original axon end-branches and sparse trajectory length in mm.

    Begin only at a non-root axon-labelled leaf of the complete original
    graph. Walk upstream to the first original full-graph bifurcation/root,
    or to a non-axon parent if reached first. Retain that final child-axon
    transition edge, flag it, and stop: never bridge across non-axon nodes.
    This uses the whole-axon map's child-label edge policy, while differing
    from the existing full-compartment terminal_branch_axon_length_mm QC.

    Rasterize each selected original edge along its trajectory using the
    reference grid's half-open voxels. Edges are disjoint across end-branches;
    whole-chain length is never deposited at a leaf. No source sampling,
    radius, arbor-size or biological acceptance threshold is introduced.
    Length describes the reference template, not calibrated native tissue.
    An unfinished unbranched trunk can satisfy this graph definition.

    No eligible ending means computable=False and null aggregate lengths,
    not zero innervation. Eligible zero-length or all-outside branches remain
    computable and retain their original identities and coverage accounting.
    Sparse entries contain only positive in-reference length, in mm/voxel.
    """
    if not isinstance(grid, ReferenceGrid):
        raise TypeError("grid must be a ReferenceGrid")
    rows = parse_swc(swc_text, source)
    by_id = {row[0]: row for row in rows}
    positions = {row[0]: i for i, row in enumerate(rows)}
    indices = np.asarray([row[2:5] for row in rows], dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        if coordinate_frame == "atlas_index_um":
            scale = np.asarray(index_scale_um, dtype=float)
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Index-coordinate SWCs require three positive um/voxel scales")
            indices = indices / scale
        elif coordinate_frame == "nifti_world_mm":
            if index_scale_um is not None:
                raise ValueError("Index scale does not apply to NIfTI world coordinates")
            inverse = np.linalg.inv(grid.affine_mm)
            indices = indices @ inverse[:3, :3].T + inverse[:3, 3]
        else:
            raise ValueError("Declare atlas_index_um or nifti_world_mm coordinates explicitly")
    if not np.isfinite(indices).all():
        raise ValueError("Mapped coordinates must be finite")

    children = {row[0]: [] for row in rows}
    root_id = next(row[0] for row in rows if row[6] == -1)
    for row in rows:
        if row[6] != -1:
            children[row[6]].append(row[0])
    leaf_ids = sorted(row[0] for row in rows
                      if row[1] == 2 and row[6] != -1 and not children[row[0]])
    branches, selected_ids = [], set()
    for leaf_id in leaf_ids:
        current = leaf_id
        chain = []
        while True:
            row = by_id[current]
            parent_id = row[6]
            if current in selected_ids:
                raise AssertionError("Original end-branches must not duplicate an edge")
            selected_ids.add(current)
            chain.append(current)
            parent = by_id[parent_id]
            at_root = parent_id == root_id
            at_bifurcation = len(children[parent_id]) >= 2
            at_compartment_boundary = parent[1] != 2
            current = parent_id
            if at_root or at_bifurcation or at_compartment_boundary:
                break
        branches.append({
            "leaf_node_id": leaf_id, "start_node_id": current,
            "start_node_type": by_id[current][1],
            "selected_child_node_ids": chain,
            "stop_reason": "original_root" if at_root else (
                "original_full_graph_bifurcation" if at_bifurcation else "nonaxon_parent"),
            "stop_at_original_root": at_root,
            "stop_at_full_graph_bifurcation": at_bifurcation,
            "stop_at_compartment_boundary": at_compartment_boundary,
            "final_transition_edge_included": at_compartment_boundary,
            "image_review_state": "unresolved",
            "biological_terminal_state": "unassessed",
        })

    selected_ids = sorted(selected_ids)
    child_positions = [positions[node_id] for node_id in selected_ids]
    parent_positions = [positions[by_id[node_id][6]] for node_id in selected_ids]
    starts = indices[parent_positions]
    ends = indices[child_positions]
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = np.linalg.norm((ends - starts) @ grid.affine_mm[:3, :3].T, axis=1)
    if not np.isfinite(lengths).all():
        raise ValueError("Template-space edge lengths must be finite")
    inside_lengths = np.zeros(len(lengths), dtype=float)
    voxel_lengths = Counter()
    # Avoid adding 0.5 before rounding: an immediately adjacent float below
    # a face must stay in the lower voxel, as in the candidate endpoint kernel.
    start_base, end_base = np.floor(starts), np.floor(ends)
    start_voxels = start_base + (starts - start_base >= 0.5)
    end_voxels = end_base + (ends - end_base >= 0.5)
    same = (np.all(start_voxels == end_voxels, axis=1)
            & np.all(start_voxels >= 0, axis=1)
            & np.all(start_voxels < grid.shape, axis=1))
    inside_lengths[same] = lengths[same]
    if np.any(same):
        voxels, inverse = np.unique(start_voxels[same].astype(int), axis=0, return_inverse=True)
        weights = np.bincount(inverse, weights=lengths[same])
        voxel_lengths.update({tuple(voxel): float(weight) for voxel, weight in zip(voxels, weights) if weight > 0})
    for edge_index in np.flatnonzero(~same & (lengths > 0)):
        pieces = []
        for voxel, fraction in segment_voxels(starts[edge_index], ends[edge_index], grid.shape):
            piece = float(lengths[edge_index] * fraction)
            voxel_lengths[voxel] += piece
            pieces.append(piece)
        inside_lengths[edge_index] = math.fsum(pieces)
    edge_records = []
    edge_lookup = {}
    for i, node_id in enumerate(selected_ids):
        parent_id = by_id[node_id][6]
        outside = max(0.0, float(lengths[i] - inside_lengths[i]))
        record = {
            "child_node_id": node_id, "parent_node_id": parent_id,
            "child_node_type": 2, "parent_node_type": by_id[parent_id][1],
            "compartment_transition": by_id[parent_id][1] != 2,
            "length_mm": float(lengths[i]),
            "in_reference_length_mm": float(inside_lengths[i]),
            "outside_reference_length_mm": outside,
        }
        edge_records.append(record)
        edge_lookup[node_id] = record
    for branch in branches:
        chain = [edge_lookup[node_id] for node_id in branch["selected_child_node_ids"]]
        for key in ("length_mm", "in_reference_length_mm", "outside_reference_length_mm"):
            branch[key] = math.fsum(edge[key] for edge in chain)

    total = math.fsum(lengths.tolist())
    inside = math.fsum(inside_lengths.tolist())
    sparse_total = math.fsum(voxel_lengths.values())
    if (not math.isclose(inside, sparse_total, rel_tol=1e-10, abs_tol=1e-10)
            or inside > total and not math.isclose(inside, total, rel_tol=1e-10, abs_tol=1e-10)):
        raise ArithmeticError("End-branch voxel allocation failed length conservation")
    computable = bool(leaf_ids)
    axon_nodes = sum(row[1] == 2 for row in rows)
    return {
        "source": str(source),
        "measurement": "reconstructed axon end-branch trajectory length; not accepted terminal arbors or synapses",
        "coordinate_frame": coordinate_frame,
        "index_scale_um": np.asarray(index_scale_um, dtype=float).tolist() if coordinate_frame == "atlas_index_um" else None,
        "reference_shape": list(grid.shape), "reference_affine_mm": grid.affine_mm.tolist(),
        "voxel_convention": "voxel i owns [i-0.5, i+0.5); half-open reference FOV",
        "compartment_policy": "child SWC type 2 selects edges; include final nonaxon-parent transition edge then stop, never bridge nonaxon chains",
        "branches": branches, "selected_edges": edge_records,
        "voxel_lengths_mm": [{"voxel": [int(value) for value in voxel], "length_mm": float(length)}
                             for voxel, length in sorted(voxel_lengths.items()) if length > 0],
        "qc": {
            "computable": computable, "map_available": computable,
            "candidate_status": "candidate_endpoints_present" if computable else (
                "no_type2_nodes" if axon_nodes == 0 else "no_eligible_axon_leaf"),
            "node_count": len(rows), "root_node_id": root_id,
            "node_type_counts": {str(kind): count for kind, count in sorted(Counter(row[1] for row in rows).items())},
            "candidate_axon_endpoint_count": len(leaf_ids), "branch_count": len(branches),
            "selected_axon_edge_count": len(selected_ids),
            "zero_length_selected_edges": int(np.count_nonzero(lengths == 0)),
            "final_transition_edge_count": sum(edge["compartment_transition"] for edge in edge_records),
            "total_length_mm": total if computable else None,
            "in_reference_length_mm": inside if computable else None,
            "outside_reference_length_mm": max(0.0, total - inside) if computable else None,
            "occupied_length_voxels": sum(length > 0 for length in voxel_lengths.values()),
            "length_space": "reference template; native tissue length not established",
            "biological_innervation_state": "unassessed",
            "image_review_state": "unresolved",
        },
    }
