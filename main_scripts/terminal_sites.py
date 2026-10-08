"""Candidate endpoint descriptors from complete, structurally validated SWCs.

Graph leaves are reconstruction ends, not accepted biological terminals,
boutons or synapses. This pure kernel does no image review, registration,
source repair, anatomical exclusion, smoothing or statistical inference.
"""

from collections import Counter

import numpy as np

from projection_maps import ReferenceGrid
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
