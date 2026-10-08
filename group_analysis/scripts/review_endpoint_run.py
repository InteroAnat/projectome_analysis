"""Independent saved-output review for candidate axonal endpoint runs.

Reconstructs leaves directly from source parent IDs and assigns voxels with a
boundary search. It does not import the endpoint detector or atlas sampler.
This verifies implementation and saved artifacts, not biological terminals,
tracing completeness or native-to-template anatomical registration.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REVIEWER_VERSION = "2.0"
IDENTITY_FIELDS = ("AnimalID", "SampleID", "NeuronID", "Subregion", "OriginalSourceLabel",
                   "AnatomyStatus", "RegistrationStatus", "SWCPath", "SWCSHA256", "ReferenceSHA256",
                   "CoordinateFrame", "IndexScaleUm")
METRICS = {"count": "candidate endpoints per computable neuron",
           "density": "candidate endpoints per template mm3 per computable neuron",
           "occupancy": "fraction of computable neurons with candidate endpoint evidence"}


def integer(value, context):
    """Accept finite integral decimal CSV values, including nullable-column 1.0."""
    try:
        number = Decimal(str(value))
        require(number.is_finite() and number == number.to_integral_value(), f"Noninteger {context}")
        return int(number)
    except (ArithmeticError, ValueError) as exc:
        raise ValueError(f"Noninteger {context}: {value!r}") from exc


def coded_grid(image, context, shape=None, affine=None, *, spatial_only=False):
    """Independently check explicit mm, coded transforms and invertible geometry."""
    require(image.header.get_xyzt_units()[0] == "mm", f"{context} units must be mm")
    sform, scode = image.header.get_sform(coded=True)
    qform, qcode = image.header.get_qform(coded=True)
    require(bool(scode or qcode), f"{context} has no coded transform")
    for matrix, code in ((sform, scode), (qform, qcode)):
        if code:
            require(np.isfinite(matrix).all(), f"{context} has nonfinite coded transform")
            require(np.allclose(matrix, image.affine, rtol=0, atol=1e-7), f"{context} coded transform differs")
    actual_shape = image.shape[:3] if spatial_only else image.shape
    require(len(actual_shape) == 3 and all(size > 0 for size in actual_shape), f"{context} is not a 3D grid")
    require(np.isfinite(image.affine).all() and np.allclose(image.affine[3], [0, 0, 0, 1]),
            f"{context} has invalid affine")
    volume = abs(float(np.linalg.det(image.affine[:3, :3])))
    require(np.isfinite(volume) and volume > 0, f"{context} has singular geometry")
    if shape is not None:
        require(tuple(actual_shape) == tuple(shape), f"{context} shape differs")
        require(np.allclose(image.affine, affine, rtol=0, atol=1e-7), f"{context} affine differs")
    return tuple(actual_shape), np.asarray(image.affine), volume


def identity_check(saved, entry, context, *, nested=False):
    for field in IDENTITY_FIELDS:
        require(saved.get(field) == entry.get(field, ""), f"{context} metadata differs: {field}")
    if nested:
        require(saved.get("source_metadata") == entry, f"{context} nested source metadata differs")
    else:
        require(json.loads(saved["source_metadata_json"]) == entry, f"{context} source metadata differs")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source_expectations(manifest, shape, affine, input_root=ROOT):
    """Read complete original graphs independently; return all source leaves.

    A source leaf ID never appears in the source parent column. Excluding roots
    and requiring type 2 then defines the candidate endpoint subset. Voxel
    lookup uses sorted faces and right-sided search rather than detector code.
    """
    affine = np.asarray(affine, dtype=float)
    inverse = np.linalg.inv(affine)
    faces = [np.arange(size + 1, dtype=float) - .5 for size in shape]
    leaves, neurons, source_paths = {}, {}, set()
    for entry in manifest:
        require(set(IDENTITY_FIELDS) <= set(entry), "Missing manifest identity/coordinate fields")
        require(all(isinstance(entry[field], str) for field in IDENTITY_FIELDS), "Nonstring manifest identity/coordinate fields")
        # An absent historical/portal label remains exactly blank even when
        # independent human evidence establishes the selected coarse group.
        require(all(entry[field].strip() for field in IDENTITY_FIELDS if field not in {"IndexScaleUm", "OriginalSourceLabel"}),
                "Blank manifest identity/coordinate fields")
        require(all(len(entry[field]) == 64 and all(char in "0123456789abcdef" for char in entry[field].lower())
                    for field in ("SWCSHA256", "ReferenceSHA256")), "Invalid manifest source/reference hash")
        uid = (entry["SampleID"], entry["NeuronID"])
        require(uid not in neurons, f"Duplicate manifest identity: {uid}")
        path = Path(entry["SWCPath"])
        if not path.is_absolute():
            path = Path(input_root) / path
        path = path.resolve()
        require(path not in source_paths, f"Reused source path: {path}")
        source_paths.add(path)
        require(sha256(path) == entry["SWCSHA256"].lower(), f"Changed source: {uid}")
        rows = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            require(len(fields) >= 7, f"Malformed source row: {uid}")
            node, kind, parent = [integer(fields[index], f"source graph field {uid}") for index in (0, 1, 6)]
            require(node > 0 and kind >= 0 and (parent == -1 or parent > 0), f"Invalid source graph field: {uid}")
            require(node not in rows, f"Duplicate source node: {uid}/{node}")
            xyz = np.asarray([float(value) for value in fields[2:5]])
            require(np.isfinite(xyz).all(), f"Nonfinite source point: {uid}/{node}")
            radius = float(fields[5])
            require(np.isfinite(radius) and radius >= 0, f"Invalid source radius: {uid}/{node}")
            rows[node] = (kind, parent, xyz)
        roots = [node for node, (_, parent, _) in rows.items() if parent == -1]
        require(len(roots) == 1, f"Source must have one root: {uid}")
        require(all(parent == -1 or parent in rows for _, parent, _ in rows.values()),
                f"Source has missing parent: {uid}")
        root = roots[0]
        reached = {root}
        for start in rows:
            node, visiting = start, set()
            while node not in reached:
                require(node not in visiting, f"Source parent cycle: {uid}/{node}")
                visiting.add(node)
                node = rows[node][1]
            reached.update(visiting)
        require(len(reached) == len(rows), f"Source does not reach root: {uid}")
        if entry["CoordinateFrame"] == "atlas_index_um":
            scale = np.asarray([float(value) for value in entry["IndexScaleUm"].split(";")])
            require(scale.shape == (3,) and np.isfinite(scale).all() and np.all(scale > 0),
                    f"Bad coordinate scale: {uid}")
        else:
            require(entry["CoordinateFrame"] == "nifti_world_mm" and not entry["IndexScaleUm"],
                    f"Unknown source frame: {uid}")
            scale = None
        indices = {node: xyz / scale if scale is not None else (inverse @ np.append(xyz, 1))[:3]
                   for node, (_, _, xyz) in rows.items()}
        require(all(np.isfinite(point).all() for point in indices.values()), f"Bad mapped source graph: {uid}")
        children = Counter(parent for _, parent, _ in rows.values() if parent != -1)
        edge_lengths = {node: float(np.linalg.norm(affine[:3, :3] @ (indices[node] - indices[parent])))
                        for node, (_, parent, _) in rows.items() if parent != -1}
        require(all(np.isfinite(length) for length in edge_lengths.values()), f"Nonfinite edge length: {uid}")
        soma_ids = [node for node, (kind, _, _) in rows.items() if kind == 1]
        soma = soma_ids[0] if len(soma_ids) == 1 else None
        soma_state = "unique_type1_node" if soma is not None else (
            "no_type1_node" if not soma_ids else "ambiguous_multiple_type1_nodes")
        parents = {parent for _, parent, _ in rows.values()}
        counts, occupied = Counter(), set()
        node_types = Counter(kind for kind, _, _ in rows.values())
        leaf_types, leaf_classes = Counter(), Counter()
        coincident = Counter(tuple(indices[node]) for node in rows if node not in parents)
        zero_candidate_edges = 0
        candidate_count, outside = 0, 0
        for node, (kind, parent, xyz) in rows.items():
            if node in parents:
                continue
            index = indices[node]
            world = (affine @ np.append(index, 1))[:3] if scale is not None else xyz
            require(np.isfinite(index).all() and np.isfinite(world).all(), f"Bad mapped point: {uid}/{node}")
            voxel = tuple(int(np.searchsorted(axis_faces, position, side="right") - 1)
                          for axis_faces, position in zip(faces, index))
            inside = all(0 <= value < size for value, size in zip(voxel, shape))
            candidate = kind == 2 and parent != -1
            leaf_class = "root_only" if parent == -1 else (
                "candidate_axon_endpoint" if candidate else "dendritic_leaf" if kind in (3, 4) else "unresolved_leaf")
            leaf_types[kind] += 1
            leaf_classes[leaf_class] += 1
            edge_length = edge_lengths.get(node)
            branch_node, branch_length, axon_length, transitions = node, 0.0, 0.0, 0
            while rows[branch_node][1] != -1:
                upstream = rows[branch_node][1]
                length = edge_lengths[branch_node]
                branch_length += length
                axon_length += length if rows[branch_node][0] == 2 else 0.0
                transitions += int(rows[branch_node][0] != rows[upstream][0])
                branch_node = upstream
                if upstream == root or children[upstream] != 1:
                    break
            if candidate:
                candidate_count += 1
                zero_candidate_edges += int(edge_length == 0.0)
                if inside:
                    counts[voxel] += 1
                    occupied.add(voxel)
                else:
                    outside += 1
            leaves[(*uid, node)] = {
                "node_id": node, "node_type": kind, "parent_id": parent,
                "source_xyz": xyz, "index_xyz": index, "world_mm": world,
                "candidate_axon_endpoint": candidate, "in_reference": inside,
                "voxel": voxel if inside else None,
                "leaf_class": leaf_class,
                "soma_distance_mm": None if soma is None else float(np.linalg.norm(affine[:3, :3] @ (index - indices[soma]))),
                "root_distance_mm": float(np.linalg.norm(affine[:3, :3] @ (index - indices[root]))),
                "terminal_edge_length_mm": edge_length,
                "terminal_branch_length_mm": branch_length,
                "terminal_branch_axon_length_mm": axon_length,
                "branch_start_node_id": branch_node,
                "terminal_branch_compartment_transitions": transitions,
                "terminal_edge_type_transition": parent != -1 and kind != rows[parent][0],
                "zero_length_terminal_edge": edge_length == 0.0,
                "coincident_leaf_count": coincident[tuple(index)],
            }
        require(sha256(path) == entry["SWCSHA256"].lower(), f"Source changed during review: {uid}")
        neurons[uid] = {
            **entry, "source_metadata": entry, "source_path": str(path), "index_scale_um": None if scale is None else scale.tolist(),
            "node_count": len(rows), "node_type_counts": dict(node_types),
            "leaf_type_counts": dict(leaf_types), "leaf_count": sum(leaf_types.values()),
            "candidate_count": candidate_count, "outside_count": outside,
            "computable": candidate_count > 0, "counts": counts, "occupied": occupied,
            "edge_count": len(rows) - 1, "root_node_id": root, "type2_node_count": node_types[2],
            "leaf_class_counts": dict(leaf_classes), "occupied_candidate_voxel_count": len(occupied),
            "zero_length_candidate_terminal_edges": zero_candidate_edges,
            "candidate_status": "candidate_endpoints_present" if candidate_count else "no_type2_nodes" if not node_types[2] else "no_eligible_axon_leaf",
            "soma_anchor_state": soma_state, "soma_anchor_node_id": soma, "type1_node_ids": soma_ids,
        }
    return leaves, neurons


def check_sparse_map(path, expected, shape, affine, scale=1.0, *, fraction=False):
    """Read every saved voxel; compare nonzero support and exact sparse values."""
    image = nib.load(path)
    coded_grid(image, f"Saved map {path}", shape, affine)
    require(image.header.get_intent()[0] == "none", f"Descriptive map has statistical intent: {path}")
    values = image.get_fdata()
    require(np.isfinite(values).all() and np.all(values >= 0), f"Invalid saved map values: {path}")
    if fraction:
        require(np.all(values <= 1 + 1e-6), f"Neuron occupancy exceeds one: {path}")
    coordinates = sorted(point for point, value in expected.items() if value > 0)
    expected_flat = np.sort(np.ravel_multi_index(np.asarray(coordinates).T, shape)) if coordinates else np.array([], dtype=int)
    np.testing.assert_array_equal(np.flatnonzero(values), expected_flat,
                                  err_msg=f"Saved nonzero support differs: {path}")
    if coordinates:
        observed = values[tuple(np.asarray(coordinates).T)]
        np.testing.assert_allclose(observed, [expected[point] * scale for point in coordinates],
                                   rtol=1e-6, atol=1e-7, err_msg=f"Saved values differ: {path}")
    return {"file": str(path), "sha256": sha256(path), "voxels": int(values.size),
            "nonzero_voxels": len(coordinates), "sum": float(values.sum())}


def atlas_expectations(inputs, record, leaves):
    """Look up all source leaves in actual atlas levels without sampler code."""
    atlas_image = nib.load(inputs["atlas"]["path"])
    shape, affine = tuple(record["shape"]), np.asarray(record["affine_mm"])
    require(atlas_image.shape == (*shape, 1, 6), "Unexpected six-level atlas dimensions")
    coded_grid(atlas_image, "Atlas", shape, affine, spatial_only=True)
    atlas = np.asanyarray(atlas_image.dataobj)[..., 0, :]
    require(np.issubdtype(atlas.dtype, np.integer) and np.all(atlas >= 0), "Invalid atlas labels")
    with Path(inputs["atlas_key"]["path"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(bool(rows) and {"Index", "Abbreviation", "Full_Name", "First_Level", "Last_Level"} <= set(rows[0]),
            "Missing atlas key fields")
    labels = {}
    for row in rows:
        index, first, last = (integer(row[field], f"atlas key {field}") for field in ("Index", "First_Level", "Last_Level"))
        require(index > 0 and 1 <= first <= last <= 6 and row["Abbreviation"].strip() and row["Full_Name"].strip(),
                "Invalid atlas key label/level/name")
        labels[index] = row
    require(len(labels) == len(rows), "Duplicate atlas key label")
    mask_image = nib.load(inputs["hemisphere_mask"]["path"])
    coded_grid(mask_image, "Hemisphere mask", shape, affine)
    hemisphere = np.asanyarray(mask_image.dataobj)
    require(np.isfinite(hemisphere).all() and np.all(hemisphere >= 0)
            and np.equal(hemisphere, np.floor(hemisphere)).all(), "Invalid hemisphere mask values")
    side_of = lambda label: next((side for prefix, side in (("CL_", "L"), ("CR_", "R"),
                                                          ("SL_", "L"), ("SR_", "R"),
                                                          ("L-", "L"), ("R-", "R"),
                                                          ("L_", "L"), ("R_", "R"))
                                if str(label or "").startswith(prefix)), "Unknown")
    # Independently derive each mask value's side using all finest-level atlas
    # labels, rather than accepting the builder's value-to-side declaration.
    mask_sides, contingency = {}, {}
    for value in np.unique(hemisphere):
        if value == 0:
            continue
        present, counts = np.unique(atlas[..., 5][hemisphere == value], return_counts=True)
        evidence = Counter()
        for index, count in zip(present, counts):
            metadata = labels.get(int(index), {})
            evidence[side_of(metadata.get("Abbreviation"))] += int(count)
        require(bool(evidence["L"]) != bool(evidence["R"]), "Hemisphere mask has mixed or absent label evidence")
        mask_sides[int(value)] = "L" if evidence["L"] else "R"
        contingency[str(int(value))] = {side: evidence[side] for side in ("L", "R")}
    require(set(mask_sides.values()) == {"L", "R"}, "Mask does not distinguish both hemispheres")
    require({str(key): value for key, value in mask_sides.items()} == record["hemisphere_value_to_side"],
            "Recorded hemisphere semantics differ from full atlas evidence")
    require(contingency == record["hemisphere_label_contingency"], "Recorded hemisphere contingency differs")
    volume = abs(float(np.linalg.det(affine[:3, :3])))
    volumes = {}
    for level in range(1, 7):
        indices, counts = np.unique(atlas[..., level - 1], return_counts=True)
        volumes[level] = {int(index): int(count) * volume for index, count in zip(indices, counts)}
    brain = None
    if "brain_mask" in inputs:
        image = nib.load(inputs["brain_mask"]["path"])
        coded_grid(image, "Brain mask", shape, affine)
        values = np.asanyarray(image.dataobj)
        require(np.isfinite(values).all() and np.all(values >= 0), "Invalid brain-mask values")
        brain = values > 0
    expected = {}
    for uid, leaf in leaves.items():
        voxel = leaf["voxel"]
        side = mask_sides.get(int(hemisphere[voxel]), "Unknown") if voxel is not None else "Unknown"
        mask_status = "out_of_FOV" if voxel is None else (
            "not_provided" if brain is None else "inside" if brain[voxel] else "outside")
        targets = []
        for level in range(1, 7):
            index = int(atlas[voxel + (level - 1,)]) if voxel is not None else None
            metadata = labels.get(index)
            abbreviation = metadata["Abbreviation"] if metadata else None
            if index is None:
                status = "out_of_FOV"
            elif index == 0:
                status = "zero_unassigned"
            elif metadata is None:
                status = "unmapped_label"
            elif int(metadata["First_Level"]) <= level <= int(metadata["Last_Level"]):
                status = "mapped"
            else:
                status = "key_level_conflict"
            label_side = side_of(abbreviation)
            targets.append({"level": level, "index": index, "abbreviation": abbreviation,
                            "full_name": metadata["Full_Name"] if metadata else None,
                            "target_status": status, "label_side": label_side,
                            "hemisphere_conflict": side in ("L", "R") and label_side in ("L", "R") and side != label_side,
                            "target_volume_mm3": volumes[level].get(index) if status == "mapped" else None})
        expected[uid] = {"hemisphere_side": side, "brain_mask_status": mask_status, "atlas_levels": targets}
    return expected, contingency


def check_tables(run_dir, neurons, leaves, targets):
    """Reconcile per-neuron QC and regional abundance/presence denominators."""
    qc_rows = csv_rows(run_dir / "per_neuron_qc.csv")
    qc = {(row["SampleID"], row["NeuronID"]): row for row in qc_rows}
    require(len(qc) == len(qc_rows) and set(qc) == set(neurons), "Per-neuron QC identity scope differs")
    candidate_targets_by_neuron = defaultdict(list)
    for key, leaf in leaves.items():
        if leaf["candidate_axon_endpoint"]:
            candidate_targets_by_neuron[key[:2]].append(targets[key])
    for uid, neuron in neurons.items():
        saved = qc[uid]
        values = {"node_count": neuron["node_count"], "leaf_count": neuron["leaf_count"],
                  "candidate_axon_endpoint_count": neuron["candidate_count"],
                  "outside_reference_candidate_count": neuron["outside_count"],
                  "in_reference_candidate_count": neuron["candidate_count"] - neuron["outside_count"],
                  "n_selected": 1, "n_computable": int(neuron["computable"])}
        values.update({field: neuron[field] for field in ("edge_count", "root_node_id", "type2_node_count",
                       "occupied_candidate_voxel_count", "zero_length_candidate_terminal_edges")})
        for field, value in values.items():
            require(integer(saved[field], field) == value, f"QC count differs: {uid}/{field}")
        identity_check(saved, neuron["source_metadata"], f"QC {uid}")
        for field in ("node_type_counts", "leaf_type_counts", "leaf_class_counts", "type1_node_ids"):
            expected = neuron[field]
            if isinstance(expected, dict):
                expected = {str(key): value for key, value in expected.items()}
            require(json.loads(saved[field + "_json"]) == expected, f"QC compartment counts differ: {uid}/{field}")
        require(saved["candidate_status"] == neuron["candidate_status"]
                and saved["map_proxy_status"] == ("computable_candidate_proxy" if neuron["computable"] else "unassessed_no_eligible_candidate"),
                f"QC candidate status differs: {uid}")
        require(saved["soma_anchor_state"] == neuron["soma_anchor_state"], f"QC soma state differs: {uid}")
        soma_id = integer(saved["soma_anchor_node_id"], "soma anchor") if saved["soma_anchor_node_id"] else None
        require(soma_id == neuron["soma_anchor_node_id"], f"QC soma anchor differs: {uid}")
        candidate_targets = candidate_targets_by_neuron[uid]
        unknown, conflicts = Counter(), Counter()
        for item in candidate_targets:
            for target in item["atlas_levels"]:
                if target["target_status"] != "mapped":
                    unknown[str(target["level"])] += 1
                if target["hemisphere_conflict"]:
                    conflicts[str(target["level"])] += 1
        for field, expected in (("unknown_target_counts_by_level_json", unknown), ("hemisphere_conflict_counts_by_level_json", conflicts)):
            require(json.loads(saved[field]) == dict(expected), f"QC target accounting differs: {uid}/{field}")
        # The presence of a mask cannot be deduced from leaves when every point is outside FOV.
        mask_present = neuron["brain_mask_provided"]
        for field, status in (("in_brain_mask_candidate_count", "inside"), ("outside_brain_mask_in_FOV_candidate_count", "outside")):
            expected = sum(item["brain_mask_status"] == status for item in candidate_targets)
            require((integer(saved[field], field) == expected) if mask_present else saved[field] == "",
                    f"QC mask accounting differs: {uid}/{field}")
        require(saved["biological_innervation_state"] == "unassessed" and saved["image_review_state"] == "unresolved",
                f"Unsupported QC acceptance: {uid}")
    expected_counts = Counter()
    metadata = {}
    for uid, leaf in leaves.items():
        if not leaf["candidate_axon_endpoint"]:
            continue
        for target in targets[uid]["atlas_levels"]:
            target_key = (target["level"], target["target_status"], target["index"])
            expected_counts[(*uid[:2], *target_key)] += 1
            metadata[target_key] = target
    def target_key(row):
        return (integer(row["level"], "target level"), row["target_status"],
                integer(row["target_index"], "target index") if row["target_index"] else None)
    def target_check(row, target, context):
        require(row["target_abbreviation"] == (target["abbreviation"] or "")
                and row["target_full_name"] == (target["full_name"] or ""), f"{context} label differs")
        if target["target_volume_mm3"] is None:
            require(row["target_volume_mm3"] == "", f"{context} unsupported target volume")
        else:
            np.testing.assert_allclose(float(row["target_volume_mm3"]), target["target_volume_mm3"], rtol=1e-10)
    observed = {}
    for row in csv_rows(run_dir / "per_neuron_target_counts.csv"):
        key = (row["SampleID"], row["NeuronID"], *target_key(row))
        require(key not in observed, f"Duplicate neuron/target row: {key}")
        require(key in expected_counts, f"Unexpected neuron/target identity: {key}")
        identity_check(row, neurons[key[:2]]["source_metadata"], f"Neuron target {key}")
        target_check(row, metadata[key[2:]], f"Neuron target {key}")
        require(integer(row["neurons_with_target_evidence"], "target evidence") == 1, "Observed target lacks binary neuron evidence")
        observed[key] = integer(row["endpoint_count"], "endpoint count")
    require(observed == dict(expected_counts), "Per-neuron regional counts differ from independently sampled leaves")
    regional_counts, regional_presence = Counter(), Counter()
    denominators = defaultdict(lambda: [0, 0])
    for neuron in neurons.values():
        key = (neuron["AnimalID"], neuron["Subregion"])
        denominators[key][0] += 1
        denominators[key][1] += int(neuron["computable"])
    for key, count in expected_counts.items():
        neuron = neurons[key[:2]]
        group_key = (neuron["AnimalID"], neuron["Subregion"], *key[2:])
        regional_counts[group_key] += count
        regional_presence[group_key] += 1
    observed_groups, unavailable_groups = set(), set()
    for row in csv_rows(run_dir / "animal_region_summaries.csv"):
        animal_key = (row["AnimalID"], row["Subregion"])
        require(animal_key in denominators, f"Unexpected regional animal/source identity: {animal_key}")
        selected, computable = denominators[animal_key]
        require(integer(row["n_selected"], "regional selected") == selected and integer(row["n_computable"], "regional computable") == computable,
                f"Regional denominator differs: {animal_key}")
        require(row["biological_innervation_state"] == "unassessed", "Unsupported regional acceptance")
        if not computable:
            require(row["target_status"] == "unavailable_no_computable_neurons", "Unassessed region status differs")
            require(row["frequency_per_computable_neuron"] == row["mean_endpoints_per_computable_neuron"] == "",
                    "Unassessed regional frequency was filled with zero")
            require(row["map_available"] == "False" and integer(row["endpoint_count"], "unavailable count") == 0
                    and integer(row["neurons_with_target_evidence"], "unavailable presence") == 0,
                    "Unavailable regional count/availability differs")
            require(all(row[field] == "" for field in ("level", "target_index", "target_abbreviation", "target_full_name", "target_volume_mm3")),
                    "Unavailable regional target fields must be blank")
            require(animal_key not in unavailable_groups, "Duplicate unavailable region")
            unavailable_groups.add(animal_key)
            continue
        key = (*animal_key, *target_key(row))
        require(key not in observed_groups and key in regional_counts, f"Unexpected/duplicate regional row: {key}")
        observed_groups.add(key)
        require(row["map_available"] == "True", f"Regional availability differs: {key}")
        require(integer(row["endpoint_count"], "regional endpoints") == regional_counts[key]
                and integer(row["neurons_with_target_evidence"], "regional presence") == regional_presence[key], f"Regional count differs: {key}")
        np.testing.assert_allclose(float(row["frequency_per_computable_neuron"]), regional_presence[key] / computable)
        np.testing.assert_allclose(float(row["mean_endpoints_per_computable_neuron"]), regional_counts[key] / computable)
        target_check(row, metadata[key[2:]], f"Regional {key}")
    require(observed_groups == set(regional_counts), "Missing regional summary rows")
    require(unavailable_groups == {key for key, (_, count) in denominators.items() if not count},
            "Missing unavailable group summaries")
    return {"qc_neurons": len(qc), "per_neuron_target_rows": len(observed),
            "animal_region_rows": len(observed_groups), "unavailable_groups": len(unavailable_groups)}


def review(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError("Use a new independent readback directory")
    record_path = run_dir / "run_provenance.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(record["status"] == "software_verified_candidate_endpoints", "Endpoint run is not successfully terminal")
    manifest_path = run_dir / "input_manifest.csv"
    inputs = record["inputs"]
    manifest_hash = inputs["manifest"]["sha256"]
    require({"manifest", "reference", "atlas", "atlas_key", "hemisphere_mask"} <= set(inputs)
            and set(inputs) <= {"manifest", "reference", "atlas", "atlas_key", "hemisphere_mask", "brain_mask"},
            "Missing or unexpected input bindings")
    require(sha256(manifest_path) == manifest_hash, "Changed preserved source manifest")
    manifest = csv_rows(manifest_path)
    for key, source in inputs.items():
        require(sha256(source["path"]) == source["sha256"], f"Changed {key}")
    reference = nib.load(inputs["reference"]["path"])
    shape, affine, volume = coded_grid(reference, "Reference")
    require(tuple(record["shape"]) == shape and np.allclose(record["affine_mm"], affine, rtol=0, atol=1e-7),
            "Recorded reference geometry differs from actual reference")
    require(record["voxel_volume_mm3"] == volume, "Recorded reference voxel volume differs")
    require(Path(record["reference"]).resolve() == Path(inputs["reference"]["path"]).resolve()
            and record["reference_sha256"] == inputs["reference"]["sha256"], "Reference identity binding differs")
    require(record["map_value_units"] == METRICS, "Recorded map value units differ")
    for field, value in (("image_review_state", "unresolved"), ("biological_innervation_state", "unassessed")):
        require(record[field] == value, f"Unsupported run acceptance: {field}")
    require(record["scientific_status"] == "exploratory candidate proxy; compartment, tracing, anatomy and registration require review"
            and record["statistical_status"] == "descriptive; no smoothing, t/p values, or synapse claims",
            "Unsupported scientific/statistical run acceptance")
    required_code = {"group_analysis/scripts/build_endpoint_maps.py", "main_scripts/endpoint_atlas.py",
                     "main_scripts/terminal_sites.py", "main_scripts/projection_maps.py", "main_scripts/swc_validation.py",
                     "group_analysis/scripts/build_projection_maps.py", "main_scripts/region_analysis/laterality.py"}
    normalized_code = [name.replace("\\", "/") for name in record["source_code_sha256"]]
    require(len(normalized_code) == len(set(normalized_code)) and set(normalized_code) == required_code,
            "Production code binding scope differs")
    for name, expected in record["source_code_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Changed bound production code: {name}")
    require(set(record["artifacts"]) == {"input_manifest.csv", "leaf_records.jsonl", "per_neuron_qc.csv",
                                       "per_neuron_target_counts.csv", "animal_region_summaries.csv"},
            "Required saved artifact binding scope differs")
    for name, artifact in record["artifacts"].items():
        require(sha256(run_dir / name) == artifact["sha256"], f"Changed saved artifact: {name}")
    require(all(row["ReferenceSHA256"].lower() == record["reference_sha256"] for row in manifest),
            "Manifest/reference binding differs")
    leaves, neurons = source_expectations(manifest, shape, affine, record.get("input_root", ROOT))
    for neuron in neurons.values():
        neuron["brain_mask_provided"] = "brain_mask" in inputs
    require(record["n_selected"] == len(neurons)
            and record["n_computable"] == sum(neuron["computable"] for neuron in neurons.values()),
            "Run-level conditional denominator differs")
    target_expectations, hemisphere_contingency = atlas_expectations(inputs, record, leaves)
    saved_leaves = {}
    with (run_dir / "leaf_records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            leaf = json.loads(line)
            uid = (str(leaf["SampleID"]), str(leaf["NeuronID"]), leaf["node_id"])
            require(uid not in saved_leaves, f"Duplicate saved leaf: {uid}")
            saved_leaves[uid] = leaf
    require(set(saved_leaves) == set(leaves), "Saved leaf identity set differs from complete source graph")
    for uid, expected in leaves.items():
        saved = saved_leaves[uid]
        source_record = neurons[uid[:2]]
        require(saved["source_sha256"] == source_record["SWCSHA256"].lower(), f"Leaf source hash differs: {uid}")
        identity_check(saved, source_record["source_metadata"], f"Leaf {uid}", nested=True)
        for key, value in (("source_path", source_record["source_path"]),
                           ("coordinate_frame", source_record["CoordinateFrame"]),
                           ("index_scale_um", source_record["index_scale_um"])):
            require(saved[key] == value, f"Leaf source contract differs: {uid}/{key}")
        for key in ("node_id", "node_type", "parent_id", "candidate_axon_endpoint", "in_reference", "leaf_class",
                    "branch_start_node_id", "terminal_branch_compartment_transitions", "terminal_edge_type_transition",
                    "zero_length_terminal_edge", "coincident_leaf_count"):
            require(saved[key] == expected[key], f"Leaf field mismatch: {uid}/{key}")
        saved_voxel = tuple(saved["voxel"]) if saved["voxel"] is not None else None
        require(saved_voxel == expected["voxel"], f"Leaf voxel mismatch: {uid}")
        for key in ("source_xyz", "index_xyz", "world_mm", "soma_distance_mm", "root_distance_mm",
                    "terminal_edge_length_mm", "terminal_branch_length_mm", "terminal_branch_axon_length_mm"):
            if expected[key] is None:
                require(saved[key] is None, f"Leaf unavailable metric differs: {uid}/{key}")
            else:
                np.testing.assert_allclose(saved[key], expected[key], rtol=1e-10, atol=1e-9,
                                           err_msg=f"Leaf coordinate/length mismatch: {uid}/{key}")
        require(saved["image_review_state"] == "unresolved", f"Unsupported biological acceptance: {uid}")
        require(saved["hemisphere_side"] == target_expectations[uid]["hemisphere_side"], f"Leaf hemisphere differs: {uid}")
        require(saved["brain_mask_status"] == target_expectations[uid]["brain_mask_status"], f"Leaf mask status differs: {uid}")
        require(saved["atlas_levels"] == target_expectations[uid]["atlas_levels"], f"Leaf atlas-level lookup differs: {uid}")
    accounting = Counter(candidate_endpoints=sum(n["candidate_count"] for n in neurons.values()),
                         in_reference_candidates=sum(n["candidate_count"] - n["outside_count"] for n in neurons.values()),
                         outside_reference_candidates=sum(n["outside_count"] for n in neurons.values()), all_leaves=len(leaves))
    for key, leaf in leaves.items():
        if leaf["candidate_axon_endpoint"]:
            accounting["candidate_brain_mask_" + target_expectations[key]["brain_mask_status"]] += 1
    require(record["accounting"] == dict(accounting), "Run leaf/mask accounting differs")
    for name, expected_uids in (("unassessed_neurons", {uid for uid, n in neurons.items() if not n["computable"]}),
                               ("unresolved_compartment_neurons", {uid for uid, n in neurons.items()
                                  if any(kind not in (1, 2, 3, 4) for kind in n["node_type_counts"])})):
        rows = record[name]
        uids = [(row["SampleID"], row["NeuronID"]) for row in rows]
        require(len(uids) == len(set(uids)) and set(uids) == expected_uids, f"Run {name} identity scope differs")
        for row, uid in zip(rows, uids):
            neuron = neurons[uid]
            identity_check(row, neuron["source_metadata"], f"Run {name}/{uid}")
            if name == "unassessed_neurons":
                require(row["reason"] == neuron["candidate_status"], f"Unassessed reason differs: {uid}")
            else:
                require(row["node_type_counts"] == {str(k): v for k, v in neuron["node_type_counts"].items() if k not in (1, 2, 3, 4)},
                        f"Unresolved compartment counts differ: {uid}")
    table_checks = check_tables(run_dir, neurons, leaves, target_expectations)
    groups = defaultdict(list)
    for neuron in neurons.values():
        groups[(neuron["AnimalID"], neuron["Subregion"])].append(neuron)
    entries = record["animal_maps"]
    actual = [(item["AnimalID"], item["Subregion"]) for item in entries]
    require(len(actual) == len(set(actual)) and set(actual) == set(groups), "Animal/source-subregion scope differs")
    checked, expected_animals = [], {}
    expected_map_paths = set()
    def map_availability(item, computable, context, reason):
        require(item["map_available"] is bool(computable), f"Map availability differs: {context}")
        fields = {kind + suffix for kind in METRICS for suffix in ("_path", "_sha256")}
        if computable:
            require(fields <= set(item) and "unavailable_reason" not in item, f"Available map fields differ: {context}")
        else:
            require(not fields.intersection(item), f"Unavailable group has map fields: {context}")
            require(item.get("unavailable_reason") == reason, f"Unavailable map reason differs: {context}")
    def map_path(item, kind):
        relative = Path(item[kind + "_path"])
        path = (run_dir / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(run_dir.resolve()), "Saved map path escapes run")
        require(path not in expected_map_paths, "Reused saved map path")
        expected_map_paths.add(path)
        return path
    for item in entries:
        key = (item["AnimalID"], item["Subregion"])
        selected = groups[key]
        computable = [neuron for neuron in selected if neuron["computable"]]
        require(item["n_selected"] == len(selected) and item["n_computable"] == len(computable),
                f"Conditional denominator differs: {key}")
        map_availability(item, computable, key, "no_computable_candidate_endpoint_neurons")
        if not computable:
            continue
        counts, occupancy = Counter(), Counter()
        for neuron in computable:
            counts.update(neuron["counts"])
            occupancy.update(neuron["occupied"])
        mean_counts = {point: count / len(computable) for point, count in counts.items()}
        mean_occupancy = {point: count / len(computable) for point, count in occupancy.items()}
        expected_animals[key] = (mean_counts, mean_occupancy)
        for kind, values, scale in (("count", mean_counts, 1), ("density", mean_counts, 1 / volume),
                                    ("occupancy", mean_occupancy, 1)):
            path = map_path(item, kind)
            require(sha256(path) == item[kind + "_sha256"], f"Changed saved {kind} map: {key}")
            checked.append(check_sparse_map(path, values, shape, affine, scale, fraction=kind == "occupancy"))
    region_entries = record["group_maps"]
    regions = {key[1] for key in groups}
    require(len(region_entries) == len(regions) and {item["Subregion"] for item in region_entries} == regions,
            "Group-map source-region coverage differs")
    for item in region_entries:
        region = item["Subregion"]
        contributors = {key[0]: value for key, value in expected_animals.items() if key[1] == region}
        selected = [neuron for key, group in groups.items() if key[1] == region for neuron in group]
        unavailable = {key[0] for key in groups if key[1] == region and key not in expected_animals}
        require(item["n_selected"] == len(selected)
                and item["n_computable"] == sum(neuron["computable"] for neuron in selected)
                and item["n_animals"] == len(contributors), f"Group conditional denominator differs: {region}")
        for field, expected in (("contributing_animals", set(contributors)), ("unavailable_animals", unavailable)):
            require(len(item[field]) == len(set(item[field])) and set(item[field]) == expected,
                    f"Group animal scope differs: {region}/{field}")
        map_availability(item, contributors, region, "no_contributing_animal_with_computable_neurons")
        if not contributors:
            continue
        means = [defaultdict(float), defaultdict(float)]
        for pair in contributors.values():
            for metric in (0, 1):
                for point, value in pair[metric].items():
                    means[metric][point] += value / len(contributors)
        for kind, values, scale in (("count", means[0], 1), ("density", means[0], 1 / volume),
                                    ("occupancy", means[1], 1)):
            path = map_path(item, kind)
            require(sha256(path) == item[kind + "_sha256"], f"Changed group {kind}: {region}")
            checked.append(check_sparse_map(path, values, shape, affine, scale, fraction=kind == "occupancy"))
    actual_map_paths = {path.resolve() for folder in ("animal_maps", "group_maps")
                        for path in (run_dir / folder).rglob("*.nii*")}
    require(actual_map_paths == expected_map_paths, "Unexpected or missing saved map files")
    # A passing report also requires all bound bytes to remain stable throughout
    # the review, after the independent source/table/pixel calculations.
    for key, source in inputs.items():
        require(sha256(source["path"]) == source["sha256"], f"Input changed during review: {key}")
    for name, expected in record["source_code_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Production code changed during review: {name}")
    for name, artifact in record["artifacts"].items():
        require(sha256(run_dir / name) == artifact["sha256"], f"Artifact changed during review: {name}")
    for neuron in neurons.values():
        require(sha256(neuron["source_path"]) == neuron["SWCSHA256"].lower(), "SWC changed during review")
    for item in checked:
        require(sha256(item["file"]) == item["sha256"], "Map changed during review")
    report = {
        "status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_version": REVIEWER_VERSION, "reviewer_sha256": sha256(__file__), "python": sys.version,
        "versions": {"numpy": np.__version__, "nibabel": nib.__version__},
        "run_provenance_sha256": sha256(record_path), "manifest_sha256": manifest_hash,
        "selected_neurons": len(neurons), "computable_neurons": sum(neuron["computable"] for neuron in neurons.values()),
        "full_graph_leaves": len(leaves),
        "candidate_axonal_leaves": sum(neuron["candidate_count"] for neuron in neurons.values()),
        "outside_reference_candidates": sum(neuron["outside_count"] for neuron in neurons.values()),
        "map_files": checked, "full_saved_voxels_read": sum(item["voxels"] for item in checked),
        "independence": "Raw parent-ID leaf sets and voxel-face search; detector and atlas sampler not imported",
        "scope": "Full source leaf identity/coordinates, all atlas-level targets, hemisphere/mask, QC/regional CSVs, map pixels/units, conditional denominators, equal-animal aggregation",
        "coordinate_contract": "Declared source frame/scale only; physical export origin and anatomical registration remain unverified",
        "not_independently_reconstructed": ["runtime/version history and timestamps", "free-text policy descriptions and CSV column declarations"],
        "atlas_target_readback": "all leaves at all six actual atlas levels",
        "hemisphere_contingency": hemisphere_contingency, "table_checks": table_checks,
        "anatomical_acceptance": "not established",
        "biological_terminal_acceptance": "not established", "statistical_inference": "none",
    }
    output.mkdir(parents=True)
    with (output / "endpoint_run_readback.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.run, args.output)
    print(f"{result['status']}: {result['selected_neurons']} sources, {result['candidate_axonal_leaves']} candidates")
