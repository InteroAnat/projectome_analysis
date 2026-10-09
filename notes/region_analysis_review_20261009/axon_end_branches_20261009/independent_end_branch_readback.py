"""Independent SWC/CSV and voxel-interval oracle; no producer/kernel imports.

Saved-run schema integration is added after the producer contract is final.
The graph and geometric oracles below have no dependency on project modules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_original_swc(text):
    """Parse exact identities and validate a finite connected single-root tree."""
    rows = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid finite seven-column SWC at line {line_number}")
        if any(values[column] != int(values[column]) for column in (0, 1, 6)):
            raise ValueError("Node/type/parent identities must be integers")
        node = int(values[0])
        if node in rows:
            raise ValueError("Duplicate node identity")
        rows[node] = values
    roots = [node for node, row in rows.items() if row[6] == -1]
    if len(roots) != 1:
        raise ValueError("Exactly one original root required")
    children = defaultdict(list)
    for node, row in rows.items():
        parent = int(row[6])
        if parent != -1:
            if parent not in rows or parent == node:
                raise ValueError("Missing or self parent")
            children[parent].append(node)
    visited = set()
    pending = roots.copy()
    while pending:
        node = pending.pop()
        if node in visited:
            raise ValueError("Cycle or duplicate graph traversal")
        visited.add(node)
        pending.extend(children[node])
    if len(visited) != len(rows):
        raise ValueError("Disconnected or cyclic graph")
    return rows, children, roots[0]


def independent_end_branches(text):
    """Select by FORWARD downstream eligibility, not producer upstream walk.

    A child-axon edge qualifies exactly when its child has an uninterrupted
    original unique-full-child axon chain ending in a nonroot axon leaf.
    All original child types contribute to the degree test. A nonaxon parent
    may supply the final boundary edge; no nonaxon downstream node is bridged.
    """
    rows, children, root = parse_original_swc(text)
    order = []
    pending = [root]
    while pending:
        node = pending.pop()
        order.append(node)
        pending.extend(children[node])
    qualifies = {}
    for node in reversed(order):
        descendants = children[node]
        qualifies[node] = bool(rows[node][1] == 2 and (
            not descendants or (len(descendants) == 1 and qualifies[descendants[0]])))
    selected = {node for node in rows if rows[node][6] != -1 and qualifies[node]}
    leaves = sorted(node for node in selected if not children[node])
    branches = []
    starts = sorted(node for node in selected if int(rows[node][6]) not in selected)
    covered = set()
    for first in starts:
        chain = [first]
        while children[chain[-1]]:
            if len(children[chain[-1]]) != 1:
                raise ValueError("Forward selected chain reached a bifurcation")
            child = children[chain[-1]][0]
            if child not in selected:
                raise ValueError("Forward selected chain interrupted before leaf")
            chain.append(child)
        if covered.intersection(chain):
            raise ValueError("Forward chain overlaps")
        covered.update(chain)
        parent = int(rows[first][6])
        branches.append({
            "leaf_node_id": chain[-1],
            "start_node_id": parent,
            "selected_child_node_ids": list(reversed(chain)),
            "stop_at_original_root": parent == root,
            "stop_at_full_graph_bifurcation": len(children[parent]) > 1,
            "stop_at_compartment_boundary": rows[parent][1] != 2,
            "final_transition_edge_included": rows[parent][1] != 2,
        })
    if covered != selected:
        raise ValueError("Forward chain membership does not cover selected edges")
    edges = [(int(rows[child][6]), child) for child in sorted(selected)]
    return {"rows": rows, "children": children, "root": root,
            "leaves": leaves, "branches": branches, "edges": edges,
            "computable": bool(leaves)}


def centre_cell(point):
    """Half-open centre-origin cell without adding .5 before floor."""
    base = np.floor(point)
    return (base + (np.asarray(point) - base >= 0.5)).astype(int)


def independent_interval_voxels(start, end, shape):
    """Slab-clip then cut a parametric segment at every internal voxel plane.

    Returns absolute original-edge fractions, rather than renormalizing after
    FOV clipping. Constant coordinates on the excluded upper face stay outside.
    This is not the producer's incremental voxel traversal.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    shape = np.asarray(shape, dtype=int)
    delta = end - start
    lower = np.full(3, -0.5)
    upper = shape - 0.5
    enter, leave = 0.0, 1.0
    for axis in range(3):
        if delta[axis] == 0:
            if start[axis] < lower[axis] or start[axis] >= upper[axis]:
                return {}
        else:
            times = sorted([(lower[axis] - start[axis]) / delta[axis],
                            (upper[axis] - start[axis]) / delta[axis]])
            enter = max(enter, times[0])
            leave = min(leave, times[1])
    if leave <= enter:
        return {}
    cuts = [enter, leave]
    for axis in range(3):
        if delta[axis] == 0:
            continue
        at_enter = start[axis] + enter * delta[axis]
        at_leave = start[axis] + leave * delta[axis]
        low, high = sorted([at_enter, at_leave])
        first = max(0, math.ceil(low - 0.5))
        last = min(int(shape[axis]) - 2, math.floor(high - 0.5))
        for cell in range(first, last + 1):
            time = (cell + 0.5 - start[axis]) / delta[axis]
            if enter < time < leave:
                cuts.append(float(time))
    cuts = sorted(set(cuts))
    result = defaultdict(float)
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right <= left:
            continue
        midpoint = start + ((left + right) / 2) * delta
        voxel = centre_cell(midpoint)
        if not ((voxel >= 0) & (voxel < shape)).all():
            raise ArithmeticError("Independent clipped interval midpoint outside FOV")
        result[tuple(int(x) for x in voxel)] += right - left
    return dict(result)


def independent_length_summary(text, affine_mm, shape, scale_um=(250, 250, 250), *, rasterize=True):
    graph = independent_end_branches(text)
    linear = np.asarray(affine_mm, dtype=float)[:3, :3]
    scale = np.asarray(scale_um, dtype=float)
    total_pieces, inside_pieces = [], []
    voxels = defaultdict(float)
    edge_records = []
    for parent, child in graph["edges"]:
        start = np.asarray(graph["rows"][parent][2:5]) / scale
        end = np.asarray(graph["rows"][child][2:5]) / scale
        length = float(np.linalg.norm(linear @ (end-start)))
        fractions = independent_interval_voxels(start, end, shape)
        inside = length * math.fsum(fractions.values())
        total_pieces.append(length)
        inside_pieces.append(inside)
        if rasterize:
            for voxel, fraction in fractions.items():
                voxels[voxel] += length * fraction
        edge_records.append({"parent_node_id": parent, "child_node_id": child,
                             "length_mm": length, "in_reference_length_mm": inside})
    return {"graph": graph, "edge_records": edge_records, "voxel_lengths_mm": dict(voxels),
            "total_length_mm": math.fsum(total_pieces) if graph["computable"] else None,
            "in_reference_length_mm": math.fsum(inside_pieces) if graph["computable"] else None}


def synthetic_oracle_checks():
    """Direct arithmetic fixtures: original false ends, transitions and cells."""
    cases = []
    graph = independent_end_branches("1 1 0 0 0 1 -1\n2 2 250 0 0 1 1\n3 3 500 0 0 1 2\n")
    if graph["leaves"] or graph["computable"]:
        raise AssertionError("An axon node with a nonaxon child is not a leaf")
    cases.append("nonaxon_child_prevents_false_end_and_preserves_missing")
    graph = independent_end_branches("1 1 0 0 0 1 -1\n2 2 250 0 0 1 1\n3 2 500 0 0 1 2\n4 3 250 250 0 1 2\n")
    if graph["edges"] != [(2,3)] or not graph["branches"][0]["stop_at_full_graph_bifurcation"]:
        raise AssertionError("Full-child bifurcation including nonaxon child must stop chain")
    cases.append("mixed_type_original_bifurcation_stops_upstream_walk")
    graph = independent_end_branches("1 1 0 0 0 1 -1\n2 2 250 0 0 1 1\n3 3 500 0 0 1 2\n4 2 750 0 0 1 3\n5 2 1000 0 0 1 4\n")
    if graph["edges"] != [(3,4),(4,5)] or not graph["branches"][0]["final_transition_edge_included"]:
        raise AssertionError("Final child-axon transition retained; no nonaxon bridge")
    cases.append("nonaxon_transition_included_then_stop_no_bridge")
    graph = independent_end_branches("9 2 500 0 0 1 7\n2 1 0 0 0 1 -1\n7 2 250 0 0 1 2\n10 2 250 250 0 1 7\n")
    if graph["edges"] != [(7,9),(7,10)]:
        raise AssertionError("Bifurcation incoming trunk must not be duplicated or selected")
    cases.append("unsorted_identity_disjoint_end_chains")
    fractions = independent_interval_voxels([-1,0,0],[3,0,0],[3,2,2])
    if fractions != {(0,0,0):.25,(1,0,0):.25,(2,0,0):.25}:
        raise AssertionError("Absolute fractions and outside FOV conservation")
    cases.append("half_open_fov_absolute_interval_fractions")
    if independent_interval_voxels([2.5,0,0],[2.5,1,0],[3,2,2]):
        raise AssertionError("Upper-FOV-face parallel edge must remain outside")
    if centre_cell([np.nextafter(.5,-np.inf),.5,1.5]).tolist()!=[0,1,2]:
        raise AssertionError("Exact versus adjacent half boundary")
    cases.append("upper_face_and_half_tie_sensitivity")
    length = independent_length_summary("1 1 0 0 0 1 -1\n2 2 250 0 0 1 1\n",np.diag([.25,.5,1,1]),[3,3,3])
    if not math.isclose(length["total_length_mm"],.25) or not math.isclose(sum(length["voxel_lengths_mm"].values()),.25):
        raise AssertionError("Affine physical millimetres and raster conservation")
    cases.append("affine_template_length_and_raster_sum")
    zero = independent_length_summary("1 1 0 0 0 1 -1\n2 2 0 0 0 1 1\n",np.eye(4),[2,2,2])
    if zero["total_length_mm"]!=0 or not zero["graph"]["computable"]:
        raise AssertionError("Eligible zero length differs from missing")
    cases.append("eligible_zero_length_not_missing")
    return cases


def raster_forward_graph(graph, affine, shape, scale):
    """Vectorize convex same-cell edges; interval-split every crossing edge."""
    edges = graph["edges"]
    if not edges:
        return {}, 0.0, 0.0, 0
    starts = np.asarray([graph["rows"][p][2:5] for p,c in edges]) / scale
    ends = np.asarray([graph["rows"][c][2:5] for p,c in edges]) / scale
    lengths = np.linalg.norm((ends-starts) @ affine[:3,:3].T,axis=1)
    start_cells = centre_cell(starts)
    end_cells = centre_cell(ends)
    same = ((start_cells==end_cells).all(axis=1)
            & (start_cells>=0).all(axis=1) & (start_cells<shape).all(axis=1))
    voxels = defaultdict(float)
    if same.any():
        flat = np.ravel_multi_index(start_cells[same].T,shape)
        unique,inverse = np.unique(flat,return_inverse=True)
        weights = np.bincount(inverse,weights=lengths[same])
        for index,weight in zip(unique,weights):
            if weight>0:
                voxels[tuple(int(x) for x in np.unravel_index(index,shape))] += float(weight)
    for index in np.flatnonzero(~same & (lengths>0)):
        for voxel,fraction in independent_interval_voxels(starts[index],ends[index],shape).items():
            voxels[voxel] += float(lengths[index]*fraction)
    return dict(voxels),math.fsum(lengths.tolist()),math.fsum(voxels.values()),int(np.count_nonzero(lengths==0))


def exact_integer(value):
    number=float(value)
    if not math.isfinite(number) or number!=int(number):
        raise ValueError("Expected finite integer-valued CSV identity")
    return int(number)


def read_csv_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def validate_saved_run(run_path, receipt_path):
    """Reconstruct ALL original edge sets, regional cells and animal maps."""
    import nibabel as nib
    from datetime import datetime
    run_path=Path(run_path).resolve()
    receipt_path=Path(receipt_path).resolve()
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    root=Path(__file__).resolve().parents[3]
    provenance_path=run_path/"run_provenance.json"
    run=json.loads(provenance_path.read_text(encoding="utf-8"))
    if run["status"]!="software_verified_descriptive_end_branches":
        raise ValueError("Require completed software-verified saved run")
    for binding in run["inputs"].values():
        if sha256(binding["path"])!=binding["sha256"]:
            raise ValueError("Input hash mismatch")
    for relative,expected in run["source_code_sha256"].items():
        if sha256(root/relative)!=expected:
            raise ValueError("Producer source hash differs")
    for relative,expected in run["artifacts"].items():
        if sha256(run_path/relative)!=expected:
            raise ValueError("Saved artifact hash mismatch: "+relative)
    manifest=read_csv_rows(run["inputs"]["manifest"]["path"])
    ledger=read_csv_rows(run_path/"input_ledger.csv")
    if ledger!=manifest or len(ledger)!=462:
        raise ValueError("Saved exact462 ledger differs from source")
    uid=lambda row:row["SampleID"]+"::"+row["NeuronID"]
    if len({uid(row) for row in ledger})!=462:
        raise ValueError("Duplicate exact identities")
    qc_rows=read_csv_rows(run_path/"per_neuron_measurements.csv")
    qc={uid(row):row for row in qc_rows}
    if set(qc)!={uid(row) for row in ledger} or len(qc_rows)!=462:
        raise ValueError("Measurement UID set differs")
    targets=read_csv_rows(run_path/"targets.csv")
    target_lookup={}
    for t in targets:
        key=(int(t["Level"]),-1 if t["TargetStatus"]=="out_of_FOV" else exact_integer(t["ARMIndex"]))
        if key in target_lookup:
            raise ValueError("Nonunique target identity")
        target_lookup[key]=t["TargetID"]
    matrices={}
    matrix_values={}
    identity_fields=["SampleID","NeuronID","AnimalID","Subregion","ARMIndex","ARMFullName","Hemisphere"]
    for level in range(1,7):
        matrix=read_csv_rows(run_path/f"ARM_L{level}_axon_end_branch_length_mm.csv")
        if len(matrix)!=462 or [uid(row) for row in matrix]!=[uid(row) for row in ledger]:
            raise ValueError("Hierarchy row order differs")
        for row,source in zip(matrix,ledger):
            if any(row[f]!=source[f] for f in identity_fields):
                raise ValueError("Matrix identity/official source label changed")
        matrices[level]={uid(row):row for row in matrix}
        matrix_values[level]=[t["TargetID"] for t in targets if int(t["Level"])==level]
    reference=nib.load(run["inputs"]["reference"]["path"])
    shape=tuple(reference.shape)
    affine=np.asarray(reference.affine)
    if shape!=tuple(run["shape"]) or not np.array_equal(affine,np.asarray(run["affine_mm"])):
        raise ValueError("Bound reference geometry disagrees")
    atlas=nib.load(run["inputs"]["atlas"]["path"])
    if atlas.shape[:3]!=shape or not np.array_equal(atlas.affine,affine):
        raise ValueError("Atlas/reference geometry differs")
    atlas_values=np.asanyarray(atlas.dataobj)
    if atlas_values.shape!=shape+(1,6):
        raise ValueError("Expected actual six-level ARM volume")
    # Key is tab-delimited, so read directly instead of relying on producer catalog.
    with Path(run["inputs"]["atlas_key"]["path"]).open(encoding="utf-8-sig",newline="") as stream:
        keyrows=list(csv.DictReader(stream,delimiter="\t"))
    official={int(row["Index"]):row for row in keyrows}
    for t in targets:
        if t["TargetStatus"] in ("mapped","key_level_conflict"):
            original=official[exact_integer(t["ARMIndex"])]
            if t["OfficialFullName"]!=original["Full_Name"] or t["Abbreviation"]!=original["Abbreviation"]:
                raise ValueError("Target official key name disagrees")
    groups=defaultdict(list)
    for row in ledger:
        groups[row["AnimalID"],row["Subregion"]].append(row)
    animal_entries={(e["AnimalID"],e["Subregion"]):e for e in run["animal_maps"]}
    if set(groups)!=set(animal_entries):
        raise ValueError("Animal/source grouping differs")
    checks=[]
    numeric_cells=missing_cells=0
    max_matrix_error=max_animal_voxel_error=0.0
    total_edges=total_leaves=eligible=0
    total_lengths=[]
    for group,rows in sorted(groups.items()):
        accumulated=np.zeros(shape,dtype=float)
        n_eligible=0
        for source in rows:
            identity=uid(source)
            path=Path(source["SWCPath"])
            if not path.is_absolute():
                path=root/path
            expected_hash=source["SWCSHA256"]
            if sha256(path)!=expected_hash:
                raise ValueError("Actual SWC hash differs: "+identity)
            if source["CoordinateFrame"]!="atlas_index_um":
                raise ValueError("Actual source index-coordinate frame required")
            scale=np.asarray([float(v) for v in source["IndexScaleUm"].split(";")])
            graph=independent_end_branches(path.read_text(encoding="utf-8"))
            sparse,total,inside,zero=raster_forward_graph(graph,affine,np.asarray(shape),scale)
            actual=qc[identity]
            children=[c for p,c in graph["edges"]]
            selection_hash=hashlib.sha256(json.dumps(children,separators=(",",":")).encode()).hexdigest()
            if actual["selected_child_ids_sha256"]!=selection_hash:
                raise ValueError("FORWARD selected edge identity differs: "+identity)
            counts={"candidate_axon_endpoint_count":len(graph["leaves"]),"branch_count":len(graph["branches"]),
                    "selected_axon_edge_count":len(children),"node_count":len(graph["rows"]),
                    "zero_length_selected_edges":zero,
                    "final_transition_edge_count":sum(graph["rows"][p][1]!=2 for p,c in graph["edges"])}
            if any(int(actual[k])!=v for k,v in counts.items()):
                raise ValueError("Independent graph counts differ: "+identity)
            if actual["computable"]!=str(graph["computable"]):
                raise ValueError("Computability differs")
            if not graph["computable"]:
                if any(actual[k]!="" for k in ["total_length_mm","in_reference_length_mm","outside_reference_length_mm"]):
                    raise ValueError("Missing graph aggregates converted to zero")
                for level in range(1,7):
                    if any(matrices[level][identity][target]!="" for target in matrix_values[level]):
                        raise ValueError("Missing hierarchy row zero-filled")
                    missing_cells+=len(matrix_values[level])
            else:
                n_eligible+=1;eligible+=1
                outside=max(0.0,total-inside)
                for field,expected in [("total_length_mm",total),("in_reference_length_mm",inside),("outside_reference_length_mm",outside)]:
                    if not math.isclose(float(actual[field]),expected,rel_tol=1e-9,abs_tol=1e-9):
                        raise ValueError("Independent physical length differs "+identity+" "+field)
                expected_targets=defaultdict(float)
                if sparse:
                    xyz=np.asarray(list(sparse),dtype=int)
                    weights=np.asarray(list(sparse.values()),dtype=float)
                    accumulated[tuple(xyz.T)]+=weights
                    for level in range(1,7):
                        labels=atlas_values[tuple(xyz.T)+(np.zeros(len(xyz),dtype=int),np.full(len(xyz),level-1,dtype=int))].astype(int)
                        bins=np.bincount(labels,weights=weights)
                        for label in np.flatnonzero(bins):
                            expected_targets[target_lookup[level,int(label)]]=float(bins[label])
                for level in range(1,7):
                    expected_targets[target_lookup[level,-1]]=outside
                    layer_sum=0.0
                    for target in matrix_values[level]:
                        expected=expected_targets[target]
                        measured=float(matrices[level][identity][target])
                        error=abs(measured-expected)
                        max_matrix_error=max(max_matrix_error,error)
                        if not math.isclose(measured,expected,rel_tol=1e-9,abs_tol=1e-9):
                            raise ValueError("Independent regional cell differs "+identity+" "+target)
                        layer_sum+=measured;numeric_cells+=1
                    if not math.isclose(layer_sum,total,rel_tol=1e-9,abs_tol=1e-9):
                        raise ValueError("Hierarchy does not conserve full selected length")
                total_lengths.append(total)
            if sha256(path)!=expected_hash:
                raise ValueError("Actual source changed during read")
            total_edges+=len(children);total_leaves+=len(graph["leaves"])
            checks.append({"NeuronUID":identity,"computable":graph["computable"],"selected_edges":len(children),
                           "original_axon_leaves":len(graph["leaves"]),"selected_child_ids_sha256":selection_hash})
        entry=animal_entries[group]
        if entry["selected_neurons"]!=len(rows) or entry["eligible_neurons"]!=n_eligible:
            raise ValueError("Conditional animal denominator differs")
        if n_eligible:
            expected=accumulated/n_eligible
            image=nib.load(run_path/entry["length_path"])
            observed=np.asarray(image.dataobj)
            if image.shape!=shape or not np.array_equal(image.affine,affine) or image.header.get_xyzt_units()[0]!="mm":
                raise ValueError("Animal map reference geometry/units differ")
            if not np.isfinite(observed).all() or (observed<0).any():
                raise ValueError("Invalid animal length map values")
            error=float(np.max(np.abs(observed.astype(float)-expected)))
            max_animal_voxel_error=max(max_animal_voxel_error,error)
            if not np.allclose(observed,expected,rtol=2e-6,atol=1e-9):
                raise ValueError("Independent complete per-animal voxel reconstruction differs: "+str(group))
            if not math.isclose(float(expected.sum()),entry["mean_in_reference_length_mm"],rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError("Animal mean length differs")
        elif "length_path" in entry:
            raise ValueError("Unassessed animal map must not be zero-filled")
        print(f"independent animal={group[0]} source={group[1]} eligible={n_eligible}/{len(rows)}",flush=True)
    max_group_voxel_error=0.0
    group_voxels=0
    for entry in run["group_maps"]:
        relevant=[e for e in run["animal_maps"] if e["Subregion"]==entry["Subregion"]]
        contributing=[e for e in relevant if e["eligible_neurons"]>0]
        if entry["contributing_animals"]!=[e["AnimalID"] for e in contributing] or entry["n_animals"]!=len(contributing):
            raise ValueError("Group equal-animal denominator differs")
        if entry["eligible_neurons"]!=sum(e["eligible_neurons"] for e in relevant) or entry["selected_neurons"]!=sum(e["selected_neurons"] for e in relevant):
            raise ValueError("Group neuron totals differ")
        if contributing:
            expected=np.zeros(shape,dtype=float)
            for animal in contributing:
                expected+=nib.load(run_path/animal["length_path"]).get_fdata()/len(contributing)
            image=nib.load(run_path/entry["length_path"])
            observed=np.asarray(image.dataobj)
            if image.shape!=shape or not np.array_equal(image.affine,affine) or image.header.get_xyzt_units()[0]!="mm":
                raise ValueError("Group map reference geometry/units differ")
            error=float(np.max(np.abs(observed.astype(float)-expected)))
            max_group_voxel_error=max(max_group_voxel_error,error)
            if not np.array_equal(observed,expected.astype(np.float32)):
                raise ValueError("Every-voxel equal-animal mean differs")
            group_voxels+=int(np.prod(shape))
        elif "length_path" in entry:
            raise ValueError("Missing group must not have a zero-filled map")
    if eligible!=run["eligible_neurons"] or total_edges!=run["selected_original_axon_edges"] or total_leaves!=run["original_axon_ends"]:
        raise ValueError("Global graph totals disagree")
    if not math.isclose(math.fsum(total_lengths),run["total_end_branch_length_mm"],rel_tol=1e-9,abs_tol=1e-9):
        raise ValueError("Global physical length differs")
    result={"status":"independent_all462_forward_edge_and_full_voxel_regional_readback_passed",
            "recorded_at":datetime.now().astimezone().isoformat(),"producer_or_kernel_imported":False,
            "scope":"All462 direct SWC parses with FORWARD unique-full-child axon-to-leaf oracle; all selected edges; full independent interval rasterization; every ARM L1-L6 CSV cell; every per-animal/group map voxel. Software/spatial descriptive validation, not biological/anatomical acceptance.",
            "synthetic_oracle_checks":synthetic_oracle_checks(),"selected_neurons":462,"eligible_neurons":eligible,
            "original_axon_leaves":total_leaves,"selected_original_edges":total_edges,
            "total_length_mm":math.fsum(total_lengths),"all_six_hierarchy_numeric_cells_checked":numeric_cells,
            "all_six_hierarchy_missing_NA_cells_checked":missing_cells,
            "max_regional_cell_abs_error_mm":max_matrix_error,"max_animal_voxel_abs_error_mm":max_animal_voxel_error,
            "max_group_voxel_abs_error_mm":max_group_voxel_error,"group_voxels_checked":group_voxels,
            "source_hashes_unchanged":True,"input_ledger_exactly_preserved":True,
            "run_provenance":{"path":str(provenance_path),"sha256":sha256(provenance_path)},
            "independent_code_sha256":sha256(Path(__file__)),"source_bindings":run["inputs"],
            "verified_artifacts":run["artifacts"],"per_neuron_edge_identity_checks":checks,
            "anatomical_or_biological_acceptance":False}
    for name,binding in run["inputs"].items():
        if sha256(binding["path"])!=binding["sha256"]:
            raise ValueError("Run input changed during independent verification")
    for relative,expected in run["artifacts"].items():
        if sha256(run_path/relative)!=expected:
            raise ValueError("Saved output changed during independent verification")
    with receipt_path.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ["status","selected_neurons","eligible_neurons","selected_original_edges","max_regional_cell_abs_error_mm","max_animal_voxel_abs_error_mm"]},indent=2))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check",action="store_true")
    parser.add_argument("--run",type=Path)
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps({"independent_synthetic_checks_passed":synthetic_oracle_checks()},indent=2))
    elif args.run and args.receipt:
        validate_saved_run(args.run,args.receipt)
    else:
        parser.error("Use --self-check or --run and --receipt")
