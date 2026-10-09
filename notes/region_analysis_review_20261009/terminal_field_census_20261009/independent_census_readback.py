"""Read saved census and independently check representative original graphs.

This validator does not import the census producer, its graph traversal, or
the production SWC parser. It uses an undirected sparse graph oracle.
"""
from pathlib import Path
import hashlib
import json
import math

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE / "selected462_ARM6_sections"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt = json.loads((RUN / "census_provenance.json").read_text())
    for binding in receipt["inputs"] + receipt["outputs"]:
        assert sha(ROOT / binding["path"]) == binding["sha256"], binding["path"]
    neurons = pd.read_csv(RUN / "neuron_morphology_and_coverage.csv", keep_default_na=False)
    sections = pd.read_csv(RUN / "connected_axon_sections.csv", keep_default_na=False)
    covered = pd.read_csv(RUN / "native_original_end_cube_coverage.csv", keep_default_na=False)
    links = pd.read_csv(RUN / "original_target_and_type_crossing_edges.csv", keep_default_na=False)
    assert len(neurons) == 462 and neurons.NeuronUID.is_unique
    assert sections.SectionUID.is_unique
    assert set(sections.NeuronUID).issubset(set(neurons.NeuronUID))
    native = neurons.Census_native_pair_available.eq(True)
    assert native.sum() == 201
    assert (neurons.loc[~native, "Census_original_axon_ends_with_cached_cube"] == "").all()
    assert (sections.loc[~sections.native_pair_available.eq(True), "original_axon_ends_with_cached_cube"] == "").all()
    assert len(covered) == 48574 and covered.cached_cube_available.eq(True).sum() == 9209
    assert covered.loc[covered.cached_cube_available.eq(True)].NeuronUID.nunique() == 184
    assert not covered.duplicated(["NeuronUID", "node_id"]).any()
    summed = sections.groupby("NeuronUID").agg(
        ends=("original_axon_end_count", "sum"),
        nodes=("node_count", "sum"),
        internal=("internal_edge_length_NMT_mm", "sum"),
        boundary=("incoming_target_crossing_length_NMT_mm", "sum"),
        compartment=("incoming_type_boundary_length_NMT_mm", "sum"))
    for row in neurons.to_dict("records"):
        uid = row["NeuronUID"]
        actual = summed.loc[uid] if uid in summed.index else pd.Series(dict.fromkeys(summed.columns, 0))
        assert actual.ends == row["Endpoint_candidate_axon_endpoint_count"]
        assert actual.nodes == row["Census_axon_node_count"]
        assert math.isclose(actual.internal + actual.boundary + actual.compartment,
                            row["Axon_selected_axon_length_mm"], rel_tol=1e-10, abs_tol=1e-7)
    atlas = np.asarray(nib.load(ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz").dataobj)[..., 0, 5]
    affine = nib.load(ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz").affine
    samples = ["252790::014.swc", "252790::032.swc", "252383::018.swc", "252383::121.swc",
               "252384::047.swc", "252718::162.swc", "252385::056.swc", "252714::099.swc",
               "251637::112.swc", "251637::458.swc"]
    samples = [uid for uid in samples if uid in set(neurons.NeuronUID)]
    members = {}
    with (RUN / "section_original_node_membership.jsonl").open() as stream:
        for line in stream:
            value = json.loads(line)
            if value["NeuronUID"] in samples:
                members.setdefault(value["NeuronUID"], []).append(value)
    checked = []
    for uid in samples:
        meta = neurons.set_index("NeuronUID").loc[uid]
        rows = np.loadtxt(ROOT / meta.SWCPath, comments="#", ndmin=2)
        ids, types, parents = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 6].astype(int)
        positions = rows[:, 2:5]
        lookup = dict(zip(ids.tolist(), range(len(ids))))
        endpoints = set(ids[(types == 2) & (parents != -1)].tolist()) - set(parents.tolist())
        axon = np.flatnonzero(types == 2)
        reduced = dict(zip(axon.tolist(), range(len(axon))))
        scale = np.array([float(x) for x in meta.IndexScaleUm.split(";")])
        voxels = np.rint(positions / scale).astype(int)
        valid = ((voxels >= 0) & (voxels < atlas.shape)).all(axis=1)
        labels = np.full(len(rows), -1, dtype=int)
        labels[valid] = atlas[tuple(voxels[valid].T)]
        pairs, total = [], []
        for child in axon:
            parent_id = parents[child]
            if parent_id == -1:
                continue
            parent = lookup[parent_id]
            delta = affine[:3, :3] @ ((positions[child] - positions[parent]) / scale)
            total.append(math.sqrt(sum(float(x) ** 2 for x in delta)))
            if types[parent] == 2 and labels[parent] == labels[child]:
                pairs.append((reduced[parent], reduced[child]))
        edge_rows = [a for a, b in pairs] + [b for a, b in pairs]
        edge_cols = [b for a, b in pairs] + [a for a, b in pairs]
        graph = coo_matrix((np.ones(len(edge_rows)), (edge_rows, edge_cols)), shape=(len(axon), len(axon)))
        _, partition = connected_components(graph, directed=False)
        expected = {frozenset(ids[axon[partition == component]].tolist()) for component in set(partition)}
        saved = members.get(uid, [])
        assert expected == {frozenset(x["original_node_ids"]) for x in saved}, uid
        assert endpoints == {x for component in saved for x in component["original_axon_end_ids"]}, uid
        assert math.isclose(math.fsum(total), meta.Axon_selected_axon_length_mm, rel_tol=1e-10, abs_tol=1e-7)
        checked.append({"NeuronUID": uid, "source_sha256": sha(ROOT / meta.SWCPath),
                        "node_count": len(rows), "sections": len(saved), "original_axon_ends": len(endpoints),
                        "independent_sparse_component_and_parent_set_match": True})
    summary = {"status": "pass", "selected_neurons": len(neurons), "sections": len(sections),
               "native_pairs": int(native.sum()), "native_original_axon_ends": len(covered),
               "covered_original_axon_ends": int(covered.cached_cube_available.eq(True).sum()),
               "neurons_with_covered_original_ends": 184,
               "all_neuron_endpoint_and_three_way_edge_length_conservation": True,
               "sampled_graphs": checked, "crossing_edges": len(links),
               "producer_receipt_sha256": sha(RUN / "census_provenance.json"),
               "validator_sha256": sha(Path(__file__))}
    with (HERE / "independent_census_readback.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
