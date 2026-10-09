"""Display existing reviewed geometry with plain-language morphology labels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

import assess_native_terminal_fields_current_rint as source


def build():
    inputs_path = source.HERE / "native_cube_pilot_current_rint/pilot_inputs.json"
    records = json.loads(inputs_path.read_text())
    output = source.HERE / "review_panels"
    receipt = output / "morphology_display_provenance.json"
    destinations = [output / (record["NeuronUID"].replace("::", "_").removesuffix(".swc")
                              + "_connected_axon_section.png") for record in records]
    for path in destinations + [receipt]:
        if path.exists():
            raise FileExistsError(path)
    bindings = [inputs_path, Path(__file__)]
    confirmations = []
    for record, destination in zip(records, destinations):
        stem = record["NeuronUID"].replace("::", "_").removesuffix(".swc")
        path = source.HERE / "native_cube_pilot_current_rint" / stem / "target_component_nodes.csv"
        nodes = pd.read_csv(path, dtype=str, keep_default_na=False)
        if len(nodes) != record["component_node_count"] or set(nodes.NeuronUID) != {record["NeuronUID"]}:
            raise ValueError("Saved reviewed node ledger disagrees with original input record")
        by_id = {int(row.node_id): np.array([float(row.native_x_um), float(row.native_y_um),
                                            float(row.native_z_um)]) for row in nodes.itertuples()}
        edges = [(int(row.parent_id), int(row.node_id)) for row in nodes.itertuples()
                 if int(row.parent_id) in by_id]
        ends = sorted(int(row.node_id) for row in nodes.itertuples() if row.full_graph_axon_leaf == "True")
        if ends != sorted(record["component_full_graph_axon_leaf_ids"]):
            raise ValueError("Original full-graph endings changed")
        origin = np.array(record["image"]["native_origin_um"])
        extent = np.array(record["image"]["nominal_spacing_um"]) * np.array([360, 360, 90])
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.subplots_adjust(left=.065, right=.99, bottom=.12, top=.69, wspace=.30)
        for ax, (x, y) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            segments = [np.array([by_id[a], by_id[b]])[:, [x, y]] for a, b in edges]
            ax.add_collection(LineCollection(segments, colors="#536b8e", linewidths=.5))
            coordinates = np.array(list(by_id.values()))
            ax.scatter(coordinates[:, x], coordinates[:, y], s=.15, color="#536b8e")
            if ends:
                tips = np.array([by_id[node] for node in ends])
                ax.scatter(tips[:, x], tips[:, y], s=9, color="#d54a35", label="Original axon ends")
            ax.plot([origin[x], origin[x]+extent[x], origin[x]+extent[x], origin[x], origin[x]],
                    [origin[y], origin[y], origin[y]+extent[y], origin[y]+extent[y], origin[y]],
                    color="#079d91", linewidth=1, label="Inspected image cube")
            ax.autoscale()
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(f"Native {'XYZ'[x]} (nominal µm)", fontsize=10)
            ax.set_ylabel(f"Native {'XYZ'[y]} (nominal µm)", fontsize=10)
            ax.tick_params(labelsize=9)
        title = ("Connected axon section within an ARM region\n"
                 f"{record['NeuronUID']} | ARM level {record['declared_target']['level']}: "
                 f"{record['declared_target']['full_name']} (index {record['declared_target']['index']})\n"
                 f"Nodes: {len(nodes):,} | original axon ends: {len(ends)} | "
                 f"branch points: {len(record['component_branch_node_ids'])}\n"
                 "Red: original axon ends; green box: inspected image cube. Regional cuts do not create ends.")
        fig.suptitle(title, fontsize=13, y=.98, linespacing=1.5)
        fig.savefig(destination, dpi=150, bbox_inches="tight", pad_inches=.25)
        plt.close(fig)
        bindings.append(path)
        confirmations.append({"NeuronUID": record["NeuronUID"], "nodes": len(nodes),
                              "original_axon_end_ids_unchanged": ends,
                              "branch_point_ids_from_original_record": record["component_branch_node_ids"],
                              "ARMFullName": record["declared_target"]["full_name"]})
    source.write_json(receipt, {
        "purpose": "display-only plain-language morphology titles; unchanged saved geometry and review decisions",
        "source_coordinate_convention": "native acquisition XYZ in nominal µm; no anatomical axis or transform acceptance",
        "inputs": [{"path": source.relative(path), "sha256": source.sha(path)} for path in bindings],
        "checks": confirmations,
        "outputs": [{"path": source.relative(path), "sha256": source.sha(path)} for path in destinations]})


if __name__ == "__main__":
    build()
