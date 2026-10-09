"""Additional local plane/continuation evidence, without changing generation."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

import assess_native_terminal_fields_current_rint as pilot


def build():
    original = pilot.HERE / "native_cube_pilot_current_rint"
    output = pilot.HERE / "optical_plane_checks_current_rint"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    records = json.loads((original / "pilot_inputs.json").read_text())
    for record in records:
        image = record["image"]
        volume = tifffile.imread(pilot.ROOT / image["path"])
        origin = np.array(image["native_origin_um"])
        point = np.array(record["selected_leaf_native_xyz_um"])
        centre = np.floor((point - origin) / pilot.SPACING + 0.5).astype(int)
        x0, x1 = max(0, centre[0]-77), min(360, centre[0]+78)
        y0, y1 = max(0, centre[1]-77), min(360, centre[1]+78)
        planes = list(range(max(0, centre[2]-4), min(90, centre[2]+5)))
        lo, hi = np.percentile(volume[planes, y0:y1, x0:x1], [1, 99.7])
        hi = max(float(hi), float(lo)+1)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
        for ax, z in zip(axes.flat, planes):
            ax.imshow(volume[z, y0:y1, x0:x1], origin="lower", cmap="gray", vmin=lo, vmax=hi,
                      extent=[origin[0]+(x0-.5)*.65, origin[0]+(x1-.5)*.65,
                              origin[1]+(y0-.5)*.65, origin[1]+(y1-.5)*.65])
            ax.scatter(point[0], point[1], s=30, facecolors="none", edgecolors="#ff654f", linewidth=0.8)
            ax.set_title(f"Native Z={origin[2]+z*3:.1f} µm; plane {z}")
            ax.set_xlabel("Native X (nominal µm)"); ax.set_ylabel("Native Y (nominal µm)")
        fig.suptitle(f"{record['NeuronUID']} | full-graph leaf {record['selected_leaf_id']}\nUnprojected cached XY planes; ring marks leaf XY only; leaf Z={point[2]:.3f} µm")
        path = output / (record["NeuronUID"].replace("::", "_").removesuffix(".swc") + "_ending_optical_planes.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
    # Passage example: an actual target-local connected type-2 component with
    # no original graph leaves or branch points. Select most nodes, no length
    # threshold; this descriptive example does not imply absence of boutons.
    record = records[1]
    native = np.array(pilot.parse_swc((pilot.ROOT / record["native_swc"]["path"]).read_text()), float)
    warped = np.array(pilot.parse_swc((pilot.ROOT / record["atlas_swc"]["path"]).read_text()), float)
    atlas = np.asarray(pilot.nib.load(pilot.ATLAS).dataobj)[..., 0, 5]
    labels = pilot.atlas_labels(warped, atlas)
    by_id, children, leaves = pilot.graph_information(native)
    options = []
    for nodes in pilot.target_components(native, labels, record["declared_target"]["index"]):
        node_set = set(nodes)
        if node_set & leaves or any(sum(child in node_set for child in children[node]) >= 2 for node in nodes):
            continue
        covered = [node for node in nodes if pilot.cube_path("252790", pilot.cube_index(by_id[node][2:5])).exists()]
        if covered:
            options.append((nodes, covered))
    nodes, covered = sorted(options, key=lambda item: (-len(item[0]), min(item[0])))[0]
    node_set = set(nodes)
    # Keep original entry/exit links visible, rather than inventing leaf ends.
    start = next(node for node in nodes if int(by_id[node][6]) not in node_set)
    exits = [(node, child) for node in nodes for child in children[node] if child not in node_set]
    selected = covered[len(covered)//2]
    cube = pilot.cube_index(by_id[selected][2:5])
    path = pilot.cube_path("252790", cube)
    edges = [(by_id[int(node[6])][2:5], node[2:5]) for node in native if node[1] == 2 and node[6] != -1]
    pilot.draw_image_views(tifffile.imread(path).transpose(2, 1, 0), np.array(cube)*pilot.BLOCK,
                           edges, by_id[selected][2:5], output / "252790_032_passage_image_and_swc.png",
                           "252790::032.swc | declared CL_granular_insula | selected ring is an internal PASSAGE node, not a leaf",
                           highlight_kind="internal passage node")
    records = {"passage": {"NeuronUID": "252790::032.swc", "target_index": 229,
                           "node_ids": sorted(nodes), "entry_link": [int(by_id[start][6]), start],
                           "exit_links": exits, "full_graph_leaves": [], "branch_nodes_within_target_component": [],
                           "selected_internal_node_id": selected,
                           "source_cube": {"path": pilot.relative(path), "sha256": pilot.sha(path)},
                           "classification": "reconstructed_target_passing_segment_candidate; en_passant_boutons_unassessed"},
               "source_generation_sha256": pilot.sha(original / "generation_provenance.json"),
               "scripts": [{"path": pilot.relative(path), "sha256": pilot.sha(path)}
                           for path in [Path(__file__), pilot.HERE / "assess_native_terminal_fields_current_rint.py"]],
               "outputs": [{"path": pilot.relative(path), "sha256": pilot.sha(path)} for path in sorted(output.glob("*.png"))]}
    pilot.write_json(output / "optical_plane_provenance.json", records)
    print(json.dumps({"passage_node_count": len(nodes), "entry_link": records["passage"]["entry_link"], "exit_links": exits,
                      "selected_internal_node_id": selected, "optical_plane_figures": 5}))


if __name__ == "__main__":
    build()
