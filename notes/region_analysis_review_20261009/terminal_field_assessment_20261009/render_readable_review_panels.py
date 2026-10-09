"""Display-only correction: readable exact optical planes and passage labels."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import tifffile

import assess_native_terminal_fields_current_rint as source


def build():
    output = source.HERE / "review_panels"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    records = json.loads((source.HERE / "native_cube_pilot_current_rint/pilot_inputs.json").read_text())
    bindings = []
    for record in records:
        image = record["image"]
        path = source.ROOT / image["path"]
        volume = tifffile.imread(path)
        origin = np.array(image["native_origin_um"])
        point = np.array(record["selected_leaf_native_xyz_um"])
        centre = np.floor((point-origin)/source.SPACING+.5).astype(int)
        x0, x1 = max(0, centre[0]-77), min(360, centre[0]+78)
        y0, y1 = max(0, centre[1]-77), min(360, centre[1]+78)
        planes = list(range(max(0, centre[2]-4), min(90, centre[2]+5)))
        lo, hi = np.percentile(volume[planes, y0:y1, x0:x1], [1, 99.7])
        hi = max(float(hi), float(lo)+1)
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        fig.subplots_adjust(left=.075, right=.985, bottom=.055, top=.87, wspace=.30, hspace=.42)
        for ax, z in zip(axes.flat, planes):
            ax.imshow(volume[z, y0:y1, x0:x1], origin="lower", cmap="gray", vmin=lo, vmax=hi,
                      extent=[origin[0]+(x0-.5)*.65, origin[0]+(x1-.5)*.65,
                              origin[1]+(y0-.5)*.65, origin[1]+(y1-.5)*.65])
            ax.scatter(point[0], point[1], s=35, facecolors="none", edgecolors="#ff654f", linewidth=.8)
            ax.set_title(f"Z {origin[2]+z*3:.1f} µm | plane {z}", fontsize=11, pad=9)
            ax.set_xlabel("Native X (nominal µm)", fontsize=10)
            ax.set_ylabel("Native Y (nominal µm)", fontsize=10)
            ax.tick_params(labelsize=9)
        fig.suptitle(f"{record['NeuronUID']} | original type-2 leaf {record['selected_leaf_id']}\n"
                     f"Unprojected XY planes; ring marks leaf XY, not a separate leaf on every plane\n"
                     f"Leaf native Z {point[2]:.3f} µm | nominal sampling 0.65 × 0.65 × 3 µm; optical resolution unverified",
                     fontsize=13, y=.975)
        stem = record["NeuronUID"].replace("::", "_").removesuffix(".swc")
        fig.savefig(output / f"{stem}_ending_optical_planes.png", dpi=150, bbox_inches="tight", pad_inches=.25)
        plt.close(fig)
        bindings.append({"path": source.relative(path), "sha256": source.sha(path)})
    optical = source.HERE / "optical_plane_checks_current_rint/optical_plane_provenance.json"
    passage = json.loads(optical.read_text())["passage"]
    record = records[1]
    native = np.loadtxt(source.ROOT / record["native_swc"]["path"])
    by_id = {int(row[0]): row for row in native}
    point = by_id[passage["selected_internal_node_id"]][2:5]
    cube = source.cube_index(point)
    origin = np.array(cube)*source.BLOCK
    path = source.ROOT / passage["source_cube"]["path"]
    volume = tifffile.imread(path).transpose(2, 1, 0)
    upper = origin + source.BLOCK
    edges = [(by_id[int(row[6])][2:5], row[2:5]) for row in native if row[1] == 2 and row[6] != -1]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.subplots_adjust(left=.065, right=.985, bottom=.07, top=.84, wspace=.32, hspace=.35)
    for col, (x, y, depth) in enumerate([(0, 1, 2), (0, 2, 1), (1, 2, 0)]):
        projection = volume.max(axis=depth)
        lo, hi = np.percentile(projection, [1, 99.7])
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(projection.T, origin="lower", cmap="gray", vmin=lo, vmax=max(hi, lo+1),
                      extent=[origin[x]-source.SPACING[x]/2, upper[x]-source.SPACING[x]/2,
                              origin[y]-source.SPACING[y]/2, upper[y]-source.SPACING[y]/2])
            if row:
                segments = []
                for a, b in edges:
                    clipped = source.clip_segment(a, b, origin, upper)
                    if clipped is not None:
                        segments.append(np.array(clipped)[:, [x, y]])
                ax.add_collection(LineCollection(segments, colors="#00cfff", linewidths=.7))
                ax.scatter(point[x], point[y], s=40, facecolors="none", edgecolors="#ff654f")
            ax.set_xlim(origin[x]-source.SPACING[x]/2, upper[x]-source.SPACING[x]/2)
            ax.set_ylim(origin[y]-source.SPACING[y]/2, upper[y]-source.SPACING[y]/2)
            ax.set_xlabel(f"Native {'XYZ'[x]} (nominal µm)", fontsize=10)
            ax.set_ylabel(f"Native {'XYZ'[y]} (nominal µm)", fontsize=10)
            ax.set_title(("Unmarked source" if not row else "SWC + internal passage node") + f"\nMaximum over {'XYZ'[depth]}", fontsize=11)
            ax.tick_params(labelsize=9)
    fig.suptitle("252790::032.swc | reconstructed passing component, declared CL_granular_insula\n"
                 "Nodes 7082–7128: no full-graph leaves or local branch nodes; entry 7081→7082, exit 7128→7129\n"
                 "Ring marks internal node 7105; this is not an ending or evidence against en passant boutons", fontsize=13, y=.975)
    fig.savefig(output / "252790_032_passage_image_and_swc.png", dpi=150, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)
    bindings += [{"path": source.relative(path), "sha256": source.sha(path)} for path in
                 [Path(__file__), source.HERE / "assess_native_terminal_fields_current_rint.py", optical]]
    source.write_json(output / "display_provenance.json", {
        "purpose": "readable source-plane and passage displays; no numerical or review changes",
        "inputs": bindings,
        "replaces_display_only": "optical_plane_checks_current_rint PNGs; original receipts preserved",
        "outputs": [{"path": source.relative(path), "sha256": source.sha(path)} for path in sorted(output.glob("*.png"))]})


if __name__ == "__main__":
    build()
