"""Display matched population MRI/map slices from an existing verified run.

Choose axon-labelled length density, candidate axon-end density, or the
fraction of endpoint-eligible neurons with an end in each voxel (occupancy).
Eligible means at least one non-root axon-labelled leaf anywhere in the
complete graph, including outside the reference field of view. Candidate
ends are not image-reviewed biological terminals.

Animal maps average within the selected source group. Group maps give equal
weight to contributing animals. Density color is log10(1 + density);
occupancy color is linear 0–1. Slice choices describe a display selection,
not full-volume anatomy or statistical significance. Source map values and
previous figures remain unchanged; the figure destination must be new.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from build_projection_maps import sha256


def render(run_dir, readback_path, output, *, metric="axon-density", scope="animal", cut_policy="per-map",
           slice_voxels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image

    run_dir, readback_path, output = map(Path, (run_dir, readback_path, output))
    if output.exists():
        raise FileExistsError("Use a new figure destination")
    record_path = run_dir / "run_provenance.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    definitions = {
        "axon-density": ("software_verified_descriptive", "density", "axonLengthDensity",
                         "Axon-labelled length (mm/reference mm³/selected neuron)"),
        "endpoint-density": ("software_verified_candidate_endpoints", "density", "candidateEndpointDensity",
                             "Candidate axon ends/reference mm³/eligible neuron"),
        "endpoint-occupancy": ("software_verified_candidate_endpoints", "occupancy", "candidateEndpointOccupancy",
                               "Fraction of eligible neurons with ≥1 candidate axon end in the voxel"),
    }
    if (metric not in definitions or scope not in ("animal", "group", "both")
            or cut_policy not in ("per-map", "shared", "fixed")):
        raise ValueError("Unknown map metric or display scope")
    if cut_policy == "fixed" and slice_voxels is None:
        raise ValueError("Fixed cuts require explicit XYZ slice indices")
    if cut_policy != "fixed" and slice_voxels is not None:
        raise ValueError("Explicit XYZ slice indices require fixed cuts")
    required_status, field, descriptor, units = definitions[metric]
    endpoint = metric.startswith("endpoint-")
    if (record["status"] != required_status or readback["status"] != "passed"
            or sha256(record_path) != readback["run_provenance_sha256"]):
        raise ValueError("Require a matching complete successful run readback")
    background_sources = record["inputs"] if endpoint else {
        key: {"path": record[key], "sha256": record[key + "_sha256"]}
        for key in ("reference", "brain_mask")}
    for path_key in ("reference", "brain_mask"):
        if path_key not in background_sources:
            raise ValueError("Matched-slice display requires a recorded brain mask")
        if sha256(background_sources[path_key]["path"]) != background_sources[path_key]["sha256"]:
            raise ValueError(f"Changed {path_key}")
    affine = np.asarray(record["affine_mm"])
    if (not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])))
            or np.any(np.diag(affine[:3, :3]) <= 0)):
        raise ValueError("This slice renderer requires the declared positive axis-aligned NMT grid")
    reference = nib.load(background_sources["reference"]["path"]).get_fdata()
    mask = nib.load(background_sources["brain_mask"]["path"]).get_fdata() > 0
    if reference.shape != tuple(record["shape"]) or mask.shape != reference.shape:
        raise ValueError("Background shape differs from the recorded grid")
    if slice_voxels is not None and (len(slice_voxels) != 3 or any(
            isinstance(value, bool) or int(value) != value or not 0 <= value < reference.shape[axis]
            for axis, value in enumerate(slice_voxels))):
        raise ValueError("Fixed slice indices must be three in-bounds integers")
    if not np.isfinite(reference).all() or not mask.any():
        raise ValueError("Background requires finite MRI intensities and a nonempty brain mask")
    gray_low, gray_high = np.percentile(reference[mask], [1, 99.5])
    if gray_high <= gray_low:
        raise ValueError("Background grayscale window is degenerate")
    prepared, colored_values = [], []
    shared_profiles = [np.zeros(reference.shape[axis]) for axis in range(3)]

    def display_planes(data, indices):
        planes = [np.take(data, indices[axis], axis=axis).T for axis in (2, 1, 0)]
        return planes if metric == "endpoint-occupancy" else [np.log10(1 + plane) for plane in planes]

    entries = ([] if scope == "group" else record["animal_maps"]) + (
        [] if scope == "animal" else record["group_maps"])
    for entry in entries:
        if endpoint and not entry["map_available"]:
            continue
        path = run_dir / entry[field + "_path"]
        if sha256(path) != entry[field + "_sha256"]:
            raise ValueError(f"Changed map: {path}")
        image = nib.load(path)
        if image.shape != reference.shape or not np.allclose(image.affine, affine):
            raise ValueError("Map/background geometry mismatch")
        data = image.get_fdata()
        if not np.isfinite(data).all() or np.any(data < 0):
            raise ValueError("Expected a finite nonnegative descriptive map")
        if metric == "endpoint-occupancy" and np.any(data > 1):
            raise ValueError("Candidate occupancy cannot exceed one")
        if cut_policy == "fixed":
            indices = list(map(int, slice_voxels))
        else:
            profiles = [data.sum(axis=tuple(a for a in range(3) if a != axis)) for axis in range(3)]
            if cut_policy == "shared":
                for axis in range(3):
                    shared_profiles[axis] += profiles[axis]
            indices = [int(np.argmax(profile)) for profile in profiles]
        planes = [] if cut_policy == "shared" else display_planes(data, indices)
        colored_values.extend(plane[plane > 0] for plane in planes)
        prepared.append((entry, indices, planes))
    if not prepared:
        raise ValueError("No available maps to render")
    if cut_policy == "shared":
        # Re-read saved maps rather than retaining full volumes in memory.
        indices = [int(np.argmax(profile)) for profile in shared_profiles]
        colored_values, common_prepared = [], []
        for entry, _, _ in prepared:
            path = run_dir / entry[field + "_path"]
            data = nib.load(path).get_fdata()
            if sha256(path) != entry[field + "_sha256"]:
                raise ValueError(f"Map changed while selecting common cuts: {path}")
            planes = display_planes(data, indices)
            colored_values.extend(plane[plane > 0] for plane in planes)
            common_prepared.append((entry, indices, planes))
        prepared = common_prepared
    positives = np.concatenate(colored_values)
    color_high = float(np.percentile(positives, 99.5)) if positives.size else 1.0
    if metric == "endpoint-occupancy":
        color_high = 1.0
    output.mkdir(parents=True)
    figures = []
    for start in range(0, len(prepared), 4):
        count = min(4, len(prepared) - start)
        figure, axes = plt.subplots(count, 3, figsize=(13, count * 3.2),
                                    squeeze=False, layout="constrained")
        for row, (entry, indices, planes) in enumerate(prepared[start:start + count]):
            selected = entry.get("n_selected", entry.get("selected_neurons", entry.get("n_neurons")))
            identity = (f"animal {entry['AnimalID']}" if "AnimalID" in entry
                        else f"equal mean of {entry['n_animals']} animals")
            denominator = f"selected neurons={selected}"
            if endpoint:
                denominator += f"; eligible neurons={entry['n_computable']}"
            for column, axis in enumerate((2, 1, 0)):
                panel = axes[row, column]
                in_plane = [a for a in range(3) if a != axis]
                extent = []
                for dimension in in_plane:
                    extent.extend([float(affine[dimension, 3] - .5 * affine[dimension, dimension]),
                                   float(affine[dimension, 3] + (reference.shape[dimension] - .5)
                                         * affine[dimension, dimension])])
                background = np.take(reference, indices[axis], axis=axis).T
                panel.imshow(background, cmap="gray", origin="lower", extent=extent,
                             norm=Normalize(gray_low, gray_high, clip=True), interpolation="nearest")
                plane = planes[column]
                plotted = panel.imshow(np.ma.masked_where(plane <= 0, plane), cmap="inferno",
                                       norm=Normalize(0, color_high), origin="lower",
                                       extent=extent, interpolation="nearest")
                cut_mm = affine[axis, 3] + indices[axis] * affine[axis, axis]
                cut_label = f"{'XYZ'[axis]}={cut_mm:.2f} mm (voxel {indices[axis]})"
                title = (f"{entry['Subregion']} · {identity}\n{denominator}\n{cut_label}"
                         if column == 0 else cut_label)
                panel.set_title(title, fontsize=9)
                panel.set_xlabel(f"{'XYZ'[in_plane[0]]} (NMT mm)", fontsize=8)
                panel.set_ylabel(f"{'XYZ'[in_plane[1]]} (NMT mm)", fontsize=8)
                panel.tick_params(labelsize=7)
        color_label = units if metric == "endpoint-occupancy" else f"log10(1 + {units})"
        figure.colorbar(plotted, ax=axes, shrink=.7, label=color_label)
        measurement = ("candidate axon ends; biological terminals unverified" if endpoint
                       else "axon-labelled segment length; includes branches")
        cuts = ("Fixed common XYZ cuts" if cut_policy == "fixed" else
                "Shared cuts maximize summed displayed-map values per axis" if cut_policy == "shared" else
                "Cuts maximize summed values per axis; cuts differ by row")
        figure.suptitle("Maps on population MRI slices · NMT v2.1 symmetric\n"
                        f"{cuts}; {measurement}\n"
                        "Descriptive values; registration and coordinate origin unverified\n"
                        f"{'Eligible: ≥1 non-root axon-labelled leaf anywhere (including outside view).' if endpoint else 'Mean per selected neuron within each animal/source group.'}", fontsize=10)
        figure.canvas.draw()
        canvas = figure.canvas.get_renderer()
        title_boxes = [panel.title.get_window_extent(canvas) for panel in axes.flat]
        if any(box.overlaps(other) for index, box in enumerate(title_boxes) for other in title_boxes[index + 1:]):
            plt.close(figure)
            raise ValueError("Panel titles overlap; shorten labels or enlarge the figure")
        path = output / f"space-{record['space_entity']}_desc-{descriptor}_matchedSlices_page-{start//4+1:02d}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        with Image.open(path) as saved:
            saved.load()
            pixels = saved.size
        figures.append({"file": path.name, "sha256": sha256(path), "pixels": pixels,
                        "panel_title_overlap_check": "passed"})
    report = {
        "status": "rendered_and_png_decoded", "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer_sha256": sha256(__file__), "run_provenance_sha256": sha256(record_path),
        "readback_path": str(readback_path), "readback_sha256": sha256(readback_path),
        "reference_sha256": background_sources["reference"]["sha256"],
        "brain_mask_sha256": background_sources["brain_mask"]["sha256"],
        "background": "matching individual NMT MRI slice; no depth projection",
        "grayscale_window": {"brain_mask_percentiles": [1, 99.5], "vmin": float(gray_low), "vmax": float(gray_high)},
        "metric": metric, "scope": scope, "units": units, "cut_policy": cut_policy,
        "map_display": "linear 0–1" if metric == "endpoint-occupancy" else "log10(1 + density)",
        "smoothing": "none; source maps unchanged",
        "common_display_vmax": color_high, "display_upper_percentile": 99.5,
        "slice_selection": ("argmax of summed displayed-map values along each axis; common cuts; first index breaks ties"
                            if cut_policy == "shared" else
                            "explicit common XYZ voxel indices" if cut_policy == "fixed" else
                            "argmax of summed per-map values along each axis; first index breaks ties"),
        "fixed_slice_voxels_xyz": list(map(int, slice_voxels)) if slice_voxels is not None else None,
        "selection_limit": "Display-selected slices do not establish full-volume targets or biological absence",
        "slices": [{"AnimalID": entry.get("AnimalID"), "Subregion": entry["Subregion"],
                    "contributing_animals": entry.get("contributing_animals"),
                    "map_path": entry[field + "_path"], "map_sha256": entry[field + "_sha256"],
                    "slice_voxels_xyz": indices} for entry, indices, _ in prepared],
        "figures": figures, "source_data_modified": False,
        "anatomical_acceptance": "not established",
        "terminal_site_measurement": "candidate graph endpoints only" if endpoint else "not measured",
    }
    (output / "matched_slices_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help='Existing verified map run; source map values will not be changed.')
    parser.add_argument("--readback", type=Path, required=True, help='Successful matching map-run review receipt.')
    parser.add_argument("--output", type=Path, required=True, help='New destination for a separate figure variant.')
    parser.add_argument("--metric", choices=("axon-density", "endpoint-density", "endpoint-occupancy"), default="axon-density", help='Display axon-labelled length density, candidate axon-end density, or endpoint occupancy (eligible-neuron fraction).')
    parser.add_argument("--scope", choices=("animal", "group", "both"), default="animal", help='Animal maps or the equal average of contributing animal maps.')
    parser.add_argument("--cut-policy", choices=("per-map", "shared", "fixed"), default="per-map", help='Choose separate cuts per map, shared data-selected cuts, or fixed common voxel indices.')
    parser.add_argument("--slice-voxels", type=int, nargs=3, metavar=("X", "Y", "Z"), help='Explicit common X Y Z voxel indices when --cut-policy=fixed; these are not millimetres.')
    args = parser.parse_args()
    result = render(args.run, args.readback, args.output, metric=args.metric, scope=args.scope,
                    cut_policy=args.cut_policy, slice_voxels=args.slice_voxels)
    print(f"{result['status']}: {len(result['figures'])} matched-slice sheets")
