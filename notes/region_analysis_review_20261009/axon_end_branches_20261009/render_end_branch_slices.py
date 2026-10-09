"""Show end-branch density on the same NMT MRI cuts as current ARM maps."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(run_directory, readback_path, brain_mask, output):
    run_directory, readback_path, brain_mask, output = map(Path, (run_directory, readback_path, brain_mask, output))
    if output.exists():
        raise FileExistsError("Preserve earlier displays; use a fresh destination")
    run_path = run_directory / "run_provenance.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    check = json.loads(readback_path.read_text(encoding="utf-8"))
    if (run["status"] != "software_verified_descriptive_end_branches"
            or check["status"] != "independent_all462_forward_edge_and_full_voxel_regional_readback_passed"
            or check["run_provenance"]["sha256"] != digest(run_path)):
        raise ValueError("Require completed maps and their matching independent readback")
    for binding in run["inputs"].values():
        if digest(binding["path"]) != binding["sha256"]:
            raise ValueError("Source/reference/ARM bytes changed")
    for relative, expected in run["artifacts"].items():
        if digest(run_directory / relative) != expected:
            raise ValueError("A saved end-branch artifact changed after independent review")
    mask_hash = digest(brain_mask)
    reference_image = nib.load(run["inputs"]["reference"]["path"])
    reference = reference_image.get_fdata()
    mask_image = nib.load(brain_mask)
    mask = mask_image.get_fdata() > 0
    affine = np.asarray(run["affine_mm"])
    if (mask.shape != reference.shape or not np.allclose(mask_image.affine, affine)
            or not np.allclose(reference_image.affine, affine) or not np.isfinite(reference).all()
            or not mask.any()):
        raise ValueError("Pinned finite MRI and matching brain mask are required")
    if not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3]))) or np.any(np.diag(affine[:3, :3]) <= 0):
        raise ValueError("This display requires the declared positive axis-aligned NMT grid")
    low, high = np.percentile(reference[mask], [1, 99.5])
    if high <= low:
        raise ValueError("Degenerate MRI grayscale window")
    cuts = [56, 200, 87]
    if any(not 0 <= cut < reference.shape[axis] for axis, cut in enumerate(cuts)):
        raise ValueError("Primary comparison cuts are outside the reference")
    ledger = pd.read_csv(run_directory / "input_ledger.csv", dtype=str, keep_default_na=False)
    source_fields = ["Subregion", "ARMFullName", "ARMIndex", "Hemisphere", "SourceARMStatus"]
    sources = ledger[source_fields].drop_duplicates().set_index("Subregion")
    if sources.index.duplicated().any():
        raise ValueError("Source groups have contradictory official names")
    prepared, positives = [], []
    for entry in run["group_maps"]:
        metadata = sources.loc[entry["Subregion"]]
        if metadata.SourceARMStatus != "mapped" or not entry["n_animals"]:
            continue
        path = run_directory / entry["length_path"]
        if digest(path) != entry["length_sha256"]:
            raise ValueError("End-branch map changed")
        image = nib.load(path)
        if image.shape != reference.shape or not np.allclose(image.affine, affine):
            raise ValueError("Map and MRI geometry differ")
        data = image.get_fdata()
        if not np.isfinite(data).all() or np.any(data < 0):
            raise ValueError("Expected finite nonnegative descriptive lengths")
        density = data / run["voxel_volume_mm3"]
        planes = [np.log10(1 + np.take(density, cuts[axis], axis=axis).T) for axis in (2, 1, 0)]
        positives.extend(plane[plane > 0] for plane in planes)
        prepared.append((entry, metadata, planes))
    if not prepared:
        raise ValueError("No named official ARM source maps")
    positive_values = np.concatenate(positives)
    color_high = float(np.percentile(positive_values, 99.5)) if positive_values.size else 1.0
    output.mkdir(parents=True)
    figures = []
    for start in range(0, len(prepared), 4):
        batch = prepared[start:start + 4]
        fig, axes = plt.subplots(len(batch), 3, figsize=(13, 3.25 * len(batch)), squeeze=False, layout="constrained")
        for row, (entry, metadata, planes) in enumerate(batch):
            name = textwrap.fill(metadata.ARMFullName.replace("_", " "), 38, break_long_words=False)
            for column, axis in enumerate((2, 1, 0)):
                panel = axes[row, column]
                panel.imshow(np.take(reference, cuts[axis], axis=axis).T, cmap="gray", origin="lower",
                             norm=Normalize(low, high, clip=True), interpolation="nearest")
                plotted = panel.imshow(np.ma.masked_less_equal(planes[column], 0), cmap="inferno", origin="lower",
                                       norm=Normalize(0, color_high, clip=True), interpolation="nearest")
                title = f"{'XYZ'[axis]} voxel {cuts[axis]}"
                if column == 0:
                    title = f"{name}\nARM index {metadata.ARMIndex}; animals={entry['n_animals']}\nend-eligible neurons={entry['eligible_neurons']}/{entry['selected_neurons']}\n" + title
                panel.set_title(title, fontsize=10)
                panel.set_xlabel("Reference voxel index")
                panel.set_ylabel("Reference voxel index")
        fig.suptitle("Reconstructed axon end-branch length density\nEqual animal mean; graph proxy, not reviewed terminal fields", fontsize=13)
        fig.colorbar(plotted, ax=axes.ravel().tolist(), shrink=.65,
                     label="log10(1 + template mm / reference mm³ / end-eligible neuron)")
        path = output / f"space-NMTv2p1_desc-axonEndBranchDensity_matchedSlices_page-{start // 4 + 1:02}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures.append({"path": path.name, "sha256": digest(path), "source_groups": [item[0]["Subregion"] for item in batch]})
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_provenance_sha256": digest(run_path),
        "readback_sha256": digest(readback_path), "renderer_sha256": digest(__file__),
        "brain_mask": {"path": str(brain_mask.resolve()), "sha256": mask_hash},
        "background": run["inputs"]["reference"], "background_description": "symmetric NMT v2.1 T1-weighted MRI",
        "MRI_percentile_window": [1, 99.5], "MRI_intensity_limits": [float(low), float(high)],
        "shared_XYZ_cuts": cuts, "MIP": False, "smoothing": None,
        "overlay": "log10(1 + stored length / reference voxel volume); display only",
        "color_percentile": 99.5, "color_upper_limit": color_high,
        "unassigned_source_groups": "retained in full ledger and numerical outputs, excluded from named anatomy figures",
        "scientific_anatomical_acceptance": False, "figures": figures}
    (output / "display_provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if digest(brain_mask) != mask_hash:
        raise RuntimeError("Brain mask changed during rendering")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "readback", "brain-mask", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.readback, args.brain_mask, args.output)
