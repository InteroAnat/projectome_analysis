"""Coverage-aware descriptive MSTIM/projectome summaries in the same ARM.

Signed fitted contrasts, T statistics, reconstructed axon length and candidate
ends retain separate meanings. This tool fits no registration or association.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main_scripts"))
from endpoint_atlas import coded_mm_grid, matching_grid
from export_arm_projection_tables import catalog, sha256
from render_projection_slices import arm_source_labels


def matching_planes(data, cuts):
    """Linear signed-data planes; no MIP, logarithm, reflection or resampling."""
    return [np.take(data, cuts[axis], axis=axis).T for axis in (2, 1, 0)]


def covered_summary(effect, statistic, coverage, labels):
    """A measured zero stays zero; a region without coverage stays missing."""
    if any(array.shape != labels.shape for array in (effect, statistic, coverage)):
        raise ValueError("Regional inputs must have identical spatial shapes")
    if (not np.isin(np.unique(coverage), [0, 1]).all()
            or not np.isfinite(effect[coverage.astype(bool)]).all()
            or not np.isfinite(statistic[coverage.astype(bool)]).all()):
        raise ValueError("Coverage must be binary and measured values finite")
    result = {}
    for index in np.unique(labels):
        region = labels == index
        measured = region & coverage.astype(bool)
        count = int(measured.sum())
        result[int(index)] = {"region_voxels": int(region.sum()), "covered_voxels": count,
                              "coverage_fraction": count / int(region.sum()),
                              "mean_signed_contrast": float(effect[measured].mean()) if count else None,
                              "median_signed_contrast": float(np.median(effect[measured])) if count else None,
                              "median_T_descriptive_only": float(np.median(statistic[measured])) if count else None}
    return result


def covered_projection(values, labels, coverage):
    """Sum additive values; voxel occupancy is deliberately not accepted here."""
    if (values.shape != labels.shape or coverage.shape != labels.shape
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.isin(np.unique(coverage), [0, 1]).all()):
        raise ValueError("Projection values must be matching finite nonnegative additive maps")
    return np.bincount(labels[coverage.astype(bool)].astype(np.int64), weights=values[coverage.astype(bool)])


def plot_qc(reference, effect, coverage, normalized_anatomy, output):
    cuts = [56, 200, 87]
    background = reference.get_fdata()
    gray = matching_planes(background, cuts)
    response = matching_planes(effect, cuts)
    support = matching_planes(coverage, cuts)
    anatomy = matching_planes(normalized_anatomy, cuts)
    bound = float(np.percentile(np.abs(effect[coverage.astype(bool)]), 99)) or 1.0
    threshold = float(np.percentile(normalized_anatomy[normalized_anatomy > 0], 99)) * 0.1
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for axis in range(3):
        for row in range(2):
            axes[row, axis].imshow(gray[axis], origin="lower", cmap="gray", vmin=75, vmax=920)
            axes[row, axis].set_axis_off()
        axes[0, axis].contour(anatomy[axis] > threshold, levels=[0.5], colors=["cyan"], linewidths=0.5)
        image = axes[1, axis].imshow(np.ma.masked_where(support[axis] == 0, response[axis]), origin="lower",
                                     cmap="coolwarm", vmin=-bound, vmax=bound, alpha=0.7)
        axes[1, axis].contour(support[axis], levels=[0.5], colors=["lime"], linewidths=0.5)
        axes[0, axis].set_title(["XY at Z=87", "XZ at Y=200", "YZ at X=56"][axis])
    figure.colorbar(image, ax=axes[1, :], label="Signed fitted contrast\nSPM global scaling; PSC unestablished", shrink=0.7)
    figure.suptitle("CM032 numerical integration QC · anatomical registration unaccepted\n"
                    "Top: normalized anatomy display contour on NMT MRI. Bottom: MSTIM contrast and SPM analysis coverage.")
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {"cuts_xyz": cuts, "contrast_abs_percentile": 99, "contrast_color_limit": bound,
            "normalized_anatomy_display_threshold": threshold, "anatomical_acceptance": False}


def main(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh integration directory")
    receipt = json.loads(args.warp_receipt.read_text(encoding="utf-8"))
    if receipt["status"] != "software_reproduction_passed_registration_unaccepted":
        raise ValueError("A matching numerical warp-reproduction receipt is required")
    for path, expected in receipt["source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("Warp-reproduction source changed")
    coverage_path = Path(receipt["outputs"]["analysis_coverage"]["path"])
    effect_path = Path(receipt["outputs"]["reproduced_contrast"]["path"])
    for item in receipt["outputs"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Reproduced image changed")
    base = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym"
    reference_path, atlas_path, key_path = base / "NMT_v2.1_sym_SS.nii.gz", base / "ARM_in_NMT_v2.1_sym.nii.gz", ROOT / "atlas/ARM_key_all.txt"
    reference = nib.load(reference_path)
    grid = coded_mm_grid(reference)
    paths = [args.warp_receipt, reference_path, atlas_path, key_path, coverage_path, effect_path, args.statistic, args.normalized_anatomy, args.label_map]
    images = [nib.load(path) for path in (effect_path, args.statistic, coverage_path, args.normalized_anatomy)]
    for image in images:
        matching_grid(image, grid)
    effect, statistic, coverage, anatomy = [image.get_fdata() for image in images]
    if not np.any(coverage) or not np.isfinite(anatomy).all() or not np.any(anatomy > 0):
        raise ValueError("Nonempty measured coverage and finite normalized anatomy are required")
    targets, levels = catalog(atlas_path, key_path, grid)
    source_labels = None
    reviewed = []
    for run, readback, kind in ((args.endpoint_run, args.endpoint_review, "endpoint"), (args.axon_run, args.axon_review, "axon")):
        record_path = run / "run_provenance.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        review = json.loads(readback.read_text(encoding="utf-8"))
        if review["status"] != "passed" or review["run_provenance_sha256"] != sha256(record_path):
            raise ValueError("Projection run is not bound to a successful independent review")
        if record["reference_sha256"] != sha256(reference_path):
            raise ValueError("Projection and MSTIM reference differ")
        manifest = run / record["preserved_manifest"]
        names, label_receipt = arm_source_labels(args.label_map, manifest)
        if label_receipt["atlas_sha256"] != sha256(atlas_path) or label_receipt["atlas_key_sha256"] != sha256(key_path):
            raise ValueError("Projection and MSTIM must use the same official ARM and key")
        if source_labels is not None and source_labels != names:
            raise ValueError("Projection runs have different source-region labels")
        source_labels = names
        paths += [record_path, readback, manifest]
        reviewed.append((run, record, kind, label_receipt))
    bindings = {str(path.resolve()): sha256(path) for path in paths}
    fmri_rows = []
    for level, labels in enumerate(levels, 1):
        summary = covered_summary(effect, statistic, coverage, labels.reshape(grid.shape))
        for target in targets.loc[(targets.Level == level) & (targets.TargetStatus != "out_of_FOV")].to_dict("records"):
            index = int(target["ARMIndex"])
            if index in summary:
                fmri_rows.append({**target, **summary[index], "subject": "cm032", "registration_status": "unaccepted"})
    fmri = pd.DataFrame(fmri_rows)
    joint = {}
    source_metadata = {row["Subregion"]: row for row in reviewed[0][3]["names"]}
    for run, record, kind, _ in reviewed:
        field = "count" if kind == "endpoint" else "length"
        for entry in record["group_maps"]:
            source = source_metadata[entry["Subregion"]]
            if source["SourceARMStatus"] != "mapped" or not entry.get("map_available", True):
                continue
            path = run / entry[field + "_path"]
            expected = entry[field + "_sha256"]
            if sha256(path) != expected:
                raise ValueError("Reviewed projection map changed")
            bindings[str(path.resolve())] = expected
            image = nib.load(path)
            matching_grid(image, grid)
            values = image.get_fdata()
            for level, labels in enumerate(levels, 1):
                sums = covered_projection(values, labels.reshape(grid.shape), coverage)
                for target in fmri.loc[fmri.Level == level].to_dict("records"):
                    key = (entry["Subregion"], target["TargetID"])
                    row = joint.setdefault(key, {**target, "SourceARMIndex": source["ARMIndex"],
                        "SourceARMFullName": source["ARMFullName"], "SourceARMGroup": entry["Subregion"],
                        "SourceHemisphere": source["Hemisphere"], "same_animal_pairing": "unknown",
                        "comparison_status": "conditional_numerical_summary_registration_unaccepted"})
                    index = int(target["ARMIndex"])
                    value = float(sums[index]) if index < len(sums) else 0.0
                    row["candidate_ends_per_eligible_neuron_in_coverage" if kind == "endpoint" else "axon_length_mm_per_selected_neuron_in_coverage"] = value if target["covered_voxels"] else None
                    row[kind + "_contributing_animals"] = entry["n_animals"]
                    row[kind + "_selected_neurons"] = entry.get("n_selected", entry.get("n_neurons"))
                    if kind == "endpoint":
                        row["endpoint_eligible_neurons"] = entry["n_computable"]
    output.mkdir(parents=True)
    fmri.to_csv(output / "cm032_signed_response_by_ARM_level.csv", index=False)
    pd.DataFrame(joint.values()).to_csv(output / "cm032_projectome_common_coverage.csv", index=False)
    display = plot_qc(reference, effect, coverage, anatomy, output / "cm032_alignment_and_response_qc.png")
    if any(sha256(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("A bound source changed during integration")
    provenance = {"status": "provisional_regional_integration_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": bindings, "script_sha256": sha256(__file__), "python": sys.version,
        "atlas_levels": list(range(1, 7)), "primary_comparison_level": 3, "joint_rows": len(joint), "fmri_region_rows": len(fmri),
        "spm_contrast": receipt["spm_contrast"], "coverage_voxels": int(np.count_nonzero(coverage)),
        "native_fMRI_sampling_mm": [0.75, 0.75, 2.0], "NMT_sampling_mm": [0.25, 0.25, 0.25],
        "coverage_policy": "Same explicit warped SPM estimation mask for signed response and additive projection summaries; no zero-value support inference",
        "endpoint_measure": "Additive mean candidate-end counts; not regional neuron frequency and not summed voxel occupancy",
        "statistical_policy": "T values summarized descriptively only; no new threshold, regional t statistic, correlation p value or independent-voxel inference",
        "spatial_policy": "No inter-template resampling: actual skull-stripped template grids/payload match; subject registrations remain unaccepted",
        "stimulation_site": "CM032 pos1, runs38-41, recorded left ventral anterior insula/Ial; exact accepted NMT stimulation coordinates unavailable",
        "source_selection_policy": "All named source ARM groups reported; no best-matching source selected",
        "CM033": "Excluded from spatial comparison: recorded orientation rejection and no accepted NMT model",
        "anatomical_acceptance": False, "causal_or_monosynaptic_claim": False, "source_writes": False, "display": display,
        "outputs": {path.name: sha256(path) for path in output.iterdir() if path.is_file()}}
    (output / "integration_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": provenance["status"], "joint_rows": len(joint), "fmri_region_rows": len(fmri)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("warp-receipt", "statistic", "normalized-anatomy", "endpoint-run", "endpoint-review", "axon-run", "axon-review", "label-map", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
