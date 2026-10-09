"""Coverage-aware descriptive MSTIM/projectome summaries in the same ARM.

Signed fitted contrasts, T statistics, reconstructed axon length and candidate
ends retain separate meanings. This tool fits no registration or association.
The additive subject/end-branch interface preserves historical CM032 outputs;
their original producer hashes remain the provenance for those saved artifacts.
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

END_BRANCH_REVIEW_STATUS = "independent_all462_forward_edge_and_full_voxel_regional_readback_passed"
CM033_CHAIN_STATUS = "software_completed_provisional_cm033_statistic_chain"
CM033_REVIEW_STATUS = "independent_saved_cm033_statistic_chain_reapplication_passed"


def local_source_path(path):
    """Resolve a recorded WSL drive path for Windows I/O, preserving receipt text."""
    text = str(path)
    if sys.platform == "win32" and text.startswith("/"):
        parts = text.split("/")
        if (len(parts) < 4 or parts[:2] != ["", "mnt"]
                or len(parts[2]) != 1 or parts[2] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                or any(part in ("", ".", "..") for part in parts[3:])):
            raise ValueError("Unsupported Unix receipt path; require /mnt/<drive>/<source>")
        return Path(parts[2].upper() + ":/" + "/".join(parts[3:]))
    return Path(path)


def reviewed_warp(receipt_path, subject, review_path=None):
    """Keep the CM032 reproduction contract; independently gate CM033's new chain."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if subject == "cm032":
        if receipt.get("status") != "software_reproduction_passed_registration_unaccepted":
            raise ValueError("A matching numerical warp-reproduction receipt is required")
        if receipt.get("subject", subject).lower() != subject:
            raise ValueError("Warp-reproduction subject differs from the requested subject")
    elif subject == "cm033":
        if (receipt.get("status") != CM033_CHAIN_STATUS
                or receipt.get("subject") != subject or review_path is None
                or not receipt.get("source_sha256")):
            raise ValueError("CM033 requires its genuine completed chain and independent warp review")
        review = json.loads(Path(review_path).read_text(encoding="utf-8"))
        bound = review.get("chain_provenance", {})
        if (review.get("status") != CM033_REVIEW_STATUS
                or bound.get("sha256") != sha256(receipt_path)
                or local_source_path(bound.get("path", "")).resolve() != receipt_path.resolve()
                or review.get("source_sha256") != receipt.get("source_sha256")
                or review.get("anatomical_acceptance") is not False):
            raise ValueError("CM033 requires an exact independently reapplied saved-chain review")
        outputs = receipt["outputs"]
        if review.get("output_sha256") != {
                key: outputs[key]["sha256"] for key in ("contrast", "T", "analysis_coverage")}:
            raise ValueError("CM033 independent review output bindings differ")
        # Genuine CM033 fields are adapted in memory only, never relabeled on disk.
        local_outputs = {
            key: {**item, "path": str(local_source_path(item["path"]))}
            for key, item in outputs.items()
        }
        receipt = {**receipt, "outputs": {**local_outputs, "reproduced_contrast": local_outputs["contrast"]}}
    else:
        raise ValueError("Subject must be cm032 or cm033")
    for path, expected in receipt["source_sha256"].items():
        if sha256(local_source_path(path)) != expected:
            raise ValueError("Warp-reproduction source changed")
    for item in receipt["outputs"].values():
        if sha256(local_source_path(item["path"])) != item["sha256"]:
            raise ValueError("Reproduced image changed")
    return receipt


def reviewed_end_branches(run, review_path, reference_path, atlas_path, key_path):
    """Adapt independently reviewed conditional length maps, without recomputing them."""
    run, review_path = Path(run), Path(review_path)
    record_path = run / "run_provenance.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (record.get("status") != "software_verified_descriptive_end_branches"
            or review.get("status") != END_BRANCH_REVIEW_STATUS
            or review.get("run_provenance", {}).get("sha256") != sha256(record_path)
            or Path(review["run_provenance"]["path"]).resolve() != record_path.resolve()
            or review.get("source_hashes_unchanged") is not True
            or review.get("input_ledger_exactly_preserved") is not True):
        raise ValueError("End-branch run requires its exact successful independent review")
    inputs = record["inputs"]
    if inputs != review.get("source_bindings"):
        raise ValueError("End-branch review source bindings differ from its run")
    paths = [record_path, review_path]
    for name, expected_path in (("reference", reference_path), ("atlas", atlas_path), ("atlas_key", key_path)):
        if inputs[name]["sha256"] != sha256(expected_path):
            raise ValueError("End-branch and MSTIM reference/ARM/key differ")
    for binding in inputs.values():
        path = Path(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise ValueError("End-branch input source changed")
        paths.append(path)
    artifacts = record["artifacts"]
    if artifacts != review.get("verified_artifacts"):
        raise ValueError("Independent end-branch artifact bindings differ")
    manifest = run / "input_ledger.csv"
    if (sha256(manifest) != artifacts["input_ledger.csv"]
            or artifacts["input_ledger.csv"] != inputs["manifest"]["sha256"]):
        raise ValueError("Preserved end-branch ledger differs from its input")
    paths.append(manifest)
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if (not {"SampleID", "NeuronID", "SWCPath", "SWCSHA256"} <= set(frame.columns)
            or frame[["SampleID", "NeuronID"]].duplicated().any()):
        raise ValueError("End-branch ledger requires unique exact neuron identities")
    for row in frame.to_dict("records"):
        path = Path(row["SWCPath"])
        if not path.is_absolute():
            path = ROOT / path
        if sha256(path) != row["SWCSHA256"]:
            raise ValueError("End-branch SWC source changed")
        paths.append(path)
    if (record["selected_neurons"] != len(frame)
            or review["selected_neurons"] != len(frame)
            or record["eligible_neurons"] != review["eligible_neurons"]
            or record["animal_aggregation"] != "eligible neuron mean within animal/source group"
            or record["group_aggregation"] != "equal mean of animals with at least one eligible neuron; absent/unassessed groups not zero-filled"
            or record["length_value_units"] != "reference-template mm per end-eligible neuron"):
        raise ValueError("End-branch conditional denominator or length contract differs")
    for entry in record["group_maps"]:
        selected, eligible, animals = (entry[key] for key in ("selected_neurons", "eligible_neurons", "n_animals"))
        if (any(type(value) is not int for value in (selected, eligible, animals))
                or not 0 <= eligible <= selected or not 0 <= animals <= eligible
                or (eligible > 0) != (animals > 0)
                or animals != len(set(entry["contributing_animals"]))):
            raise ValueError("Invalid end-branch conditional group denominator")
        has_path, has_hash = "length_path" in entry, "length_sha256" in entry
        if has_path != bool(eligible) or has_hash != bool(eligible):
            raise ValueError("End-branch map availability must match its eligible denominator")
        if eligible and artifacts.get(entry["length_path"]) != entry["length_sha256"]:
            raise ValueError("End-branch group map is absent from the independent review")
    # The existing consumer uses this common schema; saved maps stay unchanged.
    adapted = {**record, "reference_sha256": inputs["reference"]["sha256"], "preserved_manifest": "input_ledger.csv"}
    adapted["group_maps"] = [
        {**entry, "n_selected": entry["selected_neurons"], "n_computable": entry["eligible_neurons"],
         "map_available": entry["eligible_neurons"] > 0}
        for entry in record["group_maps"]
    ]
    return adapted, paths


def projection_fields(kind, entry, value, covered_voxels):
    """Saved group values already have their neuron/animal denominators applied."""
    metric = {"endpoint": "candidate_ends_per_eligible_neuron_in_coverage",
              "axon": "axon_length_mm_per_selected_neuron_in_coverage",
              "end_branch": "end_branch_length_mm_per_eligible_neuron_in_coverage"}[kind]
    available = entry.get("map_available", True)
    result = {metric: value if covered_voxels and available else None,
              kind + "_contributing_animals": entry["n_animals"],
              kind + "_selected_neurons": entry.get("n_selected", entry.get("n_neurons"))}
    if kind in ("endpoint", "end_branch"):
        result[kind + "_eligible_neurons"] = entry["n_computable"]
    return result


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


def plot_qc(reference, effect, coverage, normalized_anatomy, output, subject="cm032"):
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
    figure.suptitle(f"{subject.upper()} numerical integration QC · anatomical registration unaccepted\n"
                    "Top: normalized anatomy display contour on NMT MRI. Bottom: MSTIM contrast and SPM analysis coverage.")
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {"cuts_xyz": cuts, "contrast_abs_percentile": 99, "contrast_color_limit": bound,
            "normalized_anatomy_display_threshold": threshold, "anatomical_acceptance": False}


def main(args):
    subject = getattr(args, "subject", "cm032")
    if subject not in ("cm032", "cm033"):
        raise ValueError("Subject must be cm032 or cm033")
    end_run = getattr(args, "end_branch_run", None)
    end_review = getattr(args, "end_branch_review", None)
    if (end_run is None) != (end_review is None):
        raise ValueError("End-branch run and independent review must be supplied together")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh integration directory")
    warp_review = getattr(args, "warp_review", None)
    receipt = reviewed_warp(args.warp_receipt, subject, warp_review)
    coverage_path = Path(receipt["outputs"]["analysis_coverage"]["path"])
    effect_path = Path(receipt["outputs"]["reproduced_contrast"]["path"])
    if subject == "cm033" and sha256(args.statistic) != receipt["outputs"]["T"]["sha256"]:
        raise ValueError("CM033 statistic differs from its independently reviewed T image")
    base = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym"
    reference_path, atlas_path, key_path = base / "NMT_v2.1_sym_SS.nii.gz", base / "ARM_in_NMT_v2.1_sym.nii.gz", ROOT / "atlas/ARM_key_all.txt"
    reference = nib.load(reference_path)
    grid = coded_mm_grid(reference)
    paths = [args.warp_receipt, reference_path, atlas_path, key_path, coverage_path, effect_path, args.statistic, args.normalized_anatomy, args.label_map]
    if subject == "cm033":
        paths.append(warp_review)
        paths += [local_source_path(path) for path in receipt["source_sha256"]]
    images = [nib.load(path) for path in (effect_path, args.statistic, coverage_path, args.normalized_anatomy)]
    for image in images:
        matching_grid(image, grid)
    effect, statistic, coverage, anatomy = [image.get_fdata() for image in images]
    if not np.any(coverage) or not np.isfinite(anatomy).all() or not np.any(anatomy > 0):
        raise ValueError("Nonempty measured coverage and finite normalized anatomy are required")
    targets, levels = catalog(atlas_path, key_path, grid)
    source_labels = None
    reviewed = []
    runs = [(args.endpoint_run, args.endpoint_review, "endpoint"), (args.axon_run, args.axon_review, "axon")]
    if end_run is not None:
        runs.append((end_run, end_review, "end_branch"))
    for run, readback, kind in runs:
        record_path = run / "run_provenance.json"
        if kind == "end_branch":
            record, extra_paths = reviewed_end_branches(run, readback, reference_path, atlas_path, key_path)
            paths += extra_paths
        else:
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
                fmri_rows.append({**target, **summary[index], "subject": subject, "registration_status": "unaccepted"})
    fmri = pd.DataFrame(fmri_rows)
    joint = {}
    source_metadata = {row["Subregion"]: row for row in reviewed[0][3]["names"]}
    for run, record, kind, _ in reviewed:
        field = "count" if kind == "endpoint" else "length"
        for entry in record["group_maps"]:
            source = source_metadata[entry["Subregion"]]
            if source["SourceARMStatus"] != "mapped":
                continue
            available = entry.get("map_available", True)
            if not available and kind != "end_branch":
                continue
            values = None
            if available:
                path = run / entry[field + "_path"]
                expected = entry[field + "_sha256"]
                if sha256(path) != expected:
                    raise ValueError("Reviewed projection map changed")
                bindings[str(path.resolve())] = expected
                image = nib.load(path)
                matching_grid(image, grid)
                values = image.get_fdata()
            for level, labels in enumerate(levels, 1):
                sums = covered_projection(values, labels.reshape(grid.shape), coverage) if available else None
                for target in fmri.loc[fmri.Level == level].to_dict("records"):
                    key = (entry["Subregion"], target["TargetID"])
                    row = joint.setdefault(key, {**target, "SourceARMIndex": source["ARMIndex"],
                        "SourceARMFullName": source["ARMFullName"], "SourceARMGroup": entry["Subregion"],
                        "SourceHemisphere": source["Hemisphere"], "same_animal_pairing": "unknown",
                        "comparison_status": "conditional_numerical_summary_registration_unaccepted"})
                    index = int(target["ARMIndex"])
                    value = float(sums[index]) if available and index < len(sums) else 0.0
                    row.update(projection_fields(kind, entry, value, target["covered_voxels"]))
    output.mkdir(parents=True)
    fmri.to_csv(output / f"{subject}_signed_response_by_ARM_level.csv", index=False)
    pd.DataFrame(joint.values()).to_csv(output / f"{subject}_projectome_common_coverage.csv", index=False)
    display = plot_qc(reference, effect, coverage, anatomy, output / f"{subject}_alignment_and_response_qc.png", subject=subject)
    if any(sha256(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("A bound source changed during integration")
    provenance = {"status": "provisional_regional_integration_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": subject, "interface_version": "2_subject_and_optional_reviewed_end_branches",
        "warp_status": receipt["status"],
        "warp_review": None if subject == "cm032" else {"path": str(warp_review.resolve()), "sha256": sha256(warp_review)},
        "subject_source_limits": receipt.get("limitations", receipt.get("constraints", [])),
        "source_sha256": bindings, "script_sha256": sha256(__file__), "python": sys.version,
        "atlas_levels": list(range(1, 7)), "primary_comparison_level": 3, "joint_rows": len(joint), "fmri_region_rows": len(fmri),
        "spm_contrast": receipt["spm_contrast"], "coverage_voxels": int(np.count_nonzero(coverage)),
        "native_fMRI_sampling_mm": [0.75, 0.75, 2.0], "NMT_sampling_mm": [0.25, 0.25, 0.25],
        "coverage_policy": "Same explicit warped SPM estimation mask for signed response and additive projection summaries; no zero-value support inference",
        "endpoint_measure": "Additive mean candidate-end counts; not regional neuron frequency and not summed voxel occupancy",
        "end_branch_measure": "Optional reconstructed end-branch length in template mm; eligible-neuron means within animals, then equal contributing-animal mean; no eligible ends remain NA" if end_run is not None else "not included",
        "statistical_policy": "T values summarized descriptively only; no new threshold, regional t statistic, correlation p value or independent-voxel inference",
        "spatial_policy": "No inter-template resampling: actual skull-stripped template grids/payload match; subject registrations remain unaccepted",
        "stimulation_site": receipt.get("stimulation_site", "CM032 pos1, runs38-41, recorded left ventral anterior insula/Ial; exact accepted NMT stimulation coordinates unavailable" if subject == "cm032" else "CM033 source contrast identified by warp receipt; accepted NMT stimulation coordinates unavailable"),
        "source_selection_policy": "All mapped named source ARM groups reported; unresolved source groups remain QC strata; no best-matching source selected",
        "anatomical_acceptance": False, "causal_or_monosynaptic_claim": False, "source_writes": False, "display": display,
        "outputs": {path.name: sha256(path) for path in output.iterdir() if path.is_file()}}
    (output / "integration_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": provenance["status"], "joint_rows": len(joint), "fmri_region_rows": len(fmri)}))


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("warp-receipt", "statistic", "normalized-anatomy", "endpoint-run", "endpoint-review", "axon-run", "axon-review", "label-map", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--subject", choices=("cm032", "cm033"), default="cm032", help="Subject recorded in tables, filenames and QC; default preserves CM032 naming")
    parser.add_argument("--warp-review", type=Path, help="Required for CM033: successful independent exact saved-chain reapplication receipt")
    parser.add_argument("--end-branch-run", type=Path, help="Optional completed reconstructed end-branch map run")
    parser.add_argument("--end-branch-review", type=Path, help="Exact successful independent end-branch readback receipt; required with its run")
    return parser


if __name__ == "__main__":
    main(argument_parser().parse_args())
