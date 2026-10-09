"""Independent direct CM033 ARM-table and joint-map readback; no producer imports."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ATLAS = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym"


def local(path):
    text = str(path).replace("\\", "/")
    if text.startswith("/mnt/"):
        text = text[5].upper() + ":" + text[6:]
    return Path(text)


def sha(path):
    result = hashlib.sha256()
    with local(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return pd.read_csv(local(path), dtype=str, keep_default_na=False)


def require(condition, label):
    if not condition:
        raise ValueError(label)


def close(actual, expected):
    measured = np.array([np.nan if str(value) == "" else float(value)
                         for value in np.atleast_1d(actual)], dtype=float)
    expected = np.asarray(expected, float).reshape(-1)
    require(np.allclose(measured, expected, rtol=1e-9, atol=1e-11, equal_nan=True),
            f"Numerical mismatch: actual={measured}, expected={expected}")


def grid(image, reference, atlas=False):
    require(image.shape == ((*reference.shape, 1, 6) if atlas else reference.shape), "Exact grid shape differs")
    require(np.array_equal(image.affine, reference.affine), "Exact spatial affine differs")
    require(image.header.get_xyzt_units()[0] == "mm", "Spatial units not millimetres")
    require(int(image.header["sform_code"]) > 0 or int(image.header["qform_code"]) > 0, "Spatial form uncoded")


def identities(frame):
    result = frame.SampleID + "::" + frame.NeuronID
    require(not result.duplicated().any(), "Duplicate exact source identity")
    return result


def main():
    provenance_path = HERE / "integration_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    require(provenance["status"] == "provisional_regional_integration_completed"
            and provenance["subject"] == "cm033", "Require completed genuine CM033 summary")
    require(provenance["anatomical_acceptance"] is False, "Anatomical acceptance overclaim")
    inputs = provenance["source_sha256"]
    for path, expected in inputs.items():
        require(sha(path) == expected, "Bound source changed: " + path)
    for path, expected in provenance["outputs"].items():
        require(sha(HERE / path) == expected, "Frozen summary output changed")
    producer = ROOT / "group_analysis/scripts/summarize_mstim_arm.py"
    require(sha(producer) == provenance["script_sha256"], "Current summary producer differs")
    chain_path = next(p for p in inputs if p.endswith("statistic_chain\\chain_provenance.json")
                      or p.endswith("statistic_chain/chain_provenance.json"))
    chain = json.loads(local(chain_path).read_text())
    warp_review = json.loads(local(provenance["warp_review"]["path"]).read_text())
    require(warp_review["status"] == "independent_saved_cm033_statistic_chain_reapplication_passed"
            and warp_review["chain_provenance"]["sha256"] == sha(chain_path), "Genuine independent chain receipt mismatch")
    require(warp_review["source_sha256"] == chain["source_sha256"], "Warp source identity bindings differ")
    require(warp_review["output_sha256"] == {k: v["sha256"] for k, v in chain["outputs"].items()}, "Warp output identity bindings differ")
    reference = nib.load(ATLAS / "NMT_v2.1_sym_SS.nii.gz")
    atlas_image = nib.load(ATLAS / "ARM_in_NMT_v2.1_sym.nii.gz")
    grid(atlas_image, reference, atlas=True)
    atlas = np.asarray(atlas_image.dataobj)[:, :, :, 0, :]
    require(np.isfinite(atlas).all() and np.equal(atlas, np.floor(atlas)).all(), "Invalid observed ARM labels")
    loaded = {k: nib.load(local(chain["outputs"][k]["path"])) for k in ("contrast", "T", "analysis_coverage")}
    for image in loaded.values():
        grid(image, reference)
    mask_values = np.asanyarray(loaded["analysis_coverage"].dataobj)
    require(set(np.unique(mask_values)) == {0, 1}, "Actual model coverage not binary")
    mask = mask_values.astype(bool)
    contrast_full = loaded["contrast"].get_fdata()
    statistic_full = loaded["T"].get_fdata()
    contrast, statistic = contrast_full[mask], statistic_full[mask]
    require(np.isfinite(contrast).all() and np.isfinite(statistic).all(), "Covered values not finite")
    require(int(mask.sum()) == provenance["coverage_voxels"] == 4259693, "Coverage denominator differs")
    close(reference.header.get_zooms()[:3], provenance["NMT_sampling_mm"])
    native_source = next(row["path"] for row in chain["inputs"]["source_statistics"]
                         if local(row["path"]).name == "con_0001.nii")
    np.testing.assert_allclose(
        nib.load(local(native_source)).header.get_zooms()[:3],
        provenance["native_fMRI_sampling_mm"], rtol=2e-7, atol=1e-7,
    )  # Native float32 pixdim stores the nominal 2 mm as 2.000000238.
    key = pd.read_csv(ROOT / "atlas/ARM_key_all.txt", sep="\t", dtype=str,
                      keep_default_na=False).set_index("Index", verify_integrity=True)
    fmri = read(HERE / "cm033_signed_response_by_ARM_level.csv").set_index("TargetID", verify_integrity=True)
    joint = read(HERE / "cm033_projectome_common_coverage.csv")
    require(len(fmri) == provenance["fmri_region_rows"] == 1676, "Regional row count differs")
    require(len(joint) == provenance["joint_rows"] == 25140
            and not joint.duplicated(["SourceARMGroup", "TargetID"]).any(), "Joint identity/count differs")
    require(fmri.subject.eq("cm033").all() and fmri.registration_status.eq("unaccepted").all(), "fMRI subject/status differs")
    require(joint.same_animal_pairing.eq("unknown").all()
            and joint.comparison_status.eq("conditional_numerical_summary_registration_unaccepted").all(),
            "Unpaired/provisional status differs")
    for field in fmri.columns:
        require(np.array_equal(joint[field].to_numpy(), fmri.loc[joint.TargetID, field].to_numpy()),
                "Joint copied fMRI metadata differs: " + field)
    covered_labels, expected_targets, no_coverage, conflicts, coverage_by_level = [], set(), [], [], []
    voxel_volume = abs(float(np.linalg.det(reference.affine[:3, :3])))
    for level in range(1, 7):
        labels = atlas[:, :, :, level - 1]
        unique, sizes = np.unique(labels, return_counts=True)
        observed = labels[mask].astype(np.int64)
        covered_labels.append(observed)
        counts = np.bincount(observed)
        sums = np.bincount(observed, weights=contrast)
        order = np.argsort(observed, kind="stable")
        ordered = observed[order]
        breaks = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1, len(order)]
        medians = {int(ordered[a]): (float(np.median(contrast[order[a:b]])),
                                     float(np.median(statistic[order[a:b]])))
                   for a, b in zip(breaks[:-1], breaks[1:])}
        for index, size in zip(unique, sizes):
            index = int(index)
            if index == 0:
                status, name, abbreviation, domain, side, conflict = (
                    "zero_unassigned", "Atlas label 0 (unassigned)", "", "Unassigned", "Unknown", False)
            else:
                entry = key.loc[str(index)]
                conflict = not int(entry.First_Level) <= level <= int(entry.Last_Level)
                status = "key_level_conflict" if conflict else "mapped"
                name, abbreviation = entry.Full_Name, entry.Abbreviation
                domain = "Cortex" if abbreviation.startswith("C") else "Subcortex"
                side = abbreviation[1]
                if conflict:
                    conflicts.append({"level": level, "ARMIndex": index, "voxels": int(size)})
            target = f"L{level}:{status}:{index}"
            expected_targets.add(target)
            row = fmri.loc[target]
            require((row.OfficialFullName, row.Abbreviation, row.Domain, row.Hemisphere, row.TargetStatus)
                    == (name, abbreviation, domain, side, status), "Official observed ARM target metadata differs")
            require(row.KeyRangeConflict == str(conflict), "Official key-range conflict lost")
            require(int(float(row.ARMIndex)) == index and int(row.Level) == level, "ARM target/level identity differs")
            covered = int(counts[index]) if index < len(counts) else 0
            require(int(row.region_voxels) == int(size) and int(row.covered_voxels) == covered, "Actual regional coverage count differs")
            close(row.ActualVolumeMm3, size * voxel_volume)
            close(row.coverage_fraction, covered / size)
            if covered:
                close(row.mean_signed_contrast, sums[index] / covered)
                close(row.median_signed_contrast, medians[index][0])
                close(row.median_T_descriptive_only, medians[index][1])
            else:
                require(row.mean_signed_contrast == row.median_signed_contrast == row.median_T_descriptive_only == "", "Absent coverage zero-filled")
                no_coverage.append(target)
        require(sum(int(fmri.loc[t, "covered_voxels"]) for t in expected_targets
                    if int(fmri.loc[t, "Level"]) == level) == int(mask.sum()), "Each level must conserve coverage independently")
        coverage_by_level.append({"level": level, "covered_voxels": int(mask.sum()),
                                  "positive_ARM_covered_voxels": int(np.sum((labels > 0) & mask))})
    require(set(fmri.index) == expected_targets, "Missing/extra observed ARM rows")

    roots = {
        "endpoint": ROOT / "group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints",
        "axon": ROOT / "group_analysis/evolution_20261008/arm_mapping_20261009/main/axons",
        "end_branch": ROOT / "notes/region_analysis_review_20261009/axon_end_branches_20261009/selected462_ARM",
    }
    ledgers = {}
    totals = {}
    checked_groups, checked_animals, scalar_checks, known_zero, missing_values = 0, 0, 0, 0, 0
    max_equal_animal_regional_error = 0.0
    group_source_ids = {}
    for kind, folder in roots.items():
        record = json.loads((folder / "run_provenance.json").read_text())
        manifest = read(folder / ("input_ledger.csv" if kind == "end_branch" else record["preserved_manifest"]))
        uid = identities(manifest)
        manifest.index = uid
        ledgers[kind] = manifest
        metadata = manifest.groupby("Subregion", sort=False).first()
        if kind == "endpoint":
            qc = read(folder / "per_neuron_qc.csv")
            qc.index = identities(qc)
            require(set(qc.index) == set(uid), "Endpoint QC exact identity differs")
            eligible = qc.loc[uid, "n_computable"].eq("1")
        elif kind == "end_branch":
            qc = read(folder / "per_neuron_measurements.csv")
            qc.index = identities(qc)
            require(set(qc.index) == set(uid), "End-branch QC exact identity differs")
            eligible = qc.loc[uid, "computable"].eq("True")
        else:
            eligible = pd.Series(True, index=uid)
        require(len(manifest) == (462 if kind == "end_branch" else 436), "Original full ledger changed")
        for source in manifest.to_dict("records"):
            require(sha(source["SWCPath"]) == source["SWCSHA256"], "Selected exact SWC source changed")
        mapped_ids = set(manifest.index[manifest.SourceARMStatus.eq("mapped")])
        require(len(mapped_ids) == 410, "Mapped named source selection differs")
        group_source_ids[kind] = mapped_ids
        totals[kind + "_selected_named"] = len(mapped_ids)
        if kind != "axon":
            totals[kind + "_eligible_named"] = int(eligible.loc[list(mapped_ids)].sum())
        groups = [entry for entry in record["group_maps"]
                  if metadata.loc[entry["Subregion"], "SourceARMStatus"] == "mapped"]
        require(len(groups) == 15 and set(joint.SourceARMGroup) == {e["Subregion"] for e in groups}, "Named source-group coverage differs")
        field = "count" if kind == "endpoint" else "length"
        metric = {"endpoint": "candidate_ends_per_eligible_neuron_in_coverage",
                  "axon": "axon_length_mm_per_selected_neuron_in_coverage",
                  "end_branch": "end_branch_length_mm_per_eligible_neuron_in_coverage"}[kind]
        for entry in groups:
            group = entry["Subregion"]
            members = manifest[manifest.Subregion.eq(group)]
            local_eligible = eligible.loc[members.index]
            contributing = sorted(set(members.loc[local_eligible, "AnimalID"]))
            require(entry["contributing_animals"] == contributing and entry["n_animals"] == len(contributing),
                    "Actual contributing-animal denominator differs")
            selected = len(members)
            require(selected == entry.get("n_selected", entry.get("n_neurons", entry.get("selected_neurons"))),
                    "Actual selected-neuron denominator differs")
            if kind != "axon":
                require(int(local_eligible.sum()) == entry.get("n_computable", entry.get("eligible_neurons")),
                        "Actual eligible-neuron denominator differs")
            actual_rows = joint[joint.SourceARMGroup.eq(group)].set_index("TargetID", verify_integrity=True)
            source = metadata.loc[group]
            require(set(actual_rows.index) == expected_targets, "Missing joint target rows")
            for output_field, manifest_field in [("SourceARMIndex", "ARMIndex"),
                    ("SourceARMFullName", "ARMFullName"), ("SourceHemisphere", "Hemisphere")]:
                require(actual_rows[output_field].eq(source[manifest_field]).all(), "Source label/side/index differs")
            for output_field, value in [(kind + "_contributing_animals", len(contributing)), (kind + "_selected_neurons", selected)]:
                require(actual_rows[output_field].eq(str(value)).all(), "Copied group denominator differs")
            if kind != "axon":
                require(actual_rows[kind + "_eligible_neurons"].eq(str(int(local_eligible.sum()))).all(), "Copied eligible denominator differs")
            path = folder / entry[field + "_path"]
            require(sha(path) == entry[field + "_sha256"] == inputs[str(path.resolve())], "Group source map hash differs")
            image = nib.load(path)
            grid(image, reference)
            values = image.get_fdata()
            require(np.isfinite(values).all() and np.min(values) >= 0, "Additive group map invalid")
            weights = values[mask]
            regional = [np.bincount(labels, weights=weights) for labels in covered_labels]
            animal_sums = [np.zeros_like(arr) for arr in regional]
            animals = [animal for animal in record["animal_maps"]
                       if animal["Subregion"] == group and animal["AnimalID"] in contributing]
            require(len(animals) == len(contributing), "Missing/duplicate contributing animal maps")
            for animal in animals:
                animal_members = members[members.AnimalID.eq(animal["AnimalID"])]
                require(len(animal_members) == animal.get("n_selected", animal.get("selected_neurons")), "Animal selected denominator differs")
                if kind != "axon":
                    require(int(eligible.loc[animal_members.index].sum()) == animal.get("n_computable", animal.get("eligible_neurons")),
                            "Animal conditional eligible denominator differs")
                animal_path = folder / animal[field + "_path"]
                require(sha(animal_path) == animal[field + "_sha256"], "Original animal map hash differs")
                animal_image = nib.load(animal_path)
                grid(animal_image, reference)
                animal_weights = animal_image.get_fdata()[mask]
                for level, labels in enumerate(covered_labels):
                    arr = np.bincount(labels, weights=animal_weights)
                    animal_sums[level][:len(arr)] += arr / len(contributing)
                checked_animals += 1
            for level in range(1, 7):
                expected = regional[level - 1]
                animal_mean = animal_sums[level - 1]
                max_equal_animal_regional_error = max(max_equal_animal_regional_error,
                                                      float(np.max(np.abs(expected - animal_mean))))
                np.testing.assert_allclose(expected, animal_mean, rtol=2e-6, atol=1e-7)
                for target, row in fmri[fmri.Level.eq(str(level))].iterrows():
                    index = int(float(row.ARMIndex))
                    result = (expected[index] if index < len(expected) else 0.0) if int(row.covered_voxels) else np.nan
                    try:
                        close(actual_rows.loc[target, metric], result)
                    except ValueError as error:
                        raise ValueError(f"{kind}/{group}/{target}: {error}") from error
                    scalar_checks += 1
                    known_zero += int(np.isfinite(result) and result == 0)
                    missing_values += int(not np.isfinite(result))
            checked_groups += 1
            print(f"independent covered ARM summaries: {kind}/{group}", flush=True)
    require(group_source_ids["endpoint"] == group_source_ids["axon"] == group_source_ids["end_branch"], "Named source exact UIDs differ across map families")
    common = sorted(group_source_ids["endpoint"])
    for field in ["AnimalID", "SampleID", "NeuronID", "SWCSHA256", "SWCPath", "Hemisphere",
                  "ARMIndex", "ARMFullName", "ARMAbbreviation", "SourceARMStatus", "EvidenceSourceGroup",
                  "SourceCohort", "Henry_coarse_INS_visual_evidence", "CoordinateFrame", "IndexScaleUm",
                  "ReferenceSHA256", "AtlasSHA256", "AtlasKeySHA256"]:
        for kind in ["axon", "end_branch"]:
            require(ledgers["endpoint"].loc[common, field].equals(ledgers[kind].loc[common, field]),
                    "Source identity/evidence/frame metadata differs: " + field)
    require(totals == {"endpoint_selected_named": 410, "endpoint_eligible_named": 378,
                      "axon_selected_named": 410, "end_branch_selected_named": 410,
                      "end_branch_eligible_named": 378}, "Joint source denominator accounting differs")
    cuts = provenance["display"]["cuts_xyz"]
    require(cuts == [56, 200, 87], "Matched MRI-plane cuts differ")
    color_limit = float(np.percentile(np.abs(contrast), 99)) or 1.0
    close(provenance["display"]["contrast_color_limit"], color_limit)
    anatomy_path = next(p for p in inputs if "normalized" in Path(p).name.lower()
                        or "warped" in Path(p).name.lower() or "taskA_ana_NMT" in Path(p).name)
    anatomy = nib.load(local(anatomy_path))
    grid(anatomy, reference)
    anatomy_values = anatomy.get_fdata()
    close(provenance["display"]["normalized_anatomy_display_threshold"],
          np.percentile(anatomy_values[anatomy_values > 0], 99) * .1)
    for path, expected in inputs.items():
        require(sha(path) == expected, "Source changed during independent readback")
    for name, expected in provenance["outputs"].items():
        require(sha(HERE / name) == expected, "Frozen output changed during readback")
    report = {"status": "independent_CM033_saved_regional_and_joint_readback_passed",
              "checked_at": datetime.now().astimezone().isoformat(), "producer_functions_imported": False,
              "reader_sha256": sha(__file__), "integration_provenance_sha256": sha(provenance_path),
              "verified_source_sha256": inputs, "verified_outputs": provenance["outputs"],
              "fmri_rows": len(fmri), "joint_rows": len(joint), "all_six_levels_checked": True,
              "group_maps_checked": checked_groups, "animal_maps_checked": checked_animals,
              "joint_metric_scalars_checked": scalar_checks, "known_covered_zero_values": known_zero,
              "uncovered_missing_values": missing_values, "source_neuron_denominators": totals,
              "mapped_source_groups": 15, "same_exact_mapped_source_UIDs_all_three_families": True,
              "whole_end_branch_ledger_retained": 462, "unassigned_source_neurons_kept_QC": 52,
              "max_equal_animal_regional_sum_abs_error": max_equal_animal_regional_error,
              "coverage_by_level": coverage_by_level, "targets_without_coverage": no_coverage,
              "key_level_conflicts_retained": conflicts, "genuine_model_coverage_voxels": int(mask.sum()),
              "display_arithmetic": {"cuts_xyz": cuts, "linear_signed_color_bound": color_limit,
                  "NMT_MRI_gray_window": [75, 920], "anatomy_threshold_display_only": True,
                  "actual_PNG_viewed": True,
                  "actual_PNG_visual_review": "All six matched planes, titles, signed colorbar and legend are readable and inside image bounds. Cyan anatomy display contour follows the shown template outline; green genuine analysis coverage is partial and includes some displayed support beyond the template brain outline. This is provisional coverage/geometry evidence, not accepted physical orientation or anatomical registration.",
                  "PNG_sha256": provenance["outputs"]["cm033_alignment_and_response_qc.png"]},
              "limits": ["Conditional descriptive regional sums only; no animal pairing, causal/monosynaptic inference or biological terminal acceptance.",
                         "Interpolated T medians are descriptive and are not regional t statistics or new p values.",
                         "Pinned reference-image compatibility does not merge NMT/atlas versions or establish native resolution or physical laterality.",
                         "Warped genuine model support defines validity; zero fill and boundary mixtures outside it are invalid response evidence."],
              "anatomical_acceptance": False, "new_warp_or_GLM": False}
    with (HERE / "independent_cm033_regional_readback_20261009.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ["status", "fmri_rows", "joint_rows", "group_maps_checked",
                "animal_maps_checked", "joint_metric_scalars_checked", "known_covered_zero_values", "uncovered_missing_values"]}))


if __name__ == "__main__":
    main()
