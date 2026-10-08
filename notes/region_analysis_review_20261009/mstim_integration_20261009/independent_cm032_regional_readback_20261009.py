"""Independent direct NIfTI/CSV readback; no project analysis imports or warps."""
from pathlib import Path
import hashlib
import json

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
RUN = BASE / "cm032_regional_comparison"
ATLAS_BASE = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def close(actual, expected):
    actual = pd.to_numeric(pd.Series(actual).replace("", np.nan)).to_numpy(float)
    np.testing.assert_allclose(actual, np.asarray(expected, float).reshape(-1),
                               rtol=1e-9, atol=1e-11, equal_nan=True)


def grid(image, reference, atlas=False):
    assert image.shape[:3] == reference.shape
    assert image.header.get_xyzt_units()[0] == "mm"
    np.testing.assert_allclose(image.affine, reference.affine, rtol=0, atol=1e-6)
    assert int(image.header["sform_code"]) > 0 or int(image.header["qform_code"]) > 0
    if atlas:
        assert image.shape == (*reference.shape, 1, 6)
    else:
        assert image.shape == reference.shape


def main():
    provenance_path = RUN / "integration_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for path, expected in provenance["source_sha256"].items():
        assert sha(path) == expected, path
    for path, expected in provenance["outputs"].items():
        assert sha(RUN / path) == expected, path
    assert provenance["status"] == "provisional_regional_integration_completed"
    assert provenance["anatomical_acceptance"] is False
    warp = json.loads((BASE / "cm032_warp_reproduction/warp_reproduction.json").read_text())
    reference = nib.load(ATLAS_BASE / "NMT_v2.1_sym_SS.nii.gz")
    atlas_image = nib.load(ATLAS_BASE / "ARM_in_NMT_v2.1_sym.nii.gz")
    grid(atlas_image, reference, atlas=True)
    atlas = np.asarray(atlas_image.dataobj)[:, :, :, 0, :]
    mask_image = nib.load(warp["outputs"]["analysis_coverage"]["path"])
    contrast_image = nib.load(warp["outputs"]["reproduced_contrast"]["path"])
    statistic_path = next(p for p in provenance["source_sha256"] if p.endswith("spmT_0001_in_NMT.nii.gz"))
    statistic_image = nib.load(statistic_path)
    for image in [mask_image, contrast_image, statistic_image]:
        grid(image, reference)
    mask_values = np.asarray(mask_image.dataobj)
    assert set(np.unique(mask_values)) == {0, 1}
    mask = mask_values.astype(bool)
    contrast = contrast_image.get_fdata()[mask]
    statistic = statistic_image.get_fdata()[mask]
    assert np.isfinite(contrast).all() and np.isfinite(statistic).all()
    assert int(mask.sum()) == provenance["coverage_voxels"] == 5015104
    native_path = next(p for p in warp["source_sha256"] if p.endswith("maps\\con_0001.nii") or p.endswith("maps/con_0001.nii"))
    assert sha(native_path) == warp["source_sha256"][native_path]
    native = nib.load(native_path)
    close(native.header.get_zooms()[:3], provenance["native_fMRI_sampling_mm"])
    close(reference.header.get_zooms()[:3], provenance["NMT_sampling_mm"])
    key = pd.read_csv(ROOT / "atlas/ARM_key_all.txt", sep="\t", dtype=str, keep_default_na=False).set_index("Index")
    assert key.index.is_unique
    fmri = read(RUN / "cm032_signed_response_by_ARM_level.csv").set_index("TargetID")
    joint = read(RUN / "cm032_projectome_common_coverage.csv")
    assert fmri.index.is_unique and len(fmri) == 1676 and len(joint) == 25140
    assert not joint.duplicated(["SourceARMGroup", "TargetID"]).any()
    assert fmri.registration_status.eq("unaccepted").all() and fmri.subject.eq("cm032").all()
    assert joint.same_animal_pairing.eq("unknown").all()
    assert joint.comparison_status.eq("conditional_numerical_summary_registration_unaccepted").all()
    for field in fmri.columns:
        assert np.array_equal(joint[field].to_numpy(), fmri.loc[joint.TargetID, field].to_numpy()), field
    volume = abs(float(np.linalg.det(reference.affine[:3, :3])))
    expected_targets = set()
    covered_labels = []
    coverage_by_level = []
    conflicts = []
    no_coverage = []
    for level in range(1, 7):
        labels = atlas[:, :, :, level - 1]
        unique, region_sizes = np.unique(labels, return_counts=True)
        local = labels[mask].astype(np.int64)
        covered_labels.append(local)
        order = np.argsort(local, kind="stable")
        sorted_labels = local[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1, len(order)]
        grouped = {int(sorted_labels[a]): (int(b - a), float(np.median(contrast[order[a:b]])),
                                          float(np.median(statistic[order[a:b]])))
                   for a, b in zip(boundaries[:-1], boundaries[1:])}
        counts = np.bincount(local)
        contrast_sum = np.bincount(local, weights=contrast)
        for index, size in zip(unique, region_sizes):
            index = int(index)
            if index == 0:
                status, name, abbreviation, domain, side = "zero_unassigned", "Atlas label 0 (unassigned)", "", "Unassigned", "Unknown"
                conflict = False
            else:
                entry = key.loc[str(index)]
                conflict = not int(entry.First_Level) <= level <= int(entry.Last_Level)
                status = "key_level_conflict" if conflict else "mapped"
                name, abbreviation = entry.Full_Name, entry.Abbreviation
                domain = "Cortex" if abbreviation[0] == "C" else "Subcortex"
                side = abbreviation[1]
                if conflict:
                    conflicts.append({"level": level, "ARMIndex": index, "voxels": int(size)})
            target = f"L{level}:{status}:{index}"
            expected_targets.add(target)
            row = fmri.loc[target]
            assert (row.OfficialFullName, row.Abbreviation, row.Domain, row.Hemisphere, row.TargetStatus) == (name, abbreviation, domain, side, status)
            assert row.KeyRangeConflict == str(conflict)
            assert int(float(row.ARMIndex)) == index and int(row.Level) == level
            count = int(counts[index]) if index < len(counts) else 0
            assert (int(row.region_voxels), int(row.covered_voxels)) == (size, count)
            close(row.ActualVolumeMm3, size * volume)
            close(row.coverage_fraction, count / size)
            if count:
                assert grouped[index][0] == count
                close(row.mean_signed_contrast, contrast_sum[index] / count)
                close(row.median_signed_contrast, grouped[index][1])
                close(row.median_T_descriptive_only, grouped[index][2])
            else:
                assert row.mean_signed_contrast == row.median_signed_contrast == row.median_T_descriptive_only == ""
                no_coverage.append(target)
        positive = labels > 0
        coverage_by_level.append({"level": level, "all_covered_voxels": int(len(local)),
                                  "positive_ARM_voxels": int(positive.sum()),
                                  "positive_ARM_covered_voxels": int((positive & mask).sum()),
                                  "positive_ARM_coverage_fraction": float((positive & mask).sum() / positive.sum())})
    assert set(fmri.index) == expected_targets
    projectome = ROOT / "group_analysis/evolution_20261008/arm_mapping_20261009/main"
    selected_totals = {}
    checked_maps = 0
    for kind, folder, field, metric in [
        ("endpoint", "endpoints", "count", "candidate_ends_per_eligible_neuron_in_coverage"),
        ("axon", "axons", "length", "axon_length_mm_per_selected_neuron_in_coverage")]:
        record = json.loads((projectome / folder / "run_provenance.json").read_text())
        manifest = read(projectome / folder / record["preserved_manifest"])
        metadata = manifest.groupby("Subregion").first()
        groups = [e for e in record["group_maps"] if metadata.loc[e["Subregion"], "SourceARMStatus"] == "mapped"]
        assert len(groups) == 15
        assert set(joint.SourceARMGroup) == {e["Subregion"] for e in groups}
        selected_totals[kind] = sum(e.get("n_selected", e.get("n_neurons")) for e in groups)
        if kind == "endpoint":
            selected_totals["endpoint_eligible"] = sum(e["n_computable"] for e in groups)
        for entry in groups:
            path = projectome / folder / entry[field + "_path"]
            assert sha(path) == entry[field + "_sha256"] == provenance["source_sha256"][str(path.resolve())]
            image = nib.load(path)
            grid(image, reference)
            values = image.get_fdata()
            assert np.isfinite(values).all() and np.min(values) >= 0
            measured_values = values[mask]
            actual = joint[joint.SourceARMGroup.eq(entry["Subregion"])].set_index("TargetID")
            source = metadata.loc[entry["Subregion"]]
            for col, original in [("SourceARMFullName", "ARMFullName"), ("SourceHemisphere", "Hemisphere"), ("SourceARMIndex", "ARMIndex")]:
                assert actual[col].eq(source[original]).all()
            assert actual[kind + "_contributing_animals"].eq(str(entry["n_animals"])).all()
            assert actual[kind + "_selected_neurons"].eq(str(entry.get("n_selected", entry.get("n_neurons")))).all()
            if kind == "endpoint":
                assert actual.endpoint_eligible_neurons.eq(str(entry["n_computable"])).all()
            for level, labels in enumerate(covered_labels, 1):
                regional = np.bincount(labels, weights=measured_values)
                for target, row in fmri[fmri.Level.eq(str(level))].iterrows():
                    index = int(float(row.ARMIndex))
                    expected = (regional[index] if index < len(regional) else 0) if int(row.covered_voxels) else np.nan
                    close(actual.loc[target, metric], expected)
            checked_maps += 1
    assert selected_totals == {"endpoint": 410, "endpoint_eligible": 378, "axon": 410}
    report = {"status": "independent_CM032_saved_regional_readback_passed",
              "reader_sha256": sha(__file__), "integration_provenance_sha256": sha(provenance_path),
              "production_imports": False, "fmri_rows": len(fmri), "joint_rows": len(joint),
              "saved_group_maps": checked_maps, "source_groups": 15, "source_neuron_denominators": selected_totals,
              "coverage_by_level": coverage_by_level, "targets_without_coverage": no_coverage,
              "key_level_conflicts_retained": conflicts, "native_sampling_mm": list(map(float, native.header.get_zooms()[:3])),
              "NMT_sampling_mm": list(map(float, reference.header.get_zooms()[:3])),
              "checks": ["All source/output hashes", "All six direct ARM level labels and official key metadata",
                         "Covered and total voxel counts, means, contrast medians and descriptive T medians",
                         "All saved endpoint count and axon length values over exactly the same coverage",
                         "All source labels, selected/eligible/animal counts and copied fMRI metadata"],
              "limits": ["Numerical NMT spatial summaries only; registration and stimulation coordinates unaccepted",
                         "Saved group maps average animals equally; metric labels are not pooled-neuron estimates",
                         "Existing T image transform not independently reproduced here; median T is not a regional t test",
                         "No source-sample pairing, biological terminal, independent-voxel or causal claim",
                         "NMT upsampling does not establish native fMRI spatial resolution; no new ANTs run"]}
    with Path(__file__).with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ["checks", "limits", "targets_without_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
