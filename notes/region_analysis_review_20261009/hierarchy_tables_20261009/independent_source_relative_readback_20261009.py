"""Independent paired-target permutation and saved-cut check, no producer imports."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "notes/region_analysis_review_20261009"
RUN = BASE / "clustering_20261009/ARM6_source_relative_sensitivity"
EXPORT = BASE / "hierarchy_tables_20261009/combined_arm_projection_tables_462"
PRIMARY = BASE / "clustering_20261009/arm_L6_relative_profiles_selected462"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main():
    provenance = json.loads((RUN / "sensitivity_provenance.json").read_text())
    assert provenance["status"] == "software_verified_feature_transformation_sensitivity"
    for name in ["primary_provenance", "export_provenance", "manifest"]:
        path = PRIMARY / "run_provenance.json" if name == "primary_provenance" else provenance[name]
        assert sha(path) == provenance[name + "_SHA256"]
    for path, expected in provenance["output_hashes"].items():
        assert sha(RUN / path) == expected
    primary = json.loads((PRIMARY / "run_provenance.json").read_text())
    for path, expected in primary["output_hashes"].items():
        assert sha(PRIMARY / path) == expected
    for path, expected in primary["input_hashes"].items():
        assert sha(path) == expected
    summary = read(EXPORT / "neuron_summary.csv").set_index("NeuronUID")
    targets = read(EXPORT / "targets.csv").set_index("TargetID")
    targets = targets[targets.Level.eq("6") & targets.TargetStatus.eq("mapped")]
    mapping = read(RUN / "official_ARM_homolog_feature_mapping.csv")
    assert len(targets) == len(mapping) == 684
    assert mapping.FeatureID.is_unique
    seen = set()
    for row in mapping.itertuples(index=False):
        left, right = targets.loc[row.LeftTargetID], targets.loc[row.RightTargetID]
        assert left.Hemisphere == "L" and right.Hemisphere == "R"
        assert left.Domain == right.Domain == row.Domain
        assert left.Abbreviation[:2] in ["CL", "SL"] and right.Abbreviation[:2] in ["CR", "SR"]
        assert left.Abbreviation[2:] == right.Abbreviation[2:]
        assert left.OfficialFullName[2:] == right.OfficialFullName[2:]
        assert left.OfficialFullName.startswith(left.Abbreviation[:2] + "_")
        assert right.OfficialFullName.startswith(right.Abbreviation[:2] + "_")
        assert row.LeftOfficialFullName == left.OfficialFullName and row.RightOfficialFullName == right.OfficialFullName
        assert row.LeftAbbreviation == left.Abbreviation and row.RightAbbreviation == right.Abbreviation
        assert float(row.LeftARMIndex) == float(left.ARMIndex) and float(row.RightARMIndex) == float(right.ARMIndex)
        assert row.RelativeSide in ["ipsilateral", "contralateral"]
        assert (row.LeftTargetID, row.RightTargetID, row.RelativeSide) not in seen
        seen.add((row.LeftTargetID, row.RightTargetID, row.RelativeSide))
    assert len({(a, b) for a, b, _ in seen}) == 342
    assert {a for a, _, _ in seen} | {b for _, b, _ in seen} == set(targets.index)
    measures = read(EXPORT / "per_neuron_regional_measures.csv")
    measures = measures[measures.TargetID.isin(targets.index)]
    saved = read(RUN / "source_relative_assignments.csv").set_index("NeuronUID")
    absolute = read(PRIMARY / "all462_feature_eligibility_and_assignments.csv").set_index("NeuronUID")
    assert saved.index.equals(summary.index)
    for col in ["AnimalID", "SampleID", "NeuronID", "Hemisphere"]:
        assert saved[col].equals(summary[col])
    diagnostics = pd.read_csv(RUN / "source_relative_diagnostics.csv")
    comparison = pd.read_csv(RUN / "absolute_vs_relative_comparison.csv")
    results = []
    for family, value, eligible_name in [("axon", "AxonTemplateLengthMm", "AxonProfileEligible"),
                                         ("endpoint", "CandidateEndpointCount", "EndpointProfileEligible")]:
        eligibility = absolute[eligible_name].eq("True")
        assert saved[eligible_name].equals(absolute[eligible_name])
        raw = measures.pivot(index="NeuronUID", columns="TargetID", values=value).reindex(index=summary.index, columns=targets.index)
        raw = raw.apply(pd.to_numeric).fillna(0).loc[eligibility]
        source_left = summary.loc[eligibility].Hemisphere.eq("L").to_numpy()
        assert summary.loc[eligibility].Hemisphere.isin(["L", "R"]).all()
        columns = []
        for row in mapping.itertuples(index=False):
            choose_left = source_left if row.RelativeSide == "ipsilateral" else ~source_left
            columns.append(np.where(choose_left, raw[row.LeftTargetID], raw[row.RightTargetID]))
        permuted = np.array(columns).T
        np.testing.assert_allclose(permuted.sum(axis=1), raw.sum(axis=1), rtol=0, atol=1e-10)
        np.testing.assert_array_equal(np.sort(permuted, axis=1), np.sort(raw.to_numpy(), axis=1))
        p = permuted / permuted.sum(axis=1)[:, None]
        distance = pdist(np.sqrt(p)) / np.sqrt(2)
        tree = linkage(distance, method="average")
        for k in range(2, 9):
            labels = fcluster(tree, k, criterion="maxclust")
            np.testing.assert_array_equal(labels, saved.loc[eligibility, f"{family}_source_relative_k{k}"].astype(float).astype(int))
            row = diagnostics[diagnostics.family.eq(family) & diagnostics.requested_k.eq(k)].iloc[0]
            np.testing.assert_allclose(silhouette_score(squareform(distance), labels, metric="precomputed"), row.silhouette, atol=1e-12)
            for field in ["AnimalID", "Hemisphere", "EvidenceSourceGroup", "ARMFullName"]:
                np.testing.assert_allclose(normalized_mutual_info_score(summary.loc[eligibility, field], labels), row[field + "_cluster_NMI"], atol=1e-12)
            original = absolute.loc[eligibility, f"all_{family}_hellinger_k{k}"].astype(float).astype(int)
            ari = adjusted_rand_score(original, labels)
            expected = comparison[comparison.family.eq(family) & comparison.requested_k.eq(k)].iloc[0]
            assert expected.exact_common_UIDs == len(labels)
            np.testing.assert_allclose(ari, expected.absolute_vs_source_relative_ARI, atol=1e-12)
            results.append({"family": family, "k": k, "n": len(labels),
                            "sizes": np.unique(labels, return_counts=True)[1].tolist(),
                            "side_NMI": float(row.Hemisphere_cluster_NMI),
                            "absolute_vs_relative_ARI": ari})
    report = {"status": "independent_source_relative_saved_readback_passed",
              "reader_sha256": sha(__file__), "sensitivity_provenance_sha256": sha(RUN / "sensitivity_provenance.json"),
              "primary_provenance_sha256": sha(PRIMARY / "run_provenance.json"),
              "official_same_domain_bilateral_pairs": 342, "features": 684,
              "saved_partitions_verified": 14, "checks": results,
              "production_imports": False, "primary_outputs_hashes_unchanged": True,
              "limits": ["Side is the existing current-mask declaration, not accepted anatomical registration",
                         "Permutation preserves every per-neuron value/total and all official ARM names",
                         "No repeated-subset rerun here; all fits remain neuron weighted",
                         "Absolute/source-relative discrepancy is representation sensitivity, not a cell-type validation"]}
    with Path(__file__).with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
