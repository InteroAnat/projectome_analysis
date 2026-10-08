"""Read saved clustering independently; import no project analysis modules."""
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
EXPORT = BASE / "hierarchy_tables_20261009/combined_arm_projection_tables_462"
RUN = BASE / "clustering_20261009/arm_L6_relative_profiles_selected462"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float),
                               rtol=1e-10, atol=1e-11, equal_nan=True)


def main():
    provenance = json.loads((RUN / "run_provenance.json").read_text())
    assert provenance["status"] == "exploratory_profile_clustering_completed_software_only"
    for path, expected in provenance["input_hashes"].items():
        assert sha(path) == expected, path
    for path, expected in provenance["selected_source_hashes"].items():
        assert sha(path) == expected, path
    for path, expected in provenance["output_hashes"].items():
        assert sha(RUN / path) == expected, path
    summary = read(EXPORT / "neuron_summary.csv").set_index("NeuronUID")
    saved = read(RUN / "all462_feature_eligibility_and_assignments.csv").set_index("NeuronUID")
    assert summary.index.is_unique and saved.index.equals(summary.index)
    targets = read(EXPORT / "targets.csv")
    l6 = targets[targets.Level.eq("6")].set_index("TargetID")
    positive = l6[l6.TargetStatus.eq("mapped") & l6.ARMIndex.ne("0")]
    assert len(positive) == 684 and positive.Hemisphere.value_counts().to_dict() == {"L": 342, "R": 342}
    feature_dictionary = read(RUN / "exclusive_ARM6_feature_dictionary.csv").set_index("TargetID")
    assert feature_dictionary.index.equals(positive.index)
    close(feature_dictionary.ARMIndex, positive.ARMIndex)
    assert feature_dictionary.drop(columns="ARMIndex").equals(positive.drop(columns="ARMIndex"))
    measures = read(EXPORT / "per_neuron_regional_measures.csv")
    measures = measures[measures.Level.eq("6")]
    assert not measures.duplicated(["NeuronUID", "TargetID"]).any()
    axon = pd.DataFrame(0.0, index=summary.index, columns=l6.index)
    ends = axon.copy()
    eligible = summary.EndpointEligible.eq("True")
    ends.loc[~eligible] = np.nan
    for row in measures.itertuples(index=False):
        axon.loc[row.NeuronUID, row.TargetID] = float(row.AxonTemplateLengthMm)
        if eligible.loc[row.NeuronUID]:
            count = float(row.CandidateEndpointCount)
            assert count >= 0 and count.is_integer() and int(row.EndpointPresence) == int(count > 0)
            ends.loc[row.NeuronUID, row.TargetID] = count
        else:
            assert row.CandidateEndpointCount == row.EndpointPresence == ""
    totals = {"AxonLengthAllStatusesMm": axon.sum(axis=1),
              "AxonLengthMappedL6Mm": axon[positive.index].sum(axis=1),
              "CandidateEndsAllStatuses": ends.sum(axis=1, min_count=1),
              "CandidateEndsMappedL6": ends[positive.index].sum(axis=1, min_count=1)}
    for prefix, data in [("Axon", axon), ("CandidateEnds", ends)]:
        for status, word in [("zero_unassigned", "AtlasBackground"), ("out_of_FOV", "OutOfFOV")]:
            columns = l6[l6.TargetStatus.eq(status)].index
            name = prefix + word + ("LengthMm" if prefix == "Axon" else "")
            totals[name] = data[columns].sum(axis=1, min_count=1)
    for name, expected in totals.items():
        close(pd.to_numeric(saved[name].replace("", np.nan)), expected)
    for family, numerator, denominator in [("Axon", "AxonLengthMappedL6Mm", "AxonLengthAllStatusesMm"),
                                           ("CandidateEnds", "CandidateEndsMappedL6", "CandidateEndsAllStatuses")]:
        close(pd.to_numeric(saved[family + "MappedFraction"].replace("", np.nan)),
              totals[numerator] / totals[denominator].replace(0, np.nan))
    axon_mask = totals["AxonLengthMappedL6Mm"] > 0
    end_mask = eligible & (totals["CandidateEndsMappedL6"] > 0)
    assert np.array_equal(saved.AxonProfileEligible.eq("True"), axon_mask)
    assert np.array_equal(saved.EndpointProfileEligible.eq("True"), end_mask)
    assert (len(summary), int(eligible.sum()), int(axon_mask.sum()), int(end_mask.sum())) == (462, 429, 428, 425)
    diagnostics = pd.read_csv(RUN / "candidate_k_diagnostics.csv")
    profiles = read(RUN / "cluster_target_profiles.csv")
    comparisons = []
    checked_profile_values = 0
    for name, spec in provenance["analyses"].items():
        endpoint = "endpoint" in name
        mask = (end_mask if endpoint else axon_mask).copy()
        if name.startswith("Henry_only"):
            mask &= summary.Henry_coarse_INS_visual_evidence.eq("True")
        raw = (ends if endpoint else axon).loc[mask, positive.index]
        metadata = summary.loc[mask]
        assert len(raw) == spec["n_profiles"]
        proportions = raw.to_numpy() / raw.sum(axis=1).to_numpy()[:, None]
        mode = spec["representation"]
        if mode == "hellinger":
            condensed = pdist(np.sqrt(proportions)) / np.sqrt(2)
        elif mode == "logrelative":
            features = np.log1p(1000 * proportions)
            condensed = pdist(features / np.linalg.norm(features, axis=1)[:, None])
        else:
            assert mode == "presence_jaccard"
            condensed = pdist(raw.to_numpy() > 0, metric="jaccard")
        matrix = squareform(condensed)
        tree = linkage(condensed, method="average")
        rows = diagnostics[diagnostics.analysis.eq(name)]
        for row in rows.itertuples(index=False):
            k = int(row.requested_k)
            labels = fcluster(tree, k, criterion="maxclust")
            stored = saved.loc[mask, name + "_k" + str(k)].astype(float).astype(int).to_numpy()
            assert adjusted_rand_score(labels, stored) == 1
            assert np.array_equal(labels, stored), "Unexpected arbitrary cluster-label permutation"
            ids, sizes = np.unique(labels, return_counts=True)
            assert (len(ids), min(sizes), max(sizes), int((sizes == 1).sum())) == (
                row.realized_k, row.smallest_cluster, row.largest_cluster, row.singleton_clusters)
            close(row.silhouette, silhouette_score(matrix, labels, metric="precomputed"))
            for column in ["AnimalID", "SampleID", "Hemisphere", "EvidenceSourceGroup", "ARMFullName"]:
                close(getattr(row, column + "_cluster_NMI"),
                      normalized_mutual_info_score(metadata[column], labels))
            if k == spec["display_cut_k"]:
                for cluster in ids:
                    member = labels == cluster
                    p = proportions[member]
                    animal = metadata.AnimalID.to_numpy()[member]
                    equal = np.array([p[animal == a].mean(axis=0) for a in np.unique(animal)]).mean(axis=0)
                    actual = profiles[profiles.analysis.eq(name) & profiles.cluster.eq(str(cluster))].set_index("TargetID").loc[positive.index]
                    assert actual.requested_k.eq(str(k)).all()
                    assert actual.n_neurons.eq(str(int(member.sum()))).all()
                    assert actual.n_animals.eq(str(len(np.unique(animal)))).all()
                    close(actual.mean_neuron_relative_profile, p.mean(axis=0))
                    close(actual.equal_contributing_animal_relative_profile, equal)
                    close(actual.target_presence_fraction, (raw.to_numpy()[member] > 0).mean(axis=0))
                    assert actual.ARMFullName.equals(positive.OfficialFullName.rename("ARMFullName"))
                    assert actual.TargetHemisphere.equals(positive.Hemisphere.rename("TargetHemisphere"))
                    checked_profile_values += len(actual) * 3
            comparisons.append({"analysis": name, "k": k, "n_profiles": len(raw),
                                "partition_ARI": 1.0, "sizes": sizes.tolist()})
    assert len(comparisons) == 49
    singleton_uid = "252790::121.swc"
    assert totals["CandidateEndsAllStatuses"].loc[singleton_uid] == 1
    only = ends.loc[singleton_uid, positive.index]
    only = positive.loc[only[only > 0].index]
    assert len(only) == 1
    zero_source = summary.SourceARMStatus.eq("zero_unassigned")
    report = {
        "status": "independent_saved_profile_readback_passed",
        "producer_provenance_sha256": sha(RUN / "run_provenance.json"),
        "producer_code_sha256_recorded": provenance["code_sha256"],
        "reader_sha256": sha(__file__), "production_imports": False,
        "n_selected": len(summary), "n_endpoint_eligible": int(eligible.sum()),
        "n_axon_profiles": int(axon_mask.sum()), "n_endpoint_profiles": int(end_mask.sum()),
        "source_ARM0": {"selected": int(zero_source.sum()), "axon_profiles": int((zero_source & axon_mask).sum()),
                        "endpoint_profiles": int((zero_source & end_mask).sum())},
        "partitions": comparisons, "verified_target_profile_values": checked_profile_values,
        "singleton_endpoint_example": {"uid": singleton_uid, "count": 1, "official_target": only.OfficialFullName.iloc[0]},
        "limits": ["No repeated-subset stability rerun here; producer resampling receipts reviewed separately",
                   "All fits remain neuron weighted, conditional on mapped positive L6 composition",
                   "Absolute-side partitions largely reflect source hemisphere, not validated neuronal types",
                   "Graph leaves are candidate endpoint proxies; atlas assignment and registration remain unaccepted"]}
    output = Path(__file__).with_suffix(".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "partitions"}, indent=2))


if __name__ == "__main__":
    main()
