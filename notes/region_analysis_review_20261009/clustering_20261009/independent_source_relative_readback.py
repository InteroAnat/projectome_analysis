"""Independently rebuild the saved ipsi/contra arithmetic without producer imports."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", type=Path, required=True)
    args = parser.parse_args()
    output = args.sensitivity / "independent_arithmetic_readback.json"
    if output.exists():
        raise FileExistsError(output)
    path = args.sensitivity / "sensitivity_provenance.json"
    sensitivity = json.loads(path.read_text(encoding="utf-8"))
    primary = Path(sensitivity["primary_run"])
    producer = json.loads((primary / "run_provenance.json").read_text(encoding="utf-8"))
    protected = {**producer["input_hashes"], **producer["selected_source_hashes"]}
    protected.update({str(primary / name): value for name, value in producer["output_hashes"].items()})
    protected.update({str(args.sensitivity / name): value for name, value in sensitivity["output_hashes"].items()})
    protected[str(primary / "run_provenance.json")] = sensitivity["primary_provenance_SHA256"]
    protected[str(path)] = sha(path)
    for source, expected in protected.items():
        if sha(source) != expected:
            raise ValueError("Changed bound file: " + source)
    export = Path(sensitivity["export_provenance"]).parent
    targets = read(export / "targets.csv")
    targets = targets[(targets.Level == "6") & (targets.TargetStatus == "mapped")].set_index("TargetID")
    measures = read(export / "per_neuron_regional_measures.csv")
    measures = measures[measures.Level == "6"]
    original = read(primary / "all462_feature_eligibility_and_assignments.csv").set_index("NeuronUID")
    assignments = read(args.sensitivity / "source_relative_assignments.csv").set_index("NeuronUID")
    mapping = read(args.sensitivity / "official_ARM_homolog_feature_mapping.csv").set_index("FeatureID")
    diagnostics = pd.read_csv(args.sensitivity / "source_relative_diagnostics.csv")
    if assignments.index.tolist() != original.index.tolist() or len(assignments) != 462:
        raise ValueError("Exact462 source UID order changed")
    for column in ["SampleID", "NeuronID", "AnimalID", "Hemisphere",
                   "AxonProfileEligible", "EndpointProfileEligible"]:
        if assignments[column].tolist() != original[column].tolist():
            raise ValueError("Source ledger field changed: " + column)
    observed_targets = set()
    pairs = set()
    for row in mapping.itertuples():
        left = targets.loc[row.LeftTargetID]
        right = targets.loc[row.RightTargetID]
        for side, target in [("Left", left), ("Right", right)]:
            for field, expected in [("ARMIndex", target.ARMIndex),
                                    ("OfficialFullName", target.OfficialFullName),
                                    ("Abbreviation", target.Abbreviation)]:
                observed = getattr(row, side + field)
                equal = (float(observed) == float(expected)) if field == "ARMIndex" else (str(observed) == str(expected))
                if not equal:
                    raise ValueError("Official homolog field mismatch")
        if left.Domain != right.Domain or left.Domain != row.Domain:
            raise ValueError("Cross-domain homolog merge")
        if left.Hemisphere != "L" or right.Hemisphere != "R":
            raise ValueError("Homolog side mismatch")
        if left.Abbreviation[2:] != right.Abbreviation[2:] or left.OfficialFullName[2:] != right.OfficialFullName[2:]:
            raise ValueError("Official homolog names disagree")
        observed_targets.update([row.LeftTargetID, row.RightTargetID])
        pairs.add((row.LeftTargetID, row.RightTargetID))
    if len(pairs) != 342 or len(mapping) != 684 or observed_targets != set(targets.index):
        raise ValueError("Not an exclusive342-pair permutation of actual ARM6")
    checks = []
    for family, measure, eligibility in [
        ("axon", "AxonTemplateLengthMm", "AxonProfileEligible"),
        ("endpoint", "CandidateEndpointCount", "EndpointProfileEligible"),
    ]:
        raw = measures.pivot(index="NeuronUID", columns="TargetID", values=measure)
        raw = raw.reindex(index=original.index, columns=targets.index).replace("", np.nan).fillna(0).astype(float)
        selected = original[eligibility].eq("True")
        if not (raw.loc[selected].sum(axis=1) > 0).all():
            raise ValueError("Selected zero mapped profile")
        raw = raw.loc[selected]
        source_left = original.loc[selected, "Hemisphere"].eq("L").to_numpy()
        transformed = np.empty((len(raw), len(mapping)))
        for j, row in enumerate(mapping.itertuples()):
            use_left = source_left if row.RelativeSide == "ipsilateral" else ~source_left
            transformed[:, j] = np.where(use_left, raw[row.LeftTargetID], raw[row.RightTargetID])
        error = float(np.max(np.abs(transformed.sum(axis=1) - raw.sum(axis=1).to_numpy())))
        if error > 1e-10:
            raise ValueError("Permutation did not conserve mapped totals")
        distance = pdist(np.sqrt(transformed / transformed.sum(axis=1, keepdims=True))) / np.sqrt(2)
        tree = linkage(distance, method="average")
        for k in sensitivity["k_values"]:
            saved = assignments.loc[selected, f"{family}_source_relative_k{k}"].astype(float)
            independent = fcluster(tree, k, criterion="maxclust")
            ari = adjusted_rand_score(saved, independent)
            score = silhouette_score(squareform(distance), saved, metric="precomputed")
            expected = diagnostics[(diagnostics.family == family) & (diagnostics.requested_k == k)].silhouette.iloc[0]
            if ari != 1 or not np.isclose(score, expected, atol=1e-12, rtol=0):
                raise ValueError("Saved sensitivity partition/diagnostic mismatch")
            if assignments.loc[~selected, f"{family}_source_relative_k{k}"].ne("").any():
                raise ValueError("Unassessed row received assignment")
            checks.append({"family": family, "requested_k": k, "n_profiles": len(raw),
                           "independent_partition_ARI": ari, "silhouette_error": float(abs(score - expected)),
                           "maximum_total_conservation_error": error})
    receipt = {
        "status": "passed_independent_arithmetic_and_hash_readback_software_only",
        "sensitivity_provenance": str(path.resolve()),
        "sensitivity_provenance_SHA256": sha(path),
        "code_SHA256": sha(__file__), "exact_selected_UIDs": len(assignments),
        "official_same_domain_bilateral_pairs": len(pairs),
        "actual_positive_ARM6_features": len(observed_targets),
        "fresh_source_hashes_verified": len(producer["selected_source_hashes"]),
        "primary_numeric_outputs_unchanged": True,
        "checks": checks, "anatomical_acceptance": False,
    }
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
