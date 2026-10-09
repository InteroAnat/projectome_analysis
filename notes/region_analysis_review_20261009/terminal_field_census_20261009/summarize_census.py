"""Compact descriptive totals and availability-ranked, unreviewed case cues."""
from pathlib import Path
import hashlib
import json

import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / "selected462_ARM6_sections"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sections = pd.read_csv(RUN / "connected_axon_sections.csv", keep_default_na=False, low_memory=False)
    neurons = pd.read_csv(RUN / "neuron_morphology_and_coverage.csv", keep_default_na=False, low_memory=False)
    covered = pd.to_numeric(sections.original_axon_ends_with_cached_cube, errors="coerce")
    native = neurons.Census_native_pair_available.eq(True)
    summary = {
        "selected_neurons": len(neurons), "animals": int(neurons.AnimalID.nunique()),
        "axon_nodes": int(sections.node_count.sum()), "connected_axon_sections": len(sections),
        "section_status_counts": {str(k): int(v) for k, v in sections.ARM6Status.value_counts().items()},
        "ending_status_counts": {str(k): int(v) for k, v in sections.groupby("ARM6Status").original_axon_end_count.sum().items()},
        "topology_descriptions": {str(k): int(v) for k, v in sections.topology_description.value_counts().items()},
        "sections_with_native_identity_pair": int(sections.native_pair_available.sum()),
        "sections_with_at_least_one_end_in_cached_cube": int(covered.gt(0).sum()),
        "native_pairs": int(native.sum()), "missing_designated_native_pairs": int((~native).sum()),
        "all_selected_original_ends": int(sections.original_axon_end_count.sum()),
        "native_original_ends": int(neurons.loc[native, "Census_original_axon_end_count"].sum()),
        "covered_native_original_ends": int(covered.sum()),
        "neurons_with_native_end_cube": int(pd.to_numeric(neurons.Census_original_axon_ends_with_cached_cube, errors="coerce").gt(0).sum()),
        "availability_is_not_image_review": True,
        "inputs": {p.name: sha(p) for p in [RUN / "connected_axon_sections.csv", RUN / "neuron_morphology_and_coverage.csv"]}}
    summary["animal_totals"] = neurons.groupby("AnimalID").apply(
        lambda d: {"selected_neurons": len(d), "original_ends": int(d.Census_original_axon_end_count.sum()),
                   "native_pairs": int(d.Census_native_pair_available.sum())}, include_groups=False).to_dict()
    with (HERE / "descriptive_census_summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    # Prior small reviews do not establish complete neuron review. Excluding
    # their UIDs here merely prioritizes additional neurons for review breadth.
    prior = {"252790::014.swc", "252790::032.swc", "252383::018.swc", "252383::121.swc", "252384::047.swc",
             "252383::124.swc", "252790::037.swc", "252718::162.swc", "252385::056.swc", "252527::056.swc", "252714::099.swc"}
    fields = ["NeuronUID", "portal_region", "Henry_coarse_INS_visual_evidence", "EvidenceStratum"]
    joined = sections.merge(neurons[fields], on="NeuronUID", how="left", validate="many_to_one")
    joined["covered_ends"] = pd.to_numeric(joined.original_axon_ends_with_cached_cube, errors="coerce")
    scope = joined.SourceARMIndex.eq(0) | joined.portal_region.str.contains("PrCO", case=False, na=False)
    candidates = joined.loc[scope & joined.covered_ends.gt(0) & ~joined.NeuronUID.isin(prior)].copy()
    candidates = candidates.sort_values(["covered_ends", "local_branch_point_count", "NeuronUID", "SectionUID"],
                                        ascending=[False, False, True, True]).drop_duplicates("NeuronUID").head(20)
    columns = ["NeuronUID", "AnimalID", "SourceARMFullName", "SourceARMStatus", "SourceARMIndex", "portal_region",
               "EvidenceStratum", "Henry_coarse_INS_visual_evidence", "SectionUID", "ARM6Index", "ARM6FullName",
               "node_count", "original_axon_end_count", "local_branch_point_count", "covered_ends", "SWCSHA256"]
    candidates = candidates[columns]
    candidates["review_state"] = "not_assessed_by_census"
    candidates["ranking_basis"] = "cached_end_count_desc;local_branch_count_desc;UID;section_UID;one_section_per_new_neuron"
    candidates.to_csv(HERE / "additional_image_review_cues.csv", index=False, mode="x")
    with (HERE / "summary_provenance.json").open("x") as stream:
        json.dump({"producer_sha256": sha(Path(__file__)), "inputs": summary["inputs"],
                   "outputs": {p.name: sha(p) for p in [HERE / "descriptive_census_summary.json", HERE / "additional_image_review_cues.csv"]},
                   "selection_is_display_priority_not_biological_acceptance": True}, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(candidates[["NeuronUID", "portal_region", "ARM6FullName", "original_axon_end_count", "local_branch_point_count", "covered_ends"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
