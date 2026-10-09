"""Join existing reconstruction/image availability to saved cluster assignments.

This is a descriptive missing-evidence audit. It neither refits clusters nor
uses cache availability to classify anatomy or tracing quality. In particular,
missing native reconstructions retain missing endpoint/coverage measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSIGNMENTS = HERE / "arm_L6_relative_profiles_selected462/all462_feature_eligibility_and_assignments.csv"
CLUSTER_RECEIPT = ASSIGNMENTS.parent / "run_provenance.json"
COVERAGE = HERE.parent / "terminal_field_assessment_20261009/native_cube_pilot_current_rint/selected462_native_image_coverage.csv"
COVERAGE_RECEIPT = COVERAGE.parent / "generation_provenance.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_join(assignments, coverage):
    """Require exact identity/source agreement and preserve unknown counts."""
    for table in (assignments, coverage):
        if table.NeuronUID.duplicated().any() or table.NeuronUID.eq("").any():
            raise ValueError("Expected unique nonempty composite neuron identities")
    if set(assignments.NeuronUID) != set(coverage.NeuronUID):
        raise ValueError("Cluster and native-coverage ledgers differ in membership")
    joined = assignments.merge(coverage, on="NeuronUID", suffixes=("", "_coverage"),
                               how="inner", validate="one_to_one")
    for left, right in (("AnimalID", "AnimalID_coverage"),
                        ("SWCPath", "atlas_swc_path"),
                        ("SWCSHA256", "atlas_swc_sha256"),
                        ("ARMFullName", "SourceARMFullName")):
        if not joined[left].eq(joined[right]).all():
            raise ValueError(f"Source binding mismatch: {left}/{right}")
    native = joined.native_swc_available.map({"True": True, "False": False})
    if native.isna().any():
        raise ValueError("Unknown native availability encoding")
    ends = pd.to_numeric(joined.candidate_axon_leaves.replace("", np.nan), errors="raise")
    covered = pd.to_numeric(joined.leaves_with_cached_native_cube.replace("", np.nan), errors="raise")
    if (ends[~native].notna().any() or covered[~native].notna().any()
            or ends[native].isna().any() or covered[native].isna().any()
            or (ends.dropna() < 0).any() or (covered.dropna() < 0).any()
            or (covered[native] > ends[native]).any()
            or (ends.dropna() % 1 != 0).any() or (covered.dropna() % 1 != 0).any()):
        raise ValueError("Native counts violate availability/integer/coverage contracts")
    atlas_ends = pd.to_numeric(joined.Endpoint_candidate_axon_endpoint_count, errors="raise")
    if not ends[native].eq(atlas_ends[native]).all():
        raise ValueError("Native full-graph ends disagree with saved atlas graph counts")
    joined["NativeSWCFoundInDesignatedCache"] = native
    joined["NativeOriginalAxonEndCount"] = ends.astype("Int64")
    joined["NativeEndsWithCachedCube"] = covered.astype("Int64")
    joined["CachedEndFractionWhenNativeEndsPresent"] = covered / ends.where(ends > 0)
    joined["NativeImageEvidenceState"] = np.where(~native, "native_swc_not_found_here",
        np.where(covered > 0, "cached_end_images_available_review_not_encoded", "no_end_cube_found_here"))
    return joined


def summarize(joined):
    records = []
    strata = [("animal", "AnimalID"), ("Henry_visual_evidence", "Henry_coarse_INS_visual_evidence"),
              ("axon_Hellinger_k2", "all_axon_hellinger_k2"),
              ("endpoint_Hellinger_k2", "all_endpoint_hellinger_k2")]
    for kind, column in strata:
        for value, group in joined.groupby(column, sort=True, dropna=False):
            native = group.NativeSWCFoundInDesignatedCache
            observed = group.loc[native]
            total_ends = int(observed.NativeOriginalAxonEndCount.sum()) if len(observed) else None
            covered_ends = int(observed.NativeEndsWithCachedCube.sum()) if len(observed) else None
            records.append({"SummaryType": kind, "SummaryValue": value or "unassigned_profile",
                "SelectedNeurons": len(group), "Animals": group.AnimalID.nunique(),
                "HenryVisualINSNeurons": int(group.Henry_coarse_INS_visual_evidence.eq("True").sum()),
                "NativeSWCsFoundHere": int(native.sum()), "NativeSWCsNotFoundHere": int((~native).sum()),
                "NeuronsWithAnyCachedEndCube": int((observed.NativeEndsWithCachedCube > 0).sum()),
                "NativeEndCountAmongFoundSWCs": total_ends,
                "CachedEndCountAmongFoundSWCs": covered_ends,
                "PooledCachedEndFractionAmongFoundSWCs": covered_ends / total_ends if total_ends else None,
                "CompleteGroupNativeCoverage": bool(native.all())})
    return pd.DataFrame(records)


def main(output):
    if output.exists():
        raise FileExistsError(output)
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in
                (ASSIGNMENTS, CLUSTER_RECEIPT, COVERAGE, COVERAGE_RECEIPT, Path(__file__))}
    cluster_receipt = json.loads(CLUSTER_RECEIPT.read_text(encoding="utf-8"))
    if cluster_receipt["output_hashes"][ASSIGNMENTS.name] != sha(ASSIGNMENTS):
        raise ValueError("Saved assignment bytes do not match clustering receipt")
    coverage_receipt = json.loads(COVERAGE_RECEIPT.read_text(encoding="utf-8"))
    coverage_binding = next(item for item in coverage_receipt["outputs"]
                            if item["path"] == COVERAGE.relative_to(ROOT).as_posix())
    if coverage_binding["sha256"] != sha(COVERAGE):
        raise ValueError("Native availability bytes do not match their generation receipt")
    assignments = pd.read_csv(ASSIGNMENTS, dtype=str, keep_default_na=False)
    coverage = pd.read_csv(COVERAGE, dtype=str, keep_default_na=False)
    joined = checked_join(assignments, coverage)
    if len(joined) != 462:
        raise ValueError("Expected unchanged selected-462 ledger")
    summary = summarize(joined)
    columns = ["NeuronUID", "AnimalID", "ARMFullName", "SourceARMStatus",
               "Henry_coarse_INS_visual_evidence", "AxonProfileEligible", "EndpointProfileEligible",
               "all_axon_hellinger_k2", "all_endpoint_hellinger_k2",
               "NativeSWCFoundInDesignatedCache", "NativeOriginalAxonEndCount",
               "NativeEndsWithCachedCube", "CachedEndFractionWhenNativeEndsPresent", "NativeImageEvidenceState"]
    output.mkdir(parents=True)
    joined[columns].to_csv(output / "all462_native_image_coverage.csv", index=False)
    summary.to_csv(output / "native_image_coverage_by_animal_evidence_cluster.csv", index=False)
    native = joined.NativeSWCFoundInDesignatedCache
    receipt = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "descriptive_audit_completed",
        "source_hashes": bindings, "versions": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__},
        "selected_neurons": len(joined), "native_swcs_found_here": int(native.sum()),
        "native_swcs_not_found_here": int((~native).sum()),
        "native_original_axon_ends_among_found": int(joined.loc[native, "NativeOriginalAxonEndCount"].sum()),
        "cached_end_positions_among_found": int(joined.loc[native, "NativeEndsWithCachedCube"].sum()),
        "parameters": {"clusters": "saved primary Hellinger k2 labels; no refit", "new_cohorts": False,
                       "missing_native_measurements": "NA, never zero", "end_coverage": "cached TIFF exists at nominal original leaf position"},
        "scientific_limits": ["Native cache availability is confounded with animal; it is not tracing-quality acceptance.",
            "A cached cube is available evidence, not a completed image review or proof of termination.",
            "Pooled endpoint coverage weights neurons and ends unequally; it is not an animal-level prevalence estimate.",
            "Unknown availability elsewhere is retained; no missing case is removed or assigned zero innervation.",
            "Existing atlas assignments, cluster labels, maps, sources and their historical receipts remain unchanged."],
        "canonical_changes": False, "anatomical_acceptance": False,
        "output_hashes": {path.name: sha(path) for path in output.glob("*.csv")}}
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Input changed during audit: {relative}")
    (output / "coverage_audit_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("selected_neurons", "native_swcs_found_here",
        "native_swcs_not_found_here", "native_original_axon_ends_among_found", "cached_end_positions_among_found")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; never overwrite")
    main(parser.parse_args().output.resolve())
