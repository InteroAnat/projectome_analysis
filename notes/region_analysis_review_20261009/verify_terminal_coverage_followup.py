"""Fresh readback of the terminal census, review expansion and CM033 snapshot.

Rehash immutable source/derivative bindings and reconcile saved whole-ledger
counts. Live external process logs are time-bounded observations and are
reported separately if they change. This verifier grants no anatomical pass.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CENSUS = HERE / "terminal_field_census_20261009/selected462_ARM6_sections"
EXPANSION = HERE / "terminal_review_expansion_20261009"
COVERAGE = HERE / "clustering_20261009/native_image_coverage_20261009"
CM033 = HERE / "cm033_source_mapping_20261009"
LIVE_CM033_NOTE = Path("C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/docs/CM033_D_SOURCE_SPATIAL_REBUILD_20261009.md").resolve()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    bindings = {}

    def add(raw, expected, parent):
        path = Path(raw)
        if not path.is_absolute():
            path = parent / path if len(path.parts) == 1 or (parent / path).exists() else ROOT / path
        path = path.resolve()
        if path in bindings and bindings[path] != expected:
            raise ValueError(f"Contradictory current binding: {path}")
        bindings[path] = expected

    def records(value, parent):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                add(value["path"], value["sha256"], parent)
            for item in value.values():
                records(item, parent)
        elif isinstance(value, list):
            for item in value:
                records(item, parent)

    receipt_paths = [CENSUS / "census_provenance.json", CENSUS.parent / "independent_census_readback.json",
                     CENSUS.parent / "delivery_validation.json",
                     EXPANSION / "generation_provenance.json", EXPANSION / "assessment_inputs.json",
                     EXPANSION / "delivery_receipt.json", EXPANSION / "root_visual_review.json",
                     COVERAGE / "coverage_audit_provenance.json", COVERAGE / "independent_native_coverage_readback.json",
                     CM033 / "mapping_and_normalization_snapshot.json", CM033 / "final_snapshot_status.json"]
    for path in receipt_paths:
        data = read(path)
        add(path, sha(path), path.parent)
        records(data, path.parent)
        if isinstance(data, dict):
            for key in ("source_hashes", "source_sha256", "output_hashes", "local_audit_output_sha256"):
                for raw, expected in data.get(key, {}).items():
                    add(raw, expected, path.parent)
    for row in pd.read_csv(CENSUS / "source_swc_bindings.csv", dtype=str).to_dict("records"):
        add(row["path"], row["sha256"], CENSUS)
    failures, changed_live_records = [], []
    for path, expected in bindings.items():
        actual = sha(path) if path.is_file() else None
        if actual != expected:
            finding = {"path": str(path), "snapshot_sha256": expected, "current_sha256": actual}
            if (path.suffix.lower() == ".log" and "cm033_Dsource_20261009" in path.parts) or path == LIVE_CM033_NOTE:
                finding["interpretation"] = "Time-bounded external progress record changed; original snapshot retained. This does not replace image/model/transform bindings."
                changed_live_records.append(finding)
            else:
                failures.append(finding)
    neurons = pd.read_csv(CENSUS / "neuron_morphology_and_coverage.csv", dtype=str, keep_default_na=False)
    summary = pd.read_csv(HERE / "hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv",
                          dtype=str, keep_default_na=False).set_index("NeuronUID")
    if len(neurons) != 462 or neurons.NeuronUID.duplicated().any() or set(neurons.NeuronUID) != set(summary.index):
        raise ValueError("Census changed selected-neuron membership")
    original = neurons.NeuronUID.map(summary.Endpoint_candidate_axon_endpoint_count).astype(int)
    if not original.eq(neurons.Census_original_axon_end_count.astype(int)).all():
        raise ValueError("All-selected census endpoint counts differ from independent saved tables")
    coverage = pd.read_csv(COVERAGE / "all462_native_image_coverage.csv", dtype=str, keep_default_na=False)
    missing = coverage.NativeSWCFoundInDesignatedCache.eq("False")
    if not (missing.sum() == 261 and coverage.loc[missing, "NativeOriginalAxonEndCount"].eq("").all()
            and coverage.loc[missing, "NativeEndsWithCachedCube"].eq("").all()
            and coverage.loc[missing, "AnimalID"].eq("936").all()):
        raise ValueError("Native missingness changed or was zero-filled")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "passed" if not failures else "failed",
              "source_and_artifact_paths_rehashed": len(bindings), "failures": failures,
              "changed_time_bounded_external_records": changed_live_records, "selected_neurons": len(neurons),
              "all_selected_original_axon_ends": int(original.sum()),
              "connected_axon_sections": read(CENSUS / "census_provenance.json")["sections"],
              "missing_native_counts_preserved_NA": int(missing.sum()),
              "expanded_leaf_locations": read(EXPANSION / "delivery_receipt.json")["new_individual_leaf_locations"],
              "combined_individually_reviewed_leaves": 11, "combined_reviewed_animals": 7,
              "new_software_tests": {"census": 7, "coverage_join": 4},
              "test_evidence": "separate independent census and native-coverage receipts; prior 409-test suite not rerun",
              "verifier_sha256": sha(Path(__file__)), "receipt_hashes": {str(path.relative_to(ROOT)): sha(path) for path in receipt_paths},
              "anatomical_acceptance": False, "biological_terminal_acceptance": False,
              "full_goal_achieved": False,
              "remaining": ["Broader terminal-field completeness/segmentation remains unresolved; graph sections are not accepted arbors.",
                            "CM033 historical-left-GLM-to-new-anatomy bridge remains unverified; existing spatial job is separately monitored."]}
    output = HERE / "terminal_coverage_followup_readback_20261009.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "source_and_artifact_paths_rehashed",
                       "all_selected_original_axon_ends", "connected_axon_sections", "failures", "changed_time_bounded_external_records")}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
