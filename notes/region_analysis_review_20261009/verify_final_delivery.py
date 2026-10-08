"""Verify saved sources and delivery hashes without repeating map computation.

Historical code receipts remain immutable. Current source hashes are recorded
separately: later presentation/consumer repairs do not rewrite a producer run.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
EVOLUTION = ROOT / "group_analysis/evolution_20261008"


def verify_current_deliveries(read, rows, digest, check, errors):
    """Rehash saved deliveries; never repeat rasterization, fitting or warping."""
    result = {}

    def expect(condition, message):
        if not condition:
            errors.append(message)

    def bound_files(mapping, base=ROOT, *, outputs=False):
        expect(bool(mapping), f"Empty hash bindings: {base}")
        for name, value in mapping.items():
            path = Path(value["path"] if isinstance(value, dict) and "path" in value else name)
            path = path if path.is_absolute() else base / path
            if outputs:
                expect(path.resolve().is_relative_to(base.resolve()), f"Output escapes delivery: {path}")
            check(path, value["sha256"] if isinstance(value, dict) else value)
            if outputs and path.suffix.lower() == ".png" and path.is_file():
                with Image.open(path) as picture:
                    picture.verify()

    arm = EVOLUTION / "projection_inputs/arm_labels_20261009"
    preparation = read(arm / "preparation_provenance.json")
    delivery = read(arm / "delivery_provenance.json")
    expect(preparation["status"] == "software_verified_ARM_only_source_group_manifests"
           and delivery["status"] == "ARM_only_manifest_delivery_readback_passed", "ARM manifest delivery status changed")
    expect(preparation["source_selection_changed"] is False and preparation["evidence_metadata_promoted"] is False
           and delivery["anatomical_acceptance"] is False, "ARM preparation scope or acceptance changed")
    bound_files(preparation["inputs"])
    bound_files(preparation["outputs"], arm, outputs=True)
    bound_files(delivery["artifacts"], arm, outputs=True)
    combined = rows(arm / "combined/projection_manifest.csv")
    identities = {(row["SampleID"], row["NeuronID"]) for row in combined}
    background = {(row["SampleID"], row["NeuronID"]) for row in combined if row["ARMIndex"] == "0"}
    expect(len(combined) == len(identities) == 462 and len(background) == 52, "Combined ARM462/52 identity scope changed")
    for row in combined:
        check(row["SWCPath"], row["SWCSHA256"])
    result["combined_ARM_manifest"] = {"rows": len(combined), "unique_identities": len(identities),
        "source_ARM0_QC_rows": len(background), "manifest_sha256": digest(arm / "combined/projection_manifest.csv"),
        "delivery_receipt_sha256": digest(arm / "delivery_provenance.json")}

    cases = EVOLUTION / "classification/background_neuron_audit_20261009"
    case_review = read(cases / "final_independent_saved_readback.json")
    expect(case_review["status"] == "independent_saved_case52_readback_passed" and case_review["case_UIDs"] == 52
           and case_review["scientific_acceptance"] is False, "Final52-case QC receipt failed or changed scope")
    bound_files(case_review["artifact_hashes"], cases, outputs=True)
    case_sources = read(cases / "independent_receipt.json")
    bound_files(case_sources["inputs_and_selected_sources"])
    overlay = read(cases / "delivery_receipt_v2.json")
    bound_files(overlay["inputs"])
    check(overlay["primary_case_ledger"], overlay["primary_case_ledger_sha256"])
    case_rows = rows(cases / "selected_52_background_cases_with_case_overlays.csv")
    case_ids = {(row["SampleID"], row["NeuronID"]) for row in case_rows}
    expect(len(case_rows) == len(case_ids) == 52 and case_ids == background, "Case52 ledger differs from combined ARM0 identities")
    result["source_case_QC"] = {"receipt_sha256": digest(cases / "final_independent_saved_readback.json"),
        "cases": len(case_rows), "effective_native_pairs": case_review["effective_native_pairs"],
        "effective_visual_cases": case_review["effective_visual_cases"]}

    workflow = EVOLUTION / "arm_mapping_20261009"
    required = {"main_endpoints_build", "main_endpoints_review", "main_endpoint-density_group",
                "main_endpoint-occupancy_group", "main_axons_build", "main_axons_review", "main_axon-density_group"}
    checkpoint = read(workflow / "workflow_checkpoint.json")
    stages = [item["stage"] for item in checkpoint]
    scope = read(workflow / "scope_reduction_20261009.json")
    expect(len(stages) == len(set(stages)) and required <= set(stages), "Missing/duplicate required ARM workflow stage")
    expect(scope["status"] == "required_primary_stages_completed_optional_work_stopped"
           and set(scope["required_stages"]) == required and set(scope["completed_stages"]) == set(stages)
           and scope["source_selection_changed"] is False and scope["scientific_acceptance"] is False,
           "ARM workflow scope-reduction receipt is inconsistent")
    for item in checkpoint:
        expect(item["returncode"] == 0, f"Saved workflow stage failed: {item['stage']}")
        check(item["log"], item["log_sha256"])
    result["workflow"] = {"required_stages_completed": sorted(required), "recorded_completed_stages": stages,
        "scope_reduction_receipt_sha256": digest(workflow / "scope_reduction_20261009.json"),
        "checkpoint_sha256": digest(workflow / "workflow_checkpoint.json"),
        "full_optional_workflow_completion_claimed": False}

    upstream_bindings_valid = not errors
    map_records = []
    for family, review_name, endpoint in (("endpoints", "endpoint_run_readback.json", True),
                                          ("axons", "projection_run_readback.json", False)):
        start = len(errors)
        run = workflow / "main" / family
        review_path = workflow / "main" / (family + "_review") / review_name
        review, producer = read(review_path), read(run / "run_provenance.json")
        expect(review["status"] == "passed", f"ARM numerical review failed: {review_path}")
        expect(producer["status"] == ("software_verified_candidate_endpoints" if endpoint else "software_verified_descriptive"),
               f"ARM producer completion status changed: {family}")
        expect(review["anatomical_acceptance"] == "not established", f"Unexpected anatomical acceptance: {review_path}")
        check(run / "run_provenance.json", review["run_provenance_sha256"])
        check(run / "input_manifest.csv", review["manifest_sha256"])
        check(producer["reference"], producer["reference_sha256"])
        if endpoint:
            bound_files(producer["inputs"])
            bound_files(producer["artifacts"], run, outputs=True)
            expect((review["selected_neurons"], review["computable_neurons"]) == (436, 403), "ARM endpoint denominator changed")
        else:
            check(producer["input_manifest"], producer["manifest_sha256"])
            check(run / "per_neuron_measurements.csv", review["per_neuron_measurements_sha256"])
            if producer["brain_mask"]:
                check(producer["brain_mask"], producer["brain_mask_sha256"])
            expect(review["neurons"] == 436, "ARM axon denominator changed")
        files = review["map_files" if endpoint else "all_map_files"]
        names = []
        for item in files:
            path = Path(item["file"])
            path = path if path.is_absolute() else run / path
            expect(path.name.endswith(".nii.gz"), f"Spatial map is not a compressed NIfTI: {path}")
            names.append(str(path.resolve()))
            check(path, item["sha256"])
        expect(len(names) == len(set(names)) == (192 if endpoint else 128), f"ARM saved map coverage changed: {family}")
        present_maps = {str(path.resolve()) for folder in ("animal_maps", "group_maps")
                        for path in (run / folder).glob("*.nii.gz")}
        expect(set(names) == present_maps, f"Unreviewed or missing ARM NIfTI map: {family}")
        map_records.append({"family": family, "review_sha256": digest(review_path), "map_files_rehashed": len(files),
            "prior_complete_voxels_read": review["full_saved_voxels_read"],
            "prior_numerical_validation_reused_after_hash_checks": upstream_bindings_valid and len(errors) == start})
    result["ARM_main_map_reviews"] = map_records

    primary = {"endpoint-density_group", "endpoint-occupancy_group", "axon-density_group"}
    figure_records, primary_count, archived_count = [], 0, 0
    named_sources = {row["Subregion"] for row in rows(arm / "main/projection_manifest.csv") if row["ARMIndex"] != "0"}
    figure_paths = sorted((workflow / "main/figures").glob("*/matched_slices_provenance.json"))
    expect(primary <= {path.parent.name for path in figure_paths}, "Missing primary ARM figure family")
    for path in figure_paths:
        record = read(path)
        expect(record["status"] == "rendered_and_png_decoded" and record["fixed_slice_voxels_xyz"] == [56, 200, 87],
               f"Current ARM display contract changed: {path}")
        check(record["readback_path"], record["readback_sha256"])
        source_labels = record["source_region_labels"]
        check(source_labels["path"], source_labels["sha256"])
        check(source_labels["atlas_path"], source_labels["atlas_sha256"])
        check(source_labels["atlas_key_path"], source_labels["atlas_key_sha256"])
        mapped, named = 0, []
        for item in record["figures"]:
            png = path.parent / item["file"]
            check(png, item["sha256"])
            with Image.open(png) as picture:
                picture.verify()
            expect(item["panel_title_overlap_check"] == "passed", f"Current figure title overlap: {png}")
            expect(item["source_location_status"] in {"mapped", "unresolved"}, f"Unknown source display status: {png}")
            current_primary = path.parent.name in primary and item["source_location_status"] == "mapped"
            if current_primary:
                primary_count += 1
                mapped += 1
                named.extend(item["source_groups"])
                expect(not any(group.startswith("ARM6_0_") for group in item["source_groups"]), f"Unresolved QC included as named parcel: {png}")
            else:
                archived_count += 1
        expect({item["file"] for item in record["figures"]} == {png.name for png in path.parent.glob("*.png")},
               f"Unbound or missing current figure PNG: {path.parent}")
        if path.parent.name in primary:
            expect(mapped == 4 and len(named) == len(set(named)) == len(named_sources) and set(named) == named_sources,
                   f"Primary named-source figure coverage changed: {path}")
        figure_records.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bound_PNGs": len(record["figures"]),
                               "primary_mapped_PNGs": mapped})
    expect(primary_count == 12, "Expected12 primary mapped ARM PNGs")
    result["ARM_figures"] = {"primary_group_families": 3, "primary_mapped_PNGs": primary_count,
        "archived_current_run_PNGs_rehashed": archived_count, "families": figure_records}

    hierarchy = AUDIT / "hierarchy_tables_20261009"
    table_error_start = len(errors)
    table_dir = hierarchy / "combined_arm_projection_tables_462"
    table_path = table_dir / "export_provenance.json"
    table = read(table_path)
    expect(table["status"] == "software_verified_descriptive_tables" and (table["n_selected"], table["n_endpoint_eligible"]) == (462, 429)
           and table["scientific_anatomical_acceptance"] is False, "Hierarchy table completion or denominator changed")
    bound_files(table["source_bindings"])
    bound_files(table["artifacts"], table_dir, outputs=True)
    check(table["manifest"], table["manifest_sha256"])
    check(ROOT / "group_analysis/scripts/export_arm_projection_tables.py", table["exporter_sha256"])
    expect(table["regional_reconciliation_maps"] == 154, "Hierarchy saved-NIfTI reconciliation scope changed")
    table_review_path = hierarchy / "independent_saved_arm_tables_readback_20261009.json"
    table_review_record = {"present": table_review_path.exists()}
    expect(table_review_path.is_file(), "Required independent hierarchy readback is missing")
    if table_review_path.exists():
        review = read(table_review_path)
        expect(review["status"] == "passed" and (review["n_selected"], review["n_endpoint_eligible"]) == (462, 429)
               and review["anatomical_acceptance"] is False, "Independent hierarchy table review failed")
        check(table_path, review["export_provenance_sha256"])
        check(table_dir / "arm_projection_hierarchy_tables.xlsx", review["workbook_sha256"])
        check(hierarchy / "independent_saved_arm_tables_readback_20261009.py", review["review_script_sha256"])
        table_review_record.update(receipt_sha256=digest(table_review_path), matrix_values_checked=review["matrix_values_checked"],
            animal_target_rows_checked=review["animal_target_rows_checked"], equal_animal_target_rows_checked=review["equal_animal_target_rows_checked"],
            prior_saved_table_validation_reused_after_hash_checks=upstream_bindings_valid and len(errors) == table_error_start)
    result["hierarchy_tables"] = {"receipt_sha256": digest(table_path), "artifacts_rehashed": len(table["artifacts"]),
        "selected": 462, "endpoint_eligible": 429, "endpoint_NA": 33, "independent_readback": table_review_record}

    cluster_dir = AUDIT / "clustering_20261009/arm_L6_relative_profiles_selected462"
    cluster_error_start = len(errors)
    cluster_path = cluster_dir / "run_provenance.json"
    cluster = read(cluster_path)
    expect(cluster["status"] == "exploratory_profile_clustering_completed_software_only" and cluster["ledger_neurons"] == 462
           and cluster["anatomical_acceptance"] is False and cluster["statistical_p_values"] is False, "Clustering completion/scope changed")
    for key in ("input_hashes", "selected_source_hashes", "methods_and_baseline_hashes"):
        bound_files(cluster[key])
    bound_files(cluster["output_hashes"], cluster_dir, outputs=True)
    check(AUDIT / "clustering_20261009/primary_producer_code_20261009.py", cluster["code_sha256"])
    cluster_review_path = cluster_dir / "independent_saved_readback.json"
    cluster_review_record = {"present": cluster_review_path.exists()}
    expect(cluster_review_path.is_file(), "Required independent clustering readback is missing")
    if cluster_review_path.exists():
        review = read(cluster_review_path)
        expect(review["status"] == "independent_saved_profile_readback_passed" and review["selected_UIDs"] == review["unique_UIDs"] == 462
               and review["candidate_end_eligible"] == 429 and review["statistical_or_anatomical_acceptance"] is False,
               "Independent clustering readback failed or changed scope")
        check(cluster_path, review["run_provenance_SHA256"])
        check(arm / "combined/projection_manifest.csv", review["input_manifest_SHA256"])
        check(AUDIT / "clustering_20261009/independent_profile_readback.py", review["readback_code_SHA256"])
        cluster_review_record.update(receipt_sha256=digest(cluster_review_path), axon_profile_eligible=review["axon_profile_eligible"],
            endpoint_profile_eligible=review["endpoint_profile_eligible"],
            prior_saved_profile_validation_reused_after_hash_checks=upstream_bindings_valid and len(errors) == cluster_error_start)
    result["clustering"] = {"receipt_sha256": digest(cluster_path), "artifacts_rehashed": len(cluster["output_hashes"]),
        "selected": cluster["ledger_neurons"], "independent_readback": cluster_review_record}
    complete_profile_path = hierarchy / "independent_saved_profile_review_20261009.json"
    complete_profile = read(complete_profile_path)
    expect(complete_profile["status"] == "independent_saved_profile_readback_passed"
           and complete_profile["production_imports"] is False and len(complete_profile["partitions"]) == 49
           and all(item["partition_ARI"] == 1 for item in complete_profile["partitions"]),
           "Independent complete clustering partition review failed")
    check(cluster_path, complete_profile["producer_provenance_sha256"])
    check(hierarchy / "independent_saved_profile_review_20261009.py", complete_profile["reader_sha256"])
    result["clustering"]["independent_complete_profile_review"] = {
        "receipt_sha256": digest(complete_profile_path), "partitions_checked": 49,
        "target_profile_values_checked": complete_profile["verified_target_profile_values"]}

    sensitivity_path = AUDIT / "clustering_20261009/ARM6_source_relative_sensitivity/sensitivity_provenance.json"
    sensitivity = read(sensitivity_path)
    expect(sensitivity["status"] == "software_verified_feature_transformation_sensitivity"
           and sensitivity["official_bilateral_pairs"] == 342
           and sensitivity["primary_results_and_source_ledger_unchanged"] is True
           and sensitivity["atlas_reassignment"] is False and sensitivity["anatomical_or_statistical_acceptance"] is False,
           "Source-relative clustering sensitivity scope changed")
    check(Path(sensitivity["primary_run"]) / "run_provenance.json", sensitivity["primary_provenance_SHA256"])
    check(sensitivity["export_provenance"], sensitivity["export_provenance_SHA256"])
    check(sensitivity["manifest"], sensitivity["manifest_SHA256"])
    check(ROOT / "group_analysis/scripts/cluster_arm_projection_profiles.py", sensitivity["code_SHA256"])
    bound_files(sensitivity["output_hashes"], sensitivity_path.parent, outputs=True)
    display_path = AUDIT / "clustering_20261009/display_readability_correction/display_correction_provenance.json"
    display = read(display_path)
    expect(display["status"] == "software_verified_display_only_correction" and display["numerical_results_changed"] is False
           and display["primary_outputs_unchanged"] is True, "Clustering display-only correction changed numerical scope")
    check(Path(display["primary_run"]) / "run_provenance.json", display["primary_provenance_SHA256"])
    check(display["source_table"], display["source_table_SHA256"])
    check(AUDIT / "clustering_20261009/render_saved_profile_display.py", display["code_SHA256"])
    bound_files(display["output_hashes"], display_path.parent, outputs=True)
    result["clustering_extensions"] = {"source_relative_sensitivity_receipt_sha256": digest(sensitivity_path),
        "source_relative_sensitivity_outputs_rehashed": len(sensitivity["output_hashes"]),
        "official_bilateral_pairs": sensitivity["official_bilateral_pairs"],
        "display_only_receipt_sha256": digest(display_path), "corrected_display_PNGs_rehashed": len(display["output_hashes"]),
        "primary_cluster_run_unchanged": True, "anatomical_or_statistical_acceptance": False}
    relative_review_path = sensitivity_path.parent / "independent_arithmetic_readback.json"
    relative_review_record = {"present": relative_review_path.exists()}
    expect(relative_review_path.is_file(), "Required independent source-relative readback is missing")
    if relative_review_path.exists():
        review = read(relative_review_path)
        expect(review["status"] == "passed_independent_arithmetic_and_hash_readback_software_only"
               and review["exact_selected_UIDs"] == review["fresh_source_hashes_verified"] == 462
               and review["official_same_domain_bilateral_pairs"] == 342
               and review["primary_numeric_outputs_unchanged"] is True and review["anatomical_acceptance"] is False,
               "Independent source-relative sensitivity readback failed or changed scope")
        check(review["sensitivity_provenance"], review["sensitivity_provenance_SHA256"])
        check(AUDIT / "clustering_20261009/independent_source_relative_readback.py", review["code_SHA256"])
        cuts = [(item["family"], item["requested_k"]) for item in review["checks"]]
        expect(len(cuts) == 14 and set(cuts) == {(family, k) for family in ("axon", "endpoint") for k in range(2, 9)}
               and all(item["independent_partition_ARI"] == 1 for item in review["checks"]),
               "Independent source-relative cut coverage or arithmetic equality changed")
        relative_review_record.update(receipt_sha256=digest(relative_review_path), independent_cut_checks=len(cuts))
    result["clustering_extensions"]["independent_source_relative_readback"] = relative_review_record
    display_review_path = display_path.parent / "independent_saved_display_readback.json"
    display_review_record = {"present": display_review_path.exists()}
    expect(display_review_path.is_file(), "Required independent clustering display readback is missing")
    if display_review_path.exists():
        review = read(display_review_path)
        expect(review["status"] == "passed_independent_saved_display_readback_software_and_visual_only"
               and review["primary_outputs_unchanged"] is True and review["numerical_rerun"] is False
               and review["anatomical_acceptance"] is False, "Independent saved display review failed or changed scope")
        check(display_path, review["display_provenance_SHA256"])
        check(display["source_table"], review["source_table_SHA256"])
        check(cluster_path, review["primary_provenance_SHA256"])
        checked_names = [item["file"] for item in review["checks"]]
        expect(len(checked_names) == len(set(checked_names)) == 2 and set(checked_names) == set(display["output_hashes"]),
               "Independent display PNG coverage changed")
        for item in review["checks"]:
            png = display_path.parent / item["file"]
            check(png, item["SHA256"])
            with Image.open(png) as picture:
                expect(list(picture.size) == item["dimensions"], f"Independent display dimensions changed: {png}")
        display_review_record.update(receipt_sha256=digest(display_review_path), saved_PNGs_checked=len(checked_names))
    result["clustering_extensions"]["independent_saved_display_readback"] = display_review_record

    mstim = AUDIT / "mstim_integration_20261009"
    mst_records = []
    specifications = (("cm032_warp_reproduction/warp_reproduction.json", "software_reproduction_passed_registration_unaccepted", "source_sha256", "outputs"),
                      ("cm032_regional_comparison/integration_provenance.json", "provisional_regional_integration_completed", "source_sha256", "outputs"),
                      ("cm032_readable_qc/display_provenance.json", "display_correction_completed", "sources", "output_sha256"))
    for name, status, source_key, output_key in specifications:
        start = len(errors)
        path = mstim / name
        record = read(path)
        expect(record["status"] == status, f"MSTIM saved status changed: {path}")
        bound_files(record[source_key])
        bound_files(record[output_key], path.parent, outputs=True)
        acceptance = "scientific_acceptance" if "scientific_acceptance" in record else "anatomical_acceptance"
        expect(record[acceptance] is False, f"Unexpected MSTIM acceptance: {path}")
        if name.startswith("cm032_warp"):
            expect(record["comparison"]["exact_values_equal"] is True and record["comparison"]["maximum_absolute_difference"] == 0
                   and record["source_writes"] is False and record["new_registration_fit"] is False, "CM032 warp reproduction scope changed")
            check(mstim / "reproduce_cm032_warp.py", record["script_sha256"])
        elif name.startswith("cm032_readable"):
            check(mstim / "render_cm032_qc.py", record["script_sha256"])
            check(ROOT / "group_analysis/scripts/summarize_mstim_arm.py", record["renderer_sha256"])
        mst_records.append({"path": str(path.relative_to(ROOT)), "status": status, "sha256": digest(path),
                            "sources_rehashed": len(record[source_key]), "outputs_rehashed": len(record[output_key]),
                            "historical_producer_script_sha256": record["script_sha256"],
                            "prior_saved_validation_reused_after_hash_checks": upstream_bindings_valid and len(errors) == start})
    ancillary_path = mstim / "ancillary_source_readback.json"
    ancillary = read(ancillary_path)
    expect(ancillary["status"] == "passed" and ancillary["registration_acceptance"] is False
           and ancillary["stimulation_coordinate_acceptance"] is False and ancillary["source_writes"] is False, "MSTIM ancillary source readback failed")
    check(EVOLUTION / "crossmodal/input_readiness.json", ancillary["readiness_sha256"])
    check(mstim / "verify_mstim_sources.py", ancillary["script_sha256"])
    for item in ancillary["sources"]:
        check(item["path"], item["sha256"])
    result["MSTIM"] = {"saved_receipts": mst_records, "ancillary_receipt_sha256": digest(ancillary_path),
        "registration_and_stimulation_coordinate_acceptance": False, "new_warp_or_region_computation": False}
    independent_mstim_path = mstim / "independent_cm032_regional_readback_20261009.json"
    independent_mstim = read(independent_mstim_path)
    expect(independent_mstim["status"] == "independent_CM032_saved_regional_readback_passed"
           and independent_mstim["production_imports"] is False
           and (independent_mstim["fmri_rows"], independent_mstim["joint_rows"], independent_mstim["saved_group_maps"]) == (1676, 25140, 30),
           "Independent CM032 saved regional review failed")
    check(mstim / "cm032_regional_comparison/integration_provenance.json", independent_mstim["integration_provenance_sha256"])
    check(mstim / "independent_cm032_regional_readback_20261009.py", independent_mstim["reader_sha256"])
    result["MSTIM"]["independent_saved_regional_review"] = {
        "receipt_sha256": digest(independent_mstim_path), "fmri_rows_checked": 1676,
        "joint_rows_checked": 25140, "saved_group_maps_checked": 30}
    return result


def verify_delivery(output):
    output = output.resolve()
    if output.exists() or output.parent != AUDIT:
        raise ValueError("Use a fresh JSON receipt filename in this audit directory")
    hashes, errors = {}, []

    def digest(path):
        path = Path(path).resolve()
        if path not in hashes:
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            hashes[path] = value.hexdigest()
        return hashes[path]

    def check(path, expected):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or not path.is_file() or digest(path) != expected:
            errors.append(f"Changed or missing bound source: {path}")
            return False
        return True

    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def rows(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    source_receipt = AUDIT / "table_audit/run_20261009/source_hash_readback.csv"
    bindings = rows(source_receipt)
    expected_sources = {}
    for row in bindings:
        path, expected = row["path"], row["expected_sha256"]
        if path in expected_sources and expected_sources[path] != expected:
            errors.append(f"Conflicting recorded source hashes: {path}")
        expected_sources[path] = expected
        check(path, expected)
    corrections = AUDIT / "table_audit/corrected_diagnostics_20261009/correction_provenance.json"
    corrected = read(corrections)
    check(corrected["generator"], corrected["generator_sha256"])
    for item in corrected["results"]:
        check(item["source"], item["source_sha256"])
        check(item["derivative"], item["derivative_sha256"])

    selections = {}
    for cohort, folder in (("main", "multimonkey_coarse_20261009_v2"),
                           ("preview", "additional_candidates_20261009")):
        directory = EVOLUTION / "projection_inputs" / folder
        provenance = read(directory / "preparation_provenance.json")
        manifest = directory / "projection_manifest.csv"
        check(manifest, provenance["manifest_sha256"])
        records = rows(manifest)
        identities = {(row["SampleID"], row["NeuronID"]) for row in records}
        if len(identities) != len(records):
            errors.append(f"Duplicate composite identities in {cohort}")
        for row in records:
            check(row["SWCPath"], row["SWCSHA256"])
        selections[cohort] = identities
    if selections["main"] & selections["preview"]:
        errors.append("Main and preview identities overlap")
    if (len(selections["main"]), len(selections["preview"])) != (436, 26):
        errors.append("Recorded selection denominators changed")

    map_reviews, local_ledgers = [], []
    for cohort in ("multimonkey_coarse_20261009", "additional_candidates_20261009"):
        for family in ("endpoint_maps", "projection_maps"):
            endpoint = family == "endpoint_maps"
            suffix = "_review_v2" if endpoint and cohort.startswith("multimonkey") else "_review"
            run = EVOLUTION / family / cohort
            review_path = EVOLUTION / family / (cohort + suffix) / (
                "endpoint_run_readback.json" if endpoint else "projection_run_readback.json")
            review = read(review_path)
            if review["status"] != "passed":
                errors.append(f"Original complete numerical review failed: {review_path}")
            provenance = run / "run_provenance.json"
            check(provenance, review["run_provenance_sha256"])
            producer = read(provenance)
            check(run / "input_manifest.csv", review["manifest_sha256"])
            check(producer["reference"], producer["reference_sha256"])
            if endpoint:
                for item in producer["inputs"].values():
                    check(item["path"], item["sha256"])
                for name, item in producer["artifacts"].items():
                    check(run / name, item["sha256"])
            else:
                check(run / "per_neuron_measurements.csv", review["per_neuron_measurements_sha256"])
                if producer["brain_mask"]:
                    check(producer["brain_mask"], producer["brain_mask_sha256"])
            records = review["map_files" if endpoint else "all_map_files"]
            for item in records:
                path = Path(item["file"])
                check(path if path.is_absolute() else run / path, item["sha256"])
            map_reviews.append({"path": str(review_path.relative_to(ROOT)),
                                "sha256": digest(review_path), "maps_rehashed": len(records),
                                "prior_complete_voxels_read": review["full_saved_voxels_read"],
                                "prior_numerical_validation_reused_after_hash_checks": not errors})
            if endpoint:
                ledger = run / "leaf_records.jsonl"
                producer = read(provenance)
                ledger_bindings = [value["leaf_records.jsonl"]["sha256"] for value in producer.values()
                                   if isinstance(value, dict) and "leaf_records.jsonl" in value]
                if len(ledger_bindings) != 1:
                    raise ValueError("Expected one original endpoint-ledger hash binding")
                check(ledger, ledger_bindings[0])
                local_ledgers.append({"path": str(ledger.relative_to(ROOT)),
                                      "bytes": ledger.stat().st_size, "sha256": digest(ledger),
                                      "publication": "local derivative; reproduce with public workflow"})

    figure_records = []
    for folder in ("multimonkey_coarse_20261009_layout-v2", "additional_candidates_20261009",
                   "multimonkey_coarse_20261009_terms-v3", "additional_candidates_20261009_terms-v2"):
        for path in sorted((EVOLUTION / "figures" / folder).glob("*/matched_slices_provenance.json")):
            record = read(path)
            if record["fixed_slice_voxels_xyz"] != [56, 200, 87]:
                errors.append(f"Changed fixed display cuts: {path}")
            for item in record["figures"]:
                png = path.parent / item["file"]
                check(png, item["sha256"])
                with Image.open(png) as picture:
                    picture.verify()
                if item["panel_title_overlap_check"] != "passed":
                    errors.append(f"Panel title overlap: {png}")
            figure_records.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                                   "figures": len(record["figures"])})
    if sum(item["maps_rehashed"] for item in map_reviews) != 245:
        errors.append("Expected 245 reviewed maps")
    if (len(figure_records), sum(item["figures"] for item in figure_records)) != (18, 66):
        errors.append("Expected 18 original/readable figure families and 66 PNGs")

    current = verify_current_deliveries(read, rows, digest, check, errors)
    documents = [ROOT / "README.md", EVOLUTION / "README.md", AUDIT / "README.md", ROOT / "docs/region_analysis_terminology.md",
                 EVOLUTION / "arm_mapping_20261009/README.md", EVOLUTION / "projection_inputs/arm_labels_20261009/README.md",
                 EVOLUTION / "classification/background_neuron_audit_20261009/README.md",
                 AUDIT / "hierarchy_tables_20261009/README.md", AUDIT / "clustering_20261009/README.md",
                 AUDIT / "mstim_integration_20261009/README.md",
                 EVOLUTION / "reports/evolution_status_20261009.md", EVOLUTION / "reports/mapping_workflow_20261009.md",
                 EVOLUTION / "figures/multimonkey_coarse_20261009_terms-v3/README.md",
                 EVOLUTION / "figures/additional_candidates_20261009_terms-v2/README.md"]
    link_count = 0
    for path in documents:
        for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"[a-zA-Z]+://", target) or target.startswith("#"):
                continue
            link_count += 1
            resolved = (path.parent / target.split("#", 1)[0].strip("<>")).resolve()
            if resolved != output and not resolved.exists():
                errors.append(f"Broken report link: {path.relative_to(ROOT)} -> {target}")
    source_paths = sorted(set(ROOT.glob("main_scripts/*.py")) |
                          set((ROOT / "main_scripts/region_analysis").glob("*.py")) |
                          set((ROOT / "group_analysis/scripts").glob("*.py")) |
                          set((ROOT / "group_analysis/R_analysis").glob("*.R*")))
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    live = read(AUDIT / "validated_repairs/live_suite.json")
    methods_path = AUDIT / "methods_audit/final_live_methods_validation_summary_20261009.json"
    if live["status"] != "pass" or not live["live"] or live["errors"] != 0 or live["failures"] != 0:
        errors.append("Final live software suite did not pass")
    suite_log = ROOT / live["log"]
    suite_text = suite_log.read_text(encoding="utf-8")
    suite_counts = re.findall(r"Ran (\d+) tests? in", suite_text)
    if (not suite_counts or int(suite_counts[-1]) != live["tests_run"]
            or not re.search(r"\nOK(?: \(skipped=\d+\))?\s*$", suite_text)):
        errors.append("Final suite log and receipt do not match a completed passing run")
    test_paths = sorted(set((ROOT / "tests").rglob("test*.py")) | {
        AUDIT / relative for relative in ("code_audit/test_hierarchy_and_plots.py", "code_audit/test_standalone_identity.py",
            "methods_audit/test_final_staged_consumers_independent.py", "code_audit/test_harmonizer_preservation.py")})
    report = {"status": "passed" if not errors else "failed",
              "checked_utc": datetime.now(timezone.utc).isoformat(), "checker_sha256": digest(__file__),
              "python": sys.version, "libraries": {name: importlib.metadata.version(name) for name in
                  ("numpy", "pandas", "scipy", "nibabel", "matplotlib", "openpyxl", "Pillow")},
              "branch": subprocess.check_output(git + ["branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "head_at_validation": subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_receipt_sha256": digest(source_receipt), "source_bindings_verified": len(bindings),
              "distinct_source_paths_verified": len(expected_sources),
              "corrected_diagnostic_workbooks": len(corrected["results"]),
              "correction_provenance_sha256": digest(corrections),
              "selection_counts": {key: len(value) for key, value in selections.items()},
              "main_preview_overlap": len(selections["main"] & selections["preview"]),
              "map_reviews": map_reviews, "figure_families": figure_records,
              "historical_counts": {"reviewed_map_files_rehashed": sum(item["maps_rehashed"] for item in map_reviews),
                                    "figure_families": len(figure_records), "PNGs_rehashed": sum(item["figures"] for item in figure_records)},
              "current_counts": {"ARM_main_map_files_rehashed": sum(item["map_files_rehashed"] for item in current["ARM_main_map_reviews"]),
                                 "primary_group_figure_families": current["ARM_figures"]["primary_group_families"],
                                 "primary_mapped_PNGs": current["ARM_figures"]["primary_mapped_PNGs"],
                                 "archived_current_run_PNGs_rehashed": current["ARM_figures"]["archived_current_run_PNGs_rehashed"],
                                 "combined_ledger_neurons": current["combined_ARM_manifest"]["rows"], "source_ARM0_QC_cases": current["source_case_QC"]["cases"]},
              "current_deliveries": current,
              "local_endpoint_ledgers": local_ledgers,
              "current_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
              "current_document_sha256": {str(path.relative_to(ROOT)): digest(path) for path in documents},
              "relative_links_checked": link_count,
              "live_suite": live, "live_suite_receipt_sha256": digest(AUDIT / "validated_repairs/live_suite.json"),
              "live_suite_log_sha256": digest(suite_log),
              "live_suite_runner_sha256": digest(AUDIT / "validated_repairs/run_validation.py"),
              "current_test_sha256": {str(path.relative_to(ROOT)): digest(path) for path in test_paths},
              "methods_receipt_sha256": digest(methods_path),
              "historical_producer_receipts_rewritten": False,
              "scope": "Current hashes, identities, schemas and display readback; prior complete numerical map checks reused after hash verification",
              "scientific_acceptance": False, "source_writes": False, "errors": errors}
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "source_bindings_verified", "distinct_source_paths_verified", "corrected_diagnostic_workbooks", "selection_counts", "errors")}))
    return 0 if not errors else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUDIT / "final_delivery_receipt_20261009.json")
    raise SystemExit(verify_delivery(parser.parse_args().output))
