"""Rehash all bound local image/source artifacts after terminal source audits."""
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil

from verify_sample_regional_milestone_20261004 import HERE, ROOT, OUT, digest, read_json


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def main():
    expected, inputs = {}, {}

    def add(path, expected_sha):
        path = Path(path).resolve()
        assert ROOT == path or ROOT in path.parents, f"Unexpected external path: {path}"
        name = str(path)
        assert len(expected_sha) == 64
        assert name not in expected or expected[name] == expected_sha, f"Conflicting evidence: {name}"
        expected[name] = expected_sha

    def load(name):
        path = HERE / name
        inputs[str(path)] = digest(path)
        payload = read_json(path)
        assert payload["status"] == "passed" and not payload.get("failures"), name
        return payload

    ledger_path = OUT / "manifest/regional_context_run_20261003.json"
    ledger = read_json(ledger_path)
    queue_path = HERE / "copied_source_audit_queue_20261003.json"
    queue = read_json(queue_path)
    assert ledger["state"] == "terminal_scope_attempted" and not psutil.pid_exists(ledger["pid"])
    assert queue["state"] == "source_audit_passed" and queue["exit_code"] == 0
    assert not psutil.pid_exists(queue["source_audit_pid"])
    assert queue["regional_run_id"] == ledger["run_id"]
    manifest = OUT / "manifest/identity_manifest.csv"
    manifest_sha = digest(manifest)
    assert manifest_sha == ledger["manifest_sha256"] == queue["manifest_sha256"]
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    uids = [row["UID"] for row in rows]
    assert len(uids) == len(set(uids)) == 832
    samples = Counter(row["SampleID"] for row in rows)
    assert len(samples) == 9
    soma = load("all_soma_source_pixel_readback_20261003.json")
    graph = load("all_raw_swc_graph_readback_20261004.json")
    copied = load("copied_context_package_readback_20261003.json")
    source = load("copied_context_source_pixel_readback_20261003.json")
    regional = load("terminal_regional_scope_readback_20261004.json")
    centers = load("terminal_missing_center_reconciliation_20261004.json")
    refresh = load("review_artifact_refresh_20261003.json")
    for audit in (soma, graph, copied, source, centers, refresh):
        assert audit["manifest_sha256"] == manifest_sha
    assert [r["uid"] for r in soma["records"]] == [r["uid"] for r in graph["records"]] == uids
    assert soma["outcome_counts"] == {"complete": 782, "partial_neighbor": 28, "center_unavailable": 20, "swc_unavailable": 2}
    assert graph["outcome_counts"] == {"graph_valid": 830, "own_swc_unavailable": 2}
    assert graph["total_nodes"] == 16929320
    assert graph["full_soma_audit_sha256"] == inputs[str(HERE / "all_soma_source_pixel_readback_20261003.json")]
    add(ROOT / "main_scripts/swc_validation.py", graph["strict_parser_sha256"])
    add(HERE / "verify_all_raw_swc_graphs_20261004.py", graph["script_sha256"])
    for record in soma["records"]:
        for key in ("artifact_hashes", "cached_source_hashes"):
            for path, sha in record.get(key, {}).items():
                add(path, sha)
        if record["outcome"] == "center_unavailable":
            sample, filename = record["uid"].split("|", 1)
            add(OUT / "swc_raw" / sample / filename, record["raw_sha256"])
    for record in graph["records"]:
        sample, filename = record["uid"].split("|", 1)
        raw = OUT / "swc_raw" / sample / filename
        assert record["own_raw_url"] == f"http://10.10.31.31/swc/newswc/Monkey/{sample}/swc_raw/{filename}"
        if record["outcome"] == "graph_valid":
            add(raw, record["raw_sha256"])
        else:
            assert not raw.exists(), "Unavailable own raw source changed; re-audit the affected component"
    copy_records = {r["uid"]: r for r in copied["records"]}
    pixel_records = {r["uid"]: r for r in source["records"]}
    assert len(copy_records) == len(pixel_records) == 301 and copy_records.keys() == pixel_records.keys()
    assert source["outcome_counts"] == {"source_pixels_equal": 299, "swc_unavailable": 2}
    assert source["regional_run_id"] == ledger["run_id"]
    assert source["local_package_audit_sha256"] == inputs[str(HERE / "copied_context_package_readback_20261003.json")]
    assert inputs[str(HERE / "copied_context_source_pixel_readback_20261003.json")] == queue["report_sha256"]
    for uid, record in copy_records.items():
        for path, sha in record["artifact_hashes"].items():
            add(path, sha)
        if record["usable_context"]:
            assert pixel_records[uid]["artifact_hashes"] == record["artifact_hashes"]
    assert regional["selected"] == 531 and regional["usable_fields_verified"] == 528
    for path, sha in regional["input_hashes"].items():
        add(path, sha)
    for sample in regional["samples"]:
        report = load(f"terminal_sample_{sample}_regional_readback_20261004.json")
        for record in report["records"]:
            stem = Path(record["uid"].split("|", 1)[1]).stem
            producer_path = OUT / "manifest" / f"derived_context_pilot_{sample}_{stem}.json"
            add(producer_path, record["report_sha256"])
            producer = read_json(producer_path)
            for entry in producer["source_paths"]:
                add(entry["path"], entry["sha256"])
            for phase, sha in record.get("phase_audit_sha256", {}).items():
                path = HERE / f"derived_{phase}_readback_{sample}_{stem}_20261003.json"
                add(path, sha)
                for artifact, artifact_sha in read_json(path)["artifact_hashes"].items():
                    add(artifact, artifact_sha)
    assert centers["earlier_missing_centers_reconciled"] == 20 and not centers["requires_soma_regeneration"]
    assert refresh["rows"] == 832 and refresh["independently_audited_fields_exposed"] == 528
    assert refresh["human_fields_preserved"] and refresh["canonical_inputs_unchanged"]
    assert refresh["csv_xlsx_equal_with_numeric_readback_tolerance"] and refresh["all_manifest_fields_equal_with_numeric_readback_tolerance"]
    for kind, sha in refresh["after_sha256"].items():
        add(OUT / "tables" / f"correction_table.{kind}", sha)
    for sample, page in refresh["all_nine_pages_checked"].items():
        add(OUT / "gallery" / f"{sample}.html", page["sha256"])
    assert set(refresh["all_nine_pages_checked"]) == set(samples)
    for sidecar in refresh["lossless_diagnostic_sidecars_checked"]:
        add(OUT / sidecar["detail_file"], sidecar["sha256"])
    baseline_path = OUT / "manifest/canonical_hashes.json"
    canonical = read_json(baseline_path)
    for path, sha in canonical.get("canonical", canonical).items():
        add(path, sha)
    origins_path = HERE / "expected_staging_origins_20261003.json"
    origins = read_json(origins_path)
    assert len(origins["records"]) == 160 and len(origins["sources"]) == 10
    table = pd.read_csv(OUT / "tables/correction_table.csv", dtype=str, keep_default_na=False).set_index("UID")
    for record in origins["records"]:
        expected_lineage = {value for value in (record["expected_staging_label_origin"],
                                                record.get("earlier_recovery_label_origin")) if value}
        assert set(table.at[record["UID"], "staging_region_source"].split("|")) == expected_lineage
        assert table.at[record["UID"], "staging_harmonized_region_source"] == record["expected_staging_label_origin"]
        assert table.at[record["UID"], "staging_refined_label"] == record["expected_staging_refined_label"]
    for info in origins["sources"].values():
        add(info["path"], info["sha256"])
    clustering = ROOT / "notes/clustering_review_20261002"
    metadata_paths = sorted(clustering.glob("*/clustering_metadata.json"))
    assert len(metadata_paths) == 6
    for path in metadata_paths:
        metadata = read_json(path)
        for prefix in ("distance", "joined_fnt", "annotation"):
            add(metadata[prefix + "_file"], metadata[prefix + "_sha256"])
        for name, sha in metadata["source_sha256"].items():
            add(ROOT / "main_scripts" / name, sha)
        assert metadata["n_neurons"] == (212 if path.parent.name.startswith("single_") else 306)
        inputs[str(path)] = digest(path)
    tests_path = HERE / "excel_limit_validation_20261003.json"
    tests = read_json(tests_path)
    assert tests["tests_passed"] == 108
    for path, sha in tests["file_sha256"].items():
        add(ROOT / path, sha)
    for path in (manifest, ledger_path, queue_path, baseline_path, origins_path, tests_path,
                 ROOT / "notes/repo_review_20261002.md", ROOT / "notes/fmost_methods_research_20261002.md",
                 ROOT / "notes/fmost_review_20261002.bib", ROOT / "README.md", Path(__file__)):
        inputs[str(path)] = digest(path)
    for path, sha in inputs.items():
        add(path, sha)
    add(OUT / "gallery/index.html", digest(OUT / "gallery/index.html"))
    total_bytes = sum(Path(path).stat().st_size for path in expected)
    target = HERE / "final_component_integrity_20261004.json"
    result = {"status": "running", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "All 832 identities and all bound local raw/image/source/phase/table/gallery/research inputs",
              "manifest_sha256": manifest_sha, "run_id": ledger["run_id"], "sample_counts": dict(samples),
              "expected_files": len(expected), "expected_bytes": total_bytes,
              "checked_files": 0, "checked_bytes": 0, "server_requests": 0,
              "source_pixels_recomputed": False, "anatomical_acceptance": False,
              "input_hashes": inputs, "failures": []}
    save(target, result)
    last_checkpoint = time.monotonic()
    print(json.dumps({"status": "running", "expected_files": len(expected), "expected_bytes": total_bytes}), flush=True)
    try:
        for path, sha in sorted(expected.items()):
            assert digest(path) == sha, f"Current file differs from accepted component evidence: {path}"
            result["checked_files"] += 1
            result["checked_bytes"] += Path(path).stat().st_size
            if time.monotonic() - last_checkpoint >= 10:
                result["observed_utc"] = datetime.now(timezone.utc).isoformat()
                save(target, result)
                last_checkpoint = time.monotonic()
        assert all(digest(path) == sha for path, sha in inputs.items())
        result.update(status="passed", checked_file_hashes=expected,
                      expected_hash_manifest_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                      usable_contexts=827, usable_soma_packages=810, graph_valid_swcs=830,
                      staging_origins_checked=160, staging_sources_checked=10, clustering_runs_bound=6)
    except Exception as exc:
        result["status"] = "failed"
        result["failures"].append({"type": type(exc).__name__, "error": str(exc)})
    result["observed_utc"] = datetime.now(timezone.utc).isoformat()
    save(target, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"input_hashes", "checked_file_hashes"}}), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
