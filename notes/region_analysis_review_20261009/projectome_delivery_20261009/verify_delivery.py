"""Verify the frozen projectome delivery; no new imaging or clustering analysis."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = HERE.parent


def main():
    output = HERE / "delivery_checks.json"
    if output.exists():
        raise FileExistsError("Preserve the receipt; copy this reader to a fresh delivery directory")
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        if str(path) not in bindings:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1048576), b""):
                    h.update(block)
            bindings[str(path)] = h.hexdigest()
        if expected is not None and bindings[str(path)] != expected:
            raise ValueError(f"Changed validated binding: {path}")

    def read(path):
        bind(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def source_records(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                yield value
            for child in value.values():
                yield from source_records(child)
        elif isinstance(value, list):
            for child in value:
                yield from source_records(child)

    def rows(path):
        bind(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    checkpoint = read(AUDIT / "axon_end_branches_20261009/checkpoint_readback.json")
    if checkpoint["status"] != "frozen_end_branch_checkpoint_readback_passed":
        raise ValueError("Absent independently validated end-branch checkpoint")
    for record in checkpoint["files"]:
        bind(record["path"], record["sha256"])
    suite = read(AUDIT / "cm033_mstim_validation_20261009_transport/live_suite_20261009.json")
    if (suite["status"], suite["tests_run"], suite["errors"], suite["failures"], suite["skipped"]) != ("passed", 437, 0, 0, 0):
        raise ValueError("Unexpected final live suite")
    for path, expected in suite["source_sha256"].items():
        bind(path, expected)
    tables = AUDIT / "hierarchy_tables_20261009/combined_arm_projection_tables_462"
    hierarchy = read(tables / "export_provenance.json")
    for path, expected in hierarchy["source_bindings"].items():
        bind(path, expected)
    for path, expected in hierarchy["artifacts"].items():
        bind(tables / path, expected)
    selected = rows(tables / "neuron_summary.csv")
    branches = rows(AUDIT / "axon_end_branches_20261009/selected462_ARM/input_ledger.csv")
    identities = lambda records: {r.get("NeuronUID", r.get("uid")) for r in records}
    if len(selected) != 462 or len(branches) != 462 or len(identities(selected)) != 462 or identities(selected) != identities(branches):
        raise ValueError("Changed master-ledger identities")
    inventory = ROOT / "group_analysis/evolution_20261008/inventory/live_inventory_20261008"
    provenance = read(inventory / "provenance.json")
    independent = read(inventory / "independent_snapshot_verification.json")
    for record in source_records(provenance):
        bind(record["path"], record["sha256"])
    samples = rows(inventory / "sample_inventory.csv")
    neurons = rows(inventory / "neuron_inventory.csv")
    if len(samples) != 49 or len(neurons) != 8746 or len({r["uid"] for r in neurons}) != 8746:
        raise ValueError("Changed inventory identities or totals")
    if not independent["all_snapshot_hashes_match"] or not independent["output_exact_identity_set_matches_raw_neuron_lists"]:
        raise ValueError("Absent independent inventory agreement")
    classification = read(ROOT / "group_analysis/evolution_20261008/classification/coarse_insula_review_20261009/provenance.json")
    for record in source_records(classification["inputs"]):
        bind(record["path"], record["sha256"])
    if classification["anatomical_acceptance"] or hierarchy["scientific_anatomical_acceptance"] or checkpoint["scientific_anatomical_acceptance"]:
        raise ValueError("Software delivery cannot confer anatomical acceptance")
    links = []
    for target in re.findall(r"\]\(([^)]+)\)", (HERE / "README.md").read_text(encoding="utf-8")):
        if target.startswith(("https:", "http:", "#")):
            continue
        path = (HERE / target.split("#", 1)[0]).resolve()
        if path == output:
            continue
        bind(path)
        links.append(str(path))
    for path in (HERE / "README.md", Path(__file__), ROOT / "docs/project_note.md",
                 ROOT / "docs/region_analysis_terminology.md", AUDIT / "README.md",
                 ROOT / "group_analysis/evolution_20261008/README.md"):
        bind(path)
    command = [sys.executable, "-X", "utf8", "-B", "-m", "unittest",
               "test_publication_artifacts", "test_publication_credentials"]
    result = subprocess.run(command, cwd=AUDIT, capture_output=True, text=True, encoding="utf-8")
    log = HERE / "publication_tool_tests.log"
    with log.open("x", encoding="utf-8") as stream:
        stream.write(result.stdout + result.stderr)
    if result.returncode or "Ran 9 tests" not in result.stderr:
        raise ValueError("Publication-tool regression failure")
    bind(log)
    bind(AUDIT / "prepare_publication.py")
    bind(AUDIT / "verify_publication_index.py")
    receipt = {"status": "validated_projectome_phase_readback_passed",
               "checked_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Projectome delivery before deferred CM032/CM033 interpretation; fresh byte/identity/link readback of previously independently validated results.",
               "new_imaging_or_clustering_analysis": False,
               "scientific_anatomical_acceptance": False, "mstim_integration_deferred": True,
               "selected_neurons": 462, "end_eligible_neurons": 429,
               "inventory_sample_identities": 49, "inventory_neuron_identities": 8746,
               "end_branch_maps_bound": 67, "live_scientific_tests": 437,
               "current_scientific_source_hashes_match_suite": True,
               "fresh_publication_tool_tests": 9, "resolved_report_links": links,
               "source_binding_count": len(bindings), "source_sha256": bindings}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "bindings": len(bindings), "links": len(links)}))


if __name__ == "__main__":
    main()
