"""Read back the clarified-scope evidence without fitting or mutating sources."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    checked, errors = {}, []
    refresh = read(HERE / "mstim_source_update_20261009/cm033_source_refresh.json")
    refreshed = {str(Path(item["path"])): item
                 for item in refresh["changed_historical_bindings"]}
    acknowledged_historical_bindings = []

    def check(path, expected):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        replacement = refreshed.get(str(path))
        if replacement and expected == replacement["historical_sha256"]:
            acknowledged_historical_bindings.append(str(path))
            expected = replacement["current_sha256"]
        current = digest(path)
        if current != expected:
            errors.append(f"Changed binding: {path}")
        checked[str(path)] = current

    def pairs(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                check(value["path"], value["sha256"])
            for item in value.values():
                pairs(item)
        elif isinstance(value, list):
            for item in value:
                pairs(item)

    cm032 = read(HERE / "mstim_source_update_20261009/cm032_site_figure_review.json")
    pairs(cm032)
    check(cm032["report_path"], cm032["report_sha256"])
    cm033 = read(HERE / "mstim_source_update_20261009/cm033_updated_debfmri_evidence.json")
    for path, expected in cm033["source_sha256"].items():
        check(path, expected)
    pairs(cm033)
    pairs(refresh)
    check(refresh["historical_receipt"], refresh["historical_receipt_sha256"])
    gao = read(HERE / "literature_method_update_20261009/gao_local_methods.json")
    for paper in gao["papers"]:
        check(paper["pdf_path"], paper["pdf_sha256"])
    check(HERE / "literature_method_update_20261009/gao_local_methods.md", gao["review_md_sha256"])
    liu = read(HERE / "literature_method_update_20261009/liu_local_methods.json")
    pairs(liu)
    check(liu["note_path"], liu["note_sha256"])

    terminal_dir = HERE / "terminal_field_assessment_20261009"
    terminal = read(terminal_dir / "codex_terminal_field_assessment.json")
    for name in ["codex_terminal_field_assessment.json", "independent_terminal_review.json",
                 "native_cube_pilot_current_rint/generation_provenance.json",
                 "optical_plane_checks_current_rint/optical_plane_provenance.json",
                 "review_panels/display_provenance.json"]:
        pairs(read(terminal_dir / name))
    morphology_display = terminal_dir / "review_panels/morphology_display_provenance.json"
    if morphology_display.exists():
        pairs(read(morphology_display))
    # Bind the actual native/atlas/cube inputs in all five selected review units.
    for path in (terminal_dir / "native_cube_pilot_current_rint").glob("*/assessment_inputs.json"):
        pairs(read(path))
    assert terminal["selected_neurons"] == 462
    assert terminal["native_swcs_in_designated_cache"] == terminal["node_identity_type_parent_matches"] == 201
    assert terminal["reviewed_neurons"] + terminal["remaining_neurons_without_this_pilot_review"] == 462
    assert terminal["individually_image_reviewed_axon_leaves"] == 5
    assert terminal["biologically_accepted_terminals_or_synapses"] == 0

    # Check every selected native/NMT pair independently of the producer's
    # dictionary signatures. Integer IDs, types and parents agree after sorting.
    with (terminal_dir / "native_cube_pilot_current_rint/selected462_native_image_coverage.csv").open(
            encoding="utf-8", newline="") as stream:
        census = list(csv.DictReader(stream))
    assert len(census) == len({row["NeuronUID"] for row in census}) == 462
    cache = ROOT / "group_analysis/visual_review_20261002/cache/cubes"
    cubes = {str(path.relative_to(cache)) for path in cache.rglob("*.tif")}
    matched, leaves_total, covered_total, covered_neurons = 0, 0, 0, 0
    for row in census:
        check(row["atlas_swc_path"], row["atlas_swc_sha256"])
        if row["native_swc_available"] != "True":
            continue
        check(row["native_swc_path"], row["native_swc_sha256"])
        native = np.loadtxt(ROOT / row["native_swc_path"], ndmin=2)
        warped = np.loadtxt(ROOT / row["atlas_swc_path"], ndmin=2)
        a, b = native[:, [0, 1, 6]], warped[:, [0, 1, 6]]
        assert np.array_equal(a, a.astype(np.int64))
        assert np.array_equal(b, b.astype(np.int64))
        assert len(np.unique(a[:, 0])) == len(a) == len(b)
        assert np.array_equal(a[np.argsort(a[:, 0])], b[np.argsort(b[:, 0])])
        matched += 1
        leaf = (native[:, 1] == 2) & (native[:, 6] != -1) & ~np.isin(native[:, 0], native[:, 6])
        indices = np.floor(native[leaf, 2:5] / [234, 234, 270]).astype(np.int64)
        sample = row["NeuronUID"].split("::")[0]
        covered = sum(str(Path(sample) / "high_res_http" / str(z) / f"{x}_{y}_{z}.tif") in cubes
                      for x, y, z in indices)
        assert int(leaf.sum()) == int(row["candidate_axon_leaves"])
        assert covered == int(row["leaves_with_cached_native_cube"])
        leaves_total += int(leaf.sum())
        covered_total += covered
        covered_neurons += bool(covered)
    assert (matched, leaves_total, covered_total, covered_neurons) == (201, 48574, 9209, 184)

    # Reuse the prior whole-pipeline run only after current source/test hashes agree.
    prior = read(HERE / "final_delivery_recheck_20261009.json")
    completion = read(HERE / "goal_completion_20261009/completion_status.json")
    overrides = {item["path"]: item["current_sha256"]
                 for item in completion["source_changes_since_delivery"]}
    for path, expected in prior["current_source_sha256"].items():
        check(path, overrides.get(path.replace("\\", "/"), expected))
    for path, expected in prior["current_test_sha256"].items():
        check(path, expected)
    if overrides and not completion["executable_AST_equal_to_committed_tested_source"]:
        errors.append("The recorded comment-only source exception lacks executable equality")

    test = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s",
                           str(terminal_dir), "-p", "test_native_terminal_assessment.py", "-v"],
                          cwd=ROOT, text=True, capture_output=True, encoding="utf-8")
    if test.returncode:
        errors.append("Focused terminal tests failed")

    docs = [ROOT / "docs/project_note.md", ROOT / "group_analysis/evolution_20261008/README.md",
            ROOT / "group_analysis/evolution_20261008/terminal_projection_goal_20261008.md",
            ROOT / "group_analysis/evolution_20261008/references/scientific_basis_20261009.md",
            ROOT / "group_analysis/evolution_20261008/reports/evolution_status_20261009.md",
            HERE / "README.md", HERE / "goal_completion_20261009/README.md",
            HERE / "mstim_integration_20261009/README.md", HERE / "scope_update_20261009.md"]
    for directory in [HERE / "literature_method_update_20261009",
                      HERE / "mstim_source_update_20261009", terminal_dir]:
        docs.extend(directory.glob("*.md"))
    links = 0
    for path in docs:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#")[0]
            if not target or re.match(r"https?://|mailto:", target):
                continue
            target = re.sub(r":\d+$", "", target)
            resolved = Path(target) if Path(target).is_absolute() else path.parent / target
            links += 1
            # This receipt is the sole intentionally not-yet-written link.
            if resolved.resolve() == (HERE / "scope_update_20261009.json").resolve():
                continue
            if not resolved.exists():
                errors.append(f"Unresolved link in {path.name}: {target}")
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed", "errors": errors,
        "verifier_sha256": digest(__file__), "source_paths_rehashed": len(checked),
        "source_sha256": checked, "current_document_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in docs},
        "relative_links_checked": links, "fresh_focused_tests": {
            "returncode": test.returncode, "output": test.stdout + test.stderr},
        "saved_whole_pipeline_tests": prior["live_suite"],
        "saved_suite_rerun_in_this_update": False,
        "documented_external_source_refresh": str(HERE / "mstim_source_update_20261009/cm033_source_refresh.json"),
        "external_record_changes_acknowledged": sorted(set(acknowledged_historical_bindings)),
        "prior_failed_freshness_receipt": str(HERE / "scope_update_20261009_initial_readback.json"),
        "reviewed_neurons": 5, "remaining_without_pilot_image_review": 457,
        "independent_all_selected_census": {"selected": len(census), "native_identity_pairs": matched,
                                             "strict_native_axon_leaves": leaves_total,
                                             "cached_leaf_cubes": covered_total,
                                             "neurons_with_any_cached_leaf": covered_neurons},
        "original_fMOST_transform": "user_confirmed_unavailable_not_a_recovery_gate",
        "CM032_coarse_site": "human_session_record_supported_left_Ial_pos1_MSTIM38-41",
        "CM033_current_spatial_integration": "unsupported_by_checked_current_sources",
        "full_refined_scientific_goal_achieved": False,
        "scientific_anatomical_acceptance": False, "source_writes": False,
        "historical_receipts_rewritten": False,
    }
    (HERE / "scope_update_20261009.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key in {"status", "errors", "source_paths_rehashed", "relative_links_checked"}}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
