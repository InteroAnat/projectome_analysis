"""Inventory and scan the explicit publication scope; never stage or push files.

Only reviewed pipeline dependencies and dated evidence are selected. Original
scientific volumes, raw reconstructions and unrelated local work stay local.
"""
from pathlib import Path
import ast
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
EVOLUTION = ROOT / "group_analysis/evolution_20261008"
GIT = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
BASE_COMMIT = "78008cbaab71bf737dc7129c72e5fe284c3ba40f"
BASELINE = {
    "main_scripts/Visual_toolkit.py", "main_scripts/fnt_dist_clustering.py",
    "main_scripts/clustering_validation.py", "main_scripts/fmost_context_resources.py",
    "main_scripts/fmost_derived_context.py", "main_scripts/fmost_image_geometry.py",
    "main_scripts/fmost_review_lock.py", "main_scripts/swc_validation.py", "main_scripts/region_labels.py",
    "main_scripts/render_cached_toolkit_pair.py",
    "group_analysis/scripts/visual_review_20261002.py",
    "group_analysis/scripts/regional_context_batch_20261003.py",
    "tests/test_clustering_validation.py", "tests/test_fnt_clustering.py",
    "tests/test_fmost_image_loader.py", "tests/test_fmost_export_provenance.py",
    "tests/test_fmost_context_batch.py", "tests/test_fmost_derived_context.py",
    "tests/test_visual_review_manifest.py", "tests/test_swc_validation.py",
}
BASELINE_RECORDS = {
    "notes/bulk_visual_review_20261002/visual_methods.md",
    "notes/bulk_visual_review_20261002/derived_context_repair_validation_20261003.json",
    "notes/bulk_visual_review_20261002/regional_driver_validation_20261003.json",
    "notes/bulk_visual_review_20261002/independent_regional_resume_validation_20261003.json",
}
MAP_SCRIPTS = {
    "03_scan_insula_recovery_candidates.py", "04_refine_soma_region_by_coords.py",
    "audit_atlas_coordinate_origin_sensitivity.py", "audit_atlas_soma_locations.py",
    "audit_insula_inventory.py", "build_coarse_insula_review.py", "build_endpoint_maps.py",
    "build_projection_maps.py", "prepare_inventory_projection_manifest.py",
    "prepare_projection_manifest.py", "prepare_review_candidate_manifest.py",
    "prepare_arm_labeled_projection_manifest.py", "render_projection_slices.py",
    "review_endpoint_run.py", "review_projection_run.py", "verify_atlas_soma_audit.py",
    "export_arm_projection_tables.py", "cluster_arm_projection_profiles.py", "summarize_mstim_arm.py",
    "build_axon_end_branch_maps.py",
    "build_monkey_insula_inventory.py",
    "map_projections_by_soma_origin.py", "map_insula_origin_evidence.py",
    "map_legacy_strength_to_arm_parcels.py",
}
CORE_MODULES = {"endpoint_atlas.py", "projection_maps.py", "terminal_sites.py"}
OMIT_NAMES = {"publication_files.json", "publication_scan.json", "publication_local_artifacts.json", "publication_index_receipt.json"}
LOCAL_NODE_LEDGERS = {"leaf_records.jsonl", "section_original_node_membership.jsonl"}
OMIT_SUFFIXES = {".pyc", ".swc", ".tif", ".tiff", ".nii", ".gz", ".pdf"}
TEXT_SUFFIXES = {".py", ".ps1", ".r", ".rmd", ".md", ".txt", ".json", ".csv", ".bib", ".patch", ".log", ".yml"}
CM033_WORK_DIRECTORIES = {"nipype_work", "nipype_config", "mpl_config", "logs", "tmp"}
# This interrupted supplement has no completed independent delivery. Preserve
# it locally while the user-requested projectome phase is finalized first.
DEFERRED_WORK_PREFIX = "notes/region_analysis_review_20261009/mstim_integration_20261009/cm032_end_branch_supplement/"
# A token starts at a lexical boundary. Without this guard, BIDS task-opto
# filenames match the sk- prefix embedded in the word task.
ACCESS_TOKEN_PATTERN = (
    r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{24,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{25,})"
    r"(?![A-Za-z0-9_])"
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git_paths(arguments):
    result = subprocess.check_output(GIT + arguments, cwd=ROOT)
    return {value.decode("utf-8").replace("\\", "/") for value in result.split(b"\0") if value}


def is_cm033_work_artifact(path):
    """Exclude disposable fit caches while retaining named sources and receipts."""
    parts = Path(path).as_posix().split("/")
    prefix = ["notes", "region_analysis_review_20261009"]
    return (parts[:2] == prefix and len(parts) > 3
            and parts[2].startswith("cm033_")
            and any(part in CM033_WORK_DIRECTORIES for part in parts[3:-1]))


def is_cm033_earlier_display(path):
    """Publish the readable candidate QC set; preserve earlier displays locally."""
    parts = Path(path).as_posix().split("/")
    return (parts[:4] == ["notes", "region_analysis_review_20261009",
                         "cm033_bridge_candidate_20261009", "coarse_qc"]
            and len(parts) == 5 and Path(path).suffix.lower() in {".png", ".json"})


def main():
    if subprocess.check_output(GIT + ["diff", "--cached", "--name-only"], cwd=ROOT).strip():
        raise ValueError("Start publication inventory with an empty index; do not mix unrelated staged work")
    modified = git_paths(["ls-files", "-m", "-z"])
    candidates = set(BASELINE) | set(BASELINE_RECORDS)
    candidates |= git_paths(["diff", "--name-only", "-z", BASE_COMMIT, "HEAD"])
    candidates |= {path for path in modified if not path.startswith(".cursor/")
                   and "__pycache__" not in path and Path(path).suffix.lower() != ".pyc"}
    candidates |= {str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "tests").glob("*.py")}
    candidates |= {f"group_analysis/scripts/{name}" for name in MAP_SCRIPTS}
    candidates |= {f"main_scripts/{name}" for name in CORE_MODULES}
    candidates.add("group_analysis/data_progress/insula_inventory.py")
    candidates |= {"docs/region_analysis_terminology.md", "requirements-validation.txt",
                   "notes/whole_insula_lr_continuation_plan_v2.md"}
    for directory in (AUDIT, EVOLUTION, ROOT / "notes/clustering_review_20261002", ROOT / "notes/projection_map_review_round2", ROOT / "notes/projection_maps_by_origin"):
        for path in directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if (path.name in OMIT_NAMES or path.name in LOCAL_NODE_LEDGERS
                    or path.suffix.lower() in OMIT_SUFFIXES or path.name.startswith("publication_paths_")):
                continue
            candidates.add(str(path.relative_to(ROOT)).replace("\\", "/"))
    # Previously committed publication manifests also appear in the Git diff.
    # Exclude them here as well as during discovery to avoid self-hash cycles.
    candidates = {path for path in candidates if Path(path).name not in OMIT_NAMES | LOCAL_NODE_LEDGERS
                  and not Path(path).name.startswith("publication_paths_")}
    # Publish one current ARM figure set. Earlier display variants and animal/QC
    # panels remain immutable local derivatives with hashes in the manifest.
    def primary_figure(path):
        prefix = "group_analysis/evolution_20261008/arm_mapping_20261009/main/figures/"
        family = path[len(prefix):].split("/", 1)[0] if path.startswith(prefix) else ""
        return (path.startswith("group_analysis/evolution_20261008/soma_origin_maps_20261010/")
                or path.startswith("group_analysis/evolution_20261008/arm_target_strength_20261010/")
                or path == "group_analysis/evolution_20261008/inventory/legacy_overview_20261009/per_monkey_insula_overview.png"
                or family in {"endpoint-density_group", "endpoint-occupancy_group", "axon-density_group"}
                and "unresolvedQC" not in path)

    archived_figures = {path for path in candidates if path.startswith("group_analysis/evolution_20261008/")
                        and Path(path).suffix.lower() == ".png" and not primary_figure(path)}
    archived_figures |= {path for path in candidates if path.endswith(
        "mstim_integration_20261009/cm032_regional_comparison/cm032_alignment_and_response_qc.png")}
    archived_figures |= {path for path in candidates if
        "clustering_20261009/arm_L6_relative_profiles_selected462/" in path and
        Path(path).name in {"all_axon_hellinger_target_profiles.png", "all_endpoint_hellinger_target_profiles.png"}}
    terminal_prefix = "notes/region_analysis_review_20261009/terminal_field_assessment_20261009/"
    # Keep one readable plane/passage/morphology set. Earlier rounding/display
    # diagnostics remain local and hashed.
    archived_figures |= {path for path in candidates if path.startswith(terminal_prefix)
                        and Path(path).suffix.lower() == ".png"
                        and not path.startswith(terminal_prefix + "review_panels/")}
    archived_figures |= {path for path in candidates if is_cm033_earlier_display(path)}
    local_work = {path for path in candidates if is_cm033_work_artifact(path)
                  or path.startswith(DEFERRED_WORK_PREFIX)}
    candidates -= archived_figures
    candidates -= local_work
    for path in candidates:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
        if (ROOT / path).stat().st_size >= 100 * 1024**2:
            raise ValueError(f"Oversized Git artifact: {path}")
    # Compare candidate bytes with the inherited password in memory only. Never
    # print or persist its value, including in the diagnostic scan report.
    inherited = subprocess.check_output(GIT + ["show", f"{BASE_COMMIT}:main_scripts/Visual_toolkit.py"], cwd=ROOT).decode("utf-8")
    secret_values = []
    for node in ast.walk(ast.parse(inherited)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(target, ast.Name) and target.id == "SSH_PASS" for target in node.targets):
                if isinstance(node.value.value, str) and len(node.value.value) >= 4:
                    secret_values.append(node.value.value)
    patterns = {
        "access_token": re.compile(ACCESS_TOKEN_PATTERN),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "credential_literal": re.compile(r"(?im)^\s*(?:[A-Z_]*PASSWORD|[A-Z_]*SSH_PASS|[A-Z_]*API_KEY|[A-Z_]*ACCESS_TOKEN)\s*=\s*[\"']([^\"']{4,})[\"']"),
    }
    findings, text_count = [], 0
    for path in sorted(candidates):
        if Path(path).suffix.lower() not in TEXT_SUFFIXES:
            continue
        text_count += 1
        contents = (ROOT / path).read_text(encoding="utf-8-sig", errors="replace")
        for category, pattern in patterns.items():
            for match in pattern.finditer(contents):
                if category == "credential_literal" and match.group(1).lower() in ("password", "your_password", "test", "fixture"):
                    continue
                findings.append({"path": path, "line": contents[:match.start()].count("\n") + 1, "category": category})
        for value in secret_values:
            if value in contents:
                findings.append({"path": path, "category": "inherited_credential_bytes"})
    groups = {"baseline": [], "repairs": [], "evidence": []}
    for path in sorted(candidates):
        group = ("baseline" if path in BASELINE or path in BASELINE_RECORDS or path.startswith("notes/clustering_review_20261002/")
                 else "evidence" if path.startswith(("notes/region_analysis_review_20261009/", "group_analysis/evolution_20261008/"))
                 or path in {"README.md", "docs/project_note.md", "docs/region_analysis_terminology.md", ".gitignore"}
                 else "repairs")
        groups[group].append({"path": path, "bytes": (ROOT / path).stat().st_size, "sha256": sha(ROOT / path)})
    local = [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size,
              "sha256": sha(path), "reason": (
                  "Archived display variant; its guide identifies the current figure set"
                  if str(path.relative_to(ROOT)).replace("\\", "/") in archived_figures else
                  "Unfinished MSTIM supplement deferred by the user; preserved locally"
                  if str(path.relative_to(ROOT)).replace("\\", "/").startswith(DEFERRED_WORK_PREFIX) else
                  "Disposable registration workflow/cache; frozen commands, matrices and receipts are published"
                  if is_cm033_work_artifact(str(path.relative_to(ROOT))) else
                  "Scientific binary/large local derivative; see bound run and reproduction workflow")}
             for directory in (EVOLUTION, AUDIT) for path in directory.rglob("*") if path.is_file()
             and (path.name in LOCAL_NODE_LEDGERS or path.suffix.lower() in {".nii", ".gz", ".swc", ".tif", ".tiff"}
                  or is_cm033_work_artifact(str(path.relative_to(ROOT)))
                  or str(path.relative_to(ROOT)).replace("\\", "/").startswith(DEFERRED_WORK_PREFIX)
                  or str(path.relative_to(ROOT)).replace("\\", "/") in archived_figures)]
    stamp = datetime.now(timezone.utc).isoformat()
    (AUDIT / "publication_files.json").write_text(json.dumps({"created_utc": stamp, "groups": groups}, indent=2) + "\n", encoding="utf-8")
    (AUDIT / "publication_scan.json").write_text(json.dumps({"created_utc": stamp, "status": "passed" if not findings else "failed",
        "text_files_scanned": text_count, "findings": findings, "values_logged": False,
        "scope": "Selected task files only; repository-history credentials are not erased"}, indent=2) + "\n", encoding="utf-8")
    (AUDIT / "publication_local_artifacts.json").write_text(json.dumps({"created_utc": stamp, "artifacts": local}, indent=2) + "\n", encoding="utf-8")
    for group, rows in groups.items():
        (AUDIT / f"publication_paths_{group}.txt").write_text("\n".join(row["path"] for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"groups": {group: {"files": len(rows), "bytes": sum(row["bytes"] for row in rows)} for group, rows in groups.items()},
                      "credential_scan_status": "passed" if not findings else "failed", "findings": findings,
                      "local_scientific_artifacts": len(local)}))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
