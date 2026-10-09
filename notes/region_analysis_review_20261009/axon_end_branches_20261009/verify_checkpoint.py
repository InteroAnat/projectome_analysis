"""Read back frozen end-branch bindings without recomputing scientific results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("Preserve checkpoint receipts; select a fresh destination")
    checked = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        if path not in checked:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            checked[path] = digest.hexdigest()
        if expected is not None and checked[path] != expected:
            raise ValueError(f"Frozen binding changed: {path}")
        return checked[path]

    def read(path):
        bind(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    run_dir = HERE / "selected462_ARM"
    profile_dir = HERE / "profile_sensitivity"
    figure_dir = HERE / "matched_slices"
    run = read(run_dir / "run_provenance.json")
    graph = read(HERE / "independent_end_branch_readback.json")
    profiles = read(profile_dir / "profile_sensitivity_provenance.json")
    independent_profiles = read(profile_dir / "independent_profile_sensitivity_readback.json")
    display = read(figure_dir / "display_provenance.json")
    suite = read(HERE / "live_suite_20261009.json")
    boundary = read(HERE / "boundary_impact/validation_receipt.json")
    if graph["status"] != "independent_all462_forward_edge_and_full_voxel_regional_readback_passed":
        raise ValueError("Required independent full-voxel readback is absent")
    if independent_profiles["status"] != "independent_end_branch_profile_sensitivity_and_display_readback_passed":
        raise ValueError("Required independent profile/display readback is absent")
    if suite != {"status": "passed", "tests_run": 427, "errors": 0, "failures": 0,
                 "skipped": 0, "live_module_root": str(ROOT / "main_scripts"),
                 "scientific_acceptance": False}:
        raise ValueError("Unexpected live-suite result")
    if boundary["status"] != "pass_complete_source_allocation_no_impact" or boundary["allocation_changed_edges"]:
        raise ValueError("Actual-source boundary impact is not zero")
    if run["selected_neurons"] != 462 or run["eligible_neurons"] != 429:
        raise ValueError("Unexpected ledger/eligibility")
    if profiles["mapped_positive_profile_neurons"] != 428:
        raise ValueError("Unexpected mapped profile eligibility")
    if graph["producer_or_kernel_imported"] or independent_profiles["producer_functions_imported"]:
        raise ValueError("Independent-reader boundary violated")
    if any((run["scientific_anatomical_acceptance"], graph["anatomical_or_biological_acceptance"],
            profiles["anatomical_or_biological_acceptance"], independent_profiles["anatomical_or_biological_acceptance"],
            display["scientific_anatomical_acceptance"])):
        raise ValueError("Computational receipts cannot confer anatomical acceptance")
    bind(run_dir / "run_provenance.json", graph["run_provenance"]["sha256"])
    bind(run_dir / "run_provenance.json", display["run_provenance_sha256"])
    bind(HERE / "independent_end_branch_readback.json", display["readback_sha256"])
    bind(HERE / "independent_end_branch_readback.py", graph["independent_code_sha256"])
    bind(HERE / "independent_profile_sensitivity_readback.py", independent_profiles["independent_code_sha256"])
    bind(HERE / "render_end_branch_slices.py", display["renderer_sha256"])
    for relative, expected in run["source_code_sha256"].items():
        bind(ROOT / relative, expected)
    for source in run["inputs"].values():
        bind(source["path"], source["sha256"])
    for relative, expected in run["artifacts"].items():
        bind(run_dir / relative, expected)
        if graph["verified_artifacts"].get(relative) != expected:
            raise ValueError(f"Independent artifact binding differs: {relative}")
    map_count = 0
    for record in run["animal_maps"] + run["group_maps"]:
        if record.get("length_path"):
            bind(run_dir / record["length_path"], record["length_sha256"])
            map_count += 1
    if map_count != 67:
        raise ValueError("Unexpected eligible animal/group map count")
    for path, expected in profiles["input_bindings"].items():
        bind(path, expected)
    for source in profiles["analysis_source_code"]:
        bind(source["path"], source["sha256"])
    for relative, expected in profiles["output_hashes"].items():
        bind(profile_dir / relative, expected)
    bind(figure_dir / "display_provenance.json", independent_profiles["figure_readback"]["display_provenance_sha256"])
    for source in (display["brain_mask"], display["background"]):
        bind(source["path"], source["sha256"])
    for figure in display["figures"]:
        bind(figure_dir / figure["path"], figure["sha256"])
    for source in boundary["files"]:
        bind(ROOT / source["path"], source["sha256"])
    bind(HERE / "live_suite_20261009.log")
    bind(Path(__file__))
    result = {"status": "frozen_end_branch_checkpoint_readback_passed",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "source_binding_count": len(checked), "map_files_checked": map_count,
              "selected_neurons": 462, "eligible_neurons": 429, "profile_neurons": 428,
              "new_scientific_computation": False, "scientific_anatomical_acceptance": False,
              "scope": "Current bytes agree with independently validated saved graph, regional, voxel, profile and display receipts; not renewed anatomical review or inference.",
              "files": [{"path": str(path), "sha256": digest} for path, digest in sorted(checked.items())]}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "source_binding_count", "map_files_checked")}))


if __name__ == "__main__":
    main()
