"""Run the existing build/review/display tools on the ARM-only source manifests.

Every map must pass the independent reviewer before a figure is rendered.
Existing map directories and scientific inputs are never replaced.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EVOLUTION = ROOT / "group_analysis/evolution_20261008"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(output, cohorts, animal_figures=False):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(EVOLUTION):
        raise ValueError("Choose a fresh output root under the evolution directory")
    inputs = EVOLUTION / "projection_inputs/arm_labels_20261009"
    preparation = json.loads((inputs / "preparation_provenance.json").read_text(encoding="utf-8"))
    assets = preparation["inputs"]
    for item in assets.values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Changed ARM preparation input")
    for path, expected in preparation["outputs"].items():
        if sha(inputs / path) != expected:
            raise ValueError("Changed ARM-only manifest or label mapping")
    original_run = json.loads(Path(assets["main_run"]["path"]).read_text(encoding="utf-8"))
    brain_mask = original_run["inputs"]["brain_mask"]
    if sha(brain_mask["path"]) != brain_mask["sha256"]:
        raise ValueError("Changed original coverage/background mask")
    output.mkdir(parents=True)
    records = []

    def run(name, script, arguments):
        command = [sys.executable, "-X", "utf8", "-B", str(ROOT / "group_analysis/scripts" / script), *map(str, arguments)]
        print(f"Starting {name}", flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = output / (name + ".log")
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        item = {"stage": name, "command": command, "returncode": result.returncode,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "script_sha256": sha(ROOT / "group_analysis/scripts" / script),
                "log": str(log.relative_to(ROOT)), "log_sha256": sha(log)}
        records.append(item)
        (output / "workflow_checkpoint.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: item[key] for key in ("stage", "returncode", "elapsed_seconds")}), flush=True)
        if result.returncode:
            raise RuntimeError(f"Failed {name}; see {log}")

    for cohort in cohorts:
        manifest = inputs / cohort / "projection_manifest.csv"
        labels = inputs / cohort / "label_map.csv"
        for kind, builder, reviewer, review_name, metrics in (
            ("endpoints", "build_endpoint_maps.py", "review_endpoint_run.py", "endpoint_run_readback.json",
             ("endpoint-density", "endpoint-occupancy")),
            ("axons", "build_projection_maps.py", "review_projection_run.py", "projection_run_readback.json",
             ("axon-density",)),
        ):
            run_dir = output / cohort / kind
            review_dir = output / cohort / (kind + "_review")
            arguments = ["--manifest", manifest, "--reference", assets["reference"]["path"],
                         "--brain-mask", brain_mask["path"], "--output", run_dir]
            if kind == "endpoints":
                arguments += ["--atlas-path", assets["atlas"]["path"], "--atlas-key", assets["atlas_key"]["path"],
                              "--hemisphere-mask", assets["hemisphere_mask"]["path"]]
            run(cohort + "_" + kind + "_build", builder, arguments)
            run(cohort + "_" + kind + "_review", reviewer, ["--run", run_dir, "--output", review_dir])
            for metric in metrics:
                for scope in (("group", "animal") if animal_figures else ("group",)):
                    figure_dir = output / cohort / "figures" / (metric + "_" + scope)
                    run(cohort + "_" + metric + "_" + scope, "render_projection_slices.py", [
                        "--run", run_dir, "--readback", review_dir / review_name,
                        "--metric", metric, "--scope", scope, "--label-map", labels,
                        "--cut-policy", "fixed", "--slice-voxels", "56", "200", "87", "--output", figure_dir])
    report = {"status": "passed", "finished_utc": datetime.now(timezone.utc).isoformat(),
              "workflow_sha256": sha(__file__), "preparation_sha256": sha(inputs / "preparation_provenance.json"),
              "brain_mask": brain_mask,
              "atlas_policy": "Actual pinned ARM only; direct level-6 source labels; full names from its exact key",
              "stages": records, "cohorts": list(cohorts),
              "animal_figures": bool(animal_figures),
              "source_selection_changed": False, "source_data_modified": False,
              "scientific_acceptance": False, "inferential_maps": False}
    (output / "workflow_receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "stages": len(records)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVOLUTION / "arm_mapping_20261009")
    parser.add_argument("--cohorts", choices=("main", "preview"), nargs="+", default=["main"],
                        help="Default: one primary ARM map set. Preview sources stay in the location-QC ledger.")
    parser.add_argument("--animal-figures", action="store_true",
                        help="Optional animal QC panels; per-animal numerical maps are always retained.")
    arguments = parser.parse_args()
    if len(set(arguments.cohorts)) != len(arguments.cohorts):
        parser.error("Each cohort can be selected only once")
    main(arguments.output, arguments.cohorts, arguments.animal_figures)
