"""Read-only hash/identity/link checks of the dated delivery; not anatomy QC.

Run after the independent numerical reviewers. This checks that their recorded
inputs and outputs remain unchanged without repeating their voxel computations.
The only write is a new JSON receipt in this validation directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "group_analysis/evolution_20261008"


def check_delivery(output: Path) -> dict:
    output = output.resolve()
    if output.exists() or output.parent != Path(__file__).resolve().parent:
        raise ValueError("Use a new JSON filename in this validation directory")
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

    def verify(path, expected):
        if digest(path) != expected:
            errors.append(f"SHA-256 differs: {path}")

    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def rows(path):
        with Path(path).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    source_table = BASE / "classification/coarse_insula_review_20261009/graph_source_readback.csv"
    sources = rows(source_table)
    for row in sources:
        if row["expected_sha256"] != row["sha256"]:
            errors.append(f"Audited source hash fields disagree: {row['path']}")
        verify(ROOT / row["path"], row["expected_sha256"])
    main_inputs = BASE / "projection_inputs/multimonkey_coarse_20261009_v2"
    preview_inputs = BASE / "projection_inputs/additional_candidates_20261009"
    prior = read(main_inputs / "independent_manifest_readback_20261009.json")
    for item in prior["canonical_checks"]:
        verify(ROOT / item["path"], item["expected_sha256"])

    partitions = {}
    for name, folder, preparer in (
        ("main", main_inputs, "prepare_inventory_projection_manifest.py"),
        ("preview", preview_inputs, "prepare_review_candidate_manifest.py"),
    ):
        records = rows(folder / "projection_manifest.csv")
        preparation = read(folder / "preparation_provenance.json")
        verify(folder / "projection_manifest.csv", preparation["manifest_sha256"])
        verify(ROOT / "group_analysis/scripts" / preparer, preparation["preparer_sha256"])
        for item in preparation["inputs"].values():
            verify(Path(item["path"]), item["sha256"])
        for row in records:
            verify(Path(row["SWCPath"]), row["SWCSHA256"])
        uids = {row["SampleID"] + "|" + row["NeuronID"] for row in records}
        paths = {str(Path(row["SWCPath"]).resolve()).casefold() for row in records}
        if len(uids) != len(records) or len(paths) != len(records):
            errors.append(f"Duplicate identity/source in {name}")
        partitions[name] = {"rows": len(records), "uids": uids, "paths": paths,
                            "animals": sorted({row["AnimalID"] for row in records}),
                            "evidence_counts": dict(Counter(row["EvidenceStratum"] for row in records))}
    if partitions["main"]["uids"] & partitions["preview"]["uids"]:
        errors.append("Main/preview neuron overlap")
    if partitions["main"]["paths"] & partitions["preview"]["paths"]:
        errors.append("Main/preview source-path overlap")
    if (partitions["main"]["rows"], partitions["preview"]["rows"]) != (436, 26):
        errors.append("Dated delivery selection counts differ")

    reviews, map_count = [], 0
    for cohort in ("multimonkey_coarse_20261009", "additional_candidates_20261009"):
        for family in ("endpoint_maps", "projection_maps"):
            endpoint = family == "endpoint_maps"
            suffix = "_review_v2" if endpoint and cohort.startswith("multimonkey") else "_review"
            name = "endpoint_run_readback.json" if endpoint else "projection_run_readback.json"
            review_path = BASE / family / (cohort + suffix) / name
            review = read(review_path)
            run = BASE / family / cohort
            provenance = run / "run_provenance.json"
            if review["status"] != "passed":
                errors.append(f"Review not passed: {review_path}")
            verify(provenance, review["run_provenance_sha256"])
            record = read(provenance)
            for path, expected in record["source_code_sha256"].items():
                source = ROOT / path
                if len(Path(path).parts) == 1:
                    source = ROOT / ("group_analysis/scripts" if path == "build_projection_maps.py" else "main_scripts") / path
                verify(source, expected)
            if endpoint:
                verify(ROOT / "group_analysis/scripts/review_endpoint_run.py", review["reviewer_sha256"])
                for item in record["inputs"].values():
                    verify(Path(item["path"]), item["sha256"])
            else:
                verify(Path(review["reviewer"]["source"]), review["reviewer"]["sha256"])
                verify(Path(record["input_manifest"]), record["manifest_sha256"])
                verify(Path(record["reference"]), record["reference_sha256"])
                verify(Path(record["brain_mask"]), record["brain_mask_sha256"])
            maps = review["map_files" if endpoint else "all_map_files"]
            for item in maps:
                path = Path(item["file"])
                verify(path if path.is_absolute() else run / path, item["sha256"])
            map_count += len(maps)
            reviews.append({"path": str(review_path.relative_to(ROOT)), "sha256": digest(review_path),
                            "maps_unchanged_since_review": len(maps),
                            "prior_full_saved_voxels_read": review["full_saved_voxels_read"]})

    figures, figure_count = [], 0
    for folder in ("multimonkey_coarse_20261009_layout-v2", "additional_candidates_20261009"):
        for path in sorted((BASE / "figures" / folder).glob("*/matched_slices_provenance.json")):
            record = read(path)
            verify(ROOT / "group_analysis/scripts/render_projection_slices.py", record["renderer_sha256"])
            if record["fixed_slice_voxels_xyz"] != [56, 200, 87]:
                errors.append(f"Unexpected figure cuts: {path}")
            for item in record["figures"]:
                verify(path.parent / item["file"], item["sha256"])
                if item["panel_title_overlap_check"] != "passed":
                    errors.append(f"Title overlap: {item['file']}")
            figure_count += len(record["figures"])
            figures.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                            "figure_count": len(record["figures"])})
    if (map_count, len(figures), figure_count) != (245, 9, 33):
        errors.append("Dated map/figure delivery counts differ")

    documents = [BASE / name for name in (
        "README.md", "terminal_projection_goal_20261008.md", "reports/evolution_status_20261009.md",
        "reports/mapping_workflow_20261009.md", "figures/multimonkey_coarse_20261009_layout-v2/README.md",
        "figures/additional_candidates_20261009/README.md", "classification/coarse_insula_review_20261009/README.md")]
    link_count = 0
    for path in documents:
        for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"[a-zA-Z]+://", target) or target.startswith("#"):
                continue
            link_count += 1
            if not (path.parent / target.split("#", 1)[0].strip("<>")).exists():
                errors.append(f"Broken relative link: {path}: {target}")
    log = BASE / "validation/unittest_20261009_delivery.log"
    if not re.search(r"Ran 283 tests in [^\n]+\n\nOK", log.read_text(encoding="utf-8")):
        errors.append("Final unittest result differs")
    for partition in partitions.values():
        partition.pop("uids")
        partition.pop("paths")
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    report = {
        "status": "passed" if not errors else "failed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "checker_sha256": digest(__file__), "python": sys.version,
        "branch": subprocess.check_output(git + ["branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head": subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cached_source_files_rehashed": len(sources), "source_table_sha256": digest(source_table),
        "protected_canonical_workbooks_unchanged": len(prior["canonical_checks"]),
        "partitions": partitions, "neuron_and_source_path_overlap": 0 if not errors else "see errors",
        "independent_review_receipts": reviews, "saved_map_files_rehashed": map_count,
        "figure_families": figures, "saved_png_files_rehashed": figure_count,
        "current_document_sha256": {str(p.relative_to(ROOT)): digest(p) for p in documents},
        "relative_links_checked": link_count, "unittest_log_sha256": digest(log), "tests_passed": 283,
        "scope": "Fresh hashes/identities/links; prior independent numerical reviews not rerun",
        "scientific_acceptance": "not established by these checks", "source_writes": False,
        "errors": errors,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = check_delivery(parser.parse_args().output)
    print(result["status"], result["cached_source_files_rehashed"], "sources;",
          result["saved_map_files_rehashed"], "maps;", result["saved_png_files_rehashed"], "PNGs")
    if result["errors"]:
        print("\n".join(result["errors"]))
        raise SystemExit(1)
