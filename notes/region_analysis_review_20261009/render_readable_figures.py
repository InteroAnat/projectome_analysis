"""Create a new wording variant from previously verified maps; no map rebuild."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "group_analysis/evolution_20261008/figures"
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
from render_projection_slices import render


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    variants = [("multimonkey_coarse_20261009_layout-v2", "multimonkey_coarse_20261009_terms-v3"),
                ("additional_candidates_20261009", "additional_candidates_20261009_terms-v2")]
    protected, jobs = {}, []
    for old, new in variants:
        for provenance in sorted((FIGURES / old).glob("*/matched_slices_provenance.json")):
            source = json.loads(provenance.read_text(encoding="utf-8"))
            readback = Path(source["readback_path"])
            review = json.loads(readback.read_text(encoding="utf-8"))
            # Resolve the run by its reviewed provenance hash, never by a
            # filename guess or a machine-specific review-directory name.
            expected_run_hash = source["run_provenance_sha256"]
            if review["run_provenance_sha256"] != expected_run_hash:
                raise ValueError("Figure and review refer to different runs")
            evolution = FIGURES.parent
            matches = [path.parent for kind in ("endpoint_maps", "projection_maps")
                       for path in (evolution / kind).glob("*/run_provenance.json")
                       if sha(path) == expected_run_hash]
            if len(matches) != 1:
                raise ValueError(f"Expected one reviewed map run, found {len(matches)}")
            run = matches[0]
            for record in source["slices"]:
                path = run / record["map_path"]
                protected[str(path)] = record["map_sha256"]
            for record in source["figures"]:
                protected[str(provenance.parent / record["file"])] = record["sha256"]
            target = FIGURES / new / provenance.parent.name
            jobs.append((run, readback, target, source))
    for path, expected in protected.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Changed pre-existing map or figure: {path}")
    results = []
    for run, readback, target, source in jobs:
        result = render(run, readback, target, metric=source["metric"], scope=source["scope"],
                        cut_policy="fixed", slice_voxels=source["fixed_slice_voxels_xyz"])
        for key in ("reference_sha256", "brain_mask_sha256", "grayscale_window", "common_display_vmax", "fixed_slice_voxels_xyz"):
            if result[key] != source[key]:
                raise ValueError(f"Unexpected display change in {key}")
        results.append({"directory": str(target.relative_to(ROOT)), "figures": len(result["figures"]),
                        "provenance_sha256": sha(target / "matched_slices_provenance.json")})
        print(json.dumps(results[-1]), flush=True)
    for path, expected in protected.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Original artifact changed during display rendering: {path}")
    receipt = {"status": "passed", "source_maps_recomputed": False,
               "protected_originals": len(protected), "original_hashes_unchanged": True,
               "figure_count": sum(item["figures"] for item in results), "families": results,
               "anatomical_acceptance": False, "inferential_maps": False}
    (Path(__file__).parent / "readable_figures_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "families"}))


if __name__ == "__main__":
    main()
