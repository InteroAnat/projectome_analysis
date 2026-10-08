"""
Phase 4: Coordinate-based soma_region refinement.

Historical yield statements depended on the old raw-X reference and are not
claims about the explicitly selected geometry or current source population.

Refinement rule (conservative):
  1. Trust auto label if it is already a clear insula sub-region (IAL/IAPM/IDD5/IDM/IDV/IAI).
  2. Refine PrCO -> matched insula sub-region IFF coords inside (bbox + PAD_TOL_MM)
     of exactly one sub-region. Nearest-reference distance is a soft warning:
     candidates beyond DIST_THR_FACTOR * median_nn retain the `_distant` flag.
  3. Skip / mark excluded_ambiguous otherwise.

Geometry (2026-09-26)
---------------------
- Match in hemisphere-folded frame: dx = |Soma_NII_X - MIDLINE_NII_X|.
- PAD is expressed in mm (``PAD_TOL_MM``, env ``PROJECTOME_PAD_TOL_MM``,
  default 2.0) and converted to voxels via ``NII_VOXEL_MM`` (0.25).
- CLI requires an explicit bbox and geometry mode, with no alternate fallback.
- Folded default padding is 2 mm; explicit raw-legacy default padding is 0 mm.
- Matching uses q0.005/q0.995 bounds; outputs remain candidates, not acceptance.

Env overrides
-------------
PROJECTOME_STEP1_DIR, PROJECTOME_SAMPLES (CLI arguments take precedence).
Reference, geometry, padding and destination are controlled by CLI arguments.

Outputs:
  {RECOVERY_OUT}/{sid}_INS_HE_coord_inferred.xlsx (per new monkey)
  {RECOVERY_OUT}/all_refined_neurons.csv (consolidated long table)
"""
from __future__ import annotations

import os
import glob
import sys
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
STEP1_DIR = os.environ.get(
    "PROJECTOME_STEP1_DIR", os.path.join(GROUP_DIR, "step1_results")
)
SCRIPTS = os.path.join(GROUP_DIR, "scripts")

if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from cohort import (  # noqa: E402
    KEEP_ATLAS_LABELS,
    MIDLINE_NII_X,
    NEW_SAMPLES,
    NII_VOXEL_MM,
    fold_nii_x,
)
from insula_label_set import (build_insula_label_set, normalize_label,  # noqa: E402
                               is_explicit_unknown_label, ARM_KEY,
                               strip_prefix as _strip_prefix)

INSULA_LABELS, RESCUE_LABELS = build_insula_label_set()
KEEP_LABELS = set(INSULA_LABELS) | set(KEEP_ATLAS_LABELS)

PAD_STRICT = 0.0   # strict 99% bbox (voxels)
DIST_THR_FACTOR = 3.0  # soft sanity check; not a hard reject


def strip_prefix(s: str) -> str:
    return _strip_prefix(s)


def get_side_from_prefix(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("CL_") or s.startswith("L-") or s.startswith("L_"):
        return "L"
    if s.startswith("CR_") or s.startswith("R-") or s.startswith("R_"):
        return "R"
    return None


def find_results_xlsx(sample_id: str, step1_dir=None) -> str | None:
    pattern = os.path.join(str(step1_dir or STEP1_DIR), f"{sample_id}_*_region_analysis",
                           "tables", f"{sample_id}_results_*.xlsx")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def in_padded_bbox(x, y, z, b, pad):
    return (b["X_lo_q005"] - pad <= x <= b["X_hi_q995"] + pad
            and b["Y_lo_q005"] - pad <= y <= b["Y_hi_q995"] + pad
            and b["Z_lo_q005"] - pad <= z <= b["Z_hi_q995"] + pad)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_paths(bbox_csv, geometry_mode, anchors_csv=None, reference_neurons_csv=None):
    if geometry_mode not in {"folded-mm", "raw-legacy"}:
        raise ValueError("Explicit folded-mm or raw-legacy geometry mode is required")
    bbox = Path(bbox_csv).resolve()
    anchor_name = "251637_subregion_anchors_folded.csv" if geometry_mode == "folded-mm" else "251637_subregion_anchors.csv"
    return {"bbox": bbox, "anchors": Path(anchors_csv).resolve() if anchors_csv else bbox.parent / anchor_name,
            "reference_neurons": Path(reference_neurons_csv).resolve() if reference_neurons_csv else bbox.parent / "251637_subregion_neurons.csv"}


def _load_reference(bbox_csv, geometry_mode, anchors_csv=None, reference_neurons_csv=None):
    """Read one explicitly declared frame; missing companion files never fall back."""
    paths = reference_paths(bbox_csv, geometry_mode, anchors_csv, reference_neurons_csv)
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Reference {name} missing; no geometry fallback: {path}")
    bb, anchors, ref_neurons = (pd.read_csv(paths[name]) for name in ("bbox", "anchors", "reference_neurons"))
    bound_cols = [f"{axis}_{bound}" for axis in "XYZ" for bound in ("lo_q005", "hi_q995")]
    required = ((bb, {"sub_region", *bound_cols}, "bbox"),
                (anchors, {"sub_region", "median_nn"}, "anchors"),
                (ref_neurons, {"NeuronID", "Soma_Region_clean", "Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z"}, "neurons"))
    for table, fields, name in required:
        if table.empty or not fields <= set(table.columns):
            raise ValueError(f"Reference {name} requires nonempty fields {sorted(fields)}")
    for table, name in ((bb, "bbox"), (anchors, "anchors")):
        if table.sub_region.isna().any() or table.sub_region.astype(str).str.strip().eq("").any() or table.sub_region.duplicated().any():
            raise ValueError(f"Reference {name} requires unique nonempty regions")
    bounds = bb[bound_cols].to_numpy(float)
    if not np.isfinite(bounds).all() or np.any(bounds[:, ::2] > bounds[:, 1::2]):
        raise ValueError("Reference bbox requires finite ordered bounds")
    medians = anchors.median_nn.to_numpy(float)
    if not np.isfinite(medians).all() or np.any(medians < 0):
        raise ValueError("Reference anchors require finite nonnegative median_nn")
    x_col = "Soma_NII_X_folded" if geometry_mode == "folded-mm" else "Soma_NII_X"
    for table, name in ((bb, "bbox"), (anchors, "anchors")):
        metadata = {"x_frame", "midline_nii_x", "nii_voxel_mm"}
        if geometry_mode == "folded-mm" and not metadata <= set(table.columns):
            raise ValueError(f"Folded reference {name} requires frame, midline and voxel metadata")
        if "x_frame" in table and not table.x_frame.eq(x_col).all():
            raise ValueError(f"Reference {name} frame differs from explicit geometry mode")
        for column, expected in (("midline_nii_x", MIDLINE_NII_X), ("nii_voxel_mm", NII_VOXEL_MM)):
            if column in table and not pd.to_numeric(table[column], errors="coerce").eq(expected).all():
                raise ValueError(f"Reference {name} {column} differs from cohort geometry")
    coordinates = ref_neurons[["Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z"]].to_numpy(float)
    if not np.isfinite(coordinates).all() or ref_neurons.NeuronID.isna().any() or ref_neurons.NeuronID.duplicated().any():
        raise ValueError("Reference neurons require unique identities and finite XYZ")
    regions = set(bb.sub_region)
    if set(anchors.sub_region) != regions or set(ref_neurons.Soma_Region_clean) != regions:
        raise ValueError("Reference bbox, anchor and neuron region scopes differ")
    folded_x = np.abs(coordinates[:, 0] - MIDLINE_NII_X)
    if "Soma_NII_X_folded" in ref_neurons and not np.allclose(ref_neurons.Soma_NII_X_folded.to_numpy(float), folded_x, rtol=0, atol=1e-9):
        raise ValueError("Stored folded reference coordinates differ from abs(X-midline)")
    ref_neurons["Soma_NII_X_folded"] = folded_x
    return bb, anchors, ref_neurons, x_col


def _refine(bb, anchors, ref_neurons, x_col, source_tables, out_dir, padding_mm) -> list:
    """Existing candidate algorithm with explicit inputs and physical padding."""
    padding_vox = padding_mm / NII_VOXEL_MM

    median_nn = dict(zip(anchors["sub_region"], anchors["median_nn"]))
    ref_coords_by_region: dict[str, np.ndarray] = {}
    for reg, sub in ref_neurons.groupby("Soma_Region_clean"):
        ref_coords_by_region[reg] = sub[[x_col, "Soma_NII_Y",
                                          "Soma_NII_Z"]].to_numpy(float)

    bb_dict = {row["sub_region"]: row for _, row in bb.iterrows()}

    all_refined = []

    for sid, xlsx, source_summary in source_tables:
        print(f"\n[Phase 4] {sid}  ←  {xlsx}")
        summary = source_summary.copy()
        summary["SampleID"] = sid
        summary["Soma_Region_Auto"] = summary["Soma_Region"].astype(str)
        summary["Soma_Region_Auto_clean"] = summary["Soma_Region_Auto"].map(strip_prefix)

        refined_region = []
        refined_source = []
        refined_match = []
        refined_dist = []
        refined_side = []

        for _, row in summary.iterrows():
            auto_upper = normalize_label(row["Soma_Region_Auto"])
            x, y, z = row.get("Soma_NII_X"), row.get("Soma_NII_Y"), row.get("Soma_NII_Z")
            side = get_side_from_prefix(row["Soma_Region_Auto"]) or row.get("Soma_Side")

            if auto_upper in KEEP_LABELS:
                refined_region.append(auto_upper)
                refined_source.append("auto_atlas_insula")
                refined_match.append(auto_upper)
                refined_dist.append(np.nan)
                refined_side.append(side)
                continue

            if (auto_upper not in RESCUE_LABELS and not is_explicit_unknown_label(auto_upper)) or any(
                    pd.isna(v) for v in (x, y, z)):
                refined_region.append(np.nan)
                refined_source.append("not_rescue_candidate")
                refined_match.append("")
                refined_dist.append(np.nan)
                refined_side.append(side)
                continue

            x, y, z = float(x), float(y), float(z)
            # Fold candidate X into the same frame as the reference bbox.
            x_match = fold_nii_x(x) if x_col.endswith("folded") else float(x)

            strict_matches = [reg for reg, b in bb_dict.items()
                              if in_padded_bbox(x_match, y, z, b, PAD_STRICT)]
            tol_matches = [reg for reg, b in bb_dict.items()
                           if in_padded_bbox(x_match, y, z, b, padding_vox)]

            if len(strict_matches) == 1:
                reg_match = strict_matches[0]
                tier = "strict"
            elif len(tol_matches) == 1:
                reg_match = tol_matches[0]
                tier = "padded"
            elif len(strict_matches) > 1 or len(tol_matches) > 1:
                refined_region.append(np.nan)
                refined_source.append("excluded_ambiguous_bbox")
                refined_match.append(";".join(strict_matches or tol_matches))
                refined_dist.append(np.nan)
                refined_side.append(side)
                continue
            else:
                refined_region.append(np.nan)
                refined_source.append("excluded_outside_bbox")
                refined_match.append("")
                refined_dist.append(np.nan)
                refined_side.append(side)
                continue

            ref_arr = ref_coords_by_region.get(reg_match)
            if ref_arr is None or len(ref_arr) == 0:
                refined_region.append(np.nan)
                refined_source.append("excluded_no_reference")
                refined_match.append(reg_match)
                refined_dist.append(np.nan)
                refined_side.append(side)
                continue

            d_arr = cdist(np.array([[x_match, y, z]]), ref_arr)[0]
            d_min = float(d_arr.min())
            thr_soft = DIST_THR_FACTOR * float(median_nn.get(reg_match, np.inf))

            refined_region.append(reg_match)
            refined_source.append(
                f"coord_inferred_from_251637_{tier}"
                + (("_distant" if d_min > thr_soft else ""))
            )
            refined_match.append(reg_match)
            # Store distance in mm (coords are voxels).
            refined_dist.append(d_min * NII_VOXEL_MM)
            refined_side.append(side)

        summary["Soma_Region_Refined"] = refined_region
        summary["Soma_Region_Source"] = refined_source
        summary["Soma_Region_Match"] = refined_match
        summary["Distance_to_nearest_251637_neuron_mm"] = refined_dist
        summary["Soma_Side_Inferred"] = refined_side
        summary["PAD_TOL_MM"] = padding_mm
        summary["PAD_TOL_VOX"] = padding_vox
        summary["Reference_Geometry_Mode"] = "folded-mm" if x_col.endswith("folded") else "raw-legacy"
        summary["Anatomical_Acceptance"] = "not_assessed"

        keepers = summary[summary["Soma_Region_Source"].str.startswith(
            ("auto_atlas_insula", "coord_inferred_from_251637"), na=False)].copy()

        src_counts = summary["Soma_Region_Source"].value_counts().to_dict()
        ref_counts = (keepers["Soma_Region_Refined"]
                      .value_counts().to_dict())
        side_counts = keepers["Soma_Side_Inferred"].value_counts(dropna=False).to_dict()
        print(f"  source breakdown: {src_counts}")
        print(f"  kept sub-region counts: {ref_counts}")
        print(f"  kept side counts: {side_counts}")

        out_xlsx = Path(out_dir) / f"{sid}_INS_HE_coord_inferred.xlsx"
        with out_xlsx.open("xb") as stream:
            with pd.ExcelWriter(stream, engine="openpyxl") as w:
                summary.to_excel(w, sheet_name="Summary", index=False)
                keepers.to_excel(w, sheet_name="Insula_keepers", index=False)
        print(f"  [saved] {out_xlsx}")

        all_refined.append(keepers)

    big = pd.concat(all_refined, ignore_index=True) if all_refined else pd.DataFrame()
    if len(big):
        big_csv = Path(out_dir) / "all_refined_neurons.csv"
        with big_csv.open("x", encoding="utf-8", newline="") as stream:
            big.to_csv(stream, index=False)
        print(f"\n[saved] {big_csv}")
        print("\nCombined refined table:")
        print(f"  total = {len(big)} neurons")
        print("  by sample x sub-region:")
        print(pd.crosstab(big["SampleID"],
                          big["Soma_Region_Refined"]).to_string())
        print("  by sample x soma side:")
        print(pd.crosstab(big["SampleID"],
                          big["Soma_Side_Inferred"]).to_string())

    return all_refined


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--geometry-mode", choices=("folded-mm", "raw-legacy"), required=True)
    parser.add_argument("--anchors-csv", type=Path)
    parser.add_argument("--reference-neurons-csv", type=Path)
    parser.add_argument("--pad-mm", type=float, help="Default 2 mm folded; 0 mm raw legacy")
    parser.add_argument("--step1-dir", type=Path, default=Path(STEP1_DIR))
    parser.add_argument("--samples", nargs="+", default=list(NEW_SAMPLES))
    parser.add_argument("--out-dir", "--output-dir", type=Path, required=True, dest="out_dir")
    args = parser.parse_args(argv)
    padding = (2.0 if args.geometry_mode == "folded-mm" else 0.0) if args.pad_mm is None else args.pad_mm
    try:
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("Physical padding must be finite and nonnegative")
        if len(set(args.samples)) != len(args.samples) or any(not sid or any(c in sid for c in "*?[]/\\") for sid in args.samples):
            raise ValueError("Sample IDs must be unique exact identifiers without path/glob characters")
        output = args.out_dir.resolve()
        if output.exists():
            raise ValueError("Output destination already exists; use a new derivative directory")
        paths = reference_paths(args.bbox_csv, args.geometry_mode, args.anchors_csv, args.reference_neurons_csv)
        for name, path in paths.items():
            if not path.is_file():
                raise ValueError(f"Reference {name} missing; no geometry fallback: {path}")
        input_hashes = {name: file_sha256(path) for name, path in paths.items()}
        policy_paths = [Path(__file__), Path(SCRIPTS) / "cohort.py", Path(SCRIPTS) / "insula_label_set.py",
                        Path(PROJECT_ROOT) / "main_scripts/region_labels.py", Path(ARM_KEY)]
        policy_hashes = {str(path.resolve()): file_sha256(path) for path in policy_paths}
        reference = _load_reference(args.bbox_csv, args.geometry_mode, args.anchors_csv, args.reference_neurons_csv)
        if output.is_relative_to(args.step1_dir.resolve()) or any(output.is_relative_to(path.parent) for path in paths.values()):
            raise ValueError("Output must be outside source/reference directories")
        source_tables, source_records = [], []
        for sid in args.samples:
            xlsx = find_results_xlsx(sid, args.step1_dir)
            if not xlsx:
                source_records.append({"sample_id": sid, "status": "missing_source", "rows": None})
                print(f"[skip] no results.xlsx for {sid}")
                continue
            source = Path(xlsx).resolve()
            digest = file_sha256(source)
            frame = pd.read_excel(source, sheet_name="Summary")
            required = {"NeuronID", "Soma_Region", "Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z"}
            if not required <= set(frame.columns) or frame.NeuronID.isna().any() or frame.NeuronID.duplicated().any():
                raise ValueError(f"Source Summary requires unique neuron identities and soma fields: {source}")
            for axis in "XYZ":
                values = frame[f"Soma_NII_{axis}"]
                converted = pd.to_numeric(values, errors="coerce")
                if (values.notna() & (converted.isna() | ~np.isfinite(converted))).any():
                    raise ValueError(f"Source Summary has malformed coordinates: {source}")
            source_tables.append((sid, source, frame))
            source_records.append({"sample_id": sid, "status": "read", "path": str(source), "sha256": digest, "rows": len(frame)})
        if any(file_sha256(paths[name]) != digest for name, digest in input_hashes.items()) or any(
                file_sha256(item["path"]) != item["sha256"] for item in source_records if item["status"] == "read"):
            raise RuntimeError("Refinement input changed during validation")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=False)
    kept = _refine(*reference, source_tables, output, padding)
    if any(file_sha256(paths[name]) != digest for name, digest in input_hashes.items()) or any(
            file_sha256(item["path"]) != item["sha256"] for item in source_records if item["status"] == "read"):
        raise RuntimeError("Refinement input changed during processing; outputs are not verified")
    if any(file_sha256(path) != digest for path, digest in policy_hashes.items()):
        raise RuntimeError("Refinement code or label key changed during processing; outputs are not verified")
    provenance = {"status": "software_verified_candidate_refinement", "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "geometry_mode": args.geometry_mode, "padding_mm": padding, "padding_vox": padding / NII_VOXEL_MM,
                  "midline_nii_x": MIDLINE_NII_X, "nii_voxel_mm": NII_VOXEL_MM, "quantiles": [.005, .995],
                  "reference_inputs": {name: {"path": str(path), "sha256": input_hashes[name]} for name, path in paths.items()},
                  "sources": source_records, "sample_ids": args.samples,
                  "source_selection": "latest lexicographic matching historical step1 workbook; exact sample prefix; no portal refresh",
                  "yield_status": "incomplete_missing_sources" if len(source_tables) != len(args.samples) else "complete_selected_sources",
                  "candidate_keeper_count": sum(len(frame) for frame in kept),
                  "refinement_policy": "existing atlas keep labels; eligible rescue labels with unique strict/padded box; nearest-reference distance is a soft flag",
                  "distance_threshold_factor": DIST_THR_FACTOR, "anatomical_acceptance": "not_assessed", "canonical_promotion": False,
                  "policy_source_sha256": policy_hashes,
                  "label_sets": {"insula": sorted(INSULA_LABELS), "rescue": sorted(RESCUE_LABELS), "atlas_keep": sorted(KEEP_ATLAS_LABELS)},
                  "script_sha256": file_sha256(__file__),
                  "outputs": {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()}}
    with (output / "refinement_provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
