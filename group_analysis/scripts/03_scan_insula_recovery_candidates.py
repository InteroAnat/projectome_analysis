"""Candidate discovery using an explicitly selected 251637 reference geometry.

CLI requires --bbox-csv and --geometry-mode; there is no reference fallback.
folded-mm: |NII_X - 128| * 0.25, Y/Z * 0.25; default padding 2 mm.
raw-legacy: NII XYZ * 0.25, deliberately declared legacy frame; default pad 0.
Both modes use q0.005..q0.995 boxes. Membership is a candidate screen only.
Outputs require a new directory under group_analysis/evolution_20261008/.
"""
from __future__ import annotations

import os
import sys
import glob
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
REF_DIR = os.path.join(GROUP_DIR, "reference")
STEP1_DIR = os.path.join(GROUP_DIR, "step1_results")

SCRIPTS = os.path.join(GROUP_DIR, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from cohort import NEW_SAMPLES, MIDLINE_NII_X, NII_VOXEL_MM  # noqa: E402
from insula_label_set import (build_insula_label_set, normalize_label,
                               strip_prefix, is_explicit_unknown_label)

INSULA_LABELS, _ = build_insula_label_set()
ADJACENT_LABELS_FOR_RESCUE = {
    "PrCO",                   # critical: insula <-> PrCO mis-classification
    "Unknown", "_Unmapped",   # atlas miss
    "InsulaUnknown",
}
SOMA_REGION_PREFIXES = ("CL_", "CR_", "L-", "R-", "L_", "R_")
GATE_THRESHOLD = 30  # min total recoverable neurons across new monkeys


class ReferenceBoxes(dict):
    """Bounds in physical mm, with validated frame and source provenance."""
    def __init__(self, boxes, *, mode, pad_mm, source, source_sha256):
        super().__init__(boxes)
        self.mode, self.pad_mm = mode, pad_mm
        self.source, self.source_sha256 = source, source_sha256


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_results_xlsx(sample_id: str, step1_dir=None) -> str | None:
    step1_dir = str(step1_dir or STEP1_DIR)
    pattern = os.path.join(step1_dir,
                           f"{sample_id}_*_region_analysis", "tables",
                           f"{sample_id}_results_*.xlsx")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # fallback: untimestamped
        pattern2 = os.path.join(step1_dir,
                                f"{sample_id}_*_region_analysis", "tables",
                                f"{sample_id}_results.xlsx")
        matches = sorted(glob.glob(pattern2))
    return matches[-1] if matches else None


def load_bboxes(csv_path=None, *, geometry_mode=None, pad_mm=None) -> ReferenceBoxes:
    """Reject missing/mismatched frames rather than guessing from filenames."""
    if csv_path is None or geometry_mode not in {"folded-mm", "raw-legacy"}:
        raise ValueError("Explicit bbox CSV and folded-mm or raw-legacy geometry mode are required")
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise ValueError(f"Reference missing; no geometry fallback: {path}")
    pad_mm = float(2.0 if geometry_mode == "folded-mm" else 0.0) if pad_mm is None else float(pad_mm)
    if not np.isfinite(pad_mm) or pad_mm < 0:
        raise ValueError("Physical padding must be finite and nonnegative")
    df = pd.read_csv(path)
    required = {"sub_region", *(f"{a}_{b}" for a in "XYZ" for b in ("lo_q005", "hi_q995"))}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError("Reference requires nonempty q0.005/q0.995 bounds for XYZ")
    if geometry_mode == "folded-mm":
        if not {"x_frame", "midline_nii_x", "nii_voxel_mm"}.issubset(df.columns):
            raise ValueError("Folded reference requires explicit x_frame, midline_nii_x and nii_voxel_mm")
        if not (df.x_frame == "Soma_NII_X_folded").all():
            raise ValueError("Folded mode requires Soma_NII_X_folded reference")
    elif "x_frame" in df and not (df.x_frame == "Soma_NII_X").all():
        raise ValueError("Raw legacy mode cannot consume folded/unknown reference frames")
    for column, expected in (("midline_nii_x", MIDLINE_NII_X), ("nii_voxel_mm", NII_VOXEL_MM)):
        if column in df and not (pd.to_numeric(df[column], errors="coerce") == expected).all():
            raise ValueError(f"Reference {column} disagrees with cohort geometry")
    bb = {}
    for _, r in df.iterrows():
        label = str(r["sub_region"])
        values = [float(r[f"{a}_{b}"]) * NII_VOXEL_MM for a in "XYZ" for b in ("lo_q005", "hi_q995")]
        if label in bb or not np.isfinite(values).all() or any(values[i] > values[i+1] for i in (0, 2, 4)):
            raise ValueError("Reference labels must be unique and bounds finite and ordered")
        bb[label] = dict(zip(("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"), values))
    return ReferenceBoxes(bb, mode=geometry_mode, pad_mm=pad_mm, source=str(path), source_sha256=file_sha256(path))


def coordinates_nii(row):
    try:
        xyz = tuple(float(row.get(f"Soma_NII_{axis}")) for axis in "XYZ")
    except (ValueError, TypeError):
        return None
    return xyz if np.isfinite(xyz).all() else None


def bbox_matches(row, bboxes):
    xyz = coordinates_nii(row)
    if xyz is None:
        return None
    if not isinstance(bboxes, ReferenceBoxes):
        if not bboxes:
            return None
        raise ValueError("Coordinate screening requires an explicitly validated ReferenceBoxes object")
    x, y, z = xyz
    if bboxes.mode == "folded-mm":
        x = abs(x - MIDLINE_NII_X)
    x, y, z = (value * NII_VOXEL_MM for value in (x, y, z))
    pad = bboxes.pad_mm
    return [region for region, b in bboxes.items()
            if b["x_lo"] - pad <= x <= b["x_hi"] + pad
            and b["y_lo"] - pad <= y <= b["y_hi"] + pad
            and b["z_lo"] - pad <= z <= b["z_hi"] + pad]


def classify_neuron(row, bboxes: dict) -> tuple[str, str]:
    """Return (category, matched_subregion_or_empty)."""
    raw = row.get("Soma_Region", "")
    auto_clean = normalize_label(raw)
    matches = bbox_matches(row, bboxes)

    if auto_clean in INSULA_LABELS:
        match_str = ";".join(matches) if matches else ""
        return ("auto_insula", match_str)

    if matches is None:
        kind = "PrCO" if auto_clean == "PRCO" else "other"
        reason = "missing_coordinates" if coordinates_nii(row) is None else "missing_reference"
        return (f"auto_{kind}_{reason}", "")
    if auto_clean == "PRCO":
        if matches:
            return ("auto_PrCO_in_insula_bbox", ";".join(matches))
        return ("auto_PrCO_outside_bbox", "")

    if auto_clean in {normalize_label(x) for x in ADJACENT_LABELS_FOR_RESCUE} or auto_clean == "" \
       or is_explicit_unknown_label(raw):
        if matches:
            return ("auto_other_in_insula_bbox", ";".join(matches))
        return ("auto_other_outside_bbox", "")

    if matches:
        return ("auto_other_in_insula_bbox", ";".join(matches))
    return ("auto_other_outside_bbox", "")


def scan_sample(sample_id: str, bboxes: dict, step1_dir=None) -> tuple[pd.DataFrame, dict]:
    xlsx = find_results_xlsx(sample_id, step1_dir)
    if not xlsx:
        print(f"[WARN] no results.xlsx for {sample_id}")
        return pd.DataFrame(), dict(sample_id=sample_id, source_status="missing_source", n_total=None,
                                    auto_insula=None, auto_PrCO_in_bbox=None, auto_PrCO_outside=None,
                                    auto_other_in_bbox=None, auto_other_outside=None,
                                    candidate_lower_bound=None, recoverable=None, n_unassessed=None)

    try:
        summary = pd.read_excel(xlsx, sheet_name="Summary", dtype={"NeuronID": str, "SampleID": str})
    except (OSError, ValueError) as exc:
        return pd.DataFrame(), dict(sample_id=sample_id, source_status="unreadable_source", source_error=str(exc),
                                    n_total=None, candidate_lower_bound=None, recoverable=None, n_unassessed=None)
    if "SampleID" in summary:
        summary["Source_SampleID"] = summary["SampleID"]
        mismatch = summary.SampleID.notna() & (summary.SampleID != sample_id)
        if mismatch.any():
            raise ValueError(f"Exact sample identity mismatch in {xlsx}")
    summary["SampleID"] = sample_id
    if "NeuronID" not in summary:
        summary["NeuronID"] = None
    if "Soma_Region" not in summary:
        summary["Soma_Region"] = None
    summary["Soma_Region_clean"] = summary["Soma_Region"].map(normalize_label)

    cats, matches = [], []
    for _, r in summary.iterrows():
        c, m = classify_neuron(r, bboxes)
        cats.append(c)
        matches.append(m)
    summary["recovery_category"] = cats
    summary["bbox_match_subregion"] = matches
    summary["coordinate_status"] = ["present" if coordinates_nii(r) is not None else "missing" for _, r in summary.iterrows()]
    summary["geometry_mode"] = bboxes.mode
    summary["padding_mm"] = bboxes.pad_mm
    summary["source_workbook"] = str(Path(xlsx).resolve())
    summary["source_sha256"] = file_sha256(xlsx)
    summary["source_excel_row"] = np.arange(len(summary)) + 2
    invalid_ids = summary.NeuronID.isna() | (summary.NeuronID == "") | summary.NeuronID.duplicated(keep=False)
    summary["identity_status"] = np.where(invalid_ids, "missing_or_duplicate", "exact")
    summary["UID"] = [f"{sample_id}|{nid}" if valid else "" for nid, valid in zip(summary.NeuronID, ~invalid_ids)]
    exclude_cols = [c for c in ("exclude", "Exclude", "Excluded") if c in summary]
    summary["source_excluded"] = [any(str(row.get(c) or "").strip().lower() not in {"", "0", "false", "no", "none", "nan"} for c in exclude_cols) for _, row in summary.iterrows()]
    candidate_categories = {"auto_insula", "auto_PrCO_in_insula_bbox", "auto_other_in_insula_bbox"}
    summary["candidate_only"] = [False if excluded else None if invalid or "missing_" in cat else cat in candidate_categories
                                 for excluded, invalid, cat in zip(summary.source_excluded, invalid_ids, summary.recovery_category)]
    summary["source_label_evidence"] = summary.get("Soma_Region_Source", pd.Series("existing_source_label", index=summary.index))
    summary["anatomical_acceptance"] = "not_assessed"

    counts = summary["recovery_category"].value_counts().to_dict()
    stats = dict(
        sample_id=sample_id,
        source_status="read",
        source_workbook=str(Path(xlsx).resolve()),
        source_sha256=file_sha256(xlsx),
        n_total=len(summary),
        auto_insula=counts.get("auto_insula", 0),
        auto_PrCO_in_bbox=counts.get("auto_PrCO_in_insula_bbox", 0),
        auto_PrCO_outside=counts.get("auto_PrCO_outside_bbox", 0),
        auto_other_in_bbox=counts.get("auto_other_in_insula_bbox", 0),
        auto_other_outside=counts.get("auto_other_outside_bbox", 0),
    )
    stats["n_missing_coordinates"] = int((summary.coordinate_status == "missing").sum())
    stats["n_excluded"] = int(summary.source_excluded.sum())
    stats["n_unassessed"] = int(summary.candidate_only.isna().sum())
    stats["candidate_lower_bound"] = int((summary.candidate_only == True).sum())
    stats["recoverable"] = stats["candidate_lower_bound"] if not stats["n_unassessed"] else None
    return summary.copy(), stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--geometry-mode", choices=("folded-mm", "raw-legacy"), required=True)
    parser.add_argument("--pad-mm", type=float, help="Default 2 mm folded; 0 mm raw legacy")
    parser.add_argument("--step1-dir", type=Path, default=Path(STEP1_DIR))
    parser.add_argument("--samples", nargs="+", default=list(NEW_SAMPLES))
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory under group_analysis/evolution_20261008")
    args = parser.parse_args(argv)
    try:
        bboxes = load_bboxes(args.bbox_csv, geometry_mode=args.geometry_mode, pad_mm=args.pad_mm)
        output = args.out_dir.resolve()
        allowed = Path(GROUP_DIR, "evolution_20261008").resolve()
        if allowed not in output.parents:
            raise ValueError("Output must be a new child directory under group_analysis/evolution_20261008")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory already contains files; choose a new audit directory")
        if len(set(args.samples)) != len(args.samples) or any(not sid or any(c in sid for c in "*?[]/\\") for sid in args.samples):
            raise ValueError("Sample IDs must be unique exact identifiers, without path/glob characters")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"[discovery] {bboxes.mode}; padding={bboxes.pad_mm} mm; reference={bboxes.source}")
    all_per_neuron, all_stats = [], []
    for sid in args.samples:
        per, stats = scan_sample(sid, bboxes, args.step1_dir)
        all_per_neuron.append(per)
        all_stats.append(stats)
        print(f"{sid}: source={stats['source_status']}, rows={stats.get('n_total')}, "
              f"candidate_lower_bound={stats.get('candidate_lower_bound')}, unassessed={stats.get('n_unassessed')}")
    per_df = pd.concat(all_per_neuron, ignore_index=True) if all_per_neuron else pd.DataFrame()
    stats_df = pd.DataFrame(all_stats)
    output.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output / "discovery_scan_summary.csv", index=False)
    per_df.to_csv(output / "discovery_scan_per_neuron.csv", index=False)
    lower_bound = sum(s.get("candidate_lower_bound") or 0 for s in all_stats)
    incomplete = any(s["source_status"] != "read" or s.get("n_unassessed") for s in all_stats)
    record = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "geometry_mode": bboxes.mode, "reference_csv": bboxes.source,
              "reference_sha256": bboxes.source_sha256, "quantiles": [0.005, 0.995],
              "midline_nii_x": MIDLINE_NII_X, "nii_voxel_mm": NII_VOXEL_MM,
              "padding_mm": bboxes.pad_mm, "bbox_units": "physical_mm",
              "reference_frame_declaration": "validated_folded_metadata" if bboxes.mode == "folded-mm" else "caller_explicit_legacy_raw_X; metadata_validated_if_present",
              "sample_ids": args.samples, "step1_dir": str(args.step1_dir.resolve()),
              "source_selection": "latest lexicographic matching historical step1 workbook, exact sample prefix; no portal refresh",
              "candidate_lower_bound": lower_bound, "candidate_total": None if incomplete else lower_bound,
              "yield_status": "incomplete_unknown_sources_or_identities_or_coordinates" if incomplete else "complete_bounded_source_screen",
              "candidate_gate_threshold": GATE_THRESHOLD, "anatomical_acceptance": "not_assessed",
              "scanner_sha256": file_sha256(__file__),
              "outputs": {p.name: file_sha256(p) for p in output.glob("*.csv")}}
    (output / "discovery_provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[candidate screen] observed lower bound={lower_bound}; total={'unknown' if incomplete else lower_bound}; anatomical acceptance not assessed")
    return 3 if incomplete else 0 if lower_bound >= GATE_THRESHOLD else 2


if __name__ == "__main__":
    sys.exit(main())
