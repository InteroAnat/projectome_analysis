"""
Phase 3: Discovery scan + go/no-go gate.

For each new monkey's Summary sheet, classify every neuron into:
  - auto_insula                 : already auto-labeled IAL/IAPM/IDD5/IDM/IDV/IAI
  - auto_PrCO_in_insula_bbox    : auto PrCO and coords inside 251637 99% bbox
  - auto_other_in_insula_bbox   : auto Unknown/Other/adjacent labels, coords inside bbox
  - auto_other                  : not insula, coords outside bbox

Output:
  group_analysis/recovery/discovery_scan_summary.csv
  group_analysis/recovery/discovery_scan_per_neuron.csv
"""
from __future__ import annotations

import os
import sys
import glob
import pandas as pd
import numpy as np

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
REF_DIR = os.path.join(GROUP_DIR, "reference")
STEP1_DIR = os.path.join(GROUP_DIR, "step1_results")
OUT_DIR = os.path.join(GROUP_DIR, "recovery")
os.makedirs(OUT_DIR, exist_ok=True)

NEW_SAMPLES = ["251730", "252383", "252384", "252385"]
INSULA_LABELS = {"IAL", "IAPM", "IDD5", "IDM", "IDV", "IAI", "IA", "ID",
                 "IA/ID", "Iai"}
ADJACENT_LABELS_FOR_RESCUE = {
    "PrCO",                   # critical: insula <-> PrCO mis-classification
    "Unknown", "_Unmapped",   # atlas miss
    "InsulaUnknown",
}
SOMA_REGION_PREFIXES = ("CL_", "CR_", "L-", "R-", "L_", "R_")
GATE_THRESHOLD = 30  # min total recoverable neurons across new monkeys


def strip_prefix(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    for p in SOMA_REGION_PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    return s


def find_results_xlsx(sample_id: str) -> str | None:
    pattern = os.path.join(STEP1_DIR,
                           f"{sample_id}_*_region_analysis", "tables",
                           f"{sample_id}_results_*.xlsx")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # fallback: untimestamped
        pattern2 = os.path.join(STEP1_DIR,
                                f"{sample_id}_*_region_analysis", "tables",
                                f"{sample_id}_results.xlsx")
        matches = sorted(glob.glob(pattern2))
    return matches[-1] if matches else None


def load_bboxes() -> dict:
    """Return dict of sub_region -> dict with x_lo, x_hi, y_lo, y_hi, z_lo, z_hi (99% bbox)."""
    csv = os.path.join(REF_DIR, "251637_subregion_bboxes.csv")
    df = pd.read_csv(csv)
    bb = {}
    for _, r in df.iterrows():
        bb[r["sub_region"]] = dict(
            x_lo=float(r["X_lo_q005"]), x_hi=float(r["X_hi_q995"]),
            y_lo=float(r["Y_lo_q005"]), y_hi=float(r["Y_hi_q995"]),
            z_lo=float(r["Z_lo_q005"]), z_hi=float(r["Z_hi_q995"]),
            n=int(r["n"]),
        )
    return bb


def in_bbox(x, y, z, b) -> bool:
    if any(pd.isna(v) for v in (x, y, z)):
        return False
    return (b["x_lo"] <= x <= b["x_hi"]
            and b["y_lo"] <= y <= b["y_hi"]
            and b["z_lo"] <= z <= b["z_hi"])


def classify_neuron(row, bboxes: dict) -> tuple[str, str]:
    """Return (category, matched_subregion_or_empty)."""
    raw = str(row.get("Soma_Region", ""))
    auto_clean = strip_prefix(raw)
    x = row.get("Soma_NII_X")
    y = row.get("Soma_NII_Y")
    z = row.get("Soma_NII_Z")

    matches = [r for r, b in bboxes.items() if in_bbox(x, y, z, b)]

    if auto_clean in INSULA_LABELS:
        match_str = ";".join(matches) if matches else auto_clean
        return ("auto_insula", match_str)

    if auto_clean == "PrCO":
        if matches:
            return ("auto_PrCO_in_insula_bbox", ";".join(matches))
        return ("auto_PrCO_outside_bbox", "")

    if auto_clean in ADJACENT_LABELS_FOR_RESCUE or auto_clean == "" \
       or "Unknown" in raw:
        if matches:
            return ("auto_other_in_insula_bbox", ";".join(matches))
        return ("auto_other_outside_bbox", "")

    if matches:
        return ("auto_other_in_insula_bbox", ";".join(matches))
    return ("auto_other_outside_bbox", "")


def scan_sample(sample_id: str, bboxes: dict) -> tuple[pd.DataFrame, dict]:
    xlsx = find_results_xlsx(sample_id)
    if not xlsx:
        print(f"[WARN] no results.xlsx for {sample_id}")
        return pd.DataFrame(), dict(sample_id=sample_id, n_total=0)

    summary = pd.read_excel(xlsx, sheet_name="Summary")
    summary["SampleID"] = sample_id
    summary["Soma_Region_clean"] = summary["Soma_Region"].astype(str).map(strip_prefix)

    cats, matches = [], []
    for _, r in summary.iterrows():
        c, m = classify_neuron(r, bboxes)
        cats.append(c)
        matches.append(m)
    summary["recovery_category"] = cats
    summary["bbox_match_subregion"] = matches

    counts = summary["recovery_category"].value_counts().to_dict()
    stats = dict(
        sample_id=sample_id,
        n_total=len(summary),
        auto_insula=counts.get("auto_insula", 0),
        auto_PrCO_in_bbox=counts.get("auto_PrCO_in_insula_bbox", 0),
        auto_PrCO_outside=counts.get("auto_PrCO_outside_bbox", 0),
        auto_other_in_bbox=counts.get("auto_other_in_insula_bbox", 0),
        auto_other_outside=counts.get("auto_other_outside_bbox", 0),
    )
    stats["recoverable"] = (stats["auto_insula"]
                            + stats["auto_PrCO_in_bbox"]
                            + stats["auto_other_in_bbox"])

    cols_keep = ["SampleID", "NeuronID", "Neuron_Type", "Soma_Region",
                 "Soma_Region_clean", "Soma_Side",
                 "Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z",
                 "recovery_category", "bbox_match_subregion"]
    cols_keep = [c for c in cols_keep if c in summary.columns]
    return summary[cols_keep].copy(), stats


def main() -> int:
    bboxes = load_bboxes()
    print(f"[Phase 3] Loaded bboxes for sub-regions: {sorted(bboxes.keys())}\n")

    all_per_neuron = []
    all_stats = []
    for sid in NEW_SAMPLES:
        print(f"--- Scanning {sid} ---")
        per, stats = scan_sample(sid, bboxes)
        all_per_neuron.append(per)
        all_stats.append(stats)
        print(f"  total={stats['n_total']}, auto_insula={stats['auto_insula']}, "
              f"PrCO_in_bbox={stats['auto_PrCO_in_bbox']}, "
              f"other_in_bbox={stats['auto_other_in_bbox']}, "
              f"recoverable={stats['recoverable']}")

    per_df = pd.concat(all_per_neuron, ignore_index=True) if all_per_neuron else pd.DataFrame()
    stats_df = pd.DataFrame(all_stats)
    summary_csv = os.path.join(OUT_DIR, "discovery_scan_summary.csv")
    per_csv = os.path.join(OUT_DIR, "discovery_scan_per_neuron.csv")
    stats_df.to_csv(summary_csv, index=False)
    per_df.to_csv(per_csv, index=False)

    print(f"\n[saved] {summary_csv}")
    print(f"[saved] {per_csv}")

    print("\n" + "=" * 60)
    print("DISCOVERY SCAN SUMMARY")
    print("=" * 60)
    print(stats_df.to_string(index=False))

    total_recover = int(stats_df["recoverable"].sum())
    print(f"\nTotal recoverable insula neurons across new monkeys: {total_recover}")
    print(f"Gate threshold: {GATE_THRESHOLD}")

    if total_recover < GATE_THRESHOLD:
        print(f"\n[GATE] STOP — yield ({total_recover}) < threshold ({GATE_THRESHOLD}).")
        print("  Recommend skipping Phases 4-7 and documenting the null result.")
        return 2
    else:
        print(f"\n[GATE] PASS — proceed to Phase 4 (coord-based refinement).")
        # Per-monkey breakdown of bbox sub-region matches
        if not per_df.empty:
            recover_only = per_df[per_df["recovery_category"].isin(
                ["auto_insula", "auto_PrCO_in_insula_bbox",
                 "auto_other_in_insula_bbox"])].copy()
            print("\nPer-monkey × bbox-sub-region match counts:")
            ct = pd.crosstab(recover_only["SampleID"],
                             recover_only["bbox_match_subregion"])
            print(ct.to_string())
        return 0


if __name__ == "__main__":
    sys.exit(main())
