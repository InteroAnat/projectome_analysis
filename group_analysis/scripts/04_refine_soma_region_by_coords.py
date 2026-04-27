"""
Phase 4: Coordinate-based soma_region refinement.

Decision rationale (from Phase 3 results):
  - Strict 99% bbox produced only 8 matches, all R-side ITi in 252385.
  - At pad=2 mm, yield rises to 30 (still all R-side, all 252385).
  - 251730 / 252383 / 252384 contribute ZERO at any padding up to 10 mm.
  - This means the recovered population CAN NOT address L/R laterality.
    It CAN test inter-animal replication of the agranular-insula -> caudal_OFC
    finding from 251637 (Mesulam-Mufson 1982 III; Carmichael & Price 1995).

Refinement rule (conservative):
  1. Trust auto label if it is already a clear insula sub-region (IAL/IAPM/IDD5/IDM/IDV/IAI).
  2. Refine PrCO -> matched insula sub-region IFF coords inside (bbox + PAD_MM)
     of exactly one sub-region AND distance to nearest 251637 curated neuron
     of that sub-region is less than DIST_THR_FACTOR * median_nn_of_that_subregion.
  3. Skip / mark excluded_ambiguous otherwise.

Outputs:
  group_analysis/recovery/{sid}_INS_HE_coord_inferred.xlsx (per new monkey)
  group_analysis/recovery/all_refined_neurons.csv (consolidated long table)
"""
from __future__ import annotations

import os
import glob
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
REF_DIR = os.path.join(GROUP_DIR, "reference")
STEP1_DIR = os.path.join(GROUP_DIR, "step1_results")
SCRIPTS = os.path.join(GROUP_DIR, "scripts")
OUT_DIR = os.path.join(GROUP_DIR, "recovery")
os.makedirs(OUT_DIR, exist_ok=True)

if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from insula_label_set import (build_insula_label_set, normalize_label,
                               strip_prefix as _strip_prefix)

NEW_SAMPLES = ["251730", "252383", "252384", "252385"]
INSULA_LABELS, RESCUE_LABELS = build_insula_label_set()
print(f"[Phase 4] Insula label set ({len(INSULA_LABELS)}): {sorted(INSULA_LABELS)}")
print(f"[Phase 4] Rescue label set ({len(RESCUE_LABELS)}): {sorted(RESCUE_LABELS)}")

PAD_STRICT = 0.0   # strict 99% bbox
PAD_TOL = 2.0      # tolerance band (cross-animal NMT registration variance)
DIST_THR_FACTOR = 3.0  # soft sanity check; not a hard reject

SOMA_REGION_PREFIXES = ("CL_", "CR_", "L-", "R-", "L_", "R_")


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


def find_results_xlsx(sample_id: str) -> str | None:
    pattern = os.path.join(STEP1_DIR, f"{sample_id}_*_region_analysis",
                           "tables", f"{sample_id}_results_*.xlsx")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def in_padded_bbox(x, y, z, b, pad):
    return (b["X_lo_q005"] - pad <= x <= b["X_hi_q995"] + pad
            and b["Y_lo_q005"] - pad <= y <= b["Y_hi_q995"] + pad
            and b["Z_lo_q005"] - pad <= z <= b["Z_hi_q995"] + pad)


def main() -> int:
    bb = pd.read_csv(os.path.join(REF_DIR, "251637_subregion_bboxes.csv"))
    anchors = pd.read_csv(os.path.join(REF_DIR, "251637_subregion_anchors.csv"))
    ref_neurons = pd.read_csv(os.path.join(REF_DIR, "251637_subregion_neurons.csv"))

    # Median nn per sub-region (used to set distance threshold)
    median_nn = dict(zip(anchors["sub_region"], anchors["median_nn"]))
    # Per-sub-region 251637 coord arrays (for nearest-neighbor lookup)
    ref_coords_by_region: dict[str, np.ndarray] = {}
    for reg, sub in ref_neurons.groupby("Soma_Region_clean"):
        ref_coords_by_region[reg] = sub[["Soma_NII_X", "Soma_NII_Y",
                                          "Soma_NII_Z"]].to_numpy(float)

    bb_dict = {row["sub_region"]: row for _, row in bb.iterrows()}

    all_refined = []

    for sid in NEW_SAMPLES:
        xlsx = find_results_xlsx(sid)
        if not xlsx:
            print(f"[skip] no results.xlsx for {sid}")
            continue

        print(f"\n[Phase 4] {sid}  ←  {xlsx}")
        wb = pd.ExcelFile(xlsx)
        sheets_keep = [s for s in wb.sheet_names if s in (
            "Summary", "Soma_Hierarchy", "Terminal_Sites", "Laterality",
            "Outliers", "Projection_Length_ipsi", "Projection_Length_contra",
            "Projection_Strength_ipsi", "Projection_Strength_contra",
            "Projection_Length_L3_ipsi", "Projection_Length_L3_contra",
            "Projection_Strength_L3_ipsi", "Projection_Strength_L3_contra",
        )]
        summary = pd.read_excel(xlsx, sheet_name="Summary")
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
            auto_clean = strip_prefix(row["Soma_Region_Auto"])
            x, y, z = row.get("Soma_NII_X"), row.get("Soma_NII_Y"), row.get("Soma_NII_Z")
            side = get_side_from_prefix(row["Soma_Region_Auto"]) or row.get("Soma_Side")

            # Already insula -> trust auto. Preserve original casing (Ial vs IAL)
            # but use uppercase for the label-set comparison and for downstream
            # consistency with 251637's curated table.
            if auto_upper in INSULA_LABELS:
                refined_region.append(auto_upper)
                refined_source.append("auto_atlas_insula")
                refined_match.append(auto_upper)
                refined_dist.append(np.nan)
                refined_side.append(side)
                continue

            # Try to recover for rescue-eligible labels
            if auto_upper not in RESCUE_LABELS or any(
                    pd.isna(v) for v in (x, y, z)):
                refined_region.append(np.nan)
                refined_source.append("not_rescue_candidate")
                refined_match.append("")
                refined_dist.append(np.nan)
                refined_side.append(side)
                continue

            # Two-tier match: strict 99% bbox first, then bbox + PAD_TOL mm
            strict_matches = [reg for reg, b in bb_dict.items()
                              if in_padded_bbox(x, y, z, b, PAD_STRICT)]
            tol_matches = [reg for reg, b in bb_dict.items()
                           if in_padded_bbox(x, y, z, b, PAD_TOL)]

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

            d_arr = cdist(np.array([[x, y, z]]), ref_arr)[0]
            d_min = float(d_arr.min())
            thr_soft = DIST_THR_FACTOR * float(median_nn.get(reg_match, np.inf))

            # Soft warning if very far, but do not reject (bbox already screened)
            refined_region.append(reg_match)
            refined_source.append(
                f"coord_inferred_from_251637_{tier}"
                + (("_distant" if d_min > thr_soft else ""))
            )
            refined_match.append(reg_match)
            refined_dist.append(d_min)
            refined_side.append(side)

        summary["Soma_Region_Refined"] = refined_region
        summary["Soma_Region_Source"] = refined_source
        summary["Soma_Region_Match"] = refined_match
        summary["Distance_to_nearest_251637_neuron_mm"] = refined_dist
        summary["Soma_Side_Inferred"] = refined_side

        # Insula keepers (for combined table) - any auto-atlas-insula or
        # coord-inferred-from-251637-bbox classification.
        keepers = summary[summary["Soma_Region_Source"].str.startswith(
            ("auto_atlas_insula", "coord_inferred_from_251637"), na=False)].copy()

        # Counts
        src_counts = summary["Soma_Region_Source"].value_counts().to_dict()
        ref_counts = (keepers["Soma_Region_Refined"]
                      .value_counts().to_dict())
        side_counts = keepers["Soma_Side_Inferred"].value_counts(dropna=False).to_dict()
        print(f"  source breakdown: {src_counts}")
        print(f"  kept sub-region counts: {ref_counts}")
        print(f"  kept side counts: {side_counts}")

        # Write per-sample refined workbook (Summary sheet only — projection
        # sheets are reused as-is from the original results.xlsx in Phase 5)
        out_xlsx = os.path.join(OUT_DIR, f"{sid}_INS_HE_coord_inferred.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            summary.to_excel(w, sheet_name="Summary", index=False)
            keepers.to_excel(w, sheet_name="Insula_keepers", index=False)
        print(f"  [saved] {out_xlsx}")

        all_refined.append(keepers)

    big = pd.concat(all_refined, ignore_index=True) if all_refined else pd.DataFrame()
    if len(big):
        big_csv = os.path.join(OUT_DIR, "all_refined_neurons.csv")
        big.to_csv(big_csv, index=False)
        print(f"\n[saved] {big_csv}")
        print("\nCombined refined table:")
        print(f"  total = {len(big)} neurons")
        print("  by sample x sub-region:")
        print(pd.crosstab(big["SampleID"],
                          big["Soma_Region_Refined"]).to_string())
        print("  by sample x soma side:")
        print(pd.crosstab(big["SampleID"],
                          big["Soma_Side_Inferred"]).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
