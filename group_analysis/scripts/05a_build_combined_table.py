"""
Phase 5a: Build combined insula projection table across 251637 + new monkeys.

Outputs (under group_analysis/combined/):
  multi_monkey_INS_combined.xlsx
    - Summary               : 306 neurons (251637 untouched + new keepers)
    - Projection_Length_L3_ipsi
    - Projection_Length_L3_contra
    - Projection_Strength_L3_ipsi
    - Projection_Strength_L3_contra

The projection sheets union the target-region columns across monkeys,
filling missing values with 0. The Summary sheet adds:
    - SampleID
    - Soma_Region (final, refined for new monkeys)
    - Soma_Region_Source (auto_atlas_insula | coord_inferred_from_251637_*
                          | curated_251637)
    - All NII coords from each monkey's own step1 output

This file is the input for Phase 5c (FNT) and Phase 6 (R L/R analysis).
"""
from __future__ import annotations

import os
import sys
import glob
import pandas as pd
import numpy as np

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
SCRIPTS = os.path.join(GROUP_DIR, "scripts")
sys.path.insert(0, SCRIPTS)
from insula_label_set import normalize_label, strip_prefix as _strip_prefix

REF_INS_XLSX = os.path.join(PROJECT_ROOT, "neuron_tables_new",
                             "251637_INS_HE_inferred.xlsx")
RECOVERY_DIR = os.path.join(GROUP_DIR, "recovery")
STEP1_DIR = os.path.join(GROUP_DIR, "step1_results")
OUT_DIR = os.path.join(GROUP_DIR, "combined")
os.makedirs(OUT_DIR, exist_ok=True)

NEW_SAMPLES = ["251730", "252383", "252384", "252385"]
PROJ_SHEETS = (
    # Finest level (L6) - has Ial/Ig/Iam/Iapm etc. as separate columns
    "Projection_Length_ipsi",
    "Projection_Length_contra",
    "Projection_Strength_ipsi",
    "Projection_Strength_contra",
    # L3 grouping - has caudal_OFC/floor_of_ls/Str etc. as columns
    "Projection_Length_L3_ipsi",
    "Projection_Length_L3_contra",
    "Projection_Strength_L3_ipsi",
    "Projection_Strength_L3_contra",
)


def find_results_xlsx(sid):
    pattern = os.path.join(STEP1_DIR, f"{sid}_*_region_analysis", "tables",
                           f"{sid}_results_*.xlsx")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def union_projection_sheet(combined_meta: pd.DataFrame, source_xlsx: str,
                            sheet_name: str, sample_id: str,
                            keep_neuron_ids: list[str]) -> pd.DataFrame:
    """Read sheet_name from source_xlsx, filter to keep_neuron_ids,
    add SampleID + composite NeuronID for cross-monkey uniqueness."""
    try:
        df = pd.read_excel(source_xlsx, sheet_name=sheet_name)
    except (KeyError, ValueError) as e:
        print(f"  [warn] {sample_id}: missing sheet '{sheet_name}' -> empty")
        return pd.DataFrame()
    if "NeuronID" not in df.columns:
        print(f"  [warn] {sample_id}: no NeuronID column in {sheet_name}")
        return pd.DataFrame()
    df = df[df["NeuronID"].isin(keep_neuron_ids)].copy()
    df["SampleID"] = sample_id
    df["NeuronUID"] = df["SampleID"].astype(str) + "::" + df["NeuronID"].astype(str)
    return df


def main() -> int:
    # 1. Read 251637 untouched (Summary + projection sheets)
    print(f"[5a] Reading 251637 untouched: {REF_INS_XLSX}")
    ref_xl = pd.ExcelFile(REF_INS_XLSX)
    ref_summary = pd.read_excel(REF_INS_XLSX, sheet_name="Summary")
    ref_summary["SampleID"] = "251637"
    ref_summary["NeuronUID"] = "251637::" + ref_summary["NeuronID"].astype(str)
    # Add provenance
    ref_summary["Soma_Region_Auto"] = ref_summary["Soma_Region"]
    ref_summary["Soma_Region_Refined"] = ref_summary["Soma_Region"].astype(str).map(
        normalize_label)
    ref_summary["Soma_Region_Source"] = "curated_251637"
    print(f"  251637 summary: {len(ref_summary)} rows")

    ref_proj_sheets = {}
    for s in PROJ_SHEETS:
        if s in ref_xl.sheet_names:
            df = pd.read_excel(REF_INS_XLSX, sheet_name=s)
            df["SampleID"] = "251637"
            df["NeuronUID"] = "251637::" + df["NeuronID"].astype(str)
            ref_proj_sheets[s] = df
            print(f"  251637 {s}: {df.shape}")
        else:
            ref_proj_sheets[s] = pd.DataFrame()
            print(f"  251637 {s}: MISSING")

    # 2. New-monkey keepers (already filtered)
    new_summaries = []
    new_proj_dfs = {s: [] for s in PROJ_SHEETS}

    for sid in NEW_SAMPLES:
        kp_xlsx = os.path.join(RECOVERY_DIR, f"{sid}_INS_HE_coord_inferred.xlsx")
        if not os.path.exists(kp_xlsx):
            print(f"  [skip] {sid}: no recovery xlsx")
            continue
        keepers = pd.read_excel(kp_xlsx, sheet_name="Insula_keepers")
        if keepers.empty:
            print(f"  {sid}: 0 keepers")
            continue
        keepers["NeuronUID"] = keepers["SampleID"].astype(str) + "::" + \
                                keepers["NeuronID"].astype(str)
        # Final Soma_Region is Soma_Region_Refined; final Soma_Side from inferred
        keepers["Soma_Region_Final"] = keepers["Soma_Region_Refined"]
        new_summaries.append(keepers)

        # Projection sheets from the original results.xlsx
        src = find_results_xlsx(sid)
        if not src:
            print(f"  [skip] {sid}: no source xlsx")
            continue
        keep_ids = keepers["NeuronID"].tolist()
        for s in PROJ_SHEETS:
            df = union_projection_sheet(ref_summary, src, s, sid, keep_ids)
            if not df.empty:
                new_proj_dfs[s].append(df)
                print(f"  {sid} {s}: kept {len(df)} rows")

    # 3. Build combined Summary
    summary_cols_keep = [
        "NeuronUID", "SampleID", "NeuronID", "Neuron_Type",
        "Soma_Region_Auto", "Soma_Region_Refined", "Soma_Region_Source",
        "Soma_Side", "Soma_Side_Inferred",
        "Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z",
        "Soma_Phys_X", "Soma_Phys_Y", "Soma_Phys_Z",
        "Total_Length", "Terminal_Count",
        "N_Ipsilateral", "N_Contralateral", "N_Laterality_Unknown",
        "Laterality_Index",
    ]
    if "Soma_Side_Inferred" not in ref_summary.columns:
        ref_summary["Soma_Side_Inferred"] = ref_summary.get("Soma_Side")
    ref_keep_cols = [c for c in summary_cols_keep if c in ref_summary.columns]
    ref_block = ref_summary[ref_keep_cols].copy()
    ref_block["Soma_Region_Refined"] = ref_summary["Soma_Region_Refined"]

    new_blocks = []
    for kp in new_summaries:
        block_cols = [c for c in summary_cols_keep if c in kp.columns]
        new_blocks.append(kp[block_cols].copy())

    combined_summary = pd.concat([ref_block] + new_blocks,
                                  ignore_index=True, sort=False)

    # Use Soma_Side_Inferred when available, fall back to Soma_Side
    combined_summary["Soma_Side_Final"] = combined_summary.get(
        "Soma_Side_Inferred").fillna(combined_summary.get("Soma_Side"))

    print(f"\n[5a] Combined summary: {len(combined_summary)} neurons")
    print("  by SampleID:", combined_summary["SampleID"].value_counts().to_dict())
    print("  by Soma_Side_Final:",
          combined_summary["Soma_Side_Final"].value_counts(dropna=False).to_dict())
    print("  by Soma_Region_Refined:",
          combined_summary["Soma_Region_Refined"].value_counts().to_dict())

    # 4. Build combined projection sheets
    combined_proj = {}
    for s in PROJ_SHEETS:
        all_dfs = ([ref_proj_sheets[s]] if not ref_proj_sheets[s].empty else []) \
                  + new_proj_dfs[s]
        if not all_dfs:
            print(f"  [warn] no data for {s}")
            combined_proj[s] = pd.DataFrame()
            continue
        merged = pd.concat(all_dfs, ignore_index=True, sort=False)
        # Move SampleID + NeuronUID to the front
        front = ["NeuronUID", "SampleID", "NeuronID", "Neuron_Type"]
        front = [c for c in front if c in merged.columns]
        rest = [c for c in merged.columns if c not in front]
        merged = merged[front + rest]
        # Numeric region cells -> fill NaN with 0
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(0.0)
        combined_proj[s] = merged
        print(f"  combined {s}: {merged.shape}")

    # 5. Write workbook
    out_xlsx = os.path.join(OUT_DIR, "multi_monkey_INS_combined.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        combined_summary.to_excel(w, sheet_name="Summary", index=False)
        for s in PROJ_SHEETS:
            combined_proj[s].to_excel(w, sheet_name=s, index=False)
        # provenance summary
        prov = (combined_summary["Soma_Region_Source"]
                .value_counts().rename("n").reset_index()
                .rename(columns={"index": "Soma_Region_Source"}))
        prov.to_excel(w, sheet_name="Provenance", index=False)
    print(f"\n[saved] {out_xlsx}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
