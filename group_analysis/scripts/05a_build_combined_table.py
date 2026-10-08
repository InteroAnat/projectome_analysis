"""
Phase 5a: Build combined insula projection table across 251637 + new monkeys.

Outputs (under group_analysis/combined/):
  multi_monkey_INS_combined.xlsx
    - Summary               : validated reference neurons + selected new keepers
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
The historical canonical derivative had 306 neurons. Counts now follow the
validated inputs; an existing output is never replaced. Set
PROJECTOME_COMBINED_OUT to a fresh output directory for a derivative build.
Every configured recovery workbook and all eight quantitative sheets are
required; selected quantitative rows are aligned explicitly to Summary order.
"""
from __future__ import annotations

import os
import sys
import glob
import tempfile
from pathlib import Path
from numbers import Real
import pandas as pd
import numpy as np

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
SCRIPTS = os.path.join(GROUP_DIR, "scripts")
sys.path.insert(0, SCRIPTS)
from insula_label_set import normalize_label, strip_prefix as _strip_prefix
from cohort import NEW_SAMPLES, REFERENCE_SAMPLE

REF_INS_XLSX = os.path.join(PROJECT_ROOT, "neuron_tables_new",
                             "251637_INS_HE_inferred.xlsx")
RECOVERY_DIR = os.environ.get(
    "PROJECTOME_RECOVERY_DIR", os.path.join(GROUP_DIR, "recovery")
)
STEP1_DIR = os.environ.get(
    "PROJECTOME_STEP1_DIR", os.path.join(GROUP_DIR, "step1_results")
)
OUT_DIR = os.environ.get(
    "PROJECTOME_COMBINED_OUT", os.path.join(GROUP_DIR, "combined")
)

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


def _checked_identities(frame, sample_id, source):
    """Validate supplied identities before adding canonical sample/UID columns."""
    sample_id = str(sample_id)
    if "NeuronID" not in frame or not frame.NeuronID.map(
        lambda value: isinstance(value, str) and bool(value.strip()) and value == value.strip()
    ).all():
        raise ValueError(f"{source}: NeuronID must be present literal text")
    if frame.NeuronID.duplicated().any():
        raise ValueError(f"{source}: duplicate neuron identities")
    if "SampleID" in frame:
        def token(value):
            if pd.isna(value) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{source}: invalid SampleID")
            if isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
                return str(int(value))
            return str(value).strip()
        if not frame.SampleID.map(token).eq(sample_id).all():
            raise ValueError(f"{source}: foreign SampleID")
    expected = sample_id + "::" + frame.NeuronID
    if "NeuronUID" in frame and not frame.NeuronUID.eq(expected).all():
        raise ValueError(f"{source}: NeuronUID disagrees with sample/neuron identities")
    result = frame.copy()
    result["SampleID"] = sample_id
    result["NeuronUID"] = expected
    return result


def union_projection_sheet(combined_meta: pd.DataFrame, source_xlsx: str,
                            sheet_name: str, sample_id: str,
                            keep_neuron_ids: list[str]) -> pd.DataFrame:
    """Require all selected identities and explicitly align rows to keeper order."""
    _checked_identities(pd.DataFrame({"NeuronID": keep_neuron_ids}), sample_id, "keepers")
    df = _checked_identities(pd.read_excel(source_xlsx, sheet_name=sheet_name),
                             sample_id, f"{source_xlsx}:{sheet_name}")
    selected = df[df["NeuronID"].isin(keep_neuron_ids)]
    if set(selected.NeuronID) != set(keep_neuron_ids):
        raise ValueError(f"{sample_id}:{sheet_name}: missing selected neuron identities")
    return selected.set_index("NeuronID", drop=False).loc[keep_neuron_ids].reset_index(drop=True)


def main() -> int:
    out_xlsx = Path(OUT_DIR) / "multi_monkey_INS_combined.xlsx"
    if out_xlsx.exists():
        raise FileExistsError(f"Combined output already exists; use a fresh output directory: {out_xlsx}")
    # 1. Read 251637 untouched (Summary + projection sheets)
    print(f"[5a] Reading {REFERENCE_SAMPLE} untouched: {REF_INS_XLSX}")
    print(f"[5a] RECOVERY_DIR={RECOVERY_DIR}")
    print(f"[5a] OUT_DIR={OUT_DIR}")
    print(f"[5a] NEW_SAMPLES={NEW_SAMPLES}")
    with pd.ExcelFile(REF_INS_XLSX) as ref_xl:
        missing = set(("Summary",) + PROJ_SHEETS) - set(ref_xl.sheet_names)
    if missing:
        raise ValueError(f"Reference workbook missing required sheets: {sorted(missing)}")
    ref_summary = _checked_identities(pd.read_excel(REF_INS_XLSX, sheet_name="Summary"),
                                      REFERENCE_SAMPLE, "reference Summary")
    if ref_summary.empty:
        raise ValueError("Reference Summary cannot be empty")
    # Add provenance
    ref_summary["Soma_Region_Auto"] = ref_summary["Soma_Region"]
    ref_summary["Soma_Region_Refined"] = ref_summary["Soma_Region"].astype(str).map(
        normalize_label)
    ref_summary["Soma_Region_Source"] = "curated_251637"
    print(f"  251637 summary: {len(ref_summary)} rows")

    ref_proj_sheets = {}
    for s in PROJ_SHEETS:
        df = _checked_identities(pd.read_excel(REF_INS_XLSX, sheet_name=s), REFERENCE_SAMPLE, s)
        if set(df.NeuronID) != set(ref_summary.NeuronID):
            raise ValueError(f"Reference {s}: membership differs from Summary")
        df = df.set_index("NeuronID", drop=False).loc[ref_summary.NeuronID.tolist()].reset_index(drop=True)
        ref_proj_sheets[s] = df
        print(f"  251637 {s}: {df.shape}")

    # 2. New-monkey keepers (already filtered)
    new_summaries = []
    new_proj_dfs = {s: [] for s in PROJ_SHEETS}

    for sid in NEW_SAMPLES:
        kp_xlsx = os.path.join(RECOVERY_DIR, f"{sid}_INS_HE_coord_inferred.xlsx")
        if not os.path.exists(kp_xlsx):
            raise FileNotFoundError(f"Configured sample {sid} missing recovery workbook: {kp_xlsx}")
        keeper_input = pd.read_excel(kp_xlsx, sheet_name="Insula_keepers")
        if "SampleID" not in keeper_input:
            raise ValueError(f"{kp_xlsx}: keeper identities require SampleID")
        keepers = _checked_identities(keeper_input, sid, kp_xlsx)
        if keepers.empty:
            print(f"  {sid}: 0 keepers")
            continue
        # Final Soma_Region is Soma_Region_Refined; final Soma_Side from inferred
        keepers["Soma_Region_Final"] = keepers["Soma_Region_Refined"]
        new_summaries.append(keepers)

        # Projection sheets from the original results.xlsx
        src = find_results_xlsx(sid)
        if not src:
            raise FileNotFoundError(f"Configured sample {sid} has keepers but no source workbook")
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
    if combined_summary.NeuronUID.duplicated().any():
        raise ValueError("Combined Summary has duplicate full neuron identities")

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
        if merged.NeuronUID.tolist() != combined_summary.NeuronUID.tolist():
            raise ValueError(f"Combined {s}: ordered membership differs from Summary")
        print(f"  combined {s}: {merged.shape}")

    # 5. Write workbook
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".combined_build_", suffix=".xlsx", dir=out_xlsx.parent)
    os.close(fd)
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as w:
            combined_summary.to_excel(w, sheet_name="Summary", index=False)
            for s in PROJ_SHEETS:
                combined_proj[s].to_excel(w, sheet_name=s, index=False)
            prov = (combined_summary["Soma_Region_Source"]
                    .value_counts().rename("n").reset_index()
                    .rename(columns={"index": "Soma_Region_Source"}))
            prov.to_excel(w, sheet_name="Provenance", index=False)
        # Fail atomically if another process created the destination meanwhile.
        os.link(temporary, out_xlsx)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    print(f"\n[saved] {out_xlsx}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
