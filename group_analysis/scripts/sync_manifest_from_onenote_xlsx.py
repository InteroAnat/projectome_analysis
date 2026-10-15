"""
OPTIONAL / DEPRECATED — do not run in the normal workflow.

The project uses hand-edited dataset_status_manifest.csv only.
Run build_data_progress.py after editing the manifest.

This script is kept for one-off imports from OneNote xlsx if you explicitly need it.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DOCS = PROJECT / "group_analysis" / "docs"
MANIFEST = DOCS / "dataset_status_manifest.csv"

DEFAULT_XLSX = DOCS / "monkey_experiment_data_summary.xlsx"
ONENOTE_XLSX = Path(
    r"c:\Users\binbi\AppData\Local\Temp\OneNote\16.0\Exported"
    r"\{0DA31105-0030-4DCA-B247-41E7528928B0}\NT\2"
    r"\monkey_experiment_data_summary (1)3.xlsx"
)

PIPELINE_COLS = [
    "ion_n",
    "step1_ok",
    "corrected_insula_n",
    "tracked_insula_corrected",
    "plan_ven",
    "plan_fida",
    "plan_fidi",
    "plan_fidp",
    "notes",
]


def parse_onenote_sheet(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Monkey Data", header=None)
    header_row = None
    for i, row in raw.iterrows():
        vals = [str(v).strip() if pd.notna(v) else "" for v in row]
        if "Monkey ID" in vals and "fMOST ID" in vals:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"No header row found in {path}")

    df = pd.read_excel(path, sheet_name="Monkey Data", header=header_row)
    df = df.rename(
        columns={
            "Monkey ID": "animal",
            "fMOST ID": "fmost_id",
            "Injection Sites": "injection_sites",
            "Number for Analysis": "analysis_n",
            "Data Status": "data_status",
        }
    )
    df = df[df["animal"].notna()].copy()
    df["animal"] = pd.to_numeric(df["animal"], errors="coerce")
    df = df[df["animal"].notna()].copy()
    df["animal"] = df["animal"].astype(int).astype(str)
    df["fmost_id"] = pd.to_numeric(df["fmost_id"], errors="coerce").astype("Int64")
    df["analysis_n"] = pd.to_numeric(df["analysis_n"], errors="coerce")
    df["data_status"] = df["data_status"].astype(str).str.strip()
    df["injection_sites"] = df["injection_sites"].astype(str).str.strip()
    return df


def infer_plan_flags(sites: str) -> dict[str, int]:
    s = sites.upper()
    return {
        "plan_ven": int(any(k in s for k in ("VAIC", "CINGULATE"))),
        "plan_fida": int("IDFA" in s),
        "plan_fidi": int("IDFM" in s),
        "plan_fidp": int("IDFP" in s),
    }


def merge_pipeline_fields(new: pd.DataFrame, old: pd.DataFrame | None) -> pd.DataFrame:
    if old is not None and not old.empty:
        old = old.copy()
        old["animal"] = old["animal"].astype(str)
        new = new.merge(
            old[["animal"] + [c for c in PIPELINE_COLS if c in old.columns]],
            on="animal",
            how="left",
            suffixes=("", "_old"),
        )
    for col in PIPELINE_COLS:
        if col not in new.columns:
            new[col] = pd.NA

    # Fill plan flags from injection sites when missing
    for i, row in new.iterrows():
        flags = infer_plan_flags(str(row["injection_sites"]))
        for k, v in flags.items():
            if pd.isna(new.at[i, k]) or new.at[i, k] == "":
                new.at[i, k] = v

    # Default corrected insula from known combined cohort
    defaults = {
        "936": 260,
        "605": 5,
        "948": 5,
        "331": 36,
    }
    for animal, n in defaults.items():
        mask = new["animal"] == animal
        if mask.any() and (pd.isna(new.loc[mask, "corrected_insula_n"]).all()
                           or (new.loc[mask, "corrected_insula_n"] == 0).all()):
            new.loc[mask, "corrected_insula_n"] = n

    # ion_n: prefer existing probe; else analysis_n for Completed rows
    for i, row in new.iterrows():
        if pd.isna(row.get("ion_n")) or row.get("ion_n") == 0:
            if row["data_status"] == "Completed" and pd.notna(row["analysis_n"]):
                new.at[i, "ion_n"] = int(row["analysis_n"])
        if pd.isna(row.get("step1_ok")):
            new.at[i, "step1_ok"] = int(
                pd.notna(row.get("ion_n")) and row.get("ion_n", 0) > 0
                and row["data_status"] == "Completed"
            )

    return new


def main(xlsx_path: Path | None = None) -> Path:
    src = xlsx_path or (ONENOTE_XLSX if ONENOTE_XLSX.exists() else DEFAULT_XLSX)
    if not src.exists():
        raise FileNotFoundError(src)

    DOCS.mkdir(parents=True, exist_ok=True)
    if src.resolve() != DEFAULT_XLSX.resolve():
        shutil.copy2(src, DEFAULT_XLSX)
        print(f"[copy] {src} -> {DEFAULT_XLSX}")

    parsed = parse_onenote_sheet(src)
    old = pd.read_csv(MANIFEST) if MANIFEST.exists() else None
    merged = merge_pipeline_fields(parsed, old)

    if "tracked_insula_corrected" not in merged.columns:
        merged["tracked_insula_corrected"] = merged.get("corrected_insula_n", 0)
    else:
        merged["tracked_insula_corrected"] = merged["tracked_insula_corrected"].fillna(
            merged.get("corrected_insula_n")
        )

    out_cols = [
        "animal",
        "fmost_id",
        "injection_sites",
        "analysis_n",
        "data_status",
        "ion_n",
        "step1_ok",
        "tracked_insula_corrected",
        "corrected_insula_n",
        "plan_ven",
        "plan_fida",
        "plan_fidi",
        "plan_fidp",
        "notes",
    ]
    merged[out_cols].to_csv(MANIFEST, index=False)
    print(f"[write] {MANIFEST}")
    print(merged[out_cols].to_string(index=False))
    return MANIFEST


if __name__ == "__main__":
    main()
