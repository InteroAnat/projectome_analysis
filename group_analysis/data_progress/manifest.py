from __future__ import annotations

import pandas as pd

from .paths import MANIFEST, PLAN_ANIMALS


def five_um_yes(val) -> bool:
    s = str(val).strip().lower()
    return s.startswith("yes") or s in {"y", "1", "true"}


def five_um_label(val) -> str:
    s = str(val).strip()
    if not s or s.lower() in {"nan", "—", "none"}:
        return "—"
    if five_um_yes(s):
        return "Yes"
    if s.lower().startswith("no"):
        return s
    return s


def plan_injection_summary(row: pd.Series) -> str:
    parts = []
    if int(row.get("plan_ven", 0) or 0):
        parts.append("VEN")
    if int(row.get("plan_fida", 0) or 0):
        parts.append("FIDA")
    if int(row.get("plan_fidi", 0) or 0):
        parts.append("FIDI")
    if int(row.get("plan_fidp", 0) or 0):
        parts.append("FIDP")
    return "+".join(parts) if parts else "—"


def load_manifest(path=MANIFEST) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"animal": str, "fmost_id": "Int64"})
    df["animal"] = df["animal"].astype(str)
    df = df[~df["animal"].str.upper().eq("TOTAL")].copy()
    if "tracked_insula_corrected" not in df.columns:
        df["tracked_insula_corrected"] = df.get(
            "corrected_insula_n", pd.Series(0, index=df.index)
        )
    df["insula_corrected"] = pd.to_numeric(
        df["tracked_insula_corrected"], errors="coerce"
    ).fillna(0).astype(int)
    df["ion_n"] = pd.to_numeric(df["ion_n"], errors="coerce").fillna(0).astype(int)
    df["analysis_n"] = pd.to_numeric(df["analysis_n"], errors="coerce")
    df["step1_ok"] = pd.to_numeric(df.get("step1_ok", 0), errors="coerce").fillna(0).astype(int)
    df["five_um_label"] = df.get("5_micron_data_copied", "").map(five_um_label)
    df["five_um_ready"] = df.get("5_micron_data_copied", "").map(five_um_yes)
    df["plan_injections"] = df.apply(plan_injection_summary, axis=1)
    return df


def plan_cohort(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[manifest["animal"].isin(PLAN_ANIMALS)].copy()
