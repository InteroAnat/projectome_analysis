#!/usr/bin/env python3
"""Smoke test: Liu VEN-L/S/PC (NeuroM on ASC) vs user-confirmed 251637 VENs (SWC)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from morph_features import FEATURE_SCORE_COLS, DEFAULT_R_LOCAL, features_table

REPO = Path(__file__).resolve().parents[3]
LIU_ASC = (
    REPO
    / "external/liu2026/morph/NeuroMorph_upload260215/PatchClamp_morph"
)
OUT = REPO / "external/ven_validation/outputs"
CAND = REPO / "neuron_tables/251637_von_economo_user_confirmed.xlsx"
CANONICAL_COMBINED = (
    REPO / "group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx"
)


def load_neuron_metadata() -> pd.DataFrame:
    """Projection class + soma region from combined harmonized Summary sheet."""
    df = pd.read_excel(CANONICAL_COMBINED, sheet_name="Summary")
    df = df[df["SampleID"] == 251637].copy()
    cols = [
        "NeuronID",
        "Neuron_Type",
        "Soma_Region_Refined",
        "Soma_Area_Henry",
        "Cortical_Layer",
        "Layer_Source",
        "Total_Length",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    return out.rename(columns={"Soma_Region_Refined": "Soma_Region"})


def resolve_fmost_swc(neuron_id: str) -> Path | None:
    for p in (
        REPO / "processed_neurons/251637" / neuron_id,
        REPO / "processed_neurons/251637/raw_swcs" / neuron_id,
    ):
        if p.is_file():
            return p
    return None


def zscore_matrix(ref: pd.DataFrame, query: pd.DataFrame, cols: list[str]):
    mu = ref[cols].astype(float).mean()
    sd = ref[cols].astype(float).std(ddof=0).replace(0, np.nan)
    zr = (ref[cols].astype(float) - mu) / sd
    zq = (query[cols].astype(float) - mu) / sd
    return zr.fillna(0.0), zq.fillna(0.0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Liu features (NeuroM on ASC) ===")
    liu_rows = []
    for cls in ["VENL", "VENS", "PC-L5_ET"]:
        # Windows is case-insensitive: glob *.ASC and *.asc would double-count
        seen: dict[str, Path] = {}
        for p in sorted((LIU_ASC / cls).iterdir()):
            if p.suffix.lower() == ".asc" and p.is_file():
                seen[p.stem.lower()] = p
        paths = sorted(seen.values(), key=lambda x: x.stem)
        if not paths:
            raise SystemExit(f"missing ASC for {cls} under {LIU_ASC / cls}")
        df = features_table(paths, source=f"liu_{cls}", r_local=DEFAULT_R_LOCAL)
        df["ref_class"] = cls
        df["neuron_id"] = [p.stem for p in paths]
        liu_rows.append(df)
        n_err = int(df["error"].notna().sum()) if "error" in df else 0
        backend = (
            df.loc[df["error"].isna(), "backend"].iloc[0]
            if "error" in df.columns and df["error"].isna().any()
            else df.get("backend", pd.Series(["?"])).iloc[0]
        )
        print(cls, "n=", len(df), "errors=", n_err, "backend=", backend)
    liu = pd.concat(liu_rows, ignore_index=True)
    liu_path = OUT / "liu_ven_features.tsv"
    liu.to_csv(liu_path, sep="\t", index=False)
    print("wrote", liu_path, len(liu))

    print("=== fMOST confirmed VENs (NeuroM on SWC) ===")
    cand = pd.read_excel(CAND)
    fpaths = []
    for r in cand.itertuples():
        p = resolve_fmost_swc(str(r.NeuronID))
        if p is None:
            print("MISSING SWC", r.NeuronID)
            continue
        fpaths.append(p)
        print("use", p)
    fmost = features_table(fpaths, source="fmost_251637_ven", r_local=DEFAULT_R_LOCAL)
    fmost["neuron_id"] = [Path(p).name for p in fmost["path"]]
    if CANONICAL_COMBINED.is_file():
        summary = load_neuron_metadata()
        fmost = fmost.merge(
            summary,
            left_on="neuron_id",
            right_on="NeuronID",
            how="left",
        )
    fmost_path = OUT / "fmost_251637_ven_features.tsv"
    fmost.to_csv(fmost_path, sep="\t", index=False)
    print("wrote", fmost_path, len(fmost))

    cols = [c for c in FEATURE_SCORE_COLS if c in liu.columns and c in fmost.columns]
    print("score cols:", cols)
    liu_ok = liu.copy()
    if "error" in liu_ok:
        liu_ok = liu_ok[liu_ok["error"].isna()]
    f_ok = fmost.copy()
    if "error" in f_ok:
        f_ok = f_ok[f_ok["error"].isna()]

    scores = []
    for cls in ["VENL", "VENS", "PC-L5_ET"]:
        ref = liu_ok[liu_ok["ref_class"] == cls]
        zr, zq = zscore_matrix(ref, f_ok, cols)
        centroid = zr.mean().values
        for i, row in f_ok.iterrows():
            v = zq.loc[i].values.astype(float)
            dist = float(np.linalg.norm(v - centroid))
            scores.append(
                {
                    "neuron_id": row["neuron_id"],
                    "ref_class": cls,
                    "euclid_z": dist,
                    "Neuron_Type": row.get("Neuron_Type"),
                    "Soma_Region": row.get("Soma_Region"),
                    "Soma_Area_Henry": row.get("Soma_Area_Henry"),
                    "Cortical_Layer": row.get("Cortical_Layer"),
                    "Layer_Source": row.get("Layer_Source"),
                    "Total_Length": row.get("Total_Length"),
                    "basal_max_path": row.get("basal_max_path"),
                    "apical_max_path": row.get("apical_max_path"),
                    "bipolar_path_symmetry": row.get("bipolar_path_symmetry"),
                    "stem_opposition": row.get("stem_opposition"),
                    "basal_gt_500": row.get("basal_gt_500"),
                    "complete_morph_labels": row.get("complete_morph_labels"),
                    "backend": row.get("backend"),
                }
            )
    sc = pd.DataFrame(scores)
    wide = sc.pivot_table(index="neuron_id", columns="ref_class", values="euclid_z")
    wide.columns = [f"dist_z_{c}" for c in wide.columns]
    best = sc.sort_values(["neuron_id", "euclid_z"]).groupby("neuron_id", as_index=False).first()
    best = best.rename(columns={"ref_class": "nearest_class", "euclid_z": "nearest_dist_z"})
    out_sc = best.merge(wide.reset_index(), on="neuron_id", how="left")
    if "dist_z_VENL" in out_sc.columns and "dist_z_VENS" in out_sc.columns:
        out_sc["prefer_VENL"] = out_sc["dist_z_VENL"] < out_sc["dist_z_VENS"]
    # also keep key morph columns from fmost
    keep_cols = [
        "neuron_id",
        "apical_basal_path_ratio",
        "apical_basal_length_ratio",
        "long_pole_max_path",
        "short_pole_max_path",
        "total_dendrite_length",
    ]
    keep_cols = [c for c in keep_cols if c in f_ok.columns]
    out_sc = out_sc.merge(f_ok[keep_cols], on="neuron_id", how="left")
    sc_path = OUT / "ven_morph_scores_251637.tsv"
    out_sc.to_csv(sc_path, sep="\t", index=False)
    print("wrote", sc_path)
    print(out_sc.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
