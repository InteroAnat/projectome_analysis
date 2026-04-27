"""
Phase 1: Build 251637 insula coordinate reference (READ-ONLY on 251637).

Extracts per-sub-region NII bounding boxes, centroids, and median nearest-neighbor
distances from the manually curated 251637_INS_HE_inferred.xlsx Summary sheet.

This produces the reference template against which other monkeys' soma coords
will be screened in Phase 3-4 (recovery of insula-mis-classified neurons).

Outputs (under group_analysis/reference/):
  - 251637_subregion_bboxes.csv     - tight 99% bbox per sub-region
  - 251637_subregion_anchors.csv    - centroids + nn-distance thresholds
  - 251637_subregion_neurons.csv    - cleaned per-neuron records used
  - qc_3d_scatter.png               - sanity-check 3D scatter
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from scipy.spatial.distance import cdist

PROJECT_ROOT = r"D:\projectome_analysis"
INPUT_XLSX = os.path.join(PROJECT_ROOT, "neuron_tables_new", "251637_INS_HE_inferred.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "group_analysis", "reference")
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUANTILE_LO = 0.005   # 99% bbox lower
QUANTILE_HI = 0.995   # 99% bbox upper
SOMA_REGION_PREFIXES = ("CL_", "CR_", "L-", "R-", "L_", "R_")


def strip_prefix(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    for p in SOMA_REGION_PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    return s


def main() -> int:
    print(f"[Phase 1] Reading {INPUT_XLSX}")
    summary = pd.read_excel(INPUT_XLSX, sheet_name="Summary")
    print(f"  rows = {len(summary)}, cols = {list(summary.columns)[:8]}...")

    required = {"NeuronID", "Soma_Region", "Soma_Side",
                "Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z"}
    missing = required - set(summary.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        return 1

    df = summary[list(required)].copy()
    df["Soma_Region_clean"] = df["Soma_Region"].astype(str).map(strip_prefix)

    df = df.dropna(subset=["Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z",
                           "Soma_Region_clean"])
    df = df[df["Soma_Region_clean"].str.len() > 0]
    df = df.reset_index(drop=True)

    print("\n[Sub-region counts in 251637 (cleaned)]")
    print(df["Soma_Region_clean"].value_counts())
    print("\n[Soma_Side breakdown]")
    print(pd.crosstab(df["Soma_Region_clean"], df["Soma_Side"]))

    # Persist cleaned per-neuron table for downstream phases
    df.to_csv(os.path.join(OUTPUT_DIR, "251637_subregion_neurons.csv"),
              index=False)

    # Per-sub-region statistics
    bbox_rows = []
    anchor_rows = []
    for region, sub in df.groupby("Soma_Region_clean"):
        if len(sub) < 2:
            print(f"  [skip] {region}: only {len(sub)} neuron(s)")
            continue
        coords = sub[["Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z"]].to_numpy(float)

        # Tight 99% bbox
        lo = np.quantile(coords, QUANTILE_LO, axis=0)
        hi = np.quantile(coords, QUANTILE_HI, axis=0)
        full_lo = coords.min(axis=0)
        full_hi = coords.max(axis=0)
        bbox_rows.append(dict(
            sub_region=region, n=len(sub),
            X_lo_q005=lo[0], X_hi_q995=hi[0],
            Y_lo_q005=lo[1], Y_hi_q995=hi[1],
            Z_lo_q005=lo[2], Z_hi_q995=hi[2],
            X_min=full_lo[0], X_max=full_hi[0],
            Y_min=full_lo[1], Y_max=full_hi[1],
            Z_min=full_lo[2], Z_max=full_hi[2],
        ))

        # Centroid + median pairwise NN distance (used as ambiguity threshold)
        centroid = coords.mean(axis=0)
        dmat = cdist(coords, coords)
        np.fill_diagonal(dmat, np.inf)
        nn = dmat.min(axis=1)
        anchor_rows.append(dict(
            sub_region=region, n=len(sub),
            cx=centroid[0], cy=centroid[1], cz=centroid[2],
            median_nn=float(np.median(nn)),
            mean_nn=float(np.mean(nn)),
            q95_nn=float(np.quantile(nn, 0.95)),
            spread_X=full_hi[0] - full_lo[0],
            spread_Y=full_hi[1] - full_lo[1],
            spread_Z=full_hi[2] - full_lo[2],
        ))

    bbox_df = pd.DataFrame(bbox_rows)
    anchor_df = pd.DataFrame(anchor_rows)

    bbox_csv = os.path.join(OUTPUT_DIR, "251637_subregion_bboxes.csv")
    anchor_csv = os.path.join(OUTPUT_DIR, "251637_subregion_anchors.csv")
    bbox_df.to_csv(bbox_csv, index=False)
    anchor_df.to_csv(anchor_csv, index=False)
    print(f"\n[saved] {bbox_csv}")
    print(f"[saved] {anchor_csv}")

    print("\n[bbox table]")
    print(bbox_df.round(2).to_string(index=False))
    print("\n[anchor table]")
    print(anchor_df.round(2).to_string(index=False))

    # 3D QC scatter
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(bbox_df), 3)))
    for color, region in zip(palette, sorted(df["Soma_Region_clean"].unique())):
        sub = df[df["Soma_Region_clean"] == region]
        ax.scatter(sub["Soma_NII_X"], sub["Soma_NII_Y"], sub["Soma_NII_Z"],
                   s=18, alpha=0.7, label=f"{region} (n={len(sub)})",
                   color=color)
    ax.set_xlabel("Soma_NII_X")
    ax.set_ylabel("Soma_NII_Y")
    ax.set_zlabel("Soma_NII_Z")
    ax.set_title("251637 insula somata (NMT v2.1 NII coords) — Phase 1 reference")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9)
    plt.tight_layout()
    qc_png = os.path.join(OUTPUT_DIR, "qc_3d_scatter.png")
    plt.savefig(qc_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {qc_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
