"""
Phase 3b: Relaxed bbox scan + qualitative inspection of recovery candidates.

Goals:
  - Show what the bbox-strict matches actually look like (which sub-region,
    which side, which neuron type).
  - Add tolerance margins in mm (converted to voxels) to the 99% bbox and see
    whether yield rises into a useful range without sacrificing specificity.
  - Examine PrCO neurons that fall just outside the strict bbox.

Geometry (2026-09-26): prefer folded-X bboxes; pad expressed in mm via
``PROJECTOME_PAD_TOL_MM`` / ``NII_VOXEL_MM``.

Env: PROJECTOME_REFERENCE_DIR, PROJECTOME_RECOVERY_OUT

Output:
  {RECOVERY_OUT}/strict_recovery_inspection.csv
  {RECOVERY_OUT}/relaxed_yield_curve.csv
  {RECOVERY_OUT}/prco_distance_to_bbox.csv
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
SCRIPTS = os.path.join(GROUP_DIR, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from cohort import NII_VOXEL_MM, MIDLINE_NII_X, fold_nii_x  # noqa: E402

REF_DIR = os.environ.get(
    "PROJECTOME_REFERENCE_DIR", os.path.join(GROUP_DIR, "reference")
)
OUT_DIR = os.environ.get(
    "PROJECTOME_RECOVERY_OUT", os.path.join(GROUP_DIR, "recovery")
)
os.makedirs(OUT_DIR, exist_ok=True)

per_csv = os.path.join(OUT_DIR, "discovery_scan_per_neuron.csv")
# Prefer folded; fall back to legacy raw-X if discovery scan lives elsewhere.
if not os.path.exists(per_csv):
    per_csv = os.path.join(GROUP_DIR, "recovery", "discovery_scan_per_neuron.csv")

folded_bb = os.path.join(REF_DIR, "251637_subregion_bboxes_folded.csv")
bb_csv = (
    folded_bb if os.path.exists(folded_bb)
    else os.path.join(REF_DIR, "251637_subregion_bboxes.csv")
)
use_folded = os.path.basename(bb_csv).endswith("_folded.csv")
print(f"[3b] bbox={bb_csv} folded={use_folded}")

per = pd.read_csv(per_csv)
bb = pd.read_csv(bb_csv)


def _match_xyz(row):
    x, y, z = row["Soma_NII_X"], row["Soma_NII_Y"], row["Soma_NII_Z"]
    if use_folded and not pd.isna(x):
        x = fold_nii_x(x)
    return x, y, z


def in_bbox_padded(x, y, z, b, pad_vox):
    if any(pd.isna(v) for v in (x, y, z)):
        return False
    return (b["X_lo_q005"] - pad_vox <= x <= b["X_hi_q995"] + pad_vox
            and b["Y_lo_q005"] - pad_vox <= y <= b["Y_hi_q995"] + pad_vox
            and b["Z_lo_q005"] - pad_vox <= z <= b["Z_hi_q995"] + pad_vox)


def euclidean_distance_to_bbox(x, y, z, b) -> float:
    """0 if inside; otherwise Euclidean distance to nearest face (voxels)."""
    dx = max(b["X_lo_q005"] - x, 0, x - b["X_hi_q995"])
    dy = max(b["Y_lo_q005"] - y, 0, y - b["Y_hi_q995"])
    dz = max(b["Z_lo_q005"] - z, 0, z - b["Z_hi_q995"])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


# 1. Detailed view of strict matches
strict = per[per["recovery_category"].isin([
    "auto_insula", "auto_PrCO_in_insula_bbox", "auto_other_in_insula_bbox"
])].copy()
print("=" * 70)
print(f"STRICT RECOVERY MATCHES (n={len(strict)})")
print("=" * 70)
print(strict[["SampleID", "NeuronID", "Soma_Region", "Soma_Region_clean",
              "Soma_Side", "Neuron_Type",
              "Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z",
              "recovery_category", "bbox_match_subregion"]]
      .to_string(index=False))
strict.to_csv(os.path.join(OUT_DIR, "strict_recovery_inspection.csv"),
              index=False)

# 2. Distance of every PrCO neuron to nearest folded bbox face
prco = per[per["Soma_Region_clean"] == "PrCO"].copy()
records = []
for _, row in prco.iterrows():
    x, y, z = _match_xyz(row)
    best_d, best_r = np.inf, ""
    for _, br in bb.iterrows():
        b = br.to_dict()
        d = euclidean_distance_to_bbox(x, y, z, b)
        if d < best_d:
            best_d, best_r = d, br["sub_region"]
    records.append(dict(
        SampleID=row["SampleID"], NeuronID=row["NeuronID"],
        Soma_Side=row.get("Soma_Side"),
        Soma_NII_X=row["Soma_NII_X"], Soma_NII_Y=y, Soma_NII_Z=z,
        Soma_NII_X_match=x,
        nearest_subregion=best_r,
        distance_to_bbox_vox=best_d,
        distance_to_bbox_mm=best_d * NII_VOXEL_MM,
        recovery_category=row["recovery_category"],
    ))
dist_df = pd.DataFrame(records).sort_values("distance_to_bbox_vox")
dist_csv = os.path.join(OUT_DIR, "prco_distance_to_bbox.csv")
dist_df.to_csv(dist_csv, index=False)
print(f"\n[saved] {dist_csv}")
print("\nDistance-to-bbox distribution for all PrCO across new monkeys:")
print(dist_df["distance_to_bbox_mm"].describe())
print("\nFirst 25 sorted by distance (closest first):")
print(dist_df.head(25).to_string(index=False))

# 3. Yield as a function of bbox padding (mm → voxels)
all_per = per.copy()
yield_rows = []
for pad_mm in [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]:
    pad_vox = pad_mm / NII_VOXEL_MM
    matched = []
    for _, row in all_per.iterrows():
        x, y, z = _match_xyz(row)
        for _, br in bb.iterrows():
            b = br.to_dict()
            if in_bbox_padded(x, y, z, b, pad_vox):
                matched.append((row["SampleID"], row["Soma_Region_clean"],
                                row.get("Soma_Side")))
                break
    n = len(matched)
    n_per_sample = pd.Series([m[0] for m in matched]).value_counts().to_dict()
    row_out = dict(
        pad_mm=pad_mm, pad_vox=pad_vox, total=n,
        midline_nii_x=MIDLINE_NII_X, nii_voxel_mm=NII_VOXEL_MM,
        folded=use_folded,
    )
    for sid, cnt in n_per_sample.items():
        row_out[f"n_{sid}"] = cnt
    yield_rows.append(row_out)
yc = pd.DataFrame(yield_rows).fillna(0)
yc_csv = os.path.join(OUT_DIR, "relaxed_yield_curve.csv")
yc.to_csv(yc_csv, index=False)
print(f"\n[saved] {yc_csv}")
print("\nYield as a function of bbox padding (mm):")
print(yc.to_string(index=False))
