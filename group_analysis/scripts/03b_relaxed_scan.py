"""
Phase 3b: Relaxed bbox scan + qualitative inspection of the 8 strict-hit
recovery candidates.

Goals:
  - Show what the 8 bbox-strict matches actually look like (which sub-region,
    which side, which neuron type).
  - Add tolerance margins (0/2/4/6 mm) to the 99% bbox and see whether yield
    rises into a useful range without sacrificing specificity.
  - Examine the 15 PrCO neurons in 252385 that fall just outside the strict
    bbox: how far outside (Euclidean distance to nearest bbox face)?

Output:
  group_analysis/recovery/strict_recovery_inspection.csv
  group_analysis/recovery/relaxed_yield_curve.csv
  group_analysis/recovery/prco_distance_to_bbox.csv
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
REF_DIR = os.path.join(GROUP_DIR, "reference")
OUT_DIR = os.path.join(GROUP_DIR, "recovery")

per_csv = os.path.join(OUT_DIR, "discovery_scan_per_neuron.csv")
bb_csv = os.path.join(REF_DIR, "251637_subregion_bboxes.csv")

per = pd.read_csv(per_csv)
bb = pd.read_csv(bb_csv)


def in_bbox_padded(x, y, z, b, pad):
    if any(pd.isna(v) for v in (x, y, z)):
        return False
    return (b["X_lo_q005"] - pad <= x <= b["X_hi_q995"] + pad
            and b["Y_lo_q005"] - pad <= y <= b["Y_hi_q995"] + pad
            and b["Z_lo_q005"] - pad <= z <= b["Z_hi_q995"] + pad)


def euclidean_distance_to_bbox(x, y, z, b) -> float:
    """0 if inside; otherwise Euclidean distance to nearest face."""
    dx = max(b["X_lo_q005"] - x, 0, x - b["X_hi_q995"])
    dy = max(b["Y_lo_q005"] - y, 0, y - b["Y_hi_q995"])
    dz = max(b["Z_lo_q005"] - z, 0, z - b["Z_hi_q995"])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


# 1. Detailed view of the 8 strict matches
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

# 2. Distance of every PrCO neuron in any new monkey to the nearest insula
# sub-region bbox face (closest match wins per neuron)
prco = per[per["Soma_Region_clean"] == "PrCO"].copy()
records = []
for _, row in prco.iterrows():
    x, y, z = row["Soma_NII_X"], row["Soma_NII_Y"], row["Soma_NII_Z"]
    best_d, best_r = np.inf, ""
    for _, br in bb.iterrows():
        b = br.to_dict()
        d = euclidean_distance_to_bbox(x, y, z, b)
        if d < best_d:
            best_d, best_r = d, br["sub_region"]
    records.append(dict(
        SampleID=row["SampleID"], NeuronID=row["NeuronID"],
        Soma_Side=row.get("Soma_Side"),
        Soma_NII_X=x, Soma_NII_Y=y, Soma_NII_Z=z,
        nearest_subregion=best_r,
        distance_to_bbox=best_d,
        recovery_category=row["recovery_category"],
    ))
dist_df = pd.DataFrame(records).sort_values("distance_to_bbox")
dist_csv = os.path.join(OUT_DIR, "prco_distance_to_bbox.csv")
dist_df.to_csv(dist_csv, index=False)
print(f"\n[saved] {dist_csv}")
print("\nDistance-to-bbox distribution for all PrCO across new monkeys:")
print(dist_df["distance_to_bbox"].describe())
print("\nFirst 25 sorted by distance (closest first):")
print(dist_df.head(25).to_string(index=False))

# 3. Yield as a function of bbox padding (0-10 mm)
all_per = per.copy()
yield_rows = []
for pad in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
    matched = []
    for _, row in all_per.iterrows():
        x, y, z = row["Soma_NII_X"], row["Soma_NII_Y"], row["Soma_NII_Z"]
        for _, br in bb.iterrows():
            b = br.to_dict()
            if in_bbox_padded(x, y, z, b, pad):
                matched.append((row["SampleID"], row["Soma_Region_clean"],
                                row.get("Soma_Side")))
                break
    n = len(matched)
    n_per_sample = pd.Series([m[0] for m in matched]).value_counts().to_dict()
    yield_rows.append(dict(
        pad_mm=pad, total=n,
        n_251730=n_per_sample.get("251730", 0),
        n_252383=n_per_sample.get("252383", 0),
        n_252384=n_per_sample.get("252384", 0),
        n_252385=n_per_sample.get("252385", 0),
    ))
yc = pd.DataFrame(yield_rows)
yc_csv = os.path.join(OUT_DIR, "relaxed_yield_curve.csv")
yc.to_csv(yc_csv, index=False)
print(f"\n[saved] {yc_csv}")
print("\nYield as a function of bbox padding (mm):")
print(yc.to_string(index=False))
