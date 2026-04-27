"""
Phase 3c: Audit each new monkey's Summary sheet for the FULL insula
vocabulary (atlas-derived, not 251637-curated subset).

Counts how many neurons each new monkey has in:
  - any insula label (atlas-derived, including Ig, Iai, Iapl, etc.)
  - PrCO (rescue candidate)
  - other (potential rescue if coords match 251637 bbox + tolerance)

Prints a comparison vs the (incorrect) earlier strict-whitelist scan.
"""
from __future__ import annotations

import os
import sys
import glob
import pandas as pd

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
STEP1_DIR = os.path.join(GROUP_DIR, "step1_results")
SCRIPTS = os.path.join(GROUP_DIR, "scripts")
sys.path.insert(0, SCRIPTS)

from insula_label_set import (build_insula_label_set, normalize_label,
                               strip_prefix)

NEW_SAMPLES = ["251730", "252383", "252384", "252385"]


def find_results_xlsx(sid):
    pattern = os.path.join(STEP1_DIR, f"{sid}_*_region_analysis", "tables",
                           f"{sid}_results_*.xlsx")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def main():
    insula_set, rescue_set = build_insula_label_set()
    print(f"Insula vocabulary (n={len(insula_set)}):")
    print(f"  {sorted(insula_set)}\n")

    rows = []
    for sid in NEW_SAMPLES:
        xlsx = find_results_xlsx(sid)
        if not xlsx:
            print(f"[skip] no xlsx for {sid}")
            continue
        summary = pd.read_excel(xlsx, sheet_name="Summary")
        summary["clean_upper"] = summary["Soma_Region"].astype(str).map(
            normalize_label)

        n = len(summary)
        n_insula = int(summary["clean_upper"].isin(insula_set).sum())
        n_prco = int((summary["clean_upper"] == "PRCO").sum())
        n_other = n - n_insula - n_prco

        # Per-side breakdown of insula
        soma_side = summary["Soma_Side"] if "Soma_Side" in summary.columns else None
        ins_rows = summary[summary["clean_upper"].isin(insula_set)]
        if soma_side is not None:
            side_counts = ins_rows["Soma_Side"].value_counts(dropna=False).to_dict()
        else:
            side_counts = {}

        # Per-label counts within insula
        label_counts = ins_rows["clean_upper"].value_counts().to_dict()

        rows.append(dict(
            sample_id=sid, n_total=n,
            n_insula_atlas=n_insula,
            n_PrCO=n_prco,
            n_other=n_other,
            insula_L=side_counts.get("L", 0),
            insula_R=side_counts.get("R", 0),
            insula_unknown=side_counts.get("Unknown", 0),
            insula_label_breakdown=str(label_counts),
        ))
        print(f"--- {sid}  total={n} ---")
        print(f"  insula (atlas-derived) = {n_insula}  (L={side_counts.get('L',0)}, R={side_counts.get('R',0)})")
        print(f"    label breakdown: {label_counts}")
        print(f"  PrCO  = {n_prco}")
        print(f"  other = {n_other}\n")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(GROUP_DIR, "recovery", "atlas_insula_audit.csv")
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    print("\nSUMMARY")
    print(df[["sample_id", "n_total", "n_insula_atlas", "insula_L",
              "insula_R", "n_PrCO", "n_other"]].to_string(index=False))
    print(f"\nTotal atlas-insula across new monkeys: "
          f"{int(df['n_insula_atlas'].sum())} "
          f"(L={int(df['insula_L'].sum())}, R={int(df['insula_R'].sum())})")
    print(f"Total PrCO candidates for coord-rescue: {int(df['n_PrCO'].sum())}")


if __name__ == "__main__":
    main()
