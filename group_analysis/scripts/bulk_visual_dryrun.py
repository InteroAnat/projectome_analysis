"""Dry-run: check which neurons would be processed per sample / per group."""
import os, sys, glob
import pandas as pd

sys.path.insert(0, r"D:\projectome_analysis\group_analysis\scripts")
from insula_label_set import build_insula_label_set, normalize_label, strip_prefix

INSULA_LABELS, _ = build_insula_label_set()
NEW_SAMPLES = ["251730", "252383", "252384", "252385"]
STEP1_DIR = r"D:\projectome_analysis\group_analysis\step1_results"


def assign_group(soma_region_raw):
    if soma_region_raw is None or (isinstance(soma_region_raw, float)
                                     and pd.isna(soma_region_raw)):
        return "Unknown"
    raw = str(soma_region_raw).strip()
    if not raw or "Unknown" in raw:
        return "Unknown"
    clean_upper = normalize_label(raw)
    if clean_upper in INSULA_LABELS:
        return "INS"
    if strip_prefix(raw).upper() == "PRCO":
        return "PrCO"
    return None


def find_xlsx(sid):
    pat = os.path.join(STEP1_DIR, f"{sid}_*_region_analysis", "tables",
                        f"{sid}_results_*.xlsx")
    m = sorted(glob.glob(pat))
    return m[-1] if m else None


grand_total = {"INS": 0, "PrCO": 0, "Unknown": 0, "skip": 0, "all": 0}
for sid in NEW_SAMPLES:
    xlsx = find_xlsx(sid)
    if not xlsx:
        print(f"[{sid}] no xlsx")
        continue
    df = pd.read_excel(xlsx, sheet_name="Summary")
    df["group"] = df["Soma_Region"].map(assign_group)
    counts = df["group"].value_counts(dropna=False).to_dict()
    grand_total["all"] += len(df)
    for g in ("INS", "PrCO", "Unknown"):
        grand_total[g] += counts.get(g, 0)
    grand_total["skip"] += counts.get(None, 0) + sum(
        v for k, v in counts.items() if k not in ("INS", "PrCO", "Unknown"))

    print(f"\n=== {sid}: total {len(df)} ===")
    print(f"  INS:     {counts.get('INS', 0)}")
    print(f"  PrCO:    {counts.get('PrCO', 0)}")
    print(f"  Unknown: {counts.get('Unknown', 0)}")
    skip_n = sum(v for k, v in counts.items()
                 if k not in ("INS", "PrCO", "Unknown"))
    print(f"  (skipped non-INS/PrCO/Unknown: {skip_n})")
    # Show distribution of skipped Soma_Region values
    skip_df = df[~df["group"].isin(["INS", "PrCO", "Unknown"])]
    if len(skip_df):
        top_skip = skip_df["Soma_Region"].value_counts().head(8).to_dict()
        print(f"  top skipped regions: {top_skip}")

print("\n" + "=" * 50)
print("GRAND TOTAL")
print("=" * 50)
print(f"  INS     : {grand_total['INS']}")
print(f"  PrCO    : {grand_total['PrCO']}")
print(f"  Unknown : {grand_total['Unknown']}")
print(f"  Skipped : {grand_total['skip']}")
print(f"  ALL     : {grand_total['all']}")
print(f"  Will render: {grand_total['INS'] + grand_total['PrCO'] + grand_total['Unknown']}")
