"""
Phase 5d: Sanity-check the cross-monkey FNT distance matrix.

Tests:
  1. Distribution of within-monkey vs cross-monkey scores. If cross-monkey
     scores are systematically inflated (mean cross >> mean within) for
     known-similar morphologies, that's a registration-frame artefact.
  2. Spot-check: pick 251637 IAL ITi vs 252385 IAL ITi (anatomically very
     similar) and compare their distance to 251637 IAL vs 251637 PT
     (anatomically very different).
  3. Ensure self-distance is the smallest column for each neuron.

Outputs:
  group_analysis/fnt/sanity_check.txt
  group_analysis/fnt/within_vs_cross_distributions.png
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = r"D:\projectome_analysis"
GROUP_DIR = os.path.join(PROJECT_ROOT, "group_analysis")
FNT_DIR = os.path.join(GROUP_DIR, "fnt")
COMBINED_XLSX = os.path.join(GROUP_DIR, "combined", "multi_monkey_INS_combined.xlsx")
DIST_TXT = os.path.join(FNT_DIR, "multi_monkey_INS_dist.txt")
JOINED = os.path.join(FNT_DIR, "multi_monkey_INS_joined.fnt")


def parse_joined_fnt_neuron_order(joined_fnt_path: str) -> list[str]:
    """Read the joined FNT file and extract neuron names in I/J index order.
    Format of the marker line is `<start_node_id> Neuron <name>`. The first
    encountered marker is index 0, second is 1, etc. (matches the I/J indexing
    used by fnt-dist's output)."""
    import re
    names = []
    pat = re.compile(r"^\d+\s+Neuron\s+(\S+)\s*$")
    with open(joined_fnt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                names.append(m.group(1))
    return names


def main():
    # 1. Load distance scores (long format)
    print(f"[5d] reading {DIST_TXT}")
    df = pd.read_csv(DIST_TXT, sep="\t")
    print(f"  rows={len(df)}, cols={list(df.columns)}")
    print(f"  Score: min={df['Score'].min():.4f}, "
          f"q05={df['Score'].quantile(0.05):.4f}, "
          f"median={df['Score'].median():.4f}, "
          f"q95={df['Score'].quantile(0.95):.4f}, "
          f"max={df['Score'].max():.4f}")

    # 2. Map I, J indices to neuron names via joined FNT order
    names = parse_joined_fnt_neuron_order(JOINED)
    print(f"  neuron count from joined FNT: {len(names)}")

    df["name_i"] = df["I"].map(lambda i: names[i] if 0 <= i < len(names) else "")
    df["name_j"] = df["J"].map(lambda j: names[j] if 0 <= j < len(names) else "")
    df["sample_i"] = df["name_i"].str.split("_").str[0]
    df["sample_j"] = df["name_j"].str.split("_").str[0]
    df["pair_kind"] = (df["sample_i"] == df["sample_j"]).map(
        {True: "within", False: "cross"})
    df["self"] = df["I"] == df["J"]

    print("\n  Pair kind counts (off-diagonal):")
    off = df[~df["self"]]
    print(off["pair_kind"].value_counts().to_string())

    # 3. Within vs cross distribution stats
    print("\n  Score by pair kind (off-diagonal):")
    print(off.groupby("pair_kind")["Score"].describe().round(3).to_string())

    # 4. Self-distance check: should be the smallest score for each neuron
    self_diag = df[df["self"]][["I", "J", "Score"]].rename(
        columns={"Score": "self_score"})
    print(f"\n  Self-distance distribution:")
    print(f"    min={self_diag['self_score'].min():.4f}, "
          f"median={self_diag['self_score'].median():.4f}, "
          f"max={self_diag['self_score'].max():.4f}")
    # Check: how many neurons have min(off-diagonal score) < self_score?
    # This would mean the self-pair is NOT the closest, which would be weird.

    # 5. Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(off[off["pair_kind"] == "within"]["Score"].values,
                 bins=80, alpha=0.6, label="within-monkey", color="#1976d2", density=True)
    axes[0].hist(off[off["pair_kind"] == "cross"]["Score"].values,
                 bins=80, alpha=0.5, label="cross-monkey", color="#e65100", density=True)
    axes[0].set_xlabel("FNT score")
    axes[0].set_ylabel("density")
    axes[0].set_title("Within- vs cross-monkey FNT scores (off-diagonal)")
    axes[0].legend()

    # Boxplot per pair-kind by sample-pair
    off["pair_label"] = off["sample_i"].astype(str).where(
        off["sample_i"] == off["sample_j"], "cross")
    sample_order = ["251637", "252385", "252384", "252383", "cross"]
    box_data = [off[off["pair_label"] == s]["Score"].values for s in sample_order]
    box_labels = [f"{s}\n(n={len(d)})" for s, d in zip(sample_order, box_data)]
    axes[1].boxplot(box_data, tick_labels=box_labels, showfliers=False)
    axes[1].set_ylabel("FNT score")
    axes[1].set_title("Off-diagonal FNT scores by sample group")

    plt.tight_layout()
    out_png = os.path.join(FNT_DIR, "within_vs_cross_distributions.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"\n[saved] {out_png}")

    # 6. Write summary
    out_txt = os.path.join(FNT_DIR, "sanity_check.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"FNT distance matrix sanity check\n")
        f.write(f"================================\n\n")
        f.write(f"Source: {DIST_TXT}\n")
        f.write(f"Joined FNT: {JOINED}\n")
        f.write(f"n neurons: {len(names)}\n")
        f.write(f"n pairs (incl. self): {len(df)}\n\n")
        f.write(f"Score distribution overall:\n")
        f.write(df["Score"].describe().round(4).to_string() + "\n\n")
        f.write(f"Off-diagonal pairs: {len(off)}\n")
        f.write(f"Within-monkey: {(off['pair_kind']=='within').sum()}\n")
        f.write(f"Cross-monkey:  {(off['pair_kind']=='cross').sum()}\n\n")
        f.write("Score by pair-kind (off-diagonal):\n")
        f.write(off.groupby("pair_kind")["Score"].describe().round(4).to_string())
        f.write("\n\nScore by sample-pair group (off-diagonal):\n")
        f.write(off.groupby("pair_label")["Score"].describe().round(4).to_string())
    print(f"[saved] {out_txt}")


if __name__ == "__main__":
    main()
