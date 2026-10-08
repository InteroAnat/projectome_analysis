"""Rebuild compact tables and a standalone figure from saved diagnostic CSVs."""
from pathlib import Path
from itertools import combinations
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
RUNS = ["multi_raw", "multi_log1p", "multi_spearman_profile", "multi_type_guided",
        "single_raw", "single_spearman_profile"]
frames = []
selected = []
for run in RUNS:
    frame = pd.read_csv(ROOT / run / "clustering_diagnostics.csv")
    frame.insert(0, "run", run)
    frames.append(frame)
    meta = json.loads((ROOT / run / "clustering_metadata.json").read_text())
    row = frame.loc[frame.requested_k == meta["selected_k"]].iloc[0].to_dict()
    row.update(n=meta["n_neurons"], suggested_k=meta["suggested_k"],
               type_guided=meta["type_penalty"], mode=meta["mode"])
    selected.append(row)
pd.concat(frames).to_csv(ROOT / "mode_comparison.csv", index=False)
pd.DataFrame(selected).to_csv(ROOT / "selected_cut_summary.csv", index=False)

comparisons = []
for a, b in combinations(RUNS, 2):
    left = pd.read_csv(ROOT / a / "candidate_cluster_assignments.csv").set_index("FNT_NeuronID")
    right = pd.read_csv(ROOT / b / "candidate_cluster_assignments.csv").set_index("FNT_NeuronID")
    if set(left.index) != set(right.index):
        continue  # Compare partitions only when neuron identity sets agree.
    right = right.loc[left.index]
    for k in [2, 9, 20]:
        comparisons.append({"run_a": a, "run_b": b, "k": k,
                            "ari": adjusted_rand_score(left[f"k_{k}"], right[f"k_{k}"])})
pd.DataFrame(comparisons).to_csv(ROOT / "cross_mode_partition_ari.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
colors = {"multi_raw": "#a33e36", "multi_log1p": "#8c6d31",
          "multi_spearman_profile": "#257793"}
titles = [("silhouette", "Separation (silhouette)"),
          ("subsample_ari_mean", "Global subset stability (mean ARI)"),
          ("worst_cluster_jaccard_mean", "Lowest mean cluster Jaccard\n(assessed clusters only)"),
          ("largest_cluster_fraction", "Fraction in the largest cluster"),
          ("singleton_clusters", "Singleton clusters (unassessed Jaccard)"),
          (None, "Remove dominant monkey 251637\nARI on the remaining cohort")]
for axis, (column, title) in zip(axes.flat, titles):
    for run, color in colors.items():
        if column is None:
            data = pd.read_csv(ROOT / run / "group_holdout_diagnostics.csv")
            data = data.loc[data.held_out_group.astype(str) == "251637"]
            y = data["ari"]
        else:
            data = pd.read_csv(ROOT / run / "clustering_diagnostics.csv")
            y = data[column]
        axis.plot(data.requested_k, y, color=color, lw=1.7, marker=".", label=run[6:])
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Requested k")
    axis.grid(alpha=.2)
    if column == "largest_cluster_fraction":
        axis.set_ylim(0, 1.03)
    axis.set_xticks([2, 5, 9, 15, 20])
axes[0, 0].legend(fontsize=8)
fig.suptitle("Historical FNT cohort: 306 neurons | average linkage | no type penalty\n"
             "100 fixed-distance subsamples, 80% neurons, seed 42; exploratory diagnostics", fontsize=12)
fig.savefig(ROOT / "diagnostic_overview.png", dpi=180)
fig.savefig(ROOT / "diagnostic_overview.pdf")
plt.close(fig)

print(pd.DataFrame(selected)[["run", "n", "requested_k", "suggested_k", "silhouette",
                              "smallest_cluster", "largest_cluster", "subsample_ari_mean"]]
      .round(4).to_string(index=False))
print("\nCross-mode partition comparison:")
print(pd.DataFrame(comparisons).round(4).to_string(index=False))
