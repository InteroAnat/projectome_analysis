"""
fnt_dist_clustering.py - Hybrid Morphological Clustering

Features:
1. Natural Sorting & Robust Matrix Symmetrization.
2. Toggle between Spearman (Rank) and Log1p (Magnitude).
3. **SUPERVISED PENALTY:** Artificially increases distance between different biological types.
4. C-index Optimization for K.
5. Publication-ready Plotting with cluster size annotations on dendrogram.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from natsort import natsorted
from collections import Counter
import os
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
DIST_FILE = r'D:\projectome_analysis\main_scripts\processed_neurons\251637\fnt_processed\ins\ins_dist.txt'
TYPE_FILE = r'D:\projectome_analysis\main_scripts\neuron_tables\INS_df_v3.xlsx'
FNT_FOLDER = r'D:\projectome_analysis\main_scripts\processed_neurons\251637\fnt_processed\ins'

USE_SPEARMAN = True
USE_PENALTY = True
PENALTY_STRENGTH = 1.5


# ==========================================
# 2. LOAD & PREPARE
# ==========================================
def load_data(dist_file, type_file, fnt_folder):
    """Load distance matrix and type annotations, return symmetric matrix and type map."""
    print("--- Loading Data ---")

    if not os.path.exists(dist_file):
        sys.exit(f"Error: File not found {dist_file}")

    df_raw = pd.read_csv(dist_file, sep='\t', header=None)

    try:
        pd.to_numeric(df_raw.iloc[0, 2])
        df_raw.columns = ['i', 'j', 'score', 'm', 'nm'][:len(df_raw.columns)]
    except ValueError:
        print("    Header detected in text file.")
        df_raw = pd.read_csv(dist_file, sep='\t', header=0,
                             names=['i', 'j', 'score', 'm', 'nm'])

    print("    Constructing Square Matrix...")
    all_ids = np.union1d(df_raw['i'].unique(), df_raw['j'].unique())
    matrix = df_raw.pivot(index='i', columns='j', values='score')
    matrix = matrix.reindex(index=all_ids, columns=all_ids)

    m_values = np.nan_to_num(matrix.values.astype(float), nan=0.0)
    m_values = np.maximum(m_values, m_values.T)
    matrix = pd.DataFrame(m_values, index=all_ids, columns=all_ids)

    if os.path.exists(fnt_folder):
        file_list = natsorted(
            [f for f in os.listdir(fnt_folder) if f.endswith('.decimate.fnt')]
        )
        if len(file_list) != len(all_ids):
            print(f"WARNING: ID Mismatch. Matrix: {len(all_ids)}, Files: {len(file_list)}")
            limit = min(len(file_list), len(all_ids))
            file_list = file_list[:limit]
            matrix = matrix.iloc[:limit, :limit]

        id_to_name = {i: f.replace('.decimate.fnt', '') for i, f in enumerate(file_list)}
        matrix.index = matrix.index.map(id_to_name)
        matrix.columns = matrix.columns.map(id_to_name)

    if type_file.endswith('.xlsx'):
        bio_df = pd.read_excel(type_file)
    else:
        bio_df = pd.read_csv(type_file)

    type_map = dict(zip(bio_df['NeuronID'], bio_df['Neuron_Type']))

    common = sorted(set(matrix.index) & set(type_map.keys()))
    matrix = matrix.loc[common, common]
    np.fill_diagonal(matrix.values, 0)

    print(f"    Final Neurons: {len(matrix)}")
    return matrix, type_map


# ==========================================
# 3. TRANSFORMATIONS & PENALTY
# ==========================================
def process_matrix(raw_matrix, type_map, use_spearman=True,
                   use_penalty=True, penalty_strength=1.5):
    """Transform raw distance matrix and optionally apply supervised penalty."""

    if use_spearman:
        print("--- Mode: Spearman Correlation ---")
        corr = raw_matrix.T.corr(method='spearman').fillna(0)
        dist_matrix = (1 - corr).clip(lower=0)
    else:
        print("--- Mode: Log1p Magnitude ---")
        dist_matrix = np.log1p(raw_matrix)

    clean_vals = np.nan_to_num(dist_matrix.values, nan=0.0, posinf=None, neginf=None)
    dist_matrix = pd.DataFrame(clean_vals, index=dist_matrix.index, columns=dist_matrix.columns)
    np.fill_diagonal(dist_matrix.values, 0)

    if use_penalty:
        print(f"--- Applying Penalty ({penalty_strength}x) ---")
        max_val = dist_matrix.max().max()
        if max_val == 0:
            max_val = 1
        penalty_val = max_val * penalty_strength

        ordered_types = [type_map.get(n, 'Unknown') for n in dist_matrix.index]
        type_codes = pd.Categorical(ordered_types).codes
        diff_mask = type_codes[:, None] != type_codes[None, :]

        values = dist_matrix.values.copy()
        values[diff_mask] += penalty_val
        dist_matrix = pd.DataFrame(values, index=dist_matrix.index, columns=dist_matrix.columns)
        print("    Penalty applied.")

    return dist_matrix


# ==========================================
# 4. LINKAGE
# ==========================================
def compute_linkage(dist_matrix, method='ward'):
    """Compute hierarchical linkage from a symmetric distance matrix."""
    print("Computing Linkage...")
    mat_vals = dist_matrix.values.copy()
    mat_vals = (mat_vals + mat_vals.T) / 2
    np.fill_diagonal(mat_vals, 0)
    condensed = squareform(mat_vals, checks=False)
    return linkage(condensed, method=method)


# ==========================================
# 5. C-INDEX OPTIMIZATION
# ==========================================
def calculate_c_index(dist_matrix, linkage_matrix, max_k=65):
    """Find optimal K by minimising the C-index; returns best_k."""
    print(f"\n--- Calculating C-index (Max K={max_k}) ---")

    dists_sorted = np.sort(squareform(dist_matrix.values, checks=False))
    k_range = range(2, max_k + 1)
    c_indices = []

    for k in k_range:
        labels = fcluster(linkage_matrix, t=k, criterion='maxclust')

        S, N_intra = 0.0, 0
        for cid in range(1, k + 1):
            idx = np.where(labels == cid)[0]
            n_mem = len(idx)
            if n_mem > 1:
                sub = dist_matrix.values[np.ix_(idx, idx)]
                S += np.sum(sub[np.triu_indices(n_mem, 1)])
                N_intra += (n_mem * (n_mem - 1)) // 2

        if N_intra == 0:
            c_indices.append(1.0)
            continue

        S_min = np.sum(dists_sorted[:N_intra])
        S_max = np.sum(dists_sorted[-N_intra:])
        c = 0.0 if (S_max - S_min) == 0 else (S - S_min) / (S_max - S_min)
        c_indices.append(c)
        sys.stdout.write(f"\r    k={k} | C={c:.4f}")

    print()

    best_k = list(k_range)[int(np.argmin(c_indices))]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), c_indices, 'o-', color='teal', lw=1.5, markersize=3)
    ax.axvline(best_k, color='red', ls='--', label=f"Best k={best_k}")
    ax.set_title("C-index Optimization", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("C-index (Lower is Better)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    return best_k


# ==========================================
# 6. CLUSTER & SAVE
# ==========================================
def assign_clusters_and_save(dist_matrix, linkage_matrix, type_map,
                             k, use_penalty=True):
    """Assign cluster labels, build results DataFrame, save to Excel, print summary."""
    labels = fcluster(linkage_matrix, t=k, criterion='maxclust')

    results = pd.DataFrame({
        'NeuronID': dist_matrix.index,
        'Bio_Type': [type_map.get(n, 'Unknown') for n in dist_matrix.index],
        'Subtype_Cluster': labels
    })

    # Console summary
    print(f"\n{'=' * 55}")
    print(f"  CLUSTER SUMMARY  (k = {k},  N = {len(results)})")
    print(f"{'=' * 55}")

    cluster_counts = results['Subtype_Cluster'].value_counts().sort_index()
    for cid, count in cluster_counts.items():
        subset = results[results['Subtype_Cluster'] == cid]
        type_breakdown = subset['Bio_Type'].value_counts()
        comp_str = ", ".join(f"{t}: {n}" for t, n in type_breakdown.items())
        print(f"  Cluster {cid:>3d}  |  n = {count:>4d}  |  {comp_str}")

    print(f"{'=' * 55}\n")

    fname = f"fnt_dist_Results_Penalty{'_On' if use_penalty else '_Off'}_k{k}.xlsx"
    results.to_excel(fname, index=False)
    print(f"Saved: {fname}")

    return results


# ==========================================
# 7. VISUALIZATION
# ==========================================
def _annotate_dendrogram_clusters(g, results, Z):
    """
    Annotate cluster sizes (n=XX) on the row dendrogram.
    Draws a bracket + label at the vertical span of each cluster's leaves.
    """
    ax_dendro = g.ax_row_dendrogram
    reordered_idx = g.dendrogram_row.reordered_ind

    # Map reordered positions to cluster labels
    ordered_labels = results['Subtype_Cluster'].values[reordered_idx]

    # Find contiguous runs of each cluster in dendrogram order
    cluster_runs = []
    current_label = ordered_labels[0]
    start = 0
    for i in range(1, len(ordered_labels)):
        if ordered_labels[i] != current_label:
            cluster_runs.append((current_label, start, i - 1))
            current_label = ordered_labels[i]
            start = i
    cluster_runs.append((current_label, start, len(ordered_labels) - 1))

    # Dendrogram x-axis runs right-to-left (tree root on right, leaves on left)
    xlim = ax_dendro.get_xlim()
    x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.03

    # Scale font to neuron count so it stays readable
    n_total = len(ordered_labels)
    base_font = max(5, min(7, 350 // max(n_total, 1)))

    for cid, row_start, row_end in cluster_runs:
        n_neurons = row_end - row_start + 1
        # seaborn clustermap spaces leaves at 10-unit intervals
        y_bot = row_start * 10 + 5
        y_top = row_end * 10 + 5
        y_mid = (y_bot + y_top) / 2.0

        # Small bracket line
        bracket_x = x_pos + (xlim[1] - xlim[0]) * 0.01
        ax_dendro.plot([bracket_x, bracket_x], [y_bot, y_top],
                       color='black', lw=0.6, clip_on=False)
        ax_dendro.plot([bracket_x, bracket_x + (xlim[1] - xlim[0]) * 0.015],
                       [y_bot, y_bot], color='black', lw=0.6, clip_on=False)
        ax_dendro.plot([bracket_x, bracket_x + (xlim[1] - xlim[0]) * 0.015],
                       [y_top, y_top], color='black', lw=0.6, clip_on=False)

        # Label
        ax_dendro.text(
            x_pos, y_mid,
            f"C{cid} n={n_neurons}",
            fontsize=base_font, fontweight='bold',
            color='black',
            ha='left', va='center',
            clip_on=False,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                      edgecolor='none', alpha=0.8)
        )


def plot_heatmap(dist_matrix, Z, results, type_map,
                 use_spearman=True, use_penalty=True):
    """Publication-ready clustermap with cluster-size annotations on dendrogram."""
    print("Generating Heatmap...")

    neuron_labels = dist_matrix.index
    bio_types = [type_map.get(n, 'Unknown') for n in neuron_labels]
    unique_types = sorted(set(bio_types))

    lut = dict(zip(unique_types, sns.color_palette("tab20", len(unique_types))))
    row_colors = pd.Series(bio_types, index=neuron_labels).map(lut)

    mode_str = "Spearman Dist" if use_spearman else "Log1p FNT Dist"
    pen_str = "+ Penalty" if use_penalty else ""
    cbar_label = ("Dissimilarity (1 − Correlation)" if use_spearman
                  else "Dissimilarity (Log Distance)")

    n_clusters = results['Subtype_Cluster'].nunique()
    n_neurons = len(results)

    # --- Clustermap (original settings) ---
    g = sns.clustermap(
        dist_matrix,
        row_linkage=Z, col_linkage=Z,
        row_colors=row_colors,
        cmap='mako' if use_spearman else 'viridis_r',
        xticklabels=False, yticklabels=False,
        dendrogram_ratio=(0.15, 0.15),
        cbar_kws={'label': cbar_label, 'orientation': 'vertical'},
        figsize=(13, 13),
        rasterized=True
    )

    # Axis labels
    g.ax_heatmap.set_xlabel("Neurons", fontsize=12, labelpad=10)
    g.ax_heatmap.set_ylabel("Neurons", fontsize=12, labelpad=10)
    g.ax_heatmap.tick_params(axis='both', which='both', length=0, labelsize=0)

    # Title
    g.fig.suptitle(
        f"{mode_str} {pen_str} | Clusters: {n_clusters}",
        y=0.98, fontsize=16, fontweight='bold'
    )

    # --- Annotate cluster sizes on dendrogram ---
    _annotate_dendrogram_clusters(g, results, Z)

    # --- Legend: compact, right side ---
    handles = [mpatches.Patch(color=lut[t], label=t) for t in unique_types]

    g.fig.legend(
        handles=handles, title="Biological Type",
        loc="center right", bbox_to_anchor=(0.98, 0.8),
        borderaxespad=0., frameon=True,
        fontsize=8, title_fontsize=9,
        handlelength=0.8, handleheight=0.7,
        labelspacing=0.25, handletextpad=0.3,
        ncol=1 if len(unique_types) <= 15 else 2
    )

    plt.show()

    # --- Supplementary bar chart ---
    plot_cluster_sizes(results)


def plot_cluster_sizes(results):
    """Stacked bar chart showing neuron count per cluster, coloured by biological type."""
    print("Generating Cluster Size Chart...")

    ct = pd.crosstab(results['Subtype_Cluster'], results['Bio_Type'])
    ct = ct.sort_index()

    unique_types = sorted(results['Bio_Type'].unique())
    palette = dict(zip(unique_types, sns.color_palette("tab20", len(unique_types))))

    fig, ax = plt.subplots(figsize=(max(6, len(ct) * 0.45), 5))

    bottom = np.zeros(len(ct))
    x = np.arange(len(ct))

    for bio_type in unique_types:
        vals = ct[bio_type].values if bio_type in ct.columns else np.zeros(len(ct))
        ax.bar(x, vals, bottom=bottom, color=palette[bio_type],
               label=bio_type, edgecolor='white', linewidth=0.4)
        bottom += vals

    totals = ct.sum(axis=1).values
    for i, total in enumerate(totals):
        ax.text(i, total + max(totals) * 0.01, str(total),
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in ct.index], fontsize=8, rotation=45, ha='right')
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Neurons", fontsize=12)
    ax.set_title("Neurons per Cluster (by Biological Type)", fontsize=13, fontweight='bold')
    ax.legend(title="Bio Type", bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8, title_fontsize=9, frameon=True,
              handlelength=1.0, labelspacing=0.25)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(totals) * 1.12)

    fig.tight_layout()
    plt.show()


# ==========================================
# MAIN
# ==========================================
def main():
    # 1. Load
    raw_matrix, type_map = load_data(DIST_FILE, TYPE_FILE, FNT_FOLDER)

    # 2. Transform + optional penalty
    final_dist = process_matrix(raw_matrix, type_map,
                                use_spearman=USE_SPEARMAN,
                                use_penalty=USE_PENALTY,
                                penalty_strength=PENALTY_STRENGTH)

    # 3. Linkage
    Z = compute_linkage(final_dist, method='ward')

    # 4. Find best K
    suggested_k = calculate_c_index(final_dist, Z, max_k=65)

    # 5. Let user override
    try:
        val = input(f"Enter K (Default={suggested_k}): ").strip()
        k = int(val) if val else suggested_k
    except (ValueError, EOFError):
        k = suggested_k

    # 6. Assign clusters & save
    results = assign_clusters_and_save(final_dist, Z, type_map,
                                       k, use_penalty=USE_PENALTY)

    # 7. Plot heatmap + bar chart
    plot_heatmap(final_dist, Z, results, type_map,
                 use_spearman=USE_SPEARMAN, use_penalty=USE_PENALTY)


if __name__ == "__main__":
    main()
