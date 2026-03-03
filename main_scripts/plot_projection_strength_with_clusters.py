"""
Projection Strength Plot with Cluster Information
Combines: INS neuron data + Cluster assignments from fnt_dist clustering
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# ==========================================
# CONFIGURATION
# ==========================================
INS_TABLE = '/mnt/d/projectome_analysis/main_scripts/neuron_tables/INS_df_v3.xlsx'
CLUSTER_FILE = '/mnt/d/projectome_analysis/main_scripts/fnt_dist_Results_Penalty_On_k9.xlsx'
OUTPUT_PREFIX = 'INS_projection_strength_clustered'

# ==========================================
# DATA LOADING
# ==========================================
def load_ins_data(filepath):
    """Load INS table and parse projection lengths."""
    df = pd.read_excel(filepath)
    
    projections = {}
    for _, row in df.iterrows():
        neuron_id = row['NeuronID']
        try:
            lengths = ast.literal_eval(row['Region_projection_length'])
        except:
            lengths = {}
        projections[neuron_id] = {
            'type': row['Neuron_Type'],
            'lengths': lengths
        }
    return projections


def load_cluster_data(filepath):
    """Load cluster assignments."""
    df = pd.read_excel(filepath)
    return dict(zip(df['NeuronID'], df['Subtype_Cluster']))


# ==========================================
# PROJECTION STRENGTH CALCULATION
# ==========================================
def calc_projection_strength(projections, min_neurons=5):
    """Calculate log(axon length) matrix."""
    # Get all regions
    all_regions = set()
    for p in projections.values():
        all_regions.update(p['lengths'].keys())
    regions = sorted(all_regions)
    
    # Build matrix
    neuron_ids = list(projections.keys())
    matrix = np.zeros((len(neuron_ids), len(regions)))
    
    for i, nid in enumerate(neuron_ids):
        for j, region in enumerate(regions):
            length = projections[nid]['lengths'].get(region, 0)
            if length > 0.1:
                matrix[i, j] = np.log1p(length)
    
    # Filter to major regions
    df = pd.DataFrame(matrix, index=neuron_ids, columns=regions)
    major = (df > 0).sum() >= min_neurons
    return df.loc[:, major]


# ==========================================
# CLUSTERING & ORDERING
# ==========================================
def order_by_cluster(strength_df, cluster_map, neuron_types):
    """Reorder neurons by cluster, then by type within cluster."""
    # Create ordering DataFrame
    order_df = pd.DataFrame({
        'neuron_id': strength_df.index,
        'cluster': [cluster_map.get(nid, 0) for nid in strength_df.index],
        'type': neuron_types
    })
    
    # Sort by cluster, then by type
    type_order = {'ITi': 0, 'ITs': 1, 'ITc': 2, 'PT': 3, 'CT': 4}
    order_df['type_code'] = order_df['type'].map(type_order)
    order_df = order_df.sort_values(['cluster', 'type_code'])
    
    # Reorder strength matrix
    return strength_df.loc[order_df['neuron_id']], order_df['cluster'].values


# ==========================================
# VISUALIZATION
# ==========================================
def plot_clustered_heatmap(strength_df, clusters, neuron_types, output_prefix):
    """Plot heatmap with cluster annotations."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.02, 0.02, 1], wspace=0.05)
    
    ax_type = fig.add_subplot(gs[0, 0])
    ax_cluster = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[0, 2])
    
    # Type colors
    unique_types = list(set(neuron_types))
    type_colors = dict(zip(unique_types, sns.color_palette("tab10", len(unique_types))))
    type_array = np.array([[type_colors[t]] for t in neuron_types])
    ax_type.imshow(type_array, aspect='auto')
    ax_type.set_xticks([])
    ax_type.set_yticks([])
    ax_type.set_title('Type', fontsize=10)
    
    # Cluster colors
    unique_clusters = sorted(set(clusters))
    cluster_palette = sns.color_palette("husl", len(unique_clusters))
    cluster_colors = dict(zip(unique_clusters, cluster_palette))
    cluster_array = np.array([[cluster_colors[c]] for c in clusters])
    ax_cluster.imshow(cluster_array, aspect='auto')
    ax_cluster.set_xticks([])
    ax_cluster.set_yticks([])
    ax_cluster.set_title('Cluster', fontsize=10)
    
    # Heatmap
    sns.heatmap(strength_df, ax=ax_heatmap, cmap='mako',
                xticklabels=True, yticklabels=False,
                cbar_kws={'label': 'Projection Strength (log length)'})
    ax_heatmap.set_xlabel('Target Regions', fontsize=12)
    ax_heatmap.set_title('INS Neuron Projection Strength by Cluster (k=9)', 
                        fontsize=14, fontweight='bold')
    plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Legends
    type_patches = [plt.Rectangle((0,0),1,1, color=type_colors[t]) for t in unique_types]
    cluster_patches = [plt.Rectangle((0,0),1,1, color=cluster_colors[c]) for c in unique_clusters]
    
    fig.legend(type_patches, unique_types, title='Neuron Type',
               bbox_to_anchor=(1.02, 0.8), loc='upper left')
    fig.legend(cluster_patches, [f'C{c}' for c in unique_clusters], title='Cluster',
               bbox_to_anchor=(1.02, 0.4), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_heatmap.png")


def plot_cluster_profiles(strength_df, clusters, output_prefix):
    """Plot mean projection profile for each cluster."""
    n_clusters = len(set(clusters))
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for cid in range(1, n_clusters + 1):
        ax = axes[cid - 1]
        mask = clusters == cid
        profile = strength_df[mask].mean().sort_values(ascending=False).head(15)
        
        colors = sns.color_palette('mako', len(profile))
        bars = ax.barh(range(len(profile)), profile.values, color=colors)
        ax.set_yticks(range(len(profile)))
        ax.set_yticklabels(profile.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Projection Strength', fontsize=9)
        ax.set_title(f'Cluster {cid} (n={mask.sum()})', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Top 15 Projection Targets per Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_profiles.png")


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("Projection Strength with Cluster Information")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    projections = load_ins_data(INS_TABLE)
    cluster_map = load_cluster_data(CLUSTER_FILE)
    print(f"      {len(projections)} neurons, {len(cluster_map)} with clusters")
    
    # Calculate projection strength
    print("\n[2/4] Calculating projection strength...")
    strength_df = calc_projection_strength(projections)
    print(f"      Matrix: {strength_df.shape}")
    
    # Get neuron types and order by cluster
    print("\n[3/4] Ordering by cluster...")
    neuron_types = [projections[nid]['type'] for nid in strength_df.index]
    strength_ordered, clusters = order_by_cluster(strength_df, cluster_map, neuron_types)
    print(f"      Clusters: {sorted(set(clusters))}")
    
    # Plot
    print("\n[4/4] Generating plots...")
    plot_clustered_heatmap(strength_ordered, clusters, neuron_types, OUTPUT_PREFIX)
    plot_cluster_profiles(strength_ordered, clusters, OUTPUT_PREFIX)
    
    # Summary
    print("\n" + "=" * 60)
    print("Cluster Summary:")
    for cid in sorted(set(clusters)):
        count = (clusters == cid).sum()
        print(f"  Cluster {cid}: {count} neurons")
    print("=" * 60)


if __name__ == "__main__":
    main()
