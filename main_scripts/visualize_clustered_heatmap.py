"""
Visualize clustered projection strength heatmap similar to Gou et al. 2025 Figure 2A
Uses the existing cluster order from the table
Neurons as columns (x-axis), Regions as rows (y-axis)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load data
df = pd.read_excel('neuron_tables/251637_results_clustered_k9_spearman_penalty.xlsx')

# Get projection columns (exclude metadata)
proj_cols = [c for c in df.columns if c not in ['NeuronID', 'Neuron_Type', 'Morph_Cluster', '_Unmapped']]

# Create projection matrix
proj_matrix = df[proj_cols].values

# Log transform (add small constant to avoid log(0))
proj_matrix_log = np.log10(proj_matrix + 0.001)

# Use the existing order from the table (already clustered)
df_sorted = df.sort_values(['Morph_Cluster', 'Neuron_Type'])
ordered_indices = df_sorted.index.values

# Transpose so regions are rows (y-axis) and neurons are columns (x-axis)
ordered_matrix_T = proj_matrix_log[ordered_indices, :].T

# Define colors for neuron types (matching Gou et al. style)
type_colors = {
    'ITi': '#2E7D32',  # Green
    'ITs': '#7B1FA2',  # Purple  
    'ITc': '#F57C00',  # Orange
    'CT': '#C62828',   # Red
    'PT': '#1565C0',   # Blue
}

# Get cluster and type labels
cluster_labels = df_sorted['Morph_Cluster'].values
type_labels = df_sorted['Neuron_Type'].values
unique_clusters = sorted(df['Morph_Cluster'].unique())
unique_types = ['ITi', 'ITs', 'ITc', 'CT', 'PT']

# Create figure
fig = plt.figure(figsize=(24, 14))

# Create gridspec
gs = fig.add_gridspec(3, 3, 
                      width_ratios=[14, 0.3, 2], 
                      height_ratios=[1.2, 0.5, 10],
                      wspace=0.02, hspace=0.02,
                      left=0.06, right=0.94, top=0.90, bottom=0.06)

# 1. Morph Cluster color bar (top)
ax_cluster = fig.add_subplot(gs[0, 0])
cluster_cmap = plt.cm.tab20
cluster_numeric = cluster_labels - 1

ax_cluster.imshow(cluster_numeric.reshape(1, -1), aspect='auto', cmap=cluster_cmap, 
                  vmin=-0.5, vmax=19.5)
ax_cluster.set_xlim(-0.5, len(cluster_labels) - 0.5)
ax_cluster.set_ylim(-0.5, 0.5)
ax_cluster.axis('off')

# Add cluster labels
prev_cluster = cluster_labels[0]
cluster_start = 0
for i in range(1, len(cluster_labels) + 1):
    if i == len(cluster_labels) or cluster_labels[i] != prev_cluster:
        end_pos = i - 0.5 if i < len(cluster_labels) else len(cluster_labels) - 0.5
        mid_pos = (cluster_start + end_pos) / 2
        ax_cluster.text(mid_pos, -0.6, f'C{prev_cluster}', 
                       ha='center', va='top', fontsize=11, fontweight='bold',
                       color=cluster_cmap((prev_cluster-1)/19))
        if i < len(cluster_labels):
            ax_cluster.axvline(x=i-0.5, color='white', linewidth=2)
        cluster_start = i - 0.5
        if i < len(cluster_labels):
            prev_cluster = cluster_labels[i]

# 2. Neuron Type color bar
ax_type = fig.add_subplot(gs[1, 0])
type_cmap = plt.cm.colors.ListedColormap([type_colors[t] for t in unique_types])
type_numeric = np.array([unique_types.index(t) for t in type_labels])

ax_type.imshow(type_numeric.reshape(1, -1), aspect='auto', cmap=type_cmap, 
               vmin=-0.5, vmax=len(unique_types)-0.5)
ax_type.set_xlim(-0.5, len(type_labels) - 0.5)
ax_type.set_ylim(-0.5, 0.5)
ax_type.axis('off')

# Add type labels
for t in unique_types:
    mask = type_labels == t
    if np.any(mask):
        indices = np.where(mask)[0]
        mid_pos = (indices[0] + indices[-1]) / 2
        ax_type.text(mid_pos, -0.6, t, ha='center', va='top', 
                    fontsize=10, fontweight='bold', color=type_colors[t])

for i in range(1, len(cluster_labels)):
    if cluster_labels[i] != cluster_labels[i-1]:
        ax_type.axvline(x=i-0.5, color='white', linewidth=2)

# 3. Main heatmap
ax_heatmap = fig.add_subplot(gs[2, 0])
im = ax_heatmap.imshow(ordered_matrix_T, aspect='auto', cmap='YlGnBu', 
                       vmin=-3, vmax=2, interpolation='nearest')

ax_heatmap.set_yticks(range(len(proj_cols)))
ax_heatmap.set_yticklabels(proj_cols, fontsize=8)
ax_heatmap.tick_params(axis='y', length=0)
ax_heatmap.set_xticks([])

for i in range(1, len(cluster_labels)):
    if cluster_labels[i] != cluster_labels[i-1]:
        ax_heatmap.axvline(x=i-0.5, color='gray', linewidth=0.8, alpha=0.6)

# 4. Colorbar
ax_cbar = fig.add_subplot(gs[2, 2])
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.set_label('Log10(Projection Strength)', fontsize=11, fontweight='bold')

# 5. Legends - separate area on the right
legend_elements_type = [mpatches.Patch(color=type_colors[t], label=t) for t in unique_types]
fig.legend(handles=legend_elements_type, loc='upper right', bbox_to_anchor=(0.92, 0.88), 
           title='Neuron Type', ncol=1, fontsize=10, title_fontsize=11)

cluster_legend_elements = [mpatches.Patch(color=cluster_cmap((c-1)/19), label=f'C{c}') for c in unique_clusters]
fig.legend(handles=cluster_legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.88), 
           title='Morph Cluster', ncol=3, fontsize=9, title_fontsize=11)

# Title
fig.suptitle('Single-Neuron Projection Strength Heatmap (Gou et al. 2025 Fig 2A Style)', 
             fontsize=14, fontweight='bold', y=0.96)
fig.text(0.5, 0.93, f'{len(df)} neurons × {len(proj_cols)} regions | 9 Clusters | 5 Neuron Types',
         ha='center', fontsize=10, style='italic')

plt.savefig('output/fig2a_style_clustered_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('output/fig2a_style_clustered_heatmap.pdf', bbox_inches='tight', facecolor='white')
print("Saved to output/fig2a_style_clustered_heatmap.png")
plt.close()

# Print cluster statistics
print("\nCluster statistics:")
for cluster in unique_clusters:
    cluster_df = df[df['Morph_Cluster'] == cluster]
    type_counts = cluster_df['Neuron_Type'].value_counts()
    print(f"  Cluster {cluster}: {len(cluster_df)} neurons - {dict(type_counts)}")
