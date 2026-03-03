"""
Plot Projection Strength for INS Neurons

Reads INS_df_v3.xlsx, calculates projection strength as log(axon length),
and generates plots without clustering.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import ast

# ==========================================
# CONFIGURATION
# ==========================================
INS_TABLE = r'neuron_tables/INS_df_v3.xlsx'
OUTPUT_PREFIX = 'INS_projection_strength'

# ==========================================
# 1. LOAD AND PARSE DATA
# ==========================================
def load_ins_data(filepath):
    """Load INS table and parse projection length data."""
    print("--- Loading INS Data ---")
    
    df = pd.read_excel(filepath)
    print(f"    Loaded {len(df)} neurons")
    print(f"    Neuron types: {df['Neuron_Type'].value_counts().to_dict()}")
    
    # Parse Region_projection_length (it's stored as string representation of dict)
    projection_data = []
    
    for idx, row in df.iterrows():
        neuron_id = row['NeuronID']
        neuron_type = row['Neuron_Type']
        
        # Parse the projection length dictionary
        try:
            region_lengths = ast.literal_eval(row['Region_projection_length'])
            if not isinstance(region_lengths, dict):
                region_lengths = {}
        except:
            region_lengths = {}
        
        projection_data.append({
            'NeuronID': neuron_id,
            'Neuron_Type': neuron_type,
            'Region_Lengths': region_lengths,
            'Total_Length': row['Total_Length']
        })
    
    return projection_data


# ==========================================
# 2. CALCULATE PROJECTION STRENGTH
# ==========================================
def calculate_projection_strength(projection_data, min_length=0.1):
    """
    Calculate projection strength as log(axon length) per region.
    Returns a matrix: neurons x regions
    """
    print("\n--- Calculating Projection Strength ---")
    
    # Get all unique regions across all neurons
    all_regions = set()
    for pdata in projection_data:
        all_regions.update(pdata['Region_Lengths'].keys())
    
    all_regions = sorted(list(all_regions))
    print(f"    Found {len(all_regions)} unique target regions")
    
    # Build projection strength matrix
    neuron_ids = [pdata['NeuronID'] for pdata in projection_data]
    neuron_types = [pdata['Neuron_Type'] for pdata in projection_data]
    
    strength_matrix = np.zeros((len(neuron_ids), len(all_regions)))
    
    for i, pdata in enumerate(projection_data):
        for j, region in enumerate(all_regions):
            length = pdata['Region_Lengths'].get(region, 0)
            # Projection strength = ln(length + 1)
            if length > min_length:
                strength_matrix[i, j] = np.log1p(length)
    
    # Create DataFrame
    strength_df = pd.DataFrame(
        strength_matrix,
        index=neuron_ids,
        columns=all_regions
    )
    
    # Filter to regions with at least some projections
    non_zero_counts = (strength_df > 0).sum()
    major_regions = non_zero_counts[non_zero_counts >= 3].index.tolist()
    strength_df = strength_df[major_regions]
    
    print(f"    Selected {len(major_regions)} major regions (>=3 neurons)")
    print(f"    Projection strength range: [{strength_matrix.min():.2f}, {strength_matrix.max():.2f}]")
    
    return strength_df, neuron_ids, neuron_types


# ==========================================
# 3. PLOT 1: Heatmap of Projection Strength
# ==========================================
def plot_strength_heatmap(strength_df, neuron_types, output_prefix):
    """Create heatmap of projection strength."""
    print("\n--- Generating Heatmap ---")
    
    # Create color mapping for neuron types
    unique_types = list(set(neuron_types))
    type_colors = dict(zip(unique_types, sns.color_palette("tab10", len(unique_types))))
    row_colors = pd.Series(neuron_types, index=strength_df.index).map(type_colors)
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), 
                                    gridspec_kw={'width_ratios': [0.02, 1]})
    
    # Type color bar - FIX: don't flatten RGB tuples
    type_array = np.array([type_colors[t] for t in neuron_types])  # Shape: (n_neurons, 3)
    type_array = type_array.reshape(len(neuron_types), 1, 3)  # Shape: (n_neurons, 1, 3)
    ax1.imshow(type_array, aspect='auto', interpolation='nearest')
    ax1.set_xticks([])
    ax1.set_ylabel('Neurons (by type)')
    ax1.set_title('Type')
    
    # Heatmap
    sns.heatmap(strength_df, ax=ax2, cmap='mako', 
                xticklabels=True, yticklabels=False,
                cbar_kws={'label': 'Projection Strength (log length)'})
    ax2.set_xlabel('Target Regions', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_title('INS Neuron Projection Strength', fontsize=14, fontweight='bold')
    
    # Rotate x labels
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Legend for types
    handles = [plt.Rectangle((0,0),1,1, color=type_colors[t]) for t in unique_types]
    ax2.legend(handles, unique_types, title='Neuron Type', 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_prefix}_heatmap.png")
    plt.close()


# ==========================================
# 4. PLOT 2: Mean Projection by Neuron Type
# ==========================================
def plot_mean_by_type(strength_df, neuron_types, output_prefix):
    """Plot mean projection strength for each neuron type."""
    print("\n--- Generating Mean Projection by Type ---")
    
    # Create DataFrame with type info
    plot_df = strength_df.copy()
    plot_df['Neuron_Type'] = neuron_types
    
    # Calculate mean by type
    mean_by_type = plot_df.groupby('Neuron_Type')[plot_df.columns[:-1]].mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    mean_by_type.T.plot(kind='bar', ax=ax, colormap='tab10', width=0.8)
    ax.set_xlabel('Target Regions', fontsize=12)
    ax.set_ylabel('Mean Projection Strength', fontsize=12)
    ax.set_title('Mean Projection Strength by Neuron Type', fontsize=14, fontweight='bold')
    ax.legend(title='Neuron Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_mean_by_type.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_prefix}_mean_by_type.png")
    plt.close()


# ==========================================
# 5. PLOT 3: Top Projections Summary
# ==========================================
def plot_top_projections(strength_df, neuron_types, output_prefix, top_n=15):
    """Plot summary of top projection targets."""
    print("\n--- Generating Top Projections Summary ---")
    
    # Calculate mean projection strength per region
    mean_strength = strength_df.mean().sort_values(ascending=False)
    
    # Count neurons projecting to each region
    projection_counts = (strength_df > 0).sum().sort_values(ascending=False)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top mean projection strength
    top_mean = mean_strength.head(top_n)
    colors = sns.color_palette('mako', len(top_mean))
    axes[0].barh(range(len(top_mean)), top_mean.values, color=colors)
    axes[0].set_yticks(range(len(top_mean)))
    axes[0].set_yticklabels(top_mean.index, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Mean Projection Strength', fontsize=11)
    axes[0].set_title(f'Top {top_n} Projection Targets (by mean strength)', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Top projection counts
    top_counts = projection_counts.head(top_n)
    colors2 = sns.color_palette('viridis', len(top_counts))
    axes[1].barh(range(len(top_counts)), top_counts.values, color=colors2)
    axes[1].set_yticks(range(len(top_counts)))
    axes[1].set_yticklabels(top_counts.index, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Number of Neurons with Projections', fontsize=11)
    axes[1].set_title(f'Top {top_n} Projection Targets (by neuron count)', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_targets.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_prefix}_top_targets.png")
    plt.close()


# ==========================================
# MAIN
# ==========================================
# def main():
print("=" * 70)
print("INS Projection Strength Analysis (212 neurons)")
print("=" * 70)

# 1. Load data
projection_data = load_ins_data(INS_TABLE)

# 2. Calculate projection strength
strength_df, neuron_ids, neuron_types = calculate_projection_strength(projection_data)

# 3. Generate plots
plot_strength_heatmap(strength_df, neuron_types, OUTPUT_PREFIX)
plot_mean_by_type(strength_df, neuron_types, OUTPUT_PREFIX)
plot_top_projections(strength_df, neuron_types, OUTPUT_PREFIX)

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)


# if __name__ == "__main__":
#     main()
