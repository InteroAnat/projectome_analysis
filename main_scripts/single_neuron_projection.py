"""
Single Neuron Projection Strength Analysis

Functions for computing and visualizing projection strength 
for individual neurons or batches.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
INS_TABLE = 'neuron_tables/INS_df_v3.xlsx'


# ==========================================
# CORE FUNCTIONS
# ==========================================
def compute_single_neuron_projection(neuron_id, region_lengths_dict):
    """
    Compute projection strength for a single neuron.
    
    Parameters:
    -----------
    neuron_id : str
        Neuron identifier (e.g., '001.swc')
    region_lengths_dict : dict
        Dictionary of {region: axon_length_mm}
    
    Returns:
    --------
    dict : Projection strength data
        {
            'neuron_id': str,
            'n_regions': int,
            'total_length': float,
            'projection_strength': dict,  # {region: log(length+1)}
            'top_targets': list  # [(region, strength), ...]
        }
    """
    # Calculate projection strength (log of axon length)
    proj_strength = {}
    for region, length in region_lengths_dict.items():
        if length > 0.1:  # Filter very small values
            proj_strength[region] = np.log1p(length)
    
    # Sort by strength (descending)
    top_targets = sorted(proj_strength.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'neuron_id': neuron_id,
        'n_regions': len(proj_strength),
        'total_length': sum(region_lengths_dict.values()),
        'projection_strength': proj_strength,
        'top_targets': top_targets
    }


def plot_single_neuron_histogram(proj_data, output_path=None, top_n=15):
    """
    Plot histogram of projection strength for a single neuron.
    
    Parameters:
    -----------
    proj_data : dict
        Output from compute_single_neuron_projection()
    output_path : str, optional
        Path to save figure
    top_n : int
        Number of top targets to show
    
    Returns:
    --------
    fig : matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get top targets
    top_targets = proj_data['top_targets'][:top_n]
    regions = [t[0] for t in top_targets]
    strengths = [t[1] for t in top_targets]
    
    # Plot 1: Horizontal bar chart
    ax1 = axes[0]
    colors = sns.color_palette('mako', len(regions))
    bars = ax1.barh(range(len(regions)), strengths, color=colors)
    ax1.set_yticks(range(len(regions)))
    ax1.set_yticklabels(regions, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Projection Strength (log length)', fontsize=11)
    ax1.set_title(f"{proj_data['neuron_id']}\nTop {top_n} Projection Targets", 
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, strengths)):
        ax1.text(val + 0.1, i, f'{val:.1f}', va='center', fontsize=8)
    
    # Plot 2: Distribution histogram
    ax2 = axes[1]
    all_strengths = list(proj_data['projection_strength'].values())
    ax2.hist(all_strengths, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(all_strengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_strengths):.2f}')
    ax2.axvline(np.median(all_strengths), color='green', linestyle='--',
                label=f'Median: {np.median(all_strengths):.2f}')
    ax2.set_xlabel('Projection Strength (log length)', fontsize=11)
    ax2.set_ylabel('Number of Regions', fontsize=11)
    ax2.set_title(f"Distribution of Projection Strength\n" +
                  f"({proj_data['n_regions']} regions, " +
                  f"total: {proj_data['total_length']:.1f} mm)",
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# ==========================================
# BATCH PROCESSING
# ==========================================
def process_all_neurons(ins_table_path):
    """
    Process all neurons from INS table.
    
    Returns:
    --------
    list : List of projection data dicts for all neurons
    """
    df = pd.read_excel(ins_table_path)
    
    results = []
    for _, row in df.iterrows():
        neuron_id = row['NeuronID']
        neuron_type = row['Neuron_Type']
        
        try:
            region_lengths = ast.literal_eval(row['Region_projection_length'])
        except:
            region_lengths = {}
        
        proj_data = compute_single_neuron_projection(neuron_id, region_lengths)
        proj_data['neuron_type'] = neuron_type
        results.append(proj_data)
    
    return results


def plot_batch_summary(all_proj_data, output_path=None):
    """
    Plot summary statistics for batch of neurons.
    
    Parameters:
    -----------
    all_proj_data : list
        List of projection data dicts
    output_path : str, optional
        Path to save figure
    """
    # Extract stats
    n_regions_list = [p['n_regions'] for p in all_proj_data]
    total_lengths = [p['total_length'] for p in all_proj_data]
    types = [p.get('neuron_type', 'Unknown') for p in all_proj_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of target regions per neuron
    ax = axes[0, 0]
    ax.hist(n_regions_list, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(n_regions_list), color='red', linestyle='--',
               label=f'Mean: {np.mean(n_regions_list):.1f}')
    ax.set_xlabel('Number of Target Regions', fontsize=11)
    ax.set_ylabel('Number of Neurons', fontsize=11)
    ax.set_title('Distribution of Target Region Counts', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Total axon length per neuron
    ax = axes[0, 1]
    ax.hist(total_lengths, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(total_lengths), color='red', linestyle='--',
               label=f'Mean: {np.mean(total_lengths):.1f} mm')
    ax.set_xlabel('Total Axon Length (mm)', fontsize=11)
    ax.set_ylabel('Number of Neurons', fontsize=11)
    ax.set_title('Distribution of Total Axon Lengths', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Regions vs Length scatter
    ax = axes[1, 0]
    type_colors = {'ITi': 'blue', 'ITs': 'green', 'ITc': 'orange', 
                   'PT': 'red', 'CT': 'purple'}
    for t in set(types):
        mask = [ty == t for ty in types]
        x = np.array(n_regions_list)[mask]
        y = np.array(total_lengths)[mask]
        ax.scatter(x, y, c=type_colors.get(t, 'gray'), label=t, alpha=0.6, s=30)
    ax.set_xlabel('Number of Target Regions', fontsize=11)
    ax.set_ylabel('Total Axon Length (mm)', fontsize=11)
    ax.set_title('Regions vs Axon Length by Type', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Type composition
    ax = axes[1, 1]
    type_counts = pd.Series(types).value_counts()
    colors = [type_colors.get(t, 'gray') for t in type_counts.index]
    ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Neuron Type Composition', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Batch Summary: {len(all_proj_data)} Neurons', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# ==========================================
# MAIN - Test Functions
# ==========================================
def main():
    """Test the functions with sample neurons."""
    print("=" * 60)
    print("Single Neuron Projection Analysis - Test")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading INS data...")
    df = pd.read_excel(INS_TABLE)
    print(f"    Loaded {len(df)} neurons")
    
    # Test 1: Single neuron analysis
    print("\n[2] Testing single neuron analysis...")
    sample_neuron = df.iloc[0]
    neuron_id = sample_neuron['NeuronID']
    neuron_type = sample_neuron['Neuron_Type']
    
    try:
        region_lengths = ast.literal_eval(sample_neuron['Region_projection_length'])
    except:
        region_lengths = {}
    
    print(f"    Neuron: {neuron_id} (Type: {neuron_type})")
    print(f"    Regions: {len(region_lengths)}")
    
    # Compute projection strength
    proj_data = compute_single_neuron_projection(neuron_id, region_lengths)
    print(f"\n    Top 5 targets:")
    for region, strength in proj_data['top_targets'][:5]:
        print(f"      {region}: {strength:.2f}")
    
    # Plot single neuron
    fig = plot_single_neuron_histogram(proj_data, 
                                        'test_single_neuron.png',
                                        top_n=15)
    plt.close()
    
    # Test 2: Batch processing (first 10 neurons)
    print("\n[3] Testing batch processing (first 10 neurons)...")
    batch_data = []
    for _, row in df.head(10).iterrows():
        nid = row['NeuronID']
        try:
            lengths = ast.literal_eval(row['Region_projection_length'])
        except:
            lengths = {}
        pdata = compute_single_neuron_projection(nid, lengths)
        pdata['neuron_type'] = row['Neuron_Type']
        batch_data.append(pdata)
    
    # Plot batch summary
    fig = plot_batch_summary(batch_data, 'test_batch_summary.png')
    plt.close()
    
    # Test 3: Full batch (all neurons)
    print("\n[4] Processing all neurons...")
    all_data = process_all_neurons(INS_TABLE)
    fig = plot_batch_summary(all_data, 'full_batch_summary.png')
    plt.close()
    
    # Print summary stats
    print("\n" + "=" * 60)
    print("Summary Statistics (All Neurons):")
    print("=" * 60)
    n_regions = [p['n_regions'] for p in all_data]
    lengths = [p['total_length'] for p in all_data]
    print(f"  Target regions: {np.mean(n_regions):.1f} ± {np.std(n_regions):.1f}")
    print(f"  Total length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} mm")
    print(f"  Range: {min(lengths):.1f} - {max(lengths):.1f} mm")
    
    # Top targets across all neurons
    all_targets = {}
    for p in all_data:
        for region, strength in p['projection_strength'].items():
            all_targets[region] = all_targets.get(region, 0) + strength
    top_global = sorted(all_targets.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top 5 global targets:")
    for region, total_strength in top_global:
        print(f"    {region}: {total_strength:.1f}")
    
    print("\n" + "=" * 60)
    print("Test complete! Check output files:")
    print("  - test_single_neuron.png")
    print("  - test_batch_summary.png")
    print("  - full_batch_summary.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
