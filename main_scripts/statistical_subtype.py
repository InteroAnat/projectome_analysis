"""
compare_distributions.py - Spatial Distribution Analysis

Features:
1. Compares Anterior-Posterior (Y-axis) distribution of neuron types.
2. Compares Left-Right (X-axis) laterality.
3. Generates statistical summaries and plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIG
OUTPUT_DIR = r".\flatmap_outputs"
SOMA_CSV = os.path.join(OUTPUT_DIR, 'insula_somas_bilateral.csv')

# Colors
TYPE_COLORS = {
    'PT': '#d62728', 'CT': '#2ca02c', 
    'ITs': '#9467bd', 'ITc': '#e377c2', 'ITi': '#17becf'
}

def load_data():
    df = pd.read_csv(SOMA_CSV)
    # Filter out unclassified if needed
    df = df[df['Neuron_Type'].isin(TYPE_COLORS.keys())]
    return df

# ==========================================
# 1. ANTERIOR-POSTERIOR (A-P) ANALYSIS
# ==========================================
def analyze_AP_gradient(df):
    print("\n--- A-P Gradient Analysis ---")
    
    # NMT Convention: High Y = Anterior, Low Y = Posterior
    # We normalize Y to 0-1 range for easier reading
    y_min, y_max = df['soma_y'].min(), df['soma_y'].max()
    df['AP_Position'] = (df['soma_y'] - y_min) / (y_max - y_min)
    
    # 1. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='AP_Position', y='Neuron_Type', data=df, palette=TYPE_COLORS, order=TYPE_COLORS.keys())
    plt.title("Anterior-Posterior Distribution by Type")
    plt.xlabel("Posterior <------- Normalized Position -------> Anterior")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Density Plot (KDE)
    plt.figure(figsize=(10, 6))
    for n_type in TYPE_COLORS.keys():
        subset = df[df['Neuron_Type'] == n_type]
        if len(subset) > 1:
            sns.kdeplot(subset['AP_Position'], label=n_type, color=TYPE_COLORS[n_type], linewidth=2)
            
    plt.title("Neuron Density along A-P Axis")
    plt.xlabel("Posterior <-------------------------> Anterior")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# ==========================================
# 2. BILATERAL (L-R) ANALYSIS
# ==========================================
def analyze_laterality(df):
    print("\n--- Bilateral Analysis ---")
    
    # Identify Hemisphere based on X coordinate
    # Assuming volume width ~256 voxels. Midline ~128.
    midline = (df['soma_x'].max() + df['soma_x'].min()) / 2
    df['Hemisphere'] = np.where(df['soma_x'] < midline, 'Left', 'Right')
    
    # 1. Count Table
    counts = df.groupby(['Neuron_Type', 'Hemisphere']).size().unstack(fill_value=0)
    counts['Total'] = counts['Left'] + counts['Right']
    
    # Calculate Laterality Index (LI)
    # LI = (L - R) / (L + R)
    # +1 = Pure Left, -1 = Pure Right, 0 = Symmetric
    counts['Laterality_Index'] = (counts['Left'] - counts['Right']) / counts['Total']
    
    print(counts)
    
    # 2. Bar Plot
    counts[['Left', 'Right']].plot(kind='bar', stacked=False, color=['#3498DB', '#E74C3C'], figsize=(10, 6))
    plt.title("Neuron Counts by Hemisphere")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. SUBREGION ENRICHMENT
# ==========================================
def analyze_subregions(df):
    print("\n--- Subregion Enrichment ---")
    
    # Check if 'subregion' column exists (from your script)
    if 'subregion' not in df.columns:
        print("Subregion column not found in CSV. Generating based on AP...")
        # Simple heuristic if column missing:
        # Top 33% Y = Agranular, Mid = Dysgranular, Low = Granular
        df['subregion'] = pd.qcut(df['soma_y'], 3, labels=['Granular', 'Dysgranular', 'Agranular'])
    
    # Heatmap of Type vs Subregion
    ct = pd.crosstab(df['Neuron_Type'], df['subregion'])
    
    # Normalize by Row (Type) to see preference
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': 'Proportion'})
    plt.title("Neuron Type Preference for Insula Subregions")
    plt.ylabel("Neuron Type")
    plt.xlabel("Insula Zone")
    plt.show()

if __name__ == "__main__":
    if os.path.exists(SOMA_CSV):
        df = load_data()
        
        analyze_AP_gradient(df)
        analyze_laterality(df)
        analyze_subregions(df)
    else:
        print("Soma CSV not found.")