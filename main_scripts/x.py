"""
integrated_analysis_pipeline.py

Integrates:
1. Statistical Analysis (A-P Gradient, Laterality)
2. Interactive Flatmap Visualization (with Subtypes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import joblib
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# --- CONFIGURATION ---
INPUT_FILE = "Master_Analysis_Results_k10.xlsx" # From clustering script
FLATMAP_DIR = r".\flatmap_outputs"
NMT_DIR = r"D:\projectome_analysis\atlas\NMT_v2.1_sym\NMT_v2.1_sym"

# --- COLORS ---
TYPE_COLORS = {
    'PT': '#d62728', 'CT': '#2ca02c', 
    'ITs': '#9467bd', 'ITc': '#e377c2', 'ITi': '#17becf',
    'Unclassified': 'gray'
}

# Subtypes get gradients of these base colors
SUBTYPE_BASE_COLORS = {
    'PT': 'Reds', 'CT': 'Greens', 'ITs': 'RdPu', 
    'ITc': 'Purples', 'ITi': 'Blues', 'Unclassified': 'Greys'
}

# --- SUBREGION DEFINITIONS ---
SUBREGION_MAP = {
    'CL_Iam/Iapm': 'agranular', 'CL_lat_Ia': 'agranular', 'CL_Iai': 'agranular', 
    'CL_Ial': 'agranular', 'CL_Iapl': 'agranular', 'CL_Ia/Id': 'agranular',
    'CL_Ins/Pi': 'dysgranular', 'CL_Pi': 'dysgranular', 'CL_Ins': 'dysgranular',
    'CL_Ig': 'granular', 'CL_Ri': 'retroinsula',
    # Right omitted for brevity (same logic)
}
SUBREGION_COLORS = {'agranular': '#FFCCCC', 'dysgranular': '#CCFFCC', 'granular': '#CCCCFF', 'retroinsula': '#E0CCFF'}

# ==========================================
# 1. LOAD DATA
# ==========================================
def load_and_prep():
    print(f"Loading {INPUT_FILE}...")
    if INPUT_FILE.endswith('.xlsx'):
        df = pd.read_excel(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE)
        
    # Ensure coordinates exist
    if 'Soma_NII_X' in df.columns:
        df = df.rename(columns={'Soma_NII_X': 'soma_x', 'Soma_NII_Y': 'soma_y', 'Soma_NII_Z': 'soma_z'})
    
    # Filter valid types
    df = df[df['Neuron_Type'].isin(TYPE_COLORS.keys())]
    
    # Assign Hemisphere
    # NMT Midline approx 128
    df['Hemisphere'] = np.where(df['soma_x'] < 128, 'Left', 'Right')
    
    return df

# ==========================================
# 2. STATISTICAL ANALYSIS
# ==========================================
def run_statistics(df):
    print("\n" + "="*40)
    print("📊 STATISTICAL ANALYSIS")
    print("="*40)

    # A. A-P Gradient (Y-axis)
    print("\n[1] A-P Distribution Differences")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='soma_y', y='Neuron_Type', data=df, palette=TYPE_COLORS, order=TYPE_COLORS.keys())
    plt.title("A-P Distribution by Type")
    plt.xlabel("Posterior <---> Anterior (Voxel Index)")
    plt.show()
    
    groups = [df[df['Neuron_Type'] == t]['soma_y'].values for t in TYPE_COLORS.keys()]
    h, p = stats.kruskal(*groups)
    print(f"   Kruskal-Wallis: p={p:.4e}")

    # B. Laterality
    print("\n[2] Hemispheric Asymmetry")
    contingency = pd.crosstab(df['Neuron_Type'], df['Hemisphere'])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    print(contingency)
    print(f"   Chi-Square: p={p:.4e}")
    
    # Plot
    contingency.plot(kind='bar', stacked=False, color=['#3498DB', '#E74C3C'], figsize=(8, 5))
    plt.title("Laterality by Type")
    plt.show()

# ==========================================
# 3. INTERACTIVE FLATMAP VISUALIZATION
# ==========================================
class FlatmapVisualizer:
    def __init__(self, df):
        self.df = df
        sys.path.append(os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis'))
        from NMTFlatmap import InsulaFlatmapNMT
        self.gen = InsulaFlatmapNMT(NMT_DIR)
        
    def load_meshes(self):
        self.flatmaps = {}
        for hemi in ['left', 'right']:
            path = os.path.join(FLATMAP_DIR, f'insula_flatmap_{hemi}.pkl')
            if os.path.exists(path):
                self.flatmaps[hemi] = joblib.load(path)
            else:
                print(f"Warning: {path} not found.")

    def get_subtype_colors(self):
        palette = {}
        grouped = self.df.groupby('Neuron_Type')['Subtype_Cluster'].unique()
        for btype, subtypes in grouped.items():
            cmap = sns.color_palette(SUBTYPE_BASE_COLORS.get(btype, 'Greys'), n_colors=len(subtypes)+2)[2:]
            for i, sid in enumerate(sorted(subtypes)):
                palette[sid] = cmap[i % len(cmap)]
        return palette

    def plot_interactive(self):
        print("\nLaunching Interactive Visualization...")
        self.load_meshes()
        subtype_colors = self.get_subtype_colors()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        
        for i, hemi in enumerate(['left', 'right']):
            if hemi not in self.flatmaps: continue
            
            ax = axes[i]
            params = self.flatmaps[hemi]
            mesh = params['mesh']
            coords_2d = params['coords_2d']
            
            # Filter Data for Hemisphere
            subset = self.df[self.df['Hemisphere'] == hemi.capitalize()]
            
            # 1. Background (Hull)
            pts = np.array(list(coords_2d.values()))
            try:
                hull = ConvexHull(pts)
                ax.fill(pts[hull.vertices,0], pts[hull.vertices,1], alpha=0.1, color='gray')
            except: pass

            # 2. Plot Neurons (Colored by Subtype)
            mapped = 0
            for _, row in subset.iterrows():
                vox = np.array([row['soma_x'], row['soma_y'], row['soma_z']])
                flat = self.gen.map_point_to_flatmap(vox, mesh, coords_2d)
                
                if flat:
                    sid = row['Subtype_Cluster']
                    c = subtype_colors.get(sid, 'gray')
                    ax.scatter(flat[0], flat[1], c=[c], s=40, edgecolors='k', linewidth=0.3, alpha=0.9)
                    mapped += 1
            
            ax.set_title(f"{hemi.capitalize()} ({mapped} neurons)", fontsize=14)
            ax.set_aspect('equal')
            # ax.invert_yaxis() # Uncomment if Anterior is at bottom

        # Legend (Broad Types for simplicity)
        patches = [mpatches.Patch(color=sns.color_palette(SUBTYPE_BASE_COLORS[t])[4], label=t) for t in TYPE_COLORS]
        fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        df = load_and_prep()
        
        # 1. Stats
        run_statistics(df)
        
        # 2. Visuals
        viz = FlatmapVisualizer(df)
        viz.plot_interactive()
    else:
        print(f"Error: {INPUT_FILE} not found. Run paper_clustering_method.py first.")