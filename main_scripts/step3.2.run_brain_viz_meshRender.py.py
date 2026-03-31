#!/usr/bin/env python3
"""
Brain Visualization Pipeline
============================
IDE-friendly pipeline for brain mesh generation and neuron visualization.

INSTRUCTIONS:
    1. Edit the PARAMETERS section below
    2. Run the script in your IDE
    3. PNG will be saved to OUTPUT_DIR

EXAMPLE REGIONS:
    - Insula: CL_Ig, CL_Ial, CR_Ig, CR_Ial, etc.
    - Striatum: Cpu (caudate putamen)
    - Thalamus: various thalamic nuclei
    - Cortex: specific cortical areas
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain_viz import BrainViz, RegionExtractor

# =============================================================================
# PARAMETERS - EDIT THESE FOR YOUR SPECIFIC RUN
# =============================================================================

# ---------------------
# Paths
# ---------------------
NMT_DIR = r"D:\projectome_analysis\atlas\NMT_v2.1_sym\NMT_v2.1_sym"
ARM_ATLAS = r"D:\projectome_analysis\atlas\ARM_in_NMT_v2.1_sym.nii.gz"
ARM_KEY = r"D:\projectome_analysis\atlas\ARM_key_all.txt"
NEURON_TABLE = r"D:\projectome_analysis\main_scripts\neuron_tables\INS_df_v3.xlsx"

# ---------------------
# Output Settings
# ---------------------
OUTPUT_DIR = r".\brain_viz_output"
OUTPUT_NAME = "brain_viz.png"
DPI = 300

# ---------------------
# Region Definitions
# ---------------------
# Define regions to visualize - each key is a display name, value is list of ARM abbreviations
# Find abbreviations in ARM_key_all.txt
REGIONS = {
    # Example: Insula regions
    "agranular_l": ["CL_Ial", "CL_Iai", "CL_lat_Ia", "CL_Iapl", "CL_Iam/Iapm"],
    "granular_l": ["CL_Ig"],
    "dysgranular_l": ["CL_Ia/Id"],
    "agranular_r": ["CR_Ial", "CR_Iai", "CR_lat_Ia", "CR_Iapl", "CR_Iam/Iapm"],
    "granular_r": ["CR_Ig"],
    "dysgranular_r": ["CR_Ia/Id"],
    
    # Add more regions as needed:
    # "striatum": ["Cpu"],
    # "thalamus": ["Thal"],
}

# ---------------------
# Colors
# ---------------------
# Define colors for each region (must match keys in REGIONS)
REGION_COLORS = {
    "agranular_l": "#E74C3C",    # Red
    "granular_l": "#3498DB",     # Blue
    "dysgranular_l": "#9B59B6",  # Purple
    "agranular_r": "#E67E22",    # Orange
    "granular_r": "#2980B9",     # Dark blue
    "dysgranular_r": "#8E44AD",  # Dark purple
}

# ---------------------
# Visualization Settings
# ---------------------
# Brain mesh
BRAIN_ALPHA = 0.1          # Brain mesh transparency

# Region meshes
REGION_ALPHA = 0.1         # Region mesh transparency
REGION_POINT_SIZE = 3      # Region mesh point size

# Soma (neuron) settings
SOMA_SIZE = 20             # Soma marker size (default: 50)
SOMA_ALPHA = 0.9           # Soma transparency
SOMA_EDGEWIDTH = 0.5       # Soma edge width

# Custom soma colors by neuron type (optional)
SOMA_COLORS = {
    'PT': '#d62728',
    'CT': '#2ca02c',
    'ITs': '#9467bd',
    'ITc': '#e377c2',
    'ITi': '#17becf',
    'Unclassified': 'gray'
}

# View settings
VIEW_ELEV = 30             # Camera elevation
VIEW_AZIM = 60             # Camera azimuth
LEGEND_MARKERSCALE = 2.5   # Enlarged legend markers

# Plot title (set to None for auto-generated)
PLOT_TITLE = "251637 soma distribution"

# Show plot window after generation (True/False)
SHOW_PLOT = False

# =============================================================================
# PIPELINE - DO NOT EDIT BELOW THIS LINE
# =============================================================================

def validate_paths(paths):
    """Validate that all required paths exist."""
    errors = []
    for name, path in paths.items():
        if not os.path.exists(path):
            errors.append(f"  - {name}: {path}")
    
    if errors:
        print("ERROR: Missing required files:")
        for err in errors:
            print(err)
        return False
    return True


def run_pipeline():
    """Run the complete visualization pipeline."""
    
    # Validate paths
    paths = {
        "NMT_DIR": NMT_DIR,
        "ARM_ATLAS": ARM_ATLAS,
        "ARM_KEY": ARM_KEY,
        "NEURON_TABLE": NEURON_TABLE,
    }
    
    if not validate_paths(paths):
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("Brain Visualization Pipeline")
    print("="*60)
    
    # Initialize BrainViz
    print("\n[1/4] Initializing...")
    viz = BrainViz(nmt_dir=NMT_DIR, output_dir=OUTPUT_DIR)
    viz.legend_markerscale = LEGEND_MARKERSCALE
    viz.elev = VIEW_ELEV
    viz.azim = VIEW_AZIM
    viz.neuron_alpha = SOMA_ALPHA
    viz.neuron_linewidth = SOMA_EDGEWIDTH
    viz.neuron_colors.update(SOMA_COLORS)
    
    # Set default soma size for all types
    for ntype in SOMA_COLORS.keys():
        viz.neuron_sizes[ntype] = SOMA_SIZE
    
    # Load brain mesh
    print("\n[2/4] Loading brain mesh...")
    viz.load_brain_mesh(
        step_size=3,
        color='black',
        size=0.2,
        alpha=BRAIN_ALPHA
    )
    
    # Extract region meshes
    print("\n[3/4] Extracting region meshes...")
    extractor = RegionExtractor(nmt_dir=NMT_DIR)
    
    extracted = extractor.extract_regions(
        atlas_path=ARM_ATLAS,
        key_path=ARM_KEY,
        region_dict=REGIONS
    )
    
    successful = []
    for name, verts in extracted.items():
        if verts is not None:
            viz.add_mesh_from_array(
                name, verts,
                color=REGION_COLORS.get(name, 'gray'),
                size=REGION_POINT_SIZE,
                alpha=REGION_ALPHA
            )
            successful.append(name)
        else:
            print(f"  Warning: Failed to extract '{name}'")
    
    print(f"  Successfully added {len(successful)} regions")
    
    # Load neurons
    print("\n[4/4] Loading neurons...")
    if NEURON_TABLE.endswith('.xlsx') or NEURON_TABLE.endswith('.xls'):
        df = pd.read_excel(NEURON_TABLE)
    else:
        df = pd.read_csv(NEURON_TABLE)
    
    viz.load_neurons(df, type_col='Neuron_Type')
    print(f"  Loaded {len(df)} neurons")
    
    # Create visualization
    print("\n[Final] Creating visualization...")
    mesh_list = ['brain'] + successful
    
    title = PLOT_TITLE if PLOT_TITLE else "3D Brain Visualization"
    
    fig, ax = viz.plot(
        meshes=mesh_list,
        title=title,
        save_path=OUTPUT_NAME
    )
    
    # Save PNG
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(
        output_path,
        dpi=DPI,
        bbox_inches='tight',
        facecolor='white'
    )
    
    print(f"\n{'='*60}")
    print(f"Success! Output saved to:")
    print(f"  {output_path}")
    print(f"{'='*60}")
    
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
    
    return viz


if __name__ == "__main__":
    viz = run_pipeline()
