#!/usr/bin/env python3
"""
Example: Brain Viz Usage
========================
"""

import os
from brain_viz import BrainViz, RegionExtractor
import pandas as pd
import matplotlib.pyplot as plt

# Paths
NMT_DIR = r"D:\projectome_analysis\atlas\NMT_v2.1_sym\NMT_v2.1_sym"
OUTPUT_DIR = r".\flatmap_outputs"
SOMA_CSV = r'D:\projectome_analysis\main_scripts\neuron_tables\INS_df_v3.xlsx'
ARM_ATLAS = r"D:\projectome_analysis\atlas\ARM_in_NMT_v2.1_sym.nii.gz"
ARM_KEY = r"D:\projectome_analysis\atlas\ARM_key_all.txt"
global_id_df = pd.read_csv(ARM_KEY, delimiter='\t')
insula_abbr_table =global_id_df.loc[global_id_df['Full_Name'].str.contains('insula',case=False) & (global_id_df["Last_Level"] == 6), "Abbreviation"]

def example_original():
    """Recreate the original vs_brain.py functionality."""
    
    viz = BrainViz(nmt_dir=NMT_DIR, output_dir=OUTPUT_DIR)
    viz.load_brain_mesh(step_size=3, color='black', size=0.2, alpha=0.1)
    
    # Load insula meshes from pickle files
    left_pkl = os.path.join(OUTPUT_DIR, 'insula_flatmap_left.pkl')
    right_pkl = os.path.join(OUTPUT_DIR, 'insula_flatmap_right.pkl')
    
    if os.path.exists(left_pkl):
        viz.add_mesh_from_file('left_insula', left_pkl, 
                              color='#3498DB', size=3, alpha=0.05)
    if os.path.exists(right_pkl):
        viz.add_mesh_from_file('right_insula', right_pkl,
                              color='#E74C3C', size=3, alpha=0.05)
    
    viz.load_neurons(pd.read_excel(SOMA_CSV), type_col='Neuron_Type')
    viz.set_view(elev=30, azim=60)
    viz.plot(title="3D Insula Projectome", save_path="brain_viz_recreated.png")
    plt.show()


def example_generate_meshes():
    """Generate meshes from ARM atlas (no pickle files needed)."""
    
    print("="*60)
    print("Example: Generate Meshes from Atlas")
    print("="*60)
    
    # Initialize visualizer
    viz = BrainViz(nmt_dir=NMT_DIR, output_dir=OUTPUT_DIR)
    viz.load_brain_mesh(step_size=3, color='black', alpha=0.1)
    
    # Extract region meshes from ARM atlas
    extractor = RegionExtractor(nmt_dir=NMT_DIR)
    
    regions = extractor.extract_regions(
        atlas_path=ARM_ATLAS,
        key_path=ARM_KEY,
        region_dict={
            'left_agranular': ['CL_Ial', 'CL_Iai', 'CL_lat_Ia', 'CL_Iapl', 'CL_Iam/Iapm'],
            'left_granular': ['CL_Ig'],
            'left_dysgranular': ['CL_Ia/Id'],
            'right_agranular': ['CR_Ial', 'CR_Iai', 'CR_lat_Ia', 'CR_Iapl', 'CR_Iam/Iapm'],
            'right_granular': ['CR_Ig'],
            'right_dysgranular': ['CR_Ia/Id'],
        }
    )
    
    # Add extracted meshes to visualizer with different colors
    colors = {
        'left_agranular': '#E74C3C',   # Red
        'left_granular': '#3498DB',    # Blue
        'left_dysgranular': '#9B59B6', # Purple
        'right_agranular': '#E67E22',  # Orange
        'right_granular': '#2980B9',   # Dark blue
        'right_dysgranular': '#8E44AD', # Dark purple
    }
    
    for name, verts in regions.items():
        if verts is not None:
            viz.add_mesh_from_array(name, verts, 
                                   color=colors.get(name, 'gray'),
                                   size=3, alpha=0.1)
    
    # Load neurons
    viz.load_neurons(pd.read_excel(SOMA_CSV), type_col='Neuron_Type')
    
    # Plot all
    mesh_names = [name for name, verts in regions.items() if verts is not None]
    viz.plot(
        meshes=['brain'] + mesh_names,
        title="Generated Region Meshes",
        save_path="brain_viz_generated.png"
    )
    plt.show()


def example_simple_insula():
    """Simple example: just insula regions."""
    
    viz = BrainViz(nmt_dir=NMT_DIR, output_dir=OUTPUT_DIR)
    viz.load_brain_mesh(step_size=3, color='black', alpha=0.1)
    
    # Generate simple insula meshes
    extractor = RegionExtractor(nmt_dir=NMT_DIR)
    
    regions = extractor.extract_regions(
        atlas_path=ARM_ATLAS,
        key_path=ARM_KEY,
        region_dict={
            'left_insula': ['CL_Ig', 'CL_Ial', 'CL_Iai', 'CL_lat_Ia'],
            'right_insula': ['CR_Ig', 'CR_Ial', 'CR_Iai', 'CR_lat_Ia']
        }
    )
    
    viz.add_mesh_from_array('left_insula', regions['left_insula'], 
                           color='#3498DB', alpha=0.05)
    viz.add_mesh_from_array('right_insula', regions['right_insula'],
                           color='#E74C3C', alpha=0.05)
    
    viz.load_neurons(pd.read_excel(SOMA_CSV), type_col='Neuron_Type')
    viz.plot(save_path="brain_viz_simple.png")
    plt.show()


def example_selective_neurons():
    """Selective visualization by neuron type."""
    
    viz = BrainViz(nmt_dir=NMT_DIR, output_dir=OUTPUT_DIR)
    viz.load_brain_mesh()
    
    # Generate regions
    extractor = RegionExtractor(nmt_dir=NMT_DIR)
    regions = extractor.extract_regions(
        atlas_path=ARM_ATLAS,
        key_path=ARM_KEY,
        region_dict={'left_insula': ['CL_Ig'], 'right_insula': ['CR_Ig']}
    )
    
    for name, verts in regions.items():
        if verts is not None:
            viz.add_mesh_from_array(name, verts, alpha=0.1)
    
    # Load and filter neurons
    df = pd.read_excel(SOMA_CSV)
    viz.load_neurons(df, type_col='Neuron_Type', name='all')
    viz.filter_neurons('all', types=['PT'], new_name='PT_only')
    viz.filter_neurons('all', types=['CT'], new_name='CT_only')
    
    # Customize sizes
    viz.neuron_sizes = {'PT': 100, 'CT': 80}
    
    # Plot PT only
    viz.plot(neurons=['PT_only'], title="PT Neurons")
    plt.show()


def example_custom_colors():
    """Customize colors."""
    
    viz = BrainViz(nmt_dir=NMT_DIR, output_dir=OUTPUT_DIR)
    viz.load_brain_mesh(color='lightgray', alpha=0.15)
    
    # Generate region
    extractor = RegionExtractor(nmt_dir=NMT_DIR)
    regions = extractor.extract_regions(
        atlas_path=ARM_ATLAS,
        key_path=ARM_KEY,
        region_dict={'insula': ['CL_Ig', 'CL_Ial', 'CR_Ig', 'CR_Ial']}
    )
    
    if regions['insula'] is not None:
        viz.add_mesh_from_array('insula', regions['insula'], 
                               color='red', size=5, alpha=0.1)
    
    # Custom neuron colors
    viz.neuron_colors = {
        'PT': '#FF0000',
        'CT': '#00FF00',
        'ITs': '#0000FF',
        'ITc': '#FF00FF',
        'ITi': '#00FFFF',
    }
    
    viz.load_neurons(pd.read_excel(SOMA_CSV), type_col='Neuron_Type')
    viz.plot(title="Custom Colors")
    plt.show()


def example_find_regions_dynamically():
    """Find and visualize insula regions dynamically."""
    
    print("="*60)
    print("Example: Find Regions Dynamically")
    print("="*60)
    
    viz = BrainViz(nmt_dir=NMT_DIR, output_dir=OUTPUT_DIR)
    viz.load_brain_mesh(step_size=3, color='black', alpha=0.1)
    
    # Initialize extractor
    extractor = RegionExtractor(nmt_dir=NMT_DIR)
    
    # Find all insula regions at level 6
    insula_abbrs = extractor.find_regions_by_name(ARM_KEY, 'insula', level=6)
    print(f"\nFound insula regions: {insula_abbrs.tolist()}")
    
    # Split by hemisphere
    left_regions = [r for r in insula_abbrs if r.startswith('CL_')]
    right_regions = [r for r in insula_abbrs if r.startswith('CR_')]
    
    print(f"Left: {left_regions}")
    print(f"Right: {right_regions}")
    
    # Extract meshes
    regions = extractor.extract_regions(
        atlas_path=ARM_ATLAS,
        key_path=ARM_KEY,
        region_dict={
            'left_insula': left_regions,
            'right_insula': right_regions
        }
    )
    
    # Add to visualizer
    if regions['left_insula'] is not None:
        viz.add_mesh_from_array('left_insula', regions['left_insula'], 
                               color='#3498DB', alpha=0.05)
    if regions['right_insula'] is not None:
        viz.add_mesh_from_array('right_insula', regions['right_insula'],
                               color='#E74C3C', alpha=0.05)
    
    # Load and plot neurons
    viz.load_neurons(pd.read_excel(SOMA_CSV), type_col='Neuron_Type')
    viz.plot(title="Dynamic Insula Regions", save_path="brain_viz_dynamic.png")
    plt.show()


if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Original vs_brain.py recreation', example_original),
        '2': ('Generate meshes from atlas', example_generate_meshes),
        '3': ('Simple insula only', example_simple_insula),
        '4': ('Selective neuron types', example_selective_neurons),
        '5': ('Custom colors', example_custom_colors),
        '6': ('Find regions dynamically', example_find_regions_dynamically),
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        print(f"\nRunning: {examples[sys.argv[1]][0]}")
        examples[sys.argv[1]][1]()
    else:
        print("\nUsage: python example_usage.py [1-6]")
        print("\nExamples:")
        for k, (name, _) in examples.items():
            print(f"  {k}. {name}")
        print("\nRunning example 2 (generate meshes from atlas)...")
        example_generate_meshes()
