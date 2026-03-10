#!/usr/bin/env python3
"""
Quick Start - Minimal Working Example
=====================================
"""

from brain_viz import BrainViz, RegionExtractor
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
NMT_DIR = r"D:\projectome_analysis\atlas\NMT_v2.1_sym\NMT_v2.1_sym"
ARM_ATLAS = r"D:\projectome_analysis\atlas\ARM_in_NMT_v2.1_sym.nii.gz"
ARM_KEY = r"D:\projectome_analysis\atlas\ARM_key_all.txt"
SOMA_CSV = r'D:\projectome_analysis\main_scripts\neuron_tables\INS_df_v3.xlsx'
OUTPUT_DIR = r".\output"

# 1. Initialize visualizer
viz = BrainViz(nmt_dir=NMT_DIR, output_dir=OUTPUT_DIR)

# 2. Load brain mesh
viz.load_brain_mesh(step_size=3, color='black', alpha=0.1)

# 3. Generate region meshes from ARM atlas
extractor = RegionExtractor(nmt_dir=NMT_DIR)

# Option A: Define regions manually
regions = extractor.extract_regions(
    atlas_path=ARM_ATLAS,
    key_path=ARM_KEY,
    region_dict={
        'left_insula': ['CL_Ig', 'CL_Ial', 'CL_Iai'],
        'right_insula': ['CR_Ig', 'CR_Ial', 'CR_Iai']
    }
)

# Option B: Find insula regions dynamically
# insula_regions = extractor.find_regions_by_name(ARM_KEY, 'insula', level=6)
# print(f"Found insula regions: {insula_regions.tolist()}")

# Add generated meshes to visualizer
viz.add_mesh_from_array('left_insula', regions['left_insula'], 
                       color='#3498DB', alpha=0.05)
viz.add_mesh_from_array('right_insula', regions['right_insula'],
                       color='#E74C3C', alpha=0.05)

# 4. Load neurons
df = pd.read_excel(SOMA_CSV)
viz.load_neurons(df, type_col='Neuron_Type')

# 5. Plot
viz.plot(title="Brain with Generated Meshes", save_path="output.png")
plt.show()
