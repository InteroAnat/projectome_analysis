"""
bulk_plot_processor.py - Bulk Visualization Tool

Iterates through a DataFrame of neurons and generates plots for each using
the Visual_toolkit.

Usage:
    1. Set INPUT_FILE to your Excel/CSV path.
    2. Set OUTPUT_BASE_DIR to where you want the images.
    3. Run.
"""

import pandas as pd
import os
import sys
import traceback
from tqdm import tqdm  # pip install tqdm

# --- IMPORT TOOLKIT ---
try:
    from Visual_toolkit import Visual_toolkit
    import IONData as IT
except ImportError:
    sys.path.append(os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis'))
    from Visual_toolkit import Visual_toolkit
    import IONData as IT

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_ID = '251637'
INPUT_FILE = 'INS_df.xlsx'  # Your file from region_analysis.py
OUTPUT_BASE_DIR = r"W:\fMOST\936-251637\cube_data"

# Plotting Settings
GENERATE_HIGH_RES = True
GENERATE_LOW_RES = True

# ==========================================
# MAIN LOGIC
# ==========================================
def process_batch():
    print(f"Loading neuron list from {INPUT_FILE}...")
    if INPUT_FILE.endswith('.xlsx'):
        df = pd.read_excel(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE)
    
    # 2. Setup Toolkit
    toolkit = Visual_toolkit(SAMPLE_ID)
    ion = IT.IONData()
    
    # Create Organized Folders
    dirs = {
        'high_img': os.path.join(OUTPUT_BASE_DIR, "HighRes", "Plots"),
        'high_nii': os.path.join(OUTPUT_BASE_DIR, "HighRes", "Data"),
        'low_img':  os.path.join(OUTPUT_BASE_DIR, "LowRes", "Plots"),
        'low_nii':  os.path.join(OUTPUT_BASE_DIR, "LowRes", "Data"),
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    success_count = 0
    fail_count = 0

    # 3. Iterate
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        neuron_id = str(row['NeuronID'])
        
        try:
            # A. Get Coordinates
            if 'Soma_Phys_X' in row and pd.notnull(row['Soma_Phys_X']):
                soma_xyz = [row['Soma_Phys_X'], row['Soma_Phys_Y'], row['Soma_Phys_Z']]
            else:
                tree = ion.getRawNeuronTreeByID(SAMPLE_ID, neuron_id)
                if not tree: raise ValueError("Tree not found")
                soma_xyz = [tree.root.x, tree.root.y, tree.root.z]

            # B. Process High-Res (Soma)
            vol_h, org_h, res_h = toolkit.get_high_res_block(soma_xyz, grid_radius=1)
            
            # Save NIfTI
            toolkit.export_data(vol_h, org_h, res_h, neuron_id, suffix="SomaBlock", output_dir=dirs['high_nii'])
            # Save Plot
            toolkit.plot_soma_block(vol_h, org_h, res_h, soma_xyz, neuron_id, output_dir=dirs['high_img'])

            # C. Process Low-Res (Widefield)
            vol_l, org_l, res_l = toolkit.get_low_res_widefield(
                soma_xyz, width_um=8000, height_um=8000, depth_um=30
            )
            
            # Save NIfTI
            toolkit.export_data(vol_l, org_l, res_l, neuron_id, suffix="WideField", output_dir=dirs['low_nii'])
            
            # Save Plot (needs tree for overlay)
            tree = ion.getRawNeuronTreeByID(SAMPLE_ID, neuron_id)
            toolkit.plot_widefield_context(
                vol_l, org_l, res_l, soma_xyz, neuron_id, 
                bg_intensity=2.0, swc_tree=tree, output_dir=dirs['low_img']
            )

            success_count += 1

        except Exception as e:
            fail_count += 1
            # print(f"Error {neuron_id}: {e}")

    toolkit.close()
    print(f"Batch Finished. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    process_batch()