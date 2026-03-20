"""
bulk_plot_processor.py - Bulk Visualization Tool with Cache Support

Iterates through a DataFrame of neurons and generates plots for each using
the Visual_toolkit with caching and soma region support.

Usage:
    1. Set INPUT_FILE to your Excel/CSV path.
    2. Set OUTPUT_BASE_DIR to where you want the images.
    3. Run.

Features:
    - Uses SAME cache directory as GUI (project_root/resource/cubes/sample_id/)
    - SWC coordinates exclusively (tree.root.x/y/z)
    - Soma region in filenames and plot titles
"""

import pandas as pd
import os
import sys
import traceback
from tqdm import tqdm  # pip install tqdm
from datetime import datetime
# --- IMPORT TOOLKIT ---
try:
    from Visual_toolkit import Visual_toolkit
    import IONData as IT
except ImportError:
    sys.path.append(os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis'))
    sys.path.append(os.path.abspath(r'D:\projectome_analysis\main_scripts'))

    from Visual_toolkit import Visual_toolkit
    import IONData as IT

# ==========================================
# CONFIGURATION
# ==========================================
current_date = datetime.now().strftime('%Y%m%d')

SAMPLE_ID = '251637'
INPUT_FILE = r'D:\projectome_analysis\main_scripts\neuron_tables\251637_INS.xlsx'

# INPUT_FILE = r'D:\projectome_analysis\main_scripts\neuron_tables\251637_M1.xlsx'
# INPUT_FILE = r'D:\projectome_analysis\main_scripts\neuron_tables\251637_subset.xlsx'
PARENT_OUTPUT_DIR = r"W:\fMOST"

# ==========================================
# MAIN LOGIC
# ==========================================
def sanitize_path(name):
    """Sanitize string for use in folder/file paths."""
    if not name:
        return name
    # Replace path separators and other problematic characters
    return name.replace('/', '-').replace('\\', '-').replace(':', '-')

def process_batch():
    print(f"Loading neuron list from {INPUT_FILE}...")
    if INPUT_FILE.endswith('.xlsx'):
        df = pd.read_excel(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE)
    
    # Setup IONData for fetching neurons
    ion = IT.IONData()
    
    # Group neurons by soma_region for organized output
    if 'Soma_Region' not in df.columns:
        print("Warning: 'Soma_Region' column not found in input file. Using 'Unknown' for all neurons.")
        df['Soma_Region'] = 'Unknown'
    
    # Get unique soma regions
    regions = df['Soma_Region'].unique()
    print(f"Found {len(regions)} unique soma regions: {list(regions)}")
    
    success_count = 0
    fail_count = 0
    
    # Process each region group
    for region in regions:
        region_df = df[df['Soma_Region'] == region]
        
        # Sanitize region name for folder path
        safe_region_folder = sanitize_path(str(region))
        
        print(f"\n{'='*60}")
        print(f"Processing Region: {region} ({len(region_df)} neurons)")
        print(f"Folder: Region_{safe_region_folder}")
        print(f"{'='*60}")
        
        # Create region-specific output directory (using sanitized name)
        region_output_base = os.path.join(
            PARENT_OUTPUT_DIR, 
            SAMPLE_ID,
            f"cube_data_{os.path.splitext(os.path.basename(INPUT_FILE))[0]}_{current_date}",
            f"Region_{safe_region_folder}"
        )
        
        # Create subdirectories
        dirs = {
            'high_img': os.path.join(region_output_base, "HighRes", "Plots"),
            'high_nii': os.path.join(region_output_base, "HighRes", "Data"),
            'low_img':  os.path.join(region_output_base, "LowRes", "Plots"),
            'low_nii':  os.path.join(region_output_base, "LowRes", "Data"),
        }
        for d in dirs.values(): 
            os.makedirs(d, exist_ok=True)
        
        # Create ONE toolkit instance with DEFAULT cache (same as GUI)
        # This shares cache with GUI: project_root/resource/cubes/sample_id/
        toolkit = Visual_toolkit(SAMPLE_ID)
        
        # Process each neuron in this region
        for i, row in tqdm(region_df.iterrows(), total=len(region_df), desc=f"Region {region}"):
            neuron_id = str(row['NeuronID'])
            soma_region = str(row['Soma_Region']) if pd.notnull(row['Soma_Region']) else 'Unknown'
            
            try:
                # A. Get Coordinates from SWC ONLY (tree.root.x/y/z)
                tree = ion.getRawNeuronTreeByID(SAMPLE_ID, neuron_id)
                if not tree: 
                    raise ValueError(f"Tree not found for neuron {neuron_id}")
                soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
                
                # B. Process High-Res (Soma)
                if GENERATE_HIGH_RES:
                    vol_h, org_h, res_h = toolkit.get_high_res_block(soma_xyz, grid_radius=1)
                    
                    # Save NIfTI with region (coordinates in title only, not filename)
                    toolkit.export_data(
                        vol_h, org_h, res_h, neuron_id, 
                        suffix="SomaBlock", 
                        soma_region=soma_region,
                        soma_coords=soma_xyz,
                        output_dir=dirs['high_nii']
                    )
                    
                    # Save Plot with region and coordinates in title
                    toolkit.plot_soma_block(
                        vol_h, org_h, res_h, soma_xyz, neuron_id, 
                        suffix="SomaBlock",
                        soma_region=soma_region,
                        output_dir=dirs['high_img']
                    )

                # C. Process Low-Res (Widefield)
                if GENERATE_LOW_RES:
                    vol_l, org_l, res_l = toolkit.get_low_res_widefield(
                        soma_xyz, width_um=8000, height_um=8000, depth_um=30
                    )
                    
                    # Save NIfTI with region (coordinates in title only, not filename)
                    toolkit.export_data(
                        vol_l, org_l, res_l, neuron_id, 
                        suffix="WideField",
                        soma_region=soma_region,
                        soma_coords=soma_xyz,
                        output_dir=dirs['low_nii']
                    )
                    
                    # Save Plot with region and coordinates in title
                    toolkit.plot_widefield_context(
                        vol_l, org_l, res_l, soma_xyz, neuron_id, 
                        bg_intensity=2.0, swc_tree=tree, 
                        soma_region=soma_region,
                        output_dir=dirs['low_img']
                    )

                success_count += 1

            except Exception as e:
                fail_count += 1
                print(f"\n[ERROR] Neuron {neuron_id}: {e}")
                traceback.print_exc()
        
        # Close toolkit for this region
        toolkit.close()
    
    print(f"\n{'='*60}")
    print(f"Batch Finished. Success: {success_count}, Failed: {fail_count}")
    print(f"{'='*60}")

# ==========================================
# PLOT SETTINGS
# ==========================================
GENERATE_HIGH_RES = True
GENERATE_LOW_RES = True

if __name__ == "__main__":
    process_batch()
