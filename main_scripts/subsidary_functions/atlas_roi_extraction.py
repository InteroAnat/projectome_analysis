import nibabel as nib
import numpy as np
import nibabel as nib
import numpy as np
import pandas as pd
import os

def extract_adaptive_roi(template_path, atlas_path, atlas_table_path, target_id, output_path):
    print(f"--- Starting Adaptive ROI Extraction for ID: {target_id} ---")
    if template_path is None:
        template_path = r'D:\projectome_analysis\atlas\NMT_v2.0_sym\NMT_v2.0_sym\NMT_v2.0_sym_SS.nii.gz'
    if atlas_path is None:
        atlas_path = r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchy.nii.gz'
    if atlas_table_path is None:
        atlas_table_path = r'D:\projectome_analysis\atlas\nmt_structures_labels.txt'    
    # 1. LOAD TABLE & FIND LEVEL
    # ---------------------------------------------------------
    print(f"Reading Table: {os.path.basename(atlas_table_path)}")
    df = pd.read_csv(atlas_table_path, delimiter='\t')
    
    # Locate the row for the target ID
    # We force 'Index' to numeric to avoid string/int mismatches
    df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
    row = df[df['Index'] == target_id]
    
    if row.empty:
        print(f"Error: ID {target_id} not found in the text table.")
        return

    # LOGIC: Use 'Last_Level' for the most specific definition.
    # Level 1-6 corresponds to Array Index 0-5.
    hierarchy_level = int(row['Last_Level'].iloc[0])
    atlas_index = hierarchy_level - 1
    
    region_name = row['Abbreviation'].iloc[0]
    print(f"Target Found: '{region_name}'")
    print(f"  -> Best Hierarchy Level: {hierarchy_level}")
    print(f"  -> Atlas Array Index:    {atlas_index}")

    # 2. LOAD IMAGES
    # ---------------------------------------------------------
    print("Loading NIfTI files...")
    img_temp = nib.load(template_path)
    img_atlas = nib.load(atlas_path)
    
    data_temp = img_temp.get_fdata()
    data_atlas = img_atlas.get_fdata()

    # 3. SLICE 5D ATLAS
    # ---------------------------------------------------------
    # Shape is [X, Y, Z, Unused, Level]
    if data_atlas.ndim == 5:
        # Check bounds
        if atlas_index >= data_atlas.shape[4]:
            print(f"Error: Level {hierarchy_level} is out of bounds for this atlas.")
            return
            
        print(f"Slicing 5D Atlas at index {atlas_index}...")
        data_atlas_3d = data_atlas[:, :, :, 0, atlas_index] 
    else:
        # Fallback if someone passes a 3D file
        data_atlas_3d = data_atlas

    # 4. CHECK DIMENSIONS
    # ---------------------------------------------------------
    if data_temp.shape != data_atlas_3d.shape:
        print(f"Error: Shape mismatch! Template {data_temp.shape} vs Atlas {data_atlas_3d.shape}")
        return

    # 5. MASK & EXTRACT
    # ---------------------------------------------------------
    print("Creating Mask...")
    # Find where Atlas equals the Target ID
    mask = (data_atlas_3d == target_id)
    
    voxel_count = np.sum(mask)
    if voxel_count == 0:
        print(f"Warning: ID {target_id} exists in the table but was NOT found in the image at Level {hierarchy_level}.")
        print("Check if your atlas file matches your table (e.g. +300 offset issue).")
        return
    else:
        print(f"  -> Found {voxel_count} voxels.")

    # Create output array (Background = 0)
    extracted_roi = np.zeros_like(data_temp)
    
    # Copy pixels
    extracted_roi[mask] = data_temp[mask]

    # 6. SAVE
    # ---------------------------------------------------------
    new_img = nib.Nifti1Image(extracted_roi, img_temp.affine, img_temp.header)
    nib.save(new_img, output_path)
    print(f"Success! Saved to: {output_path}")

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # PATHS
    template_p = r'D:\projectome_analysis\atlas\NMT_v2.0_sym\NMT_v2.0_sym\NMT_v2.0_sym_SS.nii.gz'
    atlas_p    = r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchy.nii.gz'
    table_p    = r'D:\projectome_analysis\atlas\nmt_structures_labels.txt'
# --- USAGE ---
   
    # EXAMPLE 1: Extract Area 5 (ID 102)
    # The script will look up ID 102 -> See Last_Level is 4 -> Use Index 3
    extract_adaptive_roi(template_p, atlas_p, table_p, target_id=102, output_path='Area5_Extracted.nii.gz')

    # EXAMPLE 2: Extract EGP (ID 356)
    # The script will look up ID 356 -> See Last_Level is 6 -> Use Index 5
    extract_adaptive_roi(template_p, atlas_p, table_p, target_id=356, output_path='EGP_Extracted.nii.gz')

# Extract EGP (ID 356)

