import nibabel as nib
import pandas as pd
import numpy as np

def verify_atlas_integrity():
    print("--- ATLAS INTEGRITY CHECK ---")
    # PATHS
    atlas_path = r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchy.nii.gz'
    table_path = r'D:\projectome_analysis\atlas\nmt_structures_labels.txt'

    # 1. Load Data
    nii = nib.load(atlas_path)
    data = nii.get_fdata().astype(np.uint16)
    
    df = pd.read_csv(table_path, delimiter='\t')
    # Correct Mapping Logic
    id_map = dict(zip(df['Index'], df['Abbreviation']))

    # 2. Check Background
    # We check the top-left corner of Level 6 (Index 5)
    bg_val = data[0, 0, 0, 0, 5]
    print(f"1. Background Value: {bg_val} (Should be 0)")
    if bg_val == 300:
        print("   >> CRITICAL FAIL: Background is 300. You need to re-run the merge script.")
        return

    # 3. Check for Specific SARM Region (EGP)
    # EGP Original ID was 56. New ID should be 356.
    target_id = 356
    target_name = id_map.get(target_id, "Not Found in Table")
    
    print(f"2. Searching for ID {target_id} ({target_name})...")
    
    # Check if this ID exists anywhere in the Level 6 volume
    level_6 = data[:, :, :, 0, 5]
    exists = target_id in level_6
    
    if exists:
        count = np.sum(level_6 == target_id)
        print(f"   >> SUCCESS: Found {count} voxels labeled {target_name} ({target_id}).")
    else:
        print(f"   >> FAIL: ID {target_id} does not exist in the image.")

    # 4. Check for Overlap Corruption (Summation Bug)
    # If you added instead of overwritten, you might have ID 100+350 = 450.
    # Let's check for a random high ID that shouldn't exist, e.g., > 800
    max_val = np.max(level_6)
    print(f"3. Maximum ID in volume: {max_val}")
    if max_val > 618: # 618 is usually the highest NMT ID
        print(f"   >> WARNING: Max ID {max_val} is higher than expected. Potential summation artifact.")
    else:
        print("   >> OK: Max ID is within range.")

verify_atlas_integrity()