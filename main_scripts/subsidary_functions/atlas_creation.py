import nibabel as nib
from nilearn import plotting
import numpy as np
import os
"""
based on the structure key table used by ION, I found the indices are formatted as follows:
- CHARM (1- 246) but reserved 1-300 slots
- SARM (300 - 618)

Key notes:
 1. selectively increase ID of SARM-exist voxels
 2. combined atlas indices




"""


print("--- STARTING ATLAS BUILDER ---")

# 1. CONFIGURATION
# =================
base_dir = r'D:\projectome_analysis\atlas\NMT_v2.0_sym\NMT_v2.0_sym'
charm_path = os.path.join(base_dir, 'CHARM_in_NMT_v2.0_sym.nii.gz')
sarm_path  = os.path.join(base_dir, 'SARM_in_NMT_v2.0_sym.nii.gz')
output_path = r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchyy.nii.gz'

# 2. LOAD DATA
# =================
print("Loading NIfTI files...")
img_charm = nib.load(charm_path)
img_sarm  = nib.load(sarm_path)

# Convert to unsigned integers (important for ID manipulation)
# These are 5D arrays: [X, Y, Z, Level, Time]
data_charm = img_charm.get_fdata().astype(np.uint16)
data_sarm  = img_sarm.get_fdata().astype(np.uint16)

print(f"CHARM Shape: {data_charm.shape}")
print(f"SARM Shape:  {data_sarm.shape}")

# 3. SHIFT SARM IDs (+300)
# =================
print("Shifting SARM IDs by +300 (preserving background 0)...")

# Create a mask where SARM actually has data
mask_sarm_exists = data_sarm > 0

# Create an empty container
data_sarm_shifted = np.zeros_like(data_sarm)

# Apply logic: Only add 300 where data exists
# This FIXES the bug where background became 300
data_sarm_shifted[mask_sarm_exists] = data_sarm[mask_sarm_exists] + 300

# 4. MERGE (PRIORITY LOGIC)
# =================
print("Merging Atlases...")

# Step A: Lay down the Subcortex (SARM) as the base
combined_data = data_sarm_shifted.copy()

# Step B: Lay down Cortex (CHARM) on top
# If a pixel exists in CHARM, it overwrites whatever was there.
# This FIXES the Insula vs EGP conflict.
mask_charm_exists = data_charm > 0
combined_data[mask_charm_exists] = data_charm[mask_charm_exists]

# 5. VERIFICATION
# =================
bg_val = combined_data[0,0,0,0,0]
max_val = np.max(combined_data)
max_charm_val = np.max(combined_data[mask_charm_exists])
print(f"Verification:")
print(f"  - Background Value (Should be 0): {bg_val}")
print(f"  - Max ID (Should be ~618): {max_val}")
print (f" max of combined data CHARM region (should be 246) {max_charm_val}")
if bg_val != 0 or max_val != 618 or max_charm_val != 246:
    print("Error")

# 6. SAVE
# =================
print(f"Saving to {output_path}...")
# Use the affine header from CHARM so it aligns with the template
new_img = nib.Nifti1Image(combined_data, img_charm.affine, img_charm.header)
nib.save(new_img, output_path)
print("Done.")
