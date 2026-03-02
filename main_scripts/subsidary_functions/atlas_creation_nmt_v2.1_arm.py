"""
atlas_creation_nmt_v2.1_arm.py - Create combined atlas from NMT_v2.1 with ARM (Anatomical Regional Mapping)

Version: 1.0.0 (NMT_v2.1 + ARM Support)

This script creates a combined atlas using:
- ARM (Anatomical Regional Mapping) for cortical laterality (CL/CR/SL/SR)
- NMT_v2.1_sym_SS as the template

Key Features:
- CL = Cortical Left (246 regions, indices 1-246)
- CR = Cortical Right (246 regions, indices 501-746)
- SL = Subcortical Left (325 regions, indices 1001-1325)
- SR = Subcortical Right (325 regions, indices 1501-1825)

Total: 1142 regions

This allows identification of cortical laterality for projectome analysis.

Author: [Your Name]
Date: 2026-01-29
"""

import nibabel as nib
import numpy as np
import os
import pandas as pd

print("=" * 70)
print("NMT_v2.1 + ARM Atlas Builder")
print("=" * 70)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Input paths
BASE_DIR = r'D:\projectome_analysis\atlas\NMT_v2.1_sym\NMT_v2.1_sym'
ARM_PATH = os.path.join(BASE_DIR, 'ARM_in_NMT_v2.1_sym.nii.gz')
TEMPLATE_PATH = os.path.join(BASE_DIR, 'NMT_v2.1_sym_SS.nii.gz')

# Output paths
OUTPUT_DIR = r'D:\projectome_analysis\atlas\NMT_v2.1_sym'
OUTPUT_ATLAS_PATH = os.path.join(OUTPUT_DIR, 'nmt_v2.1_arm_combined.nii.gz')
OUTPUT_TABLE_PATH = os.path.join(OUTPUT_DIR, 'nmt_v2.1_arm_labels.txt')

# ARM table path
ARM_TABLE_PATH = r'D:\projectome_analysis\atlas\NMT_v2.1_sym\tables_ARM\ARM_key_all.txt'

# ID ranges for ARM
ID_RANGES = {
    'CL': (1, 246),       # Cortical Left
    'CR': (501, 746),     # Cortical Right
    'SL': (1001, 1325),   # Subcortical Left
    'SR': (1501, 1825),   # Subcortical Right
}

# =============================================================================
# 2. LOAD DATA
# =============================================================================

print("\n--- Loading NIfTI files ---")

# Load ARM atlas
print(f"Loading ARM atlas: {ARM_PATH}")
img_arm = nib.load(ARM_PATH)
data_arm = img_arm.get_fdata().astype(np.uint32)
print(f"  ARM Shape: {data_arm.shape}")
print(f"  ARM Data range: [{data_arm.min()}, {data_arm.max()}]")

# Load template (for reference)
print(f"Loading template: {TEMPLATE_PATH}")
img_template = nib.load(TEMPLATE_PATH)
print(f"  Template Shape: {img_template.shape}")

# Verify shapes match
if data_arm.shape != img_template.shape:
    print(f"  WARNING: Shape mismatch between ARM and template!")

# Load ARM table
print(f"\nLoading ARM table: {ARM_TABLE_PATH}")
arm_table = pd.read_csv(ARM_TABLE_PATH, delimiter='\t')
print(f"  Table entries: {len(arm_table)}")

# Analyze ARM indices
arm_indices = arm_table['Index'].values
print(f"  Index range: {arm_indices.min()} - {arm_indices.max()}")

# Count regions by type using ID ranges
cl_count = len(arm_table[
    (arm_table['Index'] >= ID_RANGES['CL'][0]) & 
    (arm_table['Index'] <= ID_RANGES['CL'][1])
])
cr_count = len(arm_table[
    (arm_table['Index'] >= ID_RANGES['CR'][0]) & 
    (arm_table['Index'] <= ID_RANGES['CR'][1])
])
sl_count = len(arm_table[
    (arm_table['Index'] >= ID_RANGES['SL'][0]) & 
    (arm_table['Index'] <= ID_RANGES['SL'][1])
])
sr_count = len(arm_table[
    (arm_table['Index'] >= ID_RANGES['SR'][0]) & 
    (arm_table['Index'] <= ID_RANGES['SR'][1])
])

print(f"\n  Region counts:")
print(f"    CL (Cortical Left):    {cl_count} regions (IDs {ID_RANGES['CL'][0]}-{ID_RANGES['CL'][1]})")
print(f"    CR (Cortical Right):   {cr_count} regions (IDs {ID_RANGES['CR'][0]}-{ID_RANGES['CR'][1]})")
print(f"    SL (Subcortical Left): {sl_count} regions (IDs {ID_RANGES['SL'][0]}-{ID_RANGES['SL'][1]})")
print(f"    SR (Subcortical Right):{sr_count} regions (IDs {ID_RANGES['SR'][0]}-{ID_RANGES['SR'][1]})")
print(f"    TOTAL: {cl_count + cr_count + sl_count + sr_count} regions")

# =============================================================================
# 3. VERIFY ATLAS DATA
# =============================================================================

print("\n--- Verifying Atlas Data ---")

# Get unique IDs in the ARM atlas
unique_ids = np.unique(data_arm)
unique_ids = unique_ids[unique_ids != 0]  # Remove background
print(f"  Unique region IDs in atlas: {len(unique_ids)}")

# Check which IDs from table are present in atlas
atlas_ids = set(unique_ids)
table_ids = set(arm_table['Index'].values)

present_ids = table_ids & atlas_ids
missing_ids = table_ids - atlas_ids

print(f"  IDs present in atlas: {len(present_ids)}")
print(f"  IDs missing from atlas: {len(missing_ids)}")

if missing_ids:
    print(f"    (These regions have zero volume in the symmetric template)")

# Verify ID ranges in actual data
for region_type, (min_id, max_id) in ID_RANGES.items():
    mask = (data_arm >= min_id) & (data_arm <= max_id)
    voxel_count = np.sum(mask)
    print(f"  {region_type} voxels: {voxel_count:,}")

# =============================================================================
# 4. CREATE COMBINED ATLAS
# =============================================================================

print("\n--- Creating Combined Atlas ---")

# ARM atlas already has the combined structure we need
# We just use it directly as the combined atlas
combined_data = data_arm.copy()

# Verify background is 0
bg_count = np.sum(combined_data == 0)
print(f"  Background voxels: {bg_count:,}")

# =============================================================================
# 5. CREATE HIERARCHICAL LEVELS (5D)
# =============================================================================

print("\n--- Creating Hierarchical Atlas (5D) ---")

# Create 5D atlas: [X, Y, Z, Level, Time]
# Levels 1-6 correspond to ARM's First_Level to Last_Level

max_level = arm_table['Last_Level'].max()
print(f"  Maximum hierarchy level: {max_level}")

# Initialize 5D array
hierarchical_data = np.zeros(
    (combined_data.shape[0], combined_data.shape[1], combined_data.shape[2], max_level, 1),
    dtype=np.uint32
)

# For each level, we need to map regions to their parent at that level
# This is complex for ARM since it has a true hierarchy
# For now, we'll create a simplified version where each level shows
# regions that exist at that level

print("  Building hierarchical levels...")
for level in range(1, max_level + 1):
    # Get regions that exist at this level
    regions_at_level = arm_table[
        (arm_table['First_Level'] <= level) & 
        (arm_table['Last_Level'] >= level)
    ]
    
    # Create a mapping from leaf IDs to level-specific IDs
    # For simplicity, we use the original IDs but mask out regions
    # that don't exist at this level
    
    level_data = np.zeros_like(combined_data)
    
    # For each region at this level, find its voxels
    for _, row in regions_at_level.iterrows():
        region_id = row['Index']
        mask = combined_data == region_id
        level_data[mask] = region_id
    
    hierarchical_data[:, :, :, level-1, 0] = level_data
    
    if level <= 3 or level == max_level:
        unique_at_level = len(np.unique(level_data)) - 1  # Exclude 0
        print(f"    Level {level}: {unique_at_level} unique regions")

# =============================================================================
# 6. SAVE OUTPUTS
# =============================================================================

print("\n--- Saving Outputs ---")

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save combined atlas (3D)
print(f"Saving 3D atlas: {OUTPUT_ATLAS_PATH}")
img_out = nib.Nifti1Image(combined_data, img_arm.affine, img_arm.header)
nib.save(img_out, OUTPUT_ATLAS_PATH)
print("  Done.")

# Save hierarchical atlas (5D)
HIERARCHICAL_OUTPUT_PATH = OUTPUT_ATLAS_PATH.replace('.nii.gz', '_hierarchical.nii.gz')
print(f"Saving 5D hierarchical atlas: {HIERARCHICAL_OUTPUT_PATH}")
img_hier = nib.Nifti1Image(hierarchical_data, img_arm.affine, img_arm.header)
nib.save(img_hier, HIERARCHICAL_OUTPUT_PATH)
print("  Done.")

# Save label table with laterality information
print(f"Saving label table: {OUTPUT_TABLE_PATH}")

# Add laterality column based on ID ranges
def get_laterality(index):
    if ID_RANGES['CL'][0] <= index <= ID_RANGES['CL'][1]:
        return 'Left'
    elif ID_RANGES['CR'][0] <= index <= ID_RANGES['CR'][1]:
        return 'Right'
    elif ID_RANGES['SL'][0] <= index <= ID_RANGES['SL'][1]:
        return 'Left'
    elif ID_RANGES['SR'][0] <= index <= ID_RANGES['SR'][1]:
        return 'Right'
    else:
        return 'Unknown'

def get_region_type(index):
    if ID_RANGES['CL'][0] <= index <= ID_RANGES['CL'][1]:
        return 'Cortical'
    elif ID_RANGES['CR'][0] <= index <= ID_RANGES['CR'][1]:
        return 'Cortical'
    elif ID_RANGES['SL'][0] <= index <= ID_RANGES['SL'][1]:
        return 'Subcortical'
    elif ID_RANGES['SR'][0] <= index <= ID_RANGES['SR'][1]:
        return 'Subcortical'
    else:
        return 'Unknown'

arm_table['Laterality'] = arm_table['Index'].apply(get_laterality)
arm_table['Region_Type'] = arm_table['Index'].apply(get_region_type)

# Save with tab delimiter
arm_table.to_csv(OUTPUT_TABLE_PATH, sep='\t', index=False)
print("  Done.")

# =============================================================================
# 7. VERIFICATION
# =============================================================================

print("\n--- Verification ---")

# Load saved atlas and verify
saved_atlas = nib.load(OUTPUT_ATLAS_PATH)
saved_data = saved_atlas.get_fdata()

print(f"  Saved atlas shape: {saved_data.shape}")
print(f"  Saved atlas range: [{saved_data.min()}, {saved_data.max()}]")
print(f"  Background (should be 0): {np.sum(saved_data == 0):,} voxels")

# Check laterality distribution
cl_voxels = np.sum((saved_data >= ID_RANGES['CL'][0]) & (saved_data <= ID_RANGES['CL'][1]))
cr_voxels = np.sum((saved_data >= ID_RANGES['CR'][0]) & (saved_data <= ID_RANGES['CR'][1]))
sl_voxels = np.sum((saved_data >= ID_RANGES['SL'][0]) & (saved_data <= ID_RANGES['SL'][1]))
sr_voxels = np.sum((saved_data >= ID_RANGES['SR'][0]) & (saved_data <= ID_RANGES['SR'][1]))

print(f"\n  Voxel counts by laterality:")
print(f"    CL (Cortical Left):     {cl_voxels:,} voxels")
print(f"    CR (Cortical Right):    {cr_voxels:,} voxels")
print(f"    SL (Subcortical Left):  {sl_voxels:,} voxels")
print(f"    SR (Subcortical Right): {sr_voxels:,} voxels")

# Verify table
saved_table = pd.read_csv(OUTPUT_TABLE_PATH, delimiter='\t')
print(f"\n  Saved table entries: {len(saved_table)}")
print(f"  Laterality column added: {'Laterality' in saved_table.columns}")
print(f"  Region_Type column added: {'Region_Type' in saved_table.columns}")

# =============================================================================
# 8. SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("ATLAS CREATION COMPLETE")
print("=" * 70)
print(f"\nOutput files:")
print(f"  3D Atlas:      {OUTPUT_ATLAS_PATH}")
print(f"  5D Atlas:      {HIERARCHICAL_OUTPUT_PATH}")
print(f"  Label Table:   {OUTPUT_TABLE_PATH}")
print(f"\nAtlas Statistics:")
print(f"  Total regions: {len(arm_table)}")
print(f"  Cortical Left (CL):     {cl_count} regions (IDs {ID_RANGES['CL'][0]}-{ID_RANGES['CL'][1]})")
print(f"  Cortical Right (CR):    {cr_count} regions (IDs {ID_RANGES['CR'][0]}-{ID_RANGES['CR'][1]})")
print(f"  Subcortical Left (SL):  {sl_count} regions (IDs {ID_RANGES['SL'][0]}-{ID_RANGES['SL'][1]})")
print(f"  Subcortical Right (SR): {sr_count} regions (IDs {ID_RANGES['SR'][0]}-{ID_RANGES['SR'][1]})")
print(f"\nLaterality identification is now available for region analysis!")
print("=" * 70)
