#!/usr/bin/env python3
"""
Step-by-Step Insula Flatmap with Subregion Boundaries
======================================================

Creates publication-quality flatmap figure with:
1. Subregion boundaries drawn as black lines
2. Each subregion filled with distinct color
3. Neurons plotted as colored dots
4. Brain outline for anatomical context

Each step saves output for verification.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
import nibabel as nib
from scipy.spatial import ConvexHull
from scipy import ndimage
from pathlib import Path
import trimesh
from skimage import measure

# Add paths
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
main_scripts_path = os.path.abspath(r'D:\projectome_analysis\main_scripts')
sys.path.append(neurovis_path)
sys.path.append(main_scripts_path)

from NMTFlatmap import InsulaFlatmapNMT
import neuro_tracer as nt

# Configuration
NMT_DIR = r"D:\projectome_analysis\atlas\NMT_v2.1_sym\NMT_v2.1_sym"
ARM_ATLAS = r"D:\projectome_analysis\atlas\ARM_in_NMT_v2.1_sym.nii.gz"
ARM_KEY = r"D:\projectome_analysis\atlas\ARM_key_all.txt"
INS_DF_PATH = r"D:\projectome_analysis\main_scripts\INS_df.xlsx"
OUTPUT_DIR = r".\flatmap_step_by_step"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# STEP 0: CONFIGURATION
# ============================================================================

# Subregion definitions with ARM region names and colors
SUBREGIONS = {
    'left': {
        'agranular_anterior': {
            'regions': ['CL_Iam/Iapm', 'CL_lat_Ia', 'CL_Iai'],
            'color': '#E74C3C',  # Red
            'label': 'Agranular (Ant)'
        },
        'agranular_posterior': {
            'regions': ['CL_Ial', 'CL_Iapl', 'CL_Ia/Id'],
            'color': '#E67E22',  # Orange
            'label': 'Agranular (Post)'
        },
        'granular': {
            'regions': ['CL_Ig'],
            'color': '#3498DB',  # Blue
            'label': 'Granular'
        },
        'retroinsula': {
            'regions': ['CL_Ri'],
            'color': '#9B59B6',  # Purple
            'label': 'Retroinsula'
        },
    },
    'right': {
        'agranular_anterior': {
            'regions': ['CR_Iam/Iapm', 'CR_lat_Ia', 'CR_Iai'],
            'color': '#E74C3C',
            'label': 'Agranular (Ant)'
        },
        'agranular_posterior': {
            'regions': ['CR_Ial', 'CR_Iapl', 'CR_Ia/Id'],
            'color': '#E67E22',
            'label': 'Agranular (Post)'
        },
        'granular': {
            'regions': ['CR_Ig'],
            'color': '#3498DB',
            'label': 'Granular'
        },
        'retroinsula': {
            'regions': ['CR_Ri'],
            'color': '#9B59B6',
            'label': 'Retroinsula'
        },
    }
}

# Neuron type colors
NEURON_TYPE_COLORS = {
    'PT': '#d62728',
    'CT': '#2ca02c',
    'ITs': '#9467bd',
    'ITc': '#e377c2',
    'ITi': '#17becf',
    'Unclassified': '#7f7f7f'
}

NEURON_TYPE_MARKERS = {
    'PT': '^',
    'CT': 's',
    'ITs': 'o',
    'ITc': 'o',
    'ITi': 'o',
    'Unclassified': 'o'
}


def print_step(step_num, title):
    """Print step header."""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {title}")
    print("="*70)


def save_verification_image(data, title, filename, cmap='gray'):
    """Save a verification image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if data.ndim == 3:
        # Show middle slice
        mid = data.shape[2] // 2
        im = ax.imshow(data[:, :, mid].T, cmap=cmap, origin='lower')
        ax.set_title(f'{title}\n(Z slice {mid})')
    else:
        im = ax.imshow(data.T, cmap=cmap, origin='lower')
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150)
    print(f"  Saved: {filepath}")
    plt.close()


# ============================================================================
# STEP 1: LOAD ARM ATLAS AND EXTRACT INSULA REGIONS
# ============================================================================

def step1_load_atlas():
    """Load ARM atlas and extract insula region indices."""
    print_step(1, "Load ARM Atlas and Extract Insula Regions")
    
    # Load ARM key
    arm_key = pd.read_csv(ARM_KEY, delimiter='\t')
    name_to_index = dict(zip(arm_key['Abbreviation'], arm_key['Index']))
    print(f"  Loaded ARM key with {len(name_to_index)} regions")
    
    # Load ARM atlas
    arm_nii = nib.load(ARM_ATLAS)
    arm_data = arm_nii.get_fdata().astype(int)
    print(f"  ARM atlas shape: {arm_data.shape}")
    
    # Extract level 6
    if arm_data.ndim == 5:
        arm_data = arm_data[:, :, :, 0, 5]  # level 6
    print(f"  Sliced to shape: {arm_data.shape}")
    
    # Get indices for each subregion
    subregion_indices = {}
    for hemisphere in ['left', 'right']:
        subregion_indices[hemisphere] = {}
        for subregion_name, info in SUBREGIONS[hemisphere].items():
            indices = []
            for region_name in info['regions']:
                if region_name in name_to_index:
                    indices.append(name_to_index[region_name])
            subregion_indices[hemisphere][subregion_name] = indices
            print(f"  {hemisphere} {subregion_name}: indices {indices}")
    
    return arm_data, subregion_indices


# ============================================================================
# STEP 2: CREATE SUBREGION MASKS
# ============================================================================

def step2_create_masks(arm_data, subregion_indices):
    """Create binary masks for each subregion."""
    print_step(2, "Create Subregion Masks")
    
    masks = {}
    for hemisphere in ['left', 'right']:
        masks[hemisphere] = {}
        for subregion_name, indices in subregion_indices[hemisphere].items():
            mask = np.isin(arm_data, indices).astype(np.uint8)
            masks[hemisphere][subregion_name] = mask
            print(f"  {hemisphere} {subregion_name}: {mask.sum()} voxels")
            
            # Save verification image
            save_verification_image(
                mask, 
                f'{hemisphere} {subregion_name} mask',
                f'step2_mask_{hemisphere}_{subregion_name}.png',
                cmap='Reds'
            )
    
    return masks


# ============================================================================
# STEP 3: COMBINE WITH GM MASK
# ============================================================================

def step3_combine_with_gm(masks):
    """Combine subregion masks with GM mask."""
    print_step(3, "Combine with GM Mask")
    
    # Load GM mask
    gm_path = os.path.join(NMT_DIR, 'NMT_v2.1_sym_segmentation.nii.gz')
    seg_nii = nib.load(gm_path)
    seg_data = seg_nii.get_fdata().astype(int)
    gm_mask = (seg_data == 2).astype(np.uint8)
    print(f"  GM mask: {gm_mask.sum()} voxels")
    
    save_verification_image(gm_mask, 'GM Mask', 'step3_gm_mask.png', cmap='Greens')
    
    gm_masks = {}
    for hemisphere in ['left', 'right']:
        gm_masks[hemisphere] = {}
        for subregion_name, mask in masks[hemisphere].items():
            gm_roi = mask & (gm_mask > 0)
            gm_masks[hemisphere][subregion_name] = gm_roi
            print(f"  {hemisphere} {subregion_name} GM: {gm_roi.sum()} voxels")
            
            if gm_roi.sum() > 0:
                save_verification_image(
                    gm_roi,
                    f'{hemisphere} {subregion_name} GM ROI',
                    f'step3_gmroi_{hemisphere}_{subregion_name}.png',
                    cmap='Blues'
                )
    
    return gm_masks, gm_mask


# ============================================================================
# STEP 4: EXTRACT SURFACE MESHES
# ============================================================================

def step4_extract_meshes(gm_masks):
    """Extract surface meshes for each subregion."""
    print_step(4, "Extract Surface Meshes")
    
    meshes = {}
    for hemisphere in ['left', 'right']:
        meshes[hemisphere] = {}
        for subregion_name, gm_roi in gm_masks[hemisphere].items():
            if gm_roi.sum() < 100:
                print(f"  {hemisphere} {subregion_name}: SKIPPED (too small)")
                continue
            
            try:
                # Extract surface
                verts, faces, normals, values = measure.marching_cubes(
                    gm_roi, level=0.5, method='lorensen'
                )
                
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                meshes[hemisphere][subregion_name] = mesh
                
                print(f"  {hemisphere} {subregion_name}: {len(verts)} vertices")
                print(f"    Bounds X: {verts[:,0].min():.1f} to {verts[:,0].max():.1f}")
                print(f"    Bounds Y: {verts[:,1].min():.1f} to {verts[:,1].max():.1f}")
                print(f"    Bounds Z: {verts[:,2].min():.1f} to {verts[:,2].max():.1f}")
                
            except Exception as e:
                print(f"  {hemisphere} {subregion_name}: FAILED - {e}")
                continue
    
    return meshes


# ============================================================================
# STEP 5: CREATE FLATMAP PROJECTIONS
# ============================================================================

def step5_create_flatmaps(meshes):
    """Create flatmap projections for each subregion mesh."""
    print_step(5, "Create Flatmap Projections")
    
    flatmap_gen = InsulaFlatmapNMT(nmt_template_dir=NMT_DIR)
    flatmaps = {}
    
    for hemisphere in ['left', 'right']:
        flatmaps[hemisphere] = {}
        for subregion_name, mesh in meshes[hemisphere].items():
            try:
                # Find anchor point
                anchor_idx = flatmap_gen.find_anchor_point(mesh, method='anterior_superior')
                
                # Create projection
                coords_2d = flatmap_gen.create_flatmap_projection(mesh, anchor_idx)
                
                flatmaps[hemisphere][subregion_name] = {
                    'mesh': mesh,
                    'coords_2d': coords_2d,
                    'anchor_idx': anchor_idx,
                    'color': SUBREGIONS[hemisphere][subregion_name]['color'],
                    'label': SUBREGIONS[hemisphere][subregion_name]['label']
                }
                
                print(f"  {hemisphere} {subregion_name}: {len(coords_2d)} vertices in flatmap")
                
                # Plot flatmap for verification
                fig, ax = plt.subplots(figsize=(8, 8))
                points = np.array(list(coords_2d.values()))
                ax.scatter(points[:, 0], points[:, 1], c='blue', s=1, alpha=0.5)
                ax.set_title(f'{hemisphere} {subregion_name} Flatmap')
                ax.set_aspect('equal')
                plt.tight_layout()
                filepath = os.path.join(OUTPUT_DIR, f'step5_flatmap_{hemisphere}_{subregion_name}.png')
                plt.savefig(filepath, dpi=150)
                plt.close()
                print(f"    Saved: {filepath}")
                
            except Exception as e:
                print(f"  {hemisphere} {subregion_name}: FAILED - {e}")
                continue
    
    return flatmap_gen, flatmaps


# ============================================================================
# STEP 6: LOAD AND CLASSIFY NEURONS
# ============================================================================

def step6_load_neurons():
    """Load neurons and classify by subregion."""
    print_step(6, "Load and Classify Neurons")
    
    df = pd.read_excel(INS_DF_PATH)
    print(f"  Loaded {len(df)} neurons from {INS_DF_PATH}")
    
    # Determine hemisphere and subregion for each neuron
    def classify_neuron(row):
        region = str(row['Soma_Region'])
        
        # Determine hemisphere
        if region.startswith('CL_') or region.startswith('SL_'):
            hemisphere = 'left'
        elif region.startswith('CR_') or region.startswith('SR_'):
            hemisphere = 'right'
        else:
            return 'unknown', 'unknown'
        
        # Determine subregion
        for subregion_name, info in SUBREGIONS[hemisphere].items():
            if region in info['regions']:
                return hemisphere, subregion_name
        
        # Fallback
        if 'Ia' in region or 'Id' in region:
            return hemisphere, 'agranular_posterior'
        elif 'Ig' in region:
            return hemisphere, 'granular'
        elif 'Ri' in region:
            return hemisphere, 'retroinsula'
        
        return hemisphere, 'unknown'
    
    classifications = df.apply(classify_neuron, axis=1, result_type='expand')
    df['hemisphere'] = classifications[0]
    df['subregion'] = classifications[1]
    
    print(f"  Classification:")
    print(df['subregion'].value_counts())
    
    return df


# ============================================================================
# STEP 7: EXTRACT SOMA COORDINATES
# ============================================================================

def step7_extract_somas(neurons_df, max_neurons=20):
    """Extract soma coordinates from SWC files."""
    print_step(7, f"Extract Soma Coordinates (limit: {max_neurons})")
    
    soma_data = []
    neuron_list = neurons_df.head(max_neurons) if max_neurons else neurons_df
    
    for idx, row in neuron_list.iterrows():
        neuron_id = row['NeuronID']
        sample = row.get('SampleID', '251637')
        
        try:
            neuron = nt.neuro_tracer()
            neuron.process(str(sample), str(neuron_id), nii_space='monkey')
            soma = neuron.root
            
            soma_data.append({
                'NeuronID': neuron_id,
                'SampleID': sample,
                'Neuron_Type': row['Neuron_Type'],
                'hemisphere': row['hemisphere'],
                'subregion': row['subregion'],
                'soma_x': soma.x_nii,
                'soma_y': soma.y_nii,
                'soma_z': soma.z_nii,
            })
            
        except Exception as e:
            print(f"    Failed to load {neuron_id}: {e}")
            continue
    
    soma_df = pd.DataFrame(soma_data)
    print(f"  Extracted {len(soma_df)} soma coordinates")
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'step7_soma_coordinates.csv')
    soma_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    return soma_df


# ============================================================================
# STEP 8: CREATE FINAL FLATMAP FIGURE
# ============================================================================

def step8_create_figure(flatmap_gen, flatmaps, soma_df):
    """Create final publication-quality figure."""
    print_step(8, "Create Final Figure")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    for ax_idx, hemisphere in enumerate(['left', 'right']):
        ax = axes[ax_idx]
        
        # Plot each subregion
        for subregion_name, params in flatmaps.get(hemisphere, {}).items():
            coords_2d = params['coords_2d']
            color = params['color']
            label = params['label']
            
            if len(coords_2d) < 3:
                continue
            
            points = np.array(list(coords_2d.values()))
            
            # Create boundary using convex hull
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                
                # Fill subregion
                polygon = Polygon(hull_points, closed=True, 
                                 facecolor=color, edgecolor='black',
                                 linewidth=2, alpha=0.5, label=label)
                ax.add_patch(polygon)
                
            except Exception as e:
                print(f"    Warning: Could not create boundary for {subregion_name}: {e}")
                ax.scatter(points[:, 0], points[:, 1], c=color, s=10, alpha=0.5)
        
        # Plot neurons
        hemi_neurons = soma_df[soma_df['hemisphere'] == hemisphere]
        print(f"  Plotting {len(hemi_neurons)} neurons for {hemisphere} hemisphere")
        
        for idx, row in hemi_neurons.iterrows():
            # Find the subregion flatmap for this neuron
            subregion = row['subregion']
            if subregion not in flatmaps.get(hemisphere, {}):
                continue
            
            params = flatmaps[hemisphere][subregion]
            mesh = params['mesh']
            coords_2d = params['coords_2d']
            
            # Map to flatmap
            soma_voxel = np.array([row['soma_x'], row['soma_y'], row['soma_z']])
            flat_coord = flatmap_gen.map_point_to_flatmap(soma_voxel, mesh, coords_2d)
            
            if flat_coord:
                n_type = row['Neuron_Type']
                color = NEURON_TYPE_COLORS.get(n_type, 'gray')
                marker = NEURON_TYPE_MARKERS.get(n_type, 'o')
                
                ax.scatter(flat_coord[0], flat_coord[1],
                          c=color, s=150, alpha=0.9,
                          marker=marker, edgecolors='black', linewidth=1.5,
                          zorder=10)
        
        # Styling
        ax.set_xlabel('Medial-Lateral', fontsize=12)
        ax.set_ylabel('Anterior-Posterior', fontsize=12)
        ax.set_title(f'{hemisphere.capitalize()} Insula\n(n={len(hemi_neurons)} neurons)',
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Legend for subregions
        handles = []
        for subregion_name, params in flatmaps.get(hemisphere, {}).items():
            patch = Patch(facecolor=params['color'], edgecolor='black',
                         label=params['label'], alpha=0.5)
            handles.append(patch)
        ax.legend(handles=handles, loc='upper left', fontsize=9)
    
    plt.suptitle('Monkey Insula Flatmap with Subregion Boundaries\n(Gao et al. 2025 Style)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(OUTPUT_DIR, 'step8_final_flatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    
    # Also save SVG
    svg_path = os.path.join(OUTPUT_DIR, 'step8_final_flatmap.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"  Saved: {svg_path}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all steps."""
    print("\n" + "="*70)
    print("STEP-BY-STEP INSULA FLATMAP GENERATION")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Execute steps
    arm_data, subregion_indices = step1_load_atlas()
    masks = step2_create_masks(arm_data, subregion_indices)
    gm_masks, gm_mask = step3_combine_with_gm(masks)
    meshes = step4_extract_meshes(gm_masks)
    flatmap_gen, flatmaps = step5_create_flatmaps(meshes)
    neurons_df = step6_load_neurons()
    soma_df = step7_extract_somas(neurons_df, max_neurons=20)
    step8_create_figure(flatmap_gen, flatmaps, soma_df)
    
    print("\n" + "="*70)
    print("ALL STEPS COMPLETE!")
    print(f"Check output in: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
