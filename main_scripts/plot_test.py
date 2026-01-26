from nilearn import plotting
import nibabel as nib
import numpy as np

def visualize_soma_on_atlas(neuron_obj, atlas_img, title="Soma Location"):
    """
    Plots the Atlas slice and overlays the Soma as a crosshair.
    Args:
        neuron_obj: The loaded neuro_tracer object.
        atlas_img: The NIfTI object (nib.load result) of the Atlas.
    """
    # 1. Get Soma Voxel Coordinates (Integers)
    root = neuron_obj.root
    voxel_coords = [root.x_nii, root.y_nii, root.z_nii]
    
    print(f"Soma Voxel: {voxel_coords}")

    # 2. Convert to World Coordinates (Millimeters) for Nilearn
    # We apply the affine matrix of the Atlas to the voxel coordinates
    world_coords = nib.affines.apply_affine(atlas_img.affine, voxel_coords)
    print(f"World Coords (mm): {world_coords}")

    # 3. Plot
    # cut_coords: Where to slice the brain (center on the soma)
    # roi_img: The atlas itself (displayed in color)
    # bg_img: None (black background) or an anatomical MRI if you have one
    plotting.plot_roi(
        roi_img=atlas_img, 
        bg_img=None, 
        cut_coords=world_coords,
        draw_cross=True, # Draws the crosshair at the exact point
        title=f"{title}\n{neuron_obj.swc_filename}",
        cmap='nipy_spectral', # High contrast colormap
        annotate=True
    )
    
    plotting.show()

# --- USAGE EXAMPLE ---
# Assuming 'pop.neurons' is populated
target_neuron = pop.neurons['001.swc'] 

# Load the atlas NIfTI (we need the affine)
atlas_nii = nib.load(r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchy.nii.gz')

# We need a 3D image for plotting, so slice to Level 6
# (X, Y, Z, Unused, Level)
atlas_data_l6 = atlas_nii.get_fdata()[:, :, :, 0, 5]
atlas_l6_img = nib.Nifti1Image(atlas_data_l6, atlas_nii.affine)

visualize_soma_on_atlas(target_neuron, atlas_l6_img)