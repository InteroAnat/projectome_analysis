import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
import joblib
from skimage import measure

# Configuration
NMT_DIR = r"D:\projectome_analysis\atlas\NMT_v2.1_sym\NMT_v2.1_sym"
OUTPUT_DIR = r".\flatmap_outputs"
SOMA_CSV = r'D:\projectome_analysis\main_scripts\neuron_tables\INS_df_v3.xlsx'

TYPE_COLORS = {
    'PT': '#d62728', 'CT': '#2ca02c', 
    'ITs': '#9467bd', 'ITc': '#e377c2', 'ITi': '#17becf',
    'Unclassified': 'gray'
}

def plot_full_scene():
    print("Generating Accurate Aspect Ratio Plot...")
    
    # 1. Brain Surface
    seg_path = os.path.join(NMT_DIR, "NMT_v2.1_sym_segmentation.nii.gz")
    img = nib.load(seg_path)
    data = img.get_fdata()
    brain_mask = data > 0
    verts, _, _, _ = measure.marching_cubes(brain_mask, level=0.5, step_size=3)
    
    # 2. Load Insula
    try:
        left_mesh = joblib.load(os.path.join(OUTPUT_DIR, 'insula_flatmap_left.pkl'))['mesh']
        right_mesh = joblib.load(os.path.join(OUTPUT_DIR, 'insula_flatmap_right.pkl'))['mesh']
    except:
        print("Error loading insula meshes.")
        return

    soma_df = pd.read_excel(SOMA_CSV)

    # --- PLOT ---
    fig = plt.figure(figsize=(12, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    # A. Brain Outline (Black, Sparse)
    ax.scatter(verts[:,0], verts[:,1], verts[:,2],
               c='black', s=0.2, alpha=0.1, label='Brain Surface')

    # B. Insula Meshes
    if left_mesh:
        v = left_mesh.vertices[::3]
        ax.scatter(v[:,0], v[:,1], v[:,2], c='#3498DB', s=3, alpha=0.05)
    if right_mesh:
        v = right_mesh.vertices[::3]
        ax.scatter(v[:,0], v[:,1], v[:,2], c='#E74C3C', s=3, alpha=0.05)

    # C. Neurons
    for n_type, color in TYPE_COLORS.items():
        subset = soma_df[soma_df['Neuron_Type'] == n_type]
        if not subset.empty:
            ax.scatter(
                subset['Soma_NII_X'], subset['Soma_NII_Y'], subset['Soma_NII_Z'],
                c=color, s=10, alpha=1.0, edgecolors='black', linewidth=1, label=n_type
            )

    # --- CRITICAL FIX: FORCING ASPECT RATIO ---
    # Calculate the data range for all axes
    x_range = verts[:,0].max() - verts[:,0].min()
    y_range = verts[:,1].max() - verts[:,1].min()
    z_range = verts[:,2].max() - verts[:,2].min()
    
    # Set the 'box_aspect' to match the physical dimensions of the brain
    # This prevents Matplotlib from stretching short axes to fill the square box
    ax.set_box_aspect((x_range, y_range, z_range))
    
    # Hide Axes
    ax.set_axis_off()
    
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title("3D Insula Projectome (Accurate Ratio)", fontsize=16)
    
    ax.view_init(elev=30, azim=60)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_full_scene()