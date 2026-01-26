import os
import sys
import numpy as np
import urllib.request
import tifffile
import nrrd
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# --- PATHS ---
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)

import IONData as IT

# --- CONFIG ---
SERVER_URL = 'http://bap.cebsit.ac.cn'
DATA_PATH = 'monkeydata'
BLOCK_SIZE = [360, 360, 90]    # [X, Y, Z]
RESOLUTION = [0.65, 0.65, 3.0] # [X, Y, Z] microns

class FNTCubeVis:
    def __init__(self, sample_id):
        self.sample_id = sample_id
        parent_dir = os.path.dirname(os.getcwd())
        
        # 1. Cache (Downloads)
        self.cache_dir = os.path.join(parent_dir, 'resource', 'cubes', sample_id)
        
        # 2. Output (Results)
        self.output_dir = os.path.join(parent_dir, 'resource', 'segmented_cubes', sample_id)
        
        for d in [self.cache_dir, self.output_dir]:
            if not os.path.exists(d): os.makedirs(d)
        
    def _download_block(self, idx_x, idx_y, idx_z):
        filename = f"{idx_x}_{idx_y}_{idx_z}.tif"
        local_path = os.path.join(self.cache_dir, str(idx_z), filename)
        if os.path.exists(local_path):
            try: return tifffile.imread(local_path)
            except: os.remove(local_path) 
        
        url = f"{SERVER_URL}/{DATA_PATH}/{self.sample_id}/cube/{idx_z}/{filename}"
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            with open(local_path, 'wb') as f:
                f.write(data)
            return tifffile.imread(local_path)
        except: return None

    # ==========================================
    # ACQUISITION
    # ==========================================
    def get_data(self, center_um, radius=None, width_um=None, height_um=None, depth_um=None):
        
        # --- MODE A: GRID VOLUME ---
        if radius is not None:
            print(f"--- Acquiring Grid Volume (Radius {radius}) ---")
            cx = int((center_um[0] / RESOLUTION[0]) // BLOCK_SIZE[0])
            cy = int((center_um[1] / RESOLUTION[1]) // BLOCK_SIZE[1])
            cz = int((center_um[2] / RESOLUTION[2]) // BLOCK_SIZE[2])
            
            grid_dim = (radius - 1) * 2 + 1
            vol = np.zeros((BLOCK_SIZE[2]*grid_dim, BLOCK_SIZE[1]*grid_dim, BLOCK_SIZE[0]*grid_dim), dtype=np.uint16)
            
            count = 0; total = grid_dim**3
            for k in range(1-radius, radius):
                for j in range(1-radius, radius):
                    for i in range(1-radius, radius):
                        block = self._download_block(cx+i, cy+j, cz+k)
                        if block is not None:
                            iz, iy, ix = k+radius-1, j+radius-1, i+radius-1
                            vol[iz*BLOCK_SIZE[2]:(iz+1)*BLOCK_SIZE[2], 
                                iy*BLOCK_SIZE[1]:(iy+1)*BLOCK_SIZE[1], 
                                ix*BLOCK_SIZE[0]:(ix+1)*BLOCK_SIZE[0]] = block
                        count += 1
                        print(f"\rBlock {count}/{total}", end="")
            print("\nDone.")
            ox = (cx + 1 - radius) * BLOCK_SIZE[0] * RESOLUTION[0]
            oy = (cy + 1 - radius) * BLOCK_SIZE[1] * RESOLUTION[1]
            oz = (cz + 1 - radius) * BLOCK_SIZE[2] * RESOLUTION[2]
            return vol, [ox, oy, oz]

        # --- MODE B: WIDE FIELD SLAB ---
        elif width_um and height_um:
            d_um = depth_um if depth_um else RESOLUTION[2]
            print(f"--- Acquiring Wide Slab ({int(width_um)}x{int(height_um)} um, Depth={d_um}um) ---")
            
            z_slices = max(1, int(round(d_um / RESOLUTION[2])))
            cx_px = center_um[0] / RESOLUTION[0]; cy_px = center_um[1] / RESOLUTION[1]; cz_px = center_um[2] / RESOLUTION[2]
            hw = (width_um/RESOLUTION[0])/2; hh = (height_um/RESOLUTION[1])/2
            
            min_x = int(cx_px - hw); max_x = int(cx_px + hw)
            min_y = int(cy_px - hh); max_y = int(cy_px + hh)
            
            blk_z = int(cz_px // BLOCK_SIZE[2]); loc_z = int(cz_px % BLOCK_SIZE[2])
            
            z_s = max(0, loc_z - z_slices//2); z_e = min(BLOCK_SIZE[2], loc_z + z_slices//2)
            act_thick = z_e - z_s
            
            s_bx = min_x // BLOCK_SIZE[0]; e_bx = max_x // BLOCK_SIZE[0]
            s_by = min_y // BLOCK_SIZE[1]; e_by = max_y // BLOCK_SIZE[1]
            
            gw = (e_bx - s_bx + 1) * BLOCK_SIZE[0]
            gh = (e_by - s_by + 1) * BLOCK_SIZE[1]
            canvas = np.zeros((act_thick, gh, gw), dtype=np.uint16)
            
            total = (e_bx - s_bx + 1) * (e_by - s_by + 1); count = 0
            for by in range(s_by, e_by + 1):
                for bx in range(s_bx, e_bx + 1):
                    count += 1
                    print(f"\rTile {count}/{total}", end="")
                    block = self._download_block(bx, by, blk_z)
                    if block is not None:
                        try:
                            slab = block[z_s:z_e, :, :]
                            px = (bx - s_bx) * BLOCK_SIZE[0]
                            py = (by - s_by) * BLOCK_SIZE[1]
                            canvas[:, py:py+BLOCK_SIZE[1], px:px+BLOCK_SIZE[0]] = slab
                        except: pass
            print("\nDone.")
            
            crop_x = min_x - (s_bx * BLOCK_SIZE[0])
            crop_y = min_y - (s_by * BLOCK_SIZE[1])
            final_vol = canvas[:, crop_y:crop_y+(max_y-min_y), crop_x:crop_x+(max_x-min_x)]
            
            ox = min_x * RESOLUTION[0]
            oy = min_y * RESOLUTION[1]
            oz = (blk_z * BLOCK_SIZE[2] + z_s) * RESOLUTION[2]
            return final_vol, [ox, oy, oz]
        
        return None, None

    # ==========================================
    # EXPORT & PLOT
    # ==========================================
    def export_data(self, data, origin, center_um, neuron_id, suffix="Volume"):
        """
        Saves to ../resource/segmented_cubes/sampleID/
        Name: Sample_NeuronID_Suffix.nii
        """
        # Default Name: 251637_003.swc_Volume3D.nii
        fname_base = f"{self.sample_id}_{neuron_id}_{suffix}"
        
        # Decide Extension based on data dimension
        # 3D -> .nii (NIfTI)
        # 2D -> .tif (TIFF)
        ext = ".nii" if data.ndim == 3 else ".tif"
        filename = fname_base + ext
        
        full_path = os.path.join(self.output_dir, filename)
        print(f"Exporting to: {full_path}")

        # 1. NIfTI (3D)
        if ext == '.nii':
            # Transpose [Z, Y, X] -> [X, Y, Z]
            vol_xyz = np.transpose(data, (2, 1, 0))
            
            affine = np.eye(4)
            affine[0,0]=RESOLUTION[0]; affine[1,1]=RESOLUTION[1]; affine[2,2]=RESOLUTION[2]
            affine[:3,3] = origin
            nib.save(nib.Nifti1Image(vol_xyz, affine), full_path)

        # 2. TIFF (2D)
        elif ext == '.tif':
            tifffile.imwrite(full_path, data)
            
        print("Save Complete.")
        return fname_base # Return name for plotting usage

    def plot_check(self, data_3d, origin_um, soma_um, neuron_id, title_prefix="Check"):
        print("Generating Plot...")
        
        # 1. Flatten to MIP
        img = np.max(data_3d, axis=0) if data_3d.ndim == 3 else data_3d
        h_px, w_px = img.shape
        z_slices = data_3d.shape[0] if data_3d.ndim == 3 else 1
        
        # 2. Calculate Local Soma Position
        # Local = (Global_Soma - Global_Origin) / Res
        sx = (soma_um[0] - origin_um[0]) / RESOLUTION[0]
        sy = (soma_um[1] - origin_um[1]) / RESOLUTION[1]
        
        # 3. Robust Contrast (Fixes contrast issues between Vol/Mosaic)
        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99.5)
        
        # 4. Plot Setup
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Show Image
        # origin='upper' means index (0,0) is Top-Left
        ax.imshow(img, cmap='gray', vmin=p_low, vmax=p_high, origin='upper')
        
        # Plot Soma
        # Scatter takes (X, Y). With origin='upper', Y counts down from top.
        ax.scatter(sx, sy, c='red', marker='+', s=200, linewidth=2, label='Soma')
        
        # --- CRITICAL FIX: LOCK AXES ---
        # Force the view to exactly the image size
        ax.set_xlim(0, w_px)
        ax.set_ylim(h_px, 0) # Flip 0 to Top
        
        # 5. Scale Bar (500 um)
        bar_um = 500
        bar_px = bar_um / RESOLUTION[0]
        
        # Position: Bottom-Right (Coordinates are High-X, High-Y)
        bx_start = w_px - bar_px - 50
        by_fixed = h_px - 50
        
        ax.plot([bx_start, bx_start + bar_px], [by_fixed, by_fixed], color='white', linewidth=4)
        ax.text(bx_start + bar_px/2, by_fixed - 20, f"{bar_um} µm", color='white', ha='center', fontweight='bold')

        # Titles
        w_um = w_px * RESOLUTION[0]
        h_um = h_px * RESOLUTION[1]
        z_um = z_slices * RESOLUTION[2]
        
        title_str = (
            f"Sample {self.sample_id} | {neuron_id} | {title_prefix}\n"
            f"FOV: {w_um:.0f} x {h_um:.0f} µm | Depth: {z_um:.0f} µm"
        )
        
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.axis('off')
        ax.legend(loc='upper left') # Moved legend to not block scale bar
        
        # Save
        plot_name = f"{self.sample_id}_{neuron_id}_{title_prefix.replace(' ', '')}_Plot.png"
        save_path = os.path.join(self.output_dir, plot_name)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Plot saved: {save_path}")
        plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    cube = FNTCubeVis('251637')
    ion = IT.IONData()
    
    target_neuron = '003.swc'
    tree = ion.getRawNeuronTreeByID('251637', target_neuron)
    xyz = tree.root.xyz if tree else [0,0,0]
    
    print(f"Processing {target_neuron} at {xyz}...")
    
    # 1. VOLUME (Grid) -> Saves NIfTI
    vol, org = cube.get_data(xyz, radius=1)
    cube.export_data(vol, org, xyz, target_neuron, suffix="GridVol")
    cube.plot_check(vol, org, xyz, target_neuron, "GridVol")
    
    # 2. WIDE FIELD (Mosaic) -> Saves NIfTI (if depth>1) or TIFF (if depth=1)
    slab, org_s = cube.get_data(xyz, width_um=1000, height_um=1000, depth_um=60)
    
    # Force 2D Flatten for Tiff export? Or keep 3D Slab?
    # Let's save the 3D Slab as NIfTI
    cube.export_data(slab, org_s, xyz, target_neuron, suffix="WideSlab")
    
    cube.plot_check(slab, org_s, xyz, target_neuron, "WideSlab")
    
    