"""
Visual_toolkit.py - Macaque Brain Hybrid-Resolution Visualization Toolkit

Version: 1.3.0 (MIP Visualization Fix)

UPDATE NOTES (v1.3.0):
    - Changed `plot_soma_block` from single-slice view to Maximum Intensity Projection (MIP).
      This ensures the soma is visible even if it is not perfectly centered in the Z-stack.
"""

import os
import sys
import numpy as np
import urllib.request
import tifffile
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import paramiko
from matplotlib.colors import LinearSegmentedColormap

# Ensure headless environments don't crash
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

# --- PATHS ---
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)

import IONData as IT

# --- CONFIGURATION ---

# 1. HIGH RESOLUTION (HTTP Source) - 0.65um
HTTP_HOST = 'http://bap.cebsit.ac.cn'
HTTP_PATH = 'monkeydata'
BLOCK_SIZE_PIXELS = [360, 360, 90]    # [X, Y, Z]
RESOLUTION_HIGH   = [0.65, 0.65, 3.0] # [X, Y, Z] microns

# 2. LOW RESOLUTION (SSH Source) - 5.0um Resampled
SSH_HOST = "172.20.10.250"
SSH_PORT = 20007
SSH_USER = "binbin"
SSH_PASS = "IONconnect2026"
SSH_REMOTE_BASE = "/home/binbin/share/251637CH1_projection/251637-CH1_resample/resample_5um"
RESOLUTION_LOW  = [5.0, 5.0, 3.0]     # [X, Y, Z] microns

class Visual_toolkit:
    """
    A unified tool for retrieving Macaque brain data from mixed sources.
    """
    def __init__(self, sample_id='251637'):
        self.sample_id = sample_id
        project_root = os.path.dirname(os.getcwd())
        
        self.output_dir = os.path.join(project_root, 'resource', 'segmented_cubes', sample_id)
        self.cache_http_dir = os.path.join(project_root, 'resource', 'cubes', sample_id, 'high_res_http')
        self.cache_ssh_dir  = os.path.join(project_root, 'resource', 'cubes', sample_id, 'low_res_ssh')
        
        for folder in [self.output_dir, self.cache_http_dir, self.cache_ssh_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.ssh_client = None
        self.sftp = None

    def _init_ssh(self):
        if self.sftp: return
        try:
            print(f"[INFO] Connecting to SSH ({SSH_HOST})...")
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=10)
            self.sftp = self.ssh_client.open_sftp()
            print("  > Connection Successful.")
        except Exception as e:
            print(f"  > [ERROR] SSH Connection Failed: {e}")

    def close(self):
        if self.sftp: self.sftp.close()
        if self.ssh_client: self.ssh_client.close()
        print("[INFO] Connections closed.")

    # ==========================================
    # SOURCE 1: HIGH RES (HTTP)
    # ==========================================
    def _download_http_block(self, idx_x, idx_y, idx_z):
        filename = f"{idx_x}_{idx_y}_{idx_z}.tif"
        local_path = os.path.join(self.cache_http_dir, str(idx_z), filename)
        
        if os.path.exists(local_path):
            try: return tifffile.imread(local_path)
            except: os.remove(local_path)

        url = f"{HTTP_HOST}/{HTTP_PATH}/{self.sample_id}/cube/{idx_z}/{filename}"
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            with open(local_path, 'wb') as f:
                f.write(data)
            return tifffile.imread(local_path)
        except: return None

    def get_high_res_block(self, center_um, grid_radius=2):
        print(f"\n[ACTION] Acquiring High-Res Soma Block (Radius {grid_radius})")
        
        bx = int((center_um[0] / RESOLUTION_HIGH[0]) // BLOCK_SIZE_PIXELS[0])
        by = int((center_um[1] / RESOLUTION_HIGH[1]) // BLOCK_SIZE_PIXELS[1])
        bz = int((center_um[2] / RESOLUTION_HIGH[2]) // BLOCK_SIZE_PIXELS[2])
        
        grid_dim = (grid_radius - 1) * 2 + 1
        volume = np.zeros((
            BLOCK_SIZE_PIXELS[2] * grid_dim, 
            BLOCK_SIZE_PIXELS[1] * grid_dim, 
            BLOCK_SIZE_PIXELS[0] * grid_dim
        ), dtype=np.uint16)
        
        count = 0; total = grid_dim**3
        
        for k in range(1 - grid_radius, grid_radius):
            for j in range(1 - grid_radius, grid_radius):
                for i in range(1 - grid_radius, grid_radius):
                    block = self._download_http_block(bx+i, by+j, bz+k)
                    
                    if block is not None:
                        arr_z = k + grid_radius - 1
                        arr_y = j + grid_radius - 1
                        arr_x = i + grid_radius - 1
                        
                        z_start = arr_z * BLOCK_SIZE_PIXELS[2]
                        y_start = arr_y * BLOCK_SIZE_PIXELS[1]
                        x_start = arr_x * BLOCK_SIZE_PIXELS[0]
                        
                        volume[
                            z_start : z_start + BLOCK_SIZE_PIXELS[2],
                            y_start : y_start + BLOCK_SIZE_PIXELS[1],
                            x_start : x_start + BLOCK_SIZE_PIXELS[0]
                        ] = block
                    
                    count += 1
                    print(f"\r  > Downloading Block {count}/{total}", end="")
        
        print("\n  > High-Res Volume Built.")
        
        origin_x = (bx + 1 - grid_radius) * BLOCK_SIZE_PIXELS[0] * RESOLUTION_HIGH[0]
        origin_y = (by + 1 - grid_radius) * BLOCK_SIZE_PIXELS[1] * RESOLUTION_HIGH[1]
        origin_z = (bz + 1 - grid_radius) * BLOCK_SIZE_PIXELS[2] * RESOLUTION_HIGH[2]
        
        return volume, [origin_x, origin_y, origin_z], RESOLUTION_HIGH

    # ==========================================
    # SOURCE 2: LOW RES (SSH)
    # ==========================================
    def _download_ssh_slice(self, z_index):
        self._init_ssh()
        if not self.sftp: return None
        
        filename = f"{self.sample_id}_{z_index:05d}_CH1_resample.tif"
        local_path = os.path.join(self.cache_ssh_dir, filename)
        
        if os.path.exists(local_path): return local_path
        
        try:
            remote_path = f"{SSH_REMOTE_BASE}/{filename}"
            self.sftp.get(remote_path, local_path)
            return local_path
        except: return None

    def get_low_res_widefield(self, center_um, width_um=10000, height_um=10000, depth_um=90):
        print(f"\n[ACTION] Acquiring Low-Res Wide Field ({width_um}x{height_um} um)")
        
        z_idx = int(center_um[2] / RESOLUTION_LOW[2])
        cx_px = int(center_um[0] / RESOLUTION_LOW[0])
        cy_px = int(center_um[1] / RESOLUTION_LOW[1])
        
        rad_x = int((width_um / RESOLUTION_LOW[0]) / 2)
        rad_y = int((height_um / RESOLUTION_LOW[1]) / 2)
        z_slices = max(1, int(round(depth_um / RESOLUTION_LOW[2])))
        
        min_x = max(0, cx_px - rad_x); max_x = cx_px + rad_x
        min_y = max(0, cy_px - rad_y); max_y = cy_px + rad_y
        
        z_start = max(1, z_idx - z_slices//2)
        z_end = z_idx + z_slices//2
        
        stack = []
        print(f"  > Fetching Z-Slices: {z_start} to {z_end}...")
        
        for z in range(z_start, z_end + 1):
            f_path = self._download_ssh_slice(z)
            if f_path:
                img = tifffile.imread(f_path)
                h, w = img.shape
                my = min(max_y, h); mx = min(max_x, w)
                crop = img[min_y:my, min_x:mx]
                
                th, tw = max_y - min_y, max_x - min_x
                if crop.shape != (th, tw):
                    crop = np.pad(crop, ((0, th-crop.shape[0]), (0, tw-crop.shape[1])), mode='constant')
                stack.append(crop)
            else:
                stack.append(np.zeros((max_y-min_y, max_x-min_x), dtype=np.uint16))

        volume = np.array(stack)
        
        origin_x = min_x * RESOLUTION_LOW[0]
        origin_y = min_y * RESOLUTION_LOW[1]
        origin_z = z_start * RESOLUTION_LOW[2]
        
        return volume, [origin_x, origin_y, origin_z], RESOLUTION_LOW

    # ==========================================
    # EXPORT & VISUALIZATION (UPDATED)
    # ==========================================
    def export_data(self, volume, origin, resolution, neuron_id, suffix="Volume", output_dir=None):
        target_dir = output_dir if output_dir else self.output_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        filename = f"{self.sample_id}_{neuron_id}_{suffix}"
        ext = ".nii.gz" if volume.ndim == 3 else ".tif"
        full_path = os.path.join(target_dir, filename + ext)
        
        print(f"  > Exporting: {full_path}")

        if ".nii" in ext:
            vol_xyz = np.transpose(volume, (2, 1, 0))
            affine = np.eye(4)
            affine[0,0] = resolution[0]
            affine[1,1] = -resolution[1] 
            affine[:3,3] = origin
            nib.save(nib.Nifti1Image(vol_xyz, affine), full_path)
        else:
            if volume.ndim == 3: volume = np.max(volume, axis=0) 
            tifffile.imwrite(full_path, volume)

    def plot_soma_block(self, volume_3d, origin, resolution, soma_coords, neuron_id, suffix="SomaBlock", output_dir=None):
        """
        Plots Grayscale Anatomy (High Res) using MIP.
        """
        print(f"  > Generating Grayscale Plot ({suffix})...")
        
        # --- FIXED: Use MIP (Maximum Intensity Projection) instead of Middle Slice ---
        # This collapses the Z-stack to ensure the soma is seen regardless of Z-position.
        img = np.max(volume_3d, axis=0)
        
        # Local Soma Pixel
        sx = (soma_coords[0] - origin[0]) / resolution[0]
        sy = (soma_coords[1] - origin[1]) / resolution[1]
        
        # Contrast (Gamma 0.5)
        p1, p99 = np.percentile(img, 0.5), np.percentile(img, 99.5)
        # Avoid division by zero if image is empty
        if p99 > p1:
            img_norm = np.clip((img - p1)/(p99-p1), 0, 1)
        else:
            img_norm = img
        
        img_final = np.power(img_norm, 0.5)
        
        self._save_plot(img_final, sx, sy, resolution, neuron_id, suffix, volume_3d.shape[0], 
                       cmap='gray', marker_color='cyan', output_dir=output_dir, scale_bar_um=30)

    def plot_widefield_context(self, volume_3d, origin, resolution, soma_coords, neuron_id, 
                             suffix="WideField", manual_threshold=100, bg_intensity=0.4, 
                             swc_tree=None, output_dir=None):
        """
        Plots Green Intensity on Dark Background + SWC Overlay using MIP.
        """
        print(f"  > Generating Composite Plot ({suffix})...")
        
        # Uses MIP by default
        mip = np.max(volume_3d, axis=0).astype(float)
        h, w = mip.shape
        
        # 1. Base Gray Layer
        d_max = np.max(mip)
        if d_max < manual_threshold: manual_threshold = d_max * 0.5
        p1, p99 = np.percentile(mip, 1), np.percentile(mip, 99.5)
        norm_base = np.clip((mip - p1) / (p99 - p1), 0, 1) * bg_intensity
        
        rgb = np.zeros((h, w, 3), dtype=float)
        rgb[..., 0] = norm_base; rgb[..., 1] = norm_base; rgb[..., 2] = norm_base
        
        # 2. Green Signal Layer
        mask = mip > manual_threshold
        if np.sum(mask) > 0:
            sig = mip[mask]
            s_min, s_max = np.min(sig), np.max(sig)
            if s_max > s_min:
                s_norm = (sig - s_min) / (s_max - s_min)
            else:
                s_norm = np.zeros_like(sig)
                
            bright = np.clip(bg_intensity + 0.2 + (0.4 * s_norm), 0, 1)
            
            rgb[mask, 0] *= 0.2; rgb[mask, 2] *= 0.2
            rgb[mask, 1] = bright

        # 3. Setup Figure
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black', dpi=100)
        ax.imshow(rgb, origin='upper')
        
        if swc_tree:
            self._overlay_swc(ax, swc_tree, origin, resolution, h, w)

        sx = (soma_coords[0] - origin[0]) / resolution[0]
        sy = (soma_coords[1] - origin[1]) / resolution[1]
        
        self._finalize_plot(fig, ax, sx, sy, resolution, neuron_id, suffix, volume_3d.shape[0], 
                           marker_color='white', output_dir=output_dir, scale_bar_um=500)

    def _overlay_swc(self, ax, swc_tree, origin, resolution, h, w):
        if swc_tree:
            print("Overlaying SWC Edges...")
        for edge in swc_tree.edges:
            path_x = []
            path_y = []
            for p in edge.data:
                lx = (p.x - origin[0]) / resolution[0]
                ly = (p.y - origin[1]) / resolution[1]
                if 0 <= lx < w and 0 <= ly < h:
                    path_x.append(lx)
                    path_y.append(ly)
            if path_x and path_y:
                ax.plot(path_x, path_y, color='red', linewidth=0.5, alpha=0.5)

    def _save_plot(self, img_data, sx, sy, resolution, neuron_id, suffix, z_slices, cmap, marker_color, output_dir=None, scale_bar_um=30):
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.imshow(img_data, cmap=cmap, origin='upper')
        self._finalize_plot(fig, ax, sx, sy, resolution, neuron_id, suffix, z_slices, marker_color, output_dir, scale_bar_um)

    def _finalize_plot(self, fig, ax, sx, sy, resolution, neuron_id, suffix, z_slices, marker_color, output_dir=None, scale_bar_um=30):
        h, w = ax.images[0].get_size()
        
        target_dir = output_dir if output_dir else self.output_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        marker_size_um = 20 if resolution[0] < 1.0 else 200
        marker_size_px = marker_size_um / resolution[0]
        marker_s = (marker_size_px * 0.8) ** 2
        
        ax.scatter(sx, sy, s=marker_s, marker='s', facecolors='none', edgecolors=marker_color, 
                   linewidth=0.75, label='Soma', zorder=10)
        
        label_offset_px = marker_size_px 
        ax.text(sx + label_offset_px, sy + label_offset_px, 'Target Soma', 
                color=marker_color, fontsize=9, fontweight='bold')

        bar_px = scale_bar_um / resolution[0]
        bx = w - bar_px - 50; by = h - 50
        ax.plot([bx, bx+bar_px], [by, by], color='white', linewidth=3)
        ax.text(bx+bar_px/2, by-20, f"{scale_bar_um} µm", color='white', ha='center', fontweight='bold')
        
        title = f"{self.sample_id} | {neuron_id} | {suffix}\nFOV: {w*resolution[0]:.0f}x{h*resolution[1]:.0f} µm | Depth: {z_slices*resolution[2]:.0f} µm"
        ax.set_title(title, color='white', fontsize=14)
        ax.axis('off')
        
        plot_name = f"{self.sample_id}_{neuron_id}_{suffix}_Plot.png"
        save_path = os.path.join(target_dir, plot_name)
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='black', dpi=600)
        print(f"  > Plot saved: {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    toolkit = Visual_toolkit('251637')
    ion = IT.IONData()
    NEURON = '003.swc'
    tree = ion.getRawNeuronTreeByID('251637', NEURON)
    
    if tree:
        try: soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
        except: soma_xyz = tree.root.xyz
        print(f"Target Soma: {soma_xyz}")
        
        # Test 1: High Res Soma Block (Uses new MIP logic)
        high_res_volume, high_res_origin, high_res_resolution = toolkit.get_high_res_block(soma_xyz, grid_radius=1)
        toolkit.plot_soma_block(high_res_volume, high_res_origin, high_res_resolution, soma_xyz, NEURON)
        
        # Test 2: Low Res Wide Field (Uses existing MIP logic)
        low_res_volume, low_res_origin, low_res_resolution = toolkit.get_low_res_widefield(soma_xyz, width_um=8000, height_um=8000, depth_um=30)
        toolkit.plot_widefield_context(low_res_volume, low_res_origin, low_res_resolution, soma_xyz, NEURON, bg_intensity=2.0, swc_tree=tree)
        
    toolkit.close()