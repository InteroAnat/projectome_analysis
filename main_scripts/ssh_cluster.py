import os
import sys
import numpy as np
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
import paramiko
from matplotlib.colors import LinearSegmentedColormap

# --- PATHS ---
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)

import IONData as IT
import neuro_tracer as nt

# --- CONFIG ---
HOST = "172.20.10.250"
PORT = 20007
USER = "binbin"
PASS = "q2DBX4yThL6CVVZ"
REMOTE_BASE = "/home/binbin/share/251637CH1_projection/251637-CH1_resample/resample_5um"

RES_RESAMPLED = [5.0, 5.0, 3.0] # [X, Y, Z] microns

class SSHResampledLoader:
    def __init__(self, sample_id='251637'):
        self.sample_id = sample_id
        
        project_root = os.path.dirname(os.getcwd())
        self.out_dir = os.path.join(project_root, 'resource', 'resampled_crops', sample_id)
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=10)
            self.sftp = self.ssh.open_sftp()
            print(f"Connected to {HOST}")
        except Exception as e:
            print(f"SSH Failed: {e}")
            self.sftp = None

    def close(self):
        if self.sftp: self.sftp.close()
        self.ssh.close()

    def _download_slice(self, z_index):
        if not self.sftp: return None
        filename = f"{self.sample_id}_{z_index:05d}_CH1_resample.tif"
        remote_path = f"{REMOTE_BASE}/{filename}"
        local_path = os.path.join(self.out_dir, 'cache', filename)
        
        if not os.path.exists(os.path.dirname(local_path)): os.makedirs(os.path.dirname(local_path))
        if os.path.exists(local_path): return local_path

        try:
            self.sftp.get(remote_path, local_path)
            return local_path
        except: return None

    def get_crop_stack(self, center_um, crop_size_um=20000, z_thickness=10):
        print(f"--- Processing Crop (Size: {crop_size_um/1000} cm) ---")
        
        z_idx = int(center_um[2] / RES_RESAMPLED[2])
        cx_px = int(center_um[0] / RES_RESAMPLED[0])
        cy_px = int(center_um[1] / RES_RESAMPLED[1])
        
        rad_px = int((crop_size_um / RES_RESAMPLED[0]) / 2)
        min_x = max(0, cx_px - rad_px); max_x = cx_px + rad_px
        min_y = max(0, cy_px - rad_px); max_y = cy_px + rad_px
        
        # Store Crop Origin (Global Pixels) for SWC mapping later
        self.crop_origin_px = [min_x, min_y] 
        self.crop_z_range = [0, 0] # Will set below

        z_start = max(1, z_idx - z_thickness//2)
        z_end = z_idx + z_thickness//2
        self.crop_z_range = [z_start, z_end]
        
        stack = []
        print(f"Acquiring Z={z_start} to {z_end}...")
        
        for z in range(z_start, z_end + 1):
            f_path = self._download_slice(z)
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

        vol = np.array(stack)
        org_x = min_x * RES_RESAMPLED[0]
        org_y = min_y * RES_RESAMPLED[1]
        org_z = z_start * RES_RESAMPLED[2]
        rel_x = cx_px - min_x
        rel_y = cy_px - min_y
        
        return vol, [org_x, org_y, org_z], (rel_x, rel_y)

    def export_volume(self, vol, origin, neuron_id, suffix="Resampled"):
        filename = f"{self.sample_id}_{neuron_id}_{suffix}.nii.gz"
        out_path = os.path.join(self.out_dir, filename)
        vol_xyz = np.transpose(vol, (2, 1, 0)) 
        
        affine = np.eye(4)
        affine[0,0] = RES_RESAMPLED[0]
        affine[1,1] = RES_RESAMPLED[1]
        affine[2,2] = RES_RESAMPLED[2]
        affine[:3,3] = origin
        
        nib.save(nib.Nifti1Image(vol_xyz, affine), out_path)
        print(f"Saved NIfTI: {out_path}")

    def plot_enhanced(self, vol, soma_pos, neuron_id, manual_threshold=100, bg_intensity=0.4, swc_tree=None):
        print(f"Generating Composite Plot (Thresh > {manual_threshold})...")
        
        mip = np.max(vol, axis=0).astype(float)
        h, w = mip.shape
        
        # 1. Base Gray
        d_max = np.max(mip)
        if d_max < manual_threshold: manual_threshold = d_max * 0.5
        p1, p99 = np.percentile(mip, 1), np.percentile(mip, 99.5)
        denom = p99 - p1 if p99 > p1 else 1
        
        norm_base = (mip - p1) / denom
        norm_base = np.clip(norm_base, 0, 1) * bg_intensity
        
        rgb = np.zeros((h, w, 3), dtype=float)
        rgb[..., 0] = norm_base; rgb[..., 1] = norm_base; rgb[..., 2] = norm_base
        
        # 2. Green Signal
        mask = mip > manual_threshold
        if np.sum(mask) > 0:
            signal_vals = mip[mask]
            s_min, s_max = np.min(signal_vals), np.max(signal_vals)
            s_div = (s_max - s_min) if s_max > s_min else 1
            
            signal_norm = (signal_vals - s_min) / s_div
            signal_bright = bg_intensity + 0.2 + (0.4 * signal_norm)
            signal_bright = np.clip(signal_bright, 0, 1)
            
            rgb[mask, 0] *= 0.2; rgb[mask, 2] *= 0.2; rgb[mask, 1] = signal_bright

        # Create Figure with higher internal DPI for display
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black', dpi=100)
        ax.imshow(rgb, origin='upper')
        
        # --- 3. SWC OVERLAY ---
        if swc_tree and hasattr(swc_tree, 'edges'):
            print("  Overlaying SWC Edges...")
            
            soma = swc_tree.root
            gpx_soma = soma.x / RES_RESAMPLED[0]
            gpy_soma = soma.y / RES_RESAMPLED[1]
            
            offset_x = gpx_soma - soma_pos[0]
            offset_y = gpy_soma - soma_pos[1]
            
            for edge in swc_tree.edges:
                path_x = []
                path_y = []
                
                for p in edge.data:
                    lx = (p.x / RES_RESAMPLED[0]) - offset_x
                    ly = (p.y / RES_RESAMPLED[1]) - offset_y
                    path_x.append(lx)
                    path_y.append(ly)
                
                # Check visibility
                visible = False
                for i in range(len(path_x)):
                    if 0 <= path_x[i] < w and 0 <= path_y[i] < h:
                        visible = True
                        break
                
                if visible:
                    # Plot Red Trace
                    ax.plot(path_x, path_y, color='red', linewidth=1.0, alpha=0.7)

                # --- 4. SOMA MARKER (Refined) ---
        sx, sy = soma_pos
        
        # Cyan Cross (Thinner, Smaller)
        ax.scatter(sx, sy, c='cyan',facecolor='none', marker='s', s=300, linewidth=0.5, label='Soma', zorder=10)
        ax.text(sx + 100, sy + 100, 'soma', color='white', ha='center', fontweight='bold', fontsize=8, zorder=11)
        # Scale Bar
        bar_px = 2000 / RES_RESAMPLED[0]
        bx = w - bar_px - 50; by = h - 50
        ax.plot([bx, bx+bar_px], [by, by], color='white', linewidth=3)
        ax.text(bx + bar_px/2, by - 25, "2 mm", color='white', ha='center', fontweight='bold', fontsize=12)
        
        ax.set_title(f"{neuron_id} | Signal > {manual_threshold}", color='white', fontsize=16)
        ax.axis('off')
        
        # Save High Res
        plot_path = os.path.join(self.out_dir, f"{self.sample_id}_{neuron_id}_Composite.png")
        plt.savefig(plot_path, bbox_inches='tight', facecolor='black', dpi=300)
        print(f"High-Res Plot saved: {plot_path}")
        try: plt.show()
        except: pass
# ==========================================
# RUN IT
# ==========================================
if __name__ == "__main__":
    
    SAMPLE = '251637'
    NEURON = '003.swc'
    
    loader = SSHResampledLoader(SAMPLE)
    
    # 1. Load SWC Tree
    ion = IT.IONData()
    tree = ion.getRawNeuronTreeByID(SAMPLE, NEURON)
    
    
    if tree:
        try: soma_um = [tree.root.x, tree.root.y, tree.root.z]
        except: soma_um = tree.root.xyz
        
        # 2. Get Data
        vol, origin, soma_rel = loader.get_crop_stack(soma_um, crop_size_um=8000, z_thickness=10)
        
        if vol is not None:
            loader.export_volume(vol, origin, NEURON)
            
            # 3. Plot with SWC Overlay
            # bg_intensity=0.3 (Dimmer background)
            # swc_tree=tree (Pass the object)
            loader.plot_enhanced(vol, soma_rel, NEURON, bg_intensity=1.2, swc_tree=tree)
            
        loader.close()
    else:
        print("Error: Neuron not found.")