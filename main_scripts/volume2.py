#%%
import os
import tempfile
import numpy as np
import urllib.request
import sys
import tifffile # Pip install tifffile
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)
import IONData as IT


class FNTCube:
    def __init__(self) -> None:
        self.sampleid = '251637'
        self.server = 'http://bap.cebsit.ac.cn'
        self.path = 'monkeydata'
        
        # Defaults (Updated by phraseCatalog)
        self.pixelSpace = [0.65, 0.65, 3.0] 
        self.cubesize = [360, 360, 90]
        
        self.radiu = 2
        self.volume = None
        self.centerBlockIdx = [] # Stores [100, 113, 100]

    def setSampleID(self, sampleid):
        self.sampleid = sampleid
        self.phraseCatalog()

    def phraseCatalog(self):
        print("--- Parsing Catalog ---")
        
        # 1. Download the catalog text
        url = f"{self.server}/{self.path}/{self.sampleid}/catalog"
        try:
            response = urllib.request.urlopen(url, timeout=5)
            content = bytes.decode(response.read())
            # print(content) # Uncomment to debug
        except Exception as e:
            print(f"Error downloading catalog: {e}")
            # Fallback to the values visible in your text snippet
            self.pixelSpace = [0.65, 0.65, 3.0]
            self.cubesize = [360, 360, 90]
            return

        # 2. Parse the specific sections
        # We prioritize [CH00TIFFM] because you are using TIFFs
        lines = content.splitlines()
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Detect Section Header
            if line.startswith("["):
                current_section = line
                continue
            
            # We only care about the TIFF section
            if current_section == "[CH00TIFFM]":
                
                # Parse Resolution (direction)
                # Format: direction=0.65 0 0 0 0.65 0 0 0 3
                if line.startswith("direction="):
                    parts = line.split('=')[1].strip().split()
                    # The matrix is [X 0 0, 0 Y 0, 0 0 Z]. We take indices 0, 4, 8.
                    res_x = float(parts[0])
                    res_y = float(parts[4])
                    res_z = float(parts[8])
                    self.pixelSpace = [res_x, res_y, res_z]
                    print(f"Found Resolution: {self.pixelSpace}")

                # Parse Block Size (cubesize)
                # Format: cubesize=360 360 90
                elif line.startswith("cubesize="):
                    parts = line.split('=')[1].strip().split()
                    self.cubesize = [int(parts[0]), int(parts[1]), int(parts[2])]
                    print(f"Found CubeSize: {self.cubesize}")

                # Parse Total Image Size
                elif line.startswith("size="):
                    parts = line.split('=')[1].strip().split()
                    self.imagesize = [int(parts[0]), int(parts[1]), int(parts[2])]

    def setMouseCoord(self, x, y, z):
        """
        1. Converts Microns -> Pixels
        2. Converts Pixels -> Block Index (Integer)
        """
        # Calculate GLOBAL Pixel Coordinate
        px = x / self.pixelSpace[0]
        py = y / self.pixelSpace[1]
        pz = z / self.pixelSpace[2]
        
        # Calculate BLOCK Index (Integer Division)
        self.centerBlockIdx = [
            int(px // self.cubesize[0]),
            int(py // self.cubesize[1]),
            int(pz // self.cubesize[2])
        ]
        
        print(f"Targeting Block Index: {self.centerBlockIdx}")

    def getVolumeFromIndex(self, idx_x, idx_y, idx_z):
        """
        Downloads block by direct Integer Index.
        NO coordinate conversion happens here.
        """
        filename = f"{idx_x}_{idx_y}_{idx_z}.tif"
        url = f"{self.server}/{self.path}/{self.sampleid}/cube/{idx_z}/{filename}"
        
        try:
            response = urllib.request.urlopen(url, timeout=15)
            buf = response.read()
        except:
            # Return empty black block if download fails
            return np.zeros((self.cubesize[2], self.cubesize[1], self.cubesize[0]), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as fp:
            fp.write(buf)
            fp.close()
            try:
                self.volume = tifffile.imread(fp.name)
            except:
                self.volume = np.zeros((self.cubesize[2], self.cubesize[1], self.cubesize[0]), dtype=np.uint16)
            os.remove(fp.name)
            return self.volume

    def getVolume(self):
        """
        Stitches the volume using direct Index math.
        """
        # Define grid size based on radius
        # Radius 1 = 1x1x1 (Just center)
        # Radius 2 = 3x3x3 (Center + neighbors)
        range_lim = self.radiu
        grid_dim = (range_lim - 1) * 2 + 1 # e.g. Radius 2 -> 3 blocks wide
        
        # Calculate Total Output Size
        total_z = self.cubesize[2] * grid_dim
        total_y = self.cubesize[1] * grid_dim
        total_x = self.cubesize[0] * grid_dim
        
        print(f"Acquiring {grid_dim}x{grid_dim}x{grid_dim} grid...")
        volume = np.zeros((total_z, total_y, total_x), dtype=np.uint16)
        
        count = 0
        total = grid_dim ** 3
        
        # Iterate relative offsets (e.g., -1, 0, +1)
        for i in range(1 - range_lim, range_lim): # X offset
            for j in range(1 - range_lim, range_lim): # Y offset
                for k in range(1 - range_lim, range_lim): # Z offset
                    
                    # 1. Calculate REAL Block Index (Center + Offset)
                    target_x = self.centerBlockIdx[0] + i
                    target_y = self.centerBlockIdx[1] + j
                    target_z = self.centerBlockIdx[2] + k
                    
                    # 2. Download
                    raw = self.getVolumeFromIndex(target_x, target_y, target_z)
                    
                    # 3. Calculate Paste Position in the Big Array
                    # We map offset (-1, 0, 1) to array index (0, 1, 2)
                    arr_x = i - (1 - range_lim) 
                    arr_y = j - (1 - range_lim)
                    arr_z = k - (1 - range_lim)
                    
                    bz, by, bx = self.cubesize[2], self.cubesize[1], self.cubesize[0]
                    
                    # Paste Logic
                    z_start, z_end = arr_z * bz, (arr_z + 1) * bz
                    y_start, y_end = arr_y * by, (arr_y + 1) * by
                    x_start, x_end = arr_x * bx, (arr_x + 1) * bx
                    
                    volume[z_start:z_end, y_start:y_end, x_start:x_end] = raw
                    
                    count += 1
                    print(f"\rBlock {count}/{total} (Index {target_x},{target_y},{target_z})", end="")
        range_lim = self.radiu
    
        # Calculate the Index of the Top-Left Block
        start_block_x = self.centerBlockIdx[0] + (1 - range_lim)
        start_block_y = self.centerBlockIdx[1] + (1 - range_lim)
        start_block_z = self.centerBlockIdx[2] + (1 - range_lim)
        
        # Convert Block Index -> Pixels -> Microns
        # This is the PHYSICAL location of the corner of your volume in the brain
        self.origin_um = [
            start_block_x * self.cubesize[0] * self.pixelSpace[0],
            start_block_y * self.cubesize[1] * self.pixelSpace[1],
            start_block_z * self.cubesize[2] * self.pixelSpace[2]
        ]
        print(f"\nVolume Origin (um): {self.origin_um}")
        print("\nDone.")
        self.volume = volume
        return volume
    def save_to_nrrd(self, filename):
                import nrrd
                # if any(self.volume): return
                vol = self.volume
                vol = np.transpose(vol,(2,1,0))
                # header = {'spacings': [sp[2], sp[1], sp[0]], 'units': ['microns']*3} # Z, Y, X
                
                output_dir = r'../resource/nrrds/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                print(f"Saving {filename}...")
                nrrd.write(os.path.join(output_dir,filename), vol)
import os
import numpy as np
import urllib.request
import tempfile
import tifffile
import sys

# --- CONFIGURATION FOR YOUR SAMPLE ---
SERVER_URL = 'http://bap.cebsit.ac.cn'
DATA_PATH = 'monkeydata'
SAMPLE_ID = '251637'

# Metadata from your catalog
BLOCK_SIZE = [360, 360, 90]   # [X, Y, Z] pixels
RESOLUTION = [0.65, 0.65, 3.0] # [X, Y, Z] microns

class LargeImageStitcher:
    def __init__(self, sample_id=SAMPLE_ID):
        self.sample_id = sample_id
        
    def _download_block(self, idx_x, idx_y, idx_z):
        """
        Downloads a single 3D block (approx 23MB).
        Returns the 3D numpy array.
        """
        # URL Pattern: .../cube/{Z_INDEX}/{X}_{Y}_{Z}.tif
        filename = f"{idx_x}_{idx_y}_{idx_z}.tif"
        url = f"{SERVER_URL}/{DATA_PATH}/{self.sample_id}/cube/{idx_z}/{filename}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
                
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as fp:
                fp.write(data)
                fp.close()
                # Load the 3D block. Tifffile reads as [Z, Y, X]
                block_data = tifffile.imread(fp.name)
                os.remove(fp.name)
                return block_data
        except Exception as e:
            print(f"    [Warning] Missing Block {idx_x},{idx_y},{idx_z}")
            return None

    def acquire_wide_field(self, center_um, width_um, height_um):
        """
        Stitches a large 2D image centered at 'center_um'.
        
        Args:
            center_um: [x, y, z] in microns
            width_um:  Total width to cover (microns)
            height_um: Total height to cover (microns)
        """
        print(f"--- Starting 2D Mosaic Acquisition ---")
        print(f"Center: {center_um}")
        print(f"Target Size: {width_um} x {height_um} microns")

        # 1. Convert Coordinates to Global Pixels
        # ---------------------------------------
        cx_px = center_um[0] / RESOLUTION[0]
        cy_px = center_um[1] / RESOLUTION[1]
        cz_px = center_um[2] / RESOLUTION[2]
        
        # Calculate boundaries in Pixels
        half_w_px = (width_um / RESOLUTION[0]) / 2
        half_h_px = (height_um / RESOLUTION[1]) / 2
        
        min_x_px = int(cx_px - half_w_px)
        max_x_px = int(cx_px + half_w_px)
        min_y_px = int(cy_px - half_h_px)
        max_y_px = int(cy_px + half_h_px)
        
        # 2. Determine Z-Plane
        # ---------------------------------------
        # Which block index holds this Z?
        block_z_idx = int(cz_px // BLOCK_SIZE[2])
        
        # Which slice INSIDE that block is it? (0 to 89)
        local_z_idx = int(cz_px % BLOCK_SIZE[2])
        
        print(f"Targeting Global Z: {int(cz_px)}")
        print(f" -> Block Z Index: {block_z_idx}")
        print(f" -> Internal Slice: {local_z_idx}")

        # 3. Determine Grid of Blocks
        # ---------------------------------------
        start_blk_x = min_x_px // BLOCK_SIZE[0]
        end_blk_x   = max_x_px // BLOCK_SIZE[0]
        start_blk_y = min_y_px // BLOCK_SIZE[1]
        end_blk_y   = max_y_px // BLOCK_SIZE[1]
        
        # 4. Create Canvas
        # ---------------------------------------
        # We make the canvas large enough to hold all FULL blocks involved
        # We will crop it at the end.
        
        grid_w = (end_blk_x - start_blk_x + 1) * BLOCK_SIZE[0]
        grid_h = (end_blk_y - start_blk_y + 1) * BLOCK_SIZE[1]
        
        print(f"Grid: X[{start_blk_x}-{end_blk_x}] Y[{start_blk_y}-{end_blk_y}]")
        print(f"Canvas Buffer: {grid_w} x {grid_h} pixels ({grid_w*grid_h*2/1024/1024:.1f} MB)")
        
        canvas = np.zeros((grid_h, grid_w), dtype=np.uint16)

        # 5. Download & Stitch Loop
        # ---------------------------------------
        total_blocks = (end_blk_x - start_blk_x + 1) * (end_blk_y - start_blk_y + 1)
        count = 0
        
        for by in range(start_blk_y, end_blk_y + 1):
            for bx in range(start_blk_x, end_blk_x + 1):
                count += 1
                print(f"\rProcessing tile {count}/{total_blocks}...", end="")
                
                # Fetch 3D Block
                block_3d = self._download_block(bx, by, block_z_idx)
                
                if block_3d is not None:
                    # EXTRACT 2D SLICE
                    # Shape is [Z, Y, X]. We pull out [local_z, :, :]
                    try:
                        slice_2d = block_3d[local_z_idx, :, :]
                        
                        # Calculate Paste Position (Relative to Canvas 0,0)
                        paste_x = (bx - start_blk_x) * BLOCK_SIZE[0]
                        paste_y = (by - start_blk_y) * BLOCK_SIZE[1]
                        
                        # Paste
                        h, w = slice_2d.shape
                        canvas[paste_y : paste_y+h, paste_x : paste_x+w] = slice_2d
                    except IndexError:
                        print("  [Error] Z-Index out of bounds for this block.")

        print("\nStitching complete.")

        # 6. Crop to Exact Requested ROI
        # ---------------------------------------
        # Where is the requested top-left corner relative to the canvas?
        crop_x = min_x_px - (start_blk_x * BLOCK_SIZE[0])
        crop_y = min_y_px - (start_blk_y * BLOCK_SIZE[1])
        
        req_w = max_x_px - min_x_px
        req_h = max_y_px - min_y_px
        
        final_image = canvas[crop_y : crop_y+req_h, crop_x : crop_x+req_w]
        
        return final_image

# ==========================================
# RUN IT
# ==========================================

if __name__ == '__main__':
    import vispy.io as io
    import IONData
    sampleID='251637'
    neuronID='003.swc'

    iondata=IT.IONData()
    raw_neuron_tree = iondata.getRawNeuronTreeByID(sampleID,neuronID)
    x,y,z=raw_neuron_tree.root.xyz


    # cube = FNTCube()
    # cube.setSampleID('251637')
    # cube.setMouseCoord(x,y,z) # cortex
    # # cube.setMouseCoord(6220.28,3380.16,6409.23) # th
    # # cube.setMouseCoord(6016.28,3820.16,6315.23) # th
    # # cube.setMouseCoord(3316.73,4630.76,5811.05) # th
    # cube.radiu=1
    # volume =cube.getVolume()
    
    # # cube.setMouseCoord(3459.95,1743.75,6370.94) #mean=137
    # # cube.save_to_nrrd('p.nrrd')
    # volume.transpose((2,1,0))
    
    stitcher = LargeImageStitcher()
    
    # 1. Your Soma Coordinates
    SOMA_X,SOMA_Y,SOMA_Z =raw_neuron_tree.root.xyz
    
    # 2. Define Area Size
    # 5000 microns = 5 mm (Huge area!)
    # Resolution 0.65 -> ~7700 pixels wide
    WIDTH_UM = 5000  
    HEIGHT_UM = 5000 
    
    # 3. Acquire
    large_img = stitcher.acquire_wide_field(
        center_um=[SOMA_X, SOMA_Y, SOMA_Z], 
        width_um=WIDTH_UM, 
        height_um=HEIGHT_UM
    )
    
    # 4. Save Result
    output_file = "Soma_WideField.tif"
    tifffile.imwrite(output_file, large_img)
    print(f"Saved {output_file} ({large_img.shape})")
    
    # 5. Quick View (Downsampled for screen)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    # Show decimated version for speed
    plt.imshow(large_img[::4, ::4], cmap='gray', vmin=20, vmax=3000)
    plt.title("Wide Field View (Downsampled)")
    plt.axis('off')
    plt.show()
#%%
import nrrd
# def register_to_atlas_space(high_res_vol, origin_um, high_res_spacing):
#     """
#     high_res_vol: The numpy array from getVolume()
#     origin_um: The [X, Y, Z] location of the volume's corner in microns
#     high_res_spacing: [0.65, 0.65, 3.0]
#     """
#     # 1. Target Atlas Properties
#     target_shape = (586, 506, 692) # [X, Y, Z] (Usually Z is first in numpy, check orientation!)
#     target_res = [100, 100, 90]    # Microns per pixel
    
#     # Create the empty Atlas Volume
#     # Note: Numpy is usually [Z, Y, X]. Let's assume (692, 506, 586) for safety.
#     atlas_vol = np.zeros(target_shape, dtype=np.float32) 
    
#     print("--- Registering to Atlas Space ---")
    
#     # 2. Iterate through High-Res and place into Low-Res
#     # Since High-Res is huge, we don't loop every pixel. 
#     # We calculate the START and END indices in the Atlas.
    
#     # Calculate Start Index in Atlas
#     start_idx_x = int(origin_um[0] / target_res[0])
#     start_idx_y = int(origin_um[1] / target_res[1])
#     start_idx_z = int(origin_um[2] / target_res[2])
    
#     # Calculate Size in Atlas Units
#     # (HighRes Size * HighRes Res) / Target Res
#     # Note: vol.shape is [Z, Y, X]
#     dim_x, dim_y, dim_z = high_res_vol.shape
    
#     print(dim_x,dim_y,dim_z)
#     size_x_um = dim_x * high_res_spacing[0]
#     size_y_um = dim_y * high_res_spacing[1]
#     size_z_um = dim_z * high_res_spacing[2]
#     print(size_x_um, size_y_um, size_z_um)
#     span_x = int(size_x_um / target_res[0])
#     span_y = int(size_y_um / target_res[1])
#     span_z = int(size_z_um / target_res[2])
    
#     print(f"Placing volume at Atlas Index: [{start_idx_x}, {start_idx_y}, {start_idx_z}]")
#     print(f"Volume spans in Atlas: {span_x} x {span_y} x {span_z} voxels")
    
#     # 3. Downsample High-Res to match Atlas Span
#     # We use Scipy Zoom to shrink the image
#     from scipy.ndimage import zoom
    
#     zoom_factors = [
#         span_z / dim_z,
#         span_y / dim_y,
#         span_x / dim_x
#     ]
#     print(zoom_factors)
#     small_vol = zoom(high_res_vol, zoom_factors, order=1) # Linear interpolation
    
#     # 4. Paste into Atlas
#     # Handle bounds checking (in case we are at the edge)
#     z_s, z_e = start_idx_z, start_idx_z + small_vol.shape[0]
#     y_s, y_e = start_idx_y, start_idx_y + small_vol.shape[1]
#     x_s, x_e = start_idx_x, start_idx_x + small_vol.shape[2]
#     print(z_s,z_e)
#     atlas_vol[z_s:z_e, y_s:y_e, x_s:x_e] = 10000
    
#     return atlas_vol, small_vol

# volume1=volume.transpose((2,1,0))
def register_to_atlas_space_xyz(high_res_vol, origin_um, high_res_spacing, target_shape, target_res):
    """
    Registers a chunk into an Atlas where the array is shaped [X, Y, Z].
    
    Args:
        high_res_vol: Numpy Array. Assumed [Z, Y, X] (Standard Tiff/Numpy load).
        origin_um: [X, Y, Z] microns.
        target_shape: (586, 506, 692) -> [X, Y, Z].
        target_res: [100, 100, 90] -> [X, Y, Z].
    """
    print("--- Registration (XYZ Mode) ---")
    
    # 1. Create Buffer [X, Y, Z]
    atlas_vol = np.zeros(target_shape, dtype=np.float32)
    
    # 2. Transpose Input to match Output
    # Input is [Z, Y, X]. We need [X, Y, Z].
    # Transpose order: (2, 1, 0)
    print("Transposing input [Z,Y,X] -> [X,Y,Z]...")
    vol_xyz = np.transpose(high_res_vol, (2, 1, 0))
    
    # Now input dimensions map directly:
    in_x, in_y, in_z = vol_xyz.shape
    
    # 3. Calculate Start Indices
    start_x = int(origin_um[0] / target_res[0])
    start_y = int(origin_um[1] / target_res[1])
    start_z = int(origin_um[2] / target_res[2])
    
    print(f"Insertion [X,Y,Z]: [{start_x}, {start_y}, {start_z}]")

    # 4. Calculate Span (Output Size)
    phys_x = in_x * high_res_spacing[0]
    phys_y = in_y * high_res_spacing[1]
    phys_z = in_z * high_res_spacing[2]
    
    span_x = max(1, int(phys_x / target_res[0]))
    span_y = max(1, int(phys_y / target_res[1]))
    span_z = max(1, int(phys_z / target_res[2]))
    
    # 5. Downsample (Max Pool)
    from skimage.measure import block_reduce
    
    kx = max(1, int(in_x / span_x))
    ky = max(1, int(in_y / span_y))
    kz = max(1, int(in_z / span_z))
    
    print(f"Pooling Kernel: {kx}x{ky}x{kz}")
    
    # Apply reduction on the [X,Y,Z] array
    small_vol = block_reduce(vol_xyz, block_size=(kx, ky, kz), func=np.max)
    
    # Trim to fit
    small_vol = small_vol[:span_x, :span_y, :span_z]
    
    # 6. Paste
    end_x = min(target_shape[0], start_x + span_x)
    end_y = min(target_shape[1], start_y + span_y)
    end_z = min(target_shape[2], start_z + span_z)
    
    paste_x = end_x - start_x
    paste_y = end_y - start_y
    paste_z = end_z - start_z
    
    if paste_x > 0 and paste_y > 0 and paste_z > 0:
        atlas_vol[start_x:end_x, start_y:end_y, start_z:end_z] = \
            small_vol[:paste_x, :paste_y, :paste_z]
        print("Success.")
    else:
        print("Error: Out of bounds.")

    return atlas_vol


def register_debug(high_res_vol, origin_um):
    print("\n--- DIAGNOSTIC MODE ---")
    
    # 1. INPUT CHECK
    in_max = np.max(high_res_vol)
    print(f"1. Input Volume Max Value: {in_max}")
    if in_max == 0:
        print("   CRITICAL ERROR: Input volume is empty/black!")
        return None

    # 2. COORDINATE CHECK
    # Standard Atlas Size
    ATLAS_RES = [100, 100, 90] # X, Y, Z
    ATLAS_SHAPE = [586,506,692] 
    
    # Calculate Indices
    idx_x = int(origin_um[0] / ATLAS_RES[0])
    idx_y = int(origin_um[1] / ATLAS_RES[1])
    idx_z = int(origin_um[2] / ATLAS_RES[2])
    
    print(f"2. Calculated Start Index: [Z={idx_z}, Y={idx_y}, X={idx_x}]")
    
    # Validate Bounds
    if (0 <= idx_z < ATLAS_SHAPE[0]) and (0 <= idx_y < ATLAS_SHAPE[1]) and (0 <= idx_x < ATLAS_SHAPE[2]):
        print("   Coordinates are VALID (Inside Brain Box).")
    else:
        print(f"   CRITICAL ERROR: Coordinates Out of Bounds! Max is {ATLAS_SHAPE}")
        return None

    # 3. DOWNSAMPLING CHECK
    from skimage.measure import block_reduce
    
    # Force a massive reduction just to see if ANY signal survives
    # We reduce the whole volume to 1 pixel using Max
    global_max = block_reduce(high_res_vol, block_size=high_res_vol.shape, func=np.max)
    print(f"3. Global Max Check: {global_max}")
    
    if global_max == 0:
        print("   Logic Error: Block Reduce failed to capture max value.")
        
    # 4. TEST PASTE
    atlas_vol = np.zeros(ATLAS_SHAPE, dtype=np.uint16)
    
    # Creates a 10x10x10 white box at the target location
    # This proves if the COORDINATES are correct visually
    print("4. Generating Test Cube at target location...")
    
    # Limits
    ez = min(idx_z + 10, ATLAS_SHAPE[0])
    ey = min(idx_y + 10, ATLAS_SHAPE[1])
    ex = min(idx_x + 10, ATLAS_SHAPE[2])
    
    atlas_vol[idx_z:ez, idx_y:ey, idx_x:ex] = 255
    
    print(f"   Test Cube Max: {np.max(atlas_vol)}")
    
    return atlas_vol

# --- RUN IT ---
# Pass your loaded 'volume' and 'cube.origin_um' here
test_result = register_debug(volume, cube.origin_um)
atlas_vo=register_to_atlas_space_xyz(volume,cube.origin_um,cube.pixelSpace,(586,506,692),[100,100,90])
# nrrd.write('x1.nrrd',atlas_vol)
# %%
