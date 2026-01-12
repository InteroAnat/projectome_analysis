import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from skimage import measure
import trimesh
import sys,copy,os,inspect

neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
sys.path.append(neurovis_path)

# Your custom modules
import volume
import IONData 

def CalculateMonkeySoma(sampleid, neuron_id):
    """
    Calculates and visualizes the soma volume for Monkey data.
    """
    
    # ==============================================================================
    # --- [CRITICAL MONKEY PARAMETERS] --- (PAY ATTENTION HERE)
    # ==============================================================================
    
    # 1. RESOLUTION: Check your server catalog. (e.g. 0.325 or 0.5 or 1.0)
    # If this is wrong, your volume calculation will be wrong.
    MONKEY_PIXEL_SPACE = [0.65, 0.65, 3] 
    
    # 2. DOWNLOAD RADIUS: Monkey somas are large. 
    # Radius 4 ensures we don't chop off the edges of the soma.
    DOWNLOAD_RADIUS = 4 
    
    # 3. PROTECTION RATIO: The brush size to protect the neuron from deletion.
    # Mouse = 10. Monkey = 40 (approx 26um diameter protection).
    PROTECTION_RATIO = 40 
    
    # 4. CROP SIZE: The box cut out around the soma center.
    # Needs to be larger than the soma radius (in pixels).
    CROP_WINDOW = 100 
    
    # 5. THRESHOLD: Brightness level (0-65535).
    # If the mesh is empty, LOWER this. If it's too noisy, RAISE it.
    ISO_LEVEL = 160 
    
    # ==============================================================================

    # 1. Initialize Data & Tree
    iondata = IONData.IONData()
    tree = iondata.getRawNeuronTreeByID(sampleid, neuron_id)
    
    cube = volume.FNTCube()
    cube.setSampleID(sampleid)
    cube.pixelSpace = MONKEY_PIXEL_SPACE # <--- Applied here
    cube.path='monkeydata'

    # Use Root from SWC
    print(f"Targeting Soma at: {tree.root.x}, {tree.root.y}, {tree.root.z}")
    cube.setMouseCoord(tree.root.x, tree.root.y, tree.root.z)
    
    # Trigger Download
    cube.radiu = DOWNLOAD_RADIUS # <--- Applied here
    print("Downloading Volume...")
    cube.getVolume() # Fills cube.volume

    # 2. Map SWC segments to Local Box Coordinates
    segments = []
    for edge in tree.edges[:]:
        p = []
        for point in edge.data:
            p.append(point.xyz)
        # Convert Real World -> Local Box
        pos = cube.physicalPoint2LocalPoint(p, True)
        segments.append(pos)

    # 3. Create Protective Mask
    print(f"Creating mask with ratio {PROTECTION_RATIO}...")
    blankarray = np.zeros_like(cube.volume)
    
    for seg in segments:
        for p in seg:
            # Bounds check to prevent crashing
            if (p[0] > PROTECTION_RATIO and p[0] < blankarray.shape[2] - PROTECTION_RATIO and
                p[1] > PROTECTION_RATIO and p[1] < blankarray.shape[1] - PROTECTION_RATIO and
                p[2] > PROTECTION_RATIO and p[2] < blankarray.shape[0] - PROTECTION_RATIO):
                
                # Paint the tube
                z_s, z_e = p[2]-PROTECTION_RATIO, p[2]+PROTECTION_RATIO
                y_s, y_e = p[1]-PROTECTION_RATIO, p[1]+PROTECTION_RATIO
                x_s, x_e = p[0]-PROTECTION_RATIO, p[0]+PROTECTION_RATIO
                
                blankarray[z_s:z_e, y_s:y_e, x_s:x_e] = 25

    # Smooth the mask
    blankarray2 = ndimage.gaussian_filter(blankarray, sigma=5)

    # 4. Apply Mask (Delete Background)
    clean_volume = cube.volume.copy()
    clean_volume[blankarray2 < 1.0] = 0

    # 5. Dynamic Cropping (Fixing the '152' hardcoded issue)
    # Get exact local center of the soma
    center_pos = cube.physicalPoint2LocalPoint([[tree.root.x, tree.root.y, tree.root.z]], True)[0]
    lx, ly, lz = center_pos[0], center_pos[1], center_pos[2]
    
    print(f"Cropping around Local Z-Slice: {lz}")
    
    z_start = max(0, lz - CROP_WINDOW)
    z_end   = min(clean_volume.shape[0], lz + CROP_WINDOW)
    y_start = max(0, ly - CROP_WINDOW)
    y_end   = min(clean_volume.shape[1], ly + CROP_WINDOW)
    x_start = max(0, lx - CROP_WINDOW)
    x_end   = min(clean_volume.shape[2], lx + CROP_WINDOW)

    crop = clean_volume[z_start:z_end, y_start:y_end, x_start:x_end]

    # 6. Generate Mesh
    crop_smooth = ndimage.gaussian_filter(crop.astype(float), sigma=1.5)
    
    try:
        verts, faces, t1, t2 = measure.marching_cubes(
            crop_smooth, 
            level=ISO_LEVEL, 
            spacing=(cube.pixelSpace[2], cube.pixelSpace[1], cube.pixelSpace[0]), 
            method='lorensen'
        )
    except ValueError:
        print("Error: No soma found. Try lowering ISO_LEVEL.")
        return

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # 7. Calculate Volume
    if not mesh.is_watertight:
        vol = mesh.convex_hull.volume
    else:
        vol = mesh.volume

    print(f"\n--- SOMA VOLUME: {vol:.2f} um3 ---")
    
    # 8. Visualize in Notebook (Matplotlib)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    z = mesh.vertices[:, 2]
    
    ax.plot_trisurf(x, y, z, triangles=mesh.faces, cmap='Spectral', alpha=0.6, edgecolor='none')
    
    # Hack for aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax. _xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(f"Monkey Soma Reconstruction\nVol: {vol:.0f} um3")
    plt.show()

# ==========================================================
# Run Example
# ==========================================================
CalculateMonkeySoma('251637', '002.swc')