# Monkey Atlas Visualization Guide

This guide explains how to create custom monkey brain atlas visualizations using the `subcortex_visualization` package.

---

## 📋 Overview

The `subcortex_visualization` package provides tools for creating 2D vector graphics from 3D brain atlases. While it includes 9 human atlases out-of-the-box, you can create custom monkey atlases using the workflow described below.

---

## 🗂️ Repository Structure

```
subcortex_visualization/
├── README.md                           # Main package documentation
├── tutorial.ipynb                      # Python tutorial (human atlases)
├── custom_segmentation/                # Custom atlas creation tools
│   ├── volume_to_mesh_mz3.py         # Convert NIfTI to 3D mesh
│   ├── niiAtlas2mesh.py              # Alternative mesh converter
│   ├── combinemz3.py                 # Combine mesh files
│   ├── generate_color_map.py         # Create color schemes
│   └── README.md                     # Detailed pipeline guide
├── subcortex_visualization/            # Python package
│   ├── plotting.py                   # Main plotting functions
│   ├── segmentation.py               # Atlas application
│   ├── data/                         # Atlas SVG files (human)
│   └── atlases/                      # NIfTI volumes (human)
├── subcortexVisualizationR/          # R package
└── monkey_atlas_guide/               # ⭐ THIS FOLDER
    ├── README.md                     # This guide
    ├── volume_to_mesh_mz3.py         # Copy of mesh converter
    └── CUSTOM_SEGMENTATION_GUIDE.md  # Original detailed guide
```

---

## 🚀 Quick Start: Creating a Monkey Atlas

### Prerequisites

| Software | Purpose | Download |
|----------|---------|----------|
| **niimath** | Mesh generation from volumes | https://github.com/rordenlab/niimath/releases |
| **Surf Ice** | 3D mesh visualization | https://github.com/neurolabusc/surf-ice |
| **Inkscape** | Vector graphic editing | https://inkscape.org/ |
| Python packages | `pip install nibabel numpy tifffile` | - |

---

## 📖 Step-by-Step Workflow

### **Step 1: Prepare Your Monkey Atlas**

You need a volumetric segmentation of the monkey brain:

```
Input: your_monkey_atlas.nii.gz
  - 3D NIfTI volume
  - Integer labels (1, 2, 3... = different regions)
  - Same space as your neuron data
```

**Example regions for insula visualization:**
- Area 25 (subgenual cingulate)
- Area 24 (ventral anterior cingulate)
- Area 32 (prelimbic)
- Claustrum

---

### **Step 2: Convert Volume to 3D Mesh**

Choose the appropriate script based on your atlas type:

#### **Option A: Simple 3D Atlas (no hierarchies)**

For atlases with a single volume (standard 3D NIfTI):

```bash
python volume_to_mesh_mz3.py \
    --input_volume /path/to/your_monkey_atlas.nii.gz \
    --output_path ./mesh_outputs/ \
    --out_file monkey_atlas.mz3 \
    --index_max 16 \
    --colors colors.txt \
    --delete_mz3
```

#### **Option B: Multi-Hierarchy Atlas (e.g., ARM atlas)** ⭐

For monkey atlases like **ARM** that have multiple hierarchy levels (5D NIfTI):

```bash
python volume_to_mesh_mz3_MONKEY.py \
    --input_volume ARM_in_NMT_v2.1_sym.nii.gz \
    --hierarchy_level 6 \
    --output_path ./mesh_outputs/ \
    --out_file ARM_h6.mz3 \
    --colors colors.txt \
    --delete_mz3
```

**Hierarchy Level Guide (ARM atlas):**

| Level | Regions | Description |
|-------|---------|-------------|
| 0 | 66 | Coarsest (large regions) |
| 1 | 70 | Very coarse |
| 2 | 70 | Coarse |
| 3 | 72 | Medium-coarse |
| 4 | 74 | Medium |
| 5 | 72 | Medium-fine |
| **6** | **72** | **Standard hierarchy 6** ⭐ |

**Note:** The output directory will be created automatically if it doesn't exist.

**Parameters explained:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input_volume` | Input NIfTI file | `insula_atlas.nii.gz` |
| `--output_path` | Working directory | `./mesh_outputs/` |
| `--out_file` | Final mesh filename | `monkey_atlas.mz3` |
| `--index_max` | Maximum region index | 16 (for 16 regions) |
| `--colors` | Color scheme file | `colors.txt` |
| `--delete_mz3` | Clean intermediate files | - |

**Generate custom colors:**

```python
import numpy as np
import matplotlib

# Sample 16 colors from plasma colormap
cmap = matplotlib.colormaps.get_cmap('plasma')
colors = cmap(np.linspace(0, 1, 16))
rgb_colors = (colors[:, :3] * 256).astype(int)
np.savetxt('colors.txt', rgb_colors, fmt='%d')
```

---

### **Step 3: Visualize in Surf Ice**

1. **Launch Surf Ice**
2. **Open mesh**: `File > Open` → select `monkey_atlas.mz3`
3. **Adjust view**: Rotate to show lateral/medial surfaces
4. **Take screenshots**: Save PNG images for both views

**Tips:**
- Use mouse to rotate (left-click + drag)
- Zoom with scroll wheel
- Save screenshot: `File > Save Bitmap`

---

### **Step 4: Create SVG Outlines (Inkscape)**

#### 4.1 Setup
- Open Inkscape
- Import your Surf Ice screenshot: `File > Import`

#### 4.2 Trace Regions
- Select **Freehand Lines** tool (F6)
- Set **Smoothing**: 20-25 in top toolbar
- Trace around each region carefully

#### 4.3 Label Regions (Critical!)

For each traced region:
1. Select the path
2. Open `Object > Object Properties`
3. Set **Title** format: `regionname_face_hemisphere`

**Examples:**
```
insula_lateral_L
insula_medial_L
putamen_lateral_L
caudate_medial_R
```

#### 4.4 Save Files

Create 3 SVG files:
```
monkey_atlas_L.svg    # Left hemisphere
monkey_atlas_R.svg    # Right hemisphere  
monkey_atlas_both.svg # Combined (copy L+R, flip R)
```

---

### **Step 5: Create Lookup Tables**

Create CSV files defining plot order:

**`monkey_atlas_L_ordering.csv`:**
```csv
region,face,plot_order,Hemisphere
insula,lateral,1,L
putamen,lateral,2,L
caudate,lateral,3,L
...
```

**`monkey_atlas_R_ordering.csv`:**
```csv
region,face,plot_order,Hemisphere
insula,lateral,1,R
putamen,lateral,2,R
caudate,lateral,3,R
...
```

**`monkey_atlas_both_ordering.csv`:**
```csv
region,face,plot_order,Hemisphere
insula,lateral,1,L
insula,lateral,2,R
putamen,lateral,3,L
putamen,lateral,4,R
...
```

**Note:** `plot_order` controls layering (1 = bottom, higher = top)

---

### **Step 6: Install Custom Atlas**

Copy your files to the package data directory:

```bash
# Copy SVG files
cp monkey_atlas_*.svg \
   subcortex_visualization/subcortex_visualization/data/

# Copy lookup tables
cp monkey_atlas_*_ordering.csv \
   subcortex_visualization/subcortex_visualization/data/

# Reinstall package
cd subcortex_visualization
pip install .
```

---

### **Step 7: Use in Your Code**

```python
from subcortex_visualization import plot_subcortical_data
import pandas as pd
import matplotlib.pyplot as plt

# Prepare your monkey neuron data
monkey_data = pd.DataFrame({
    "region": ["insula", "putamen", "caudate", "claustrum"],
    "value": [1.5, 2.3, 0.8, 1.2],
    "Hemisphere": ["L", "L", "L", "L"]
})

# Create visualization
plot_subcortical_data(
    subcortex_data=monkey_data,
    atlas='monkey_atlas',      # Your custom atlas name
    hemisphere='L',            # 'L', 'R', or 'both'
    value_column='value',
    cmap='plasma',
    fill_title='Projection Strength',
    line_color='black',
    line_thickness=1.5
)

plt.show()
```

---

## 📊 Example: Visualizing Insula Neurons

```python
import pandas as pd
from subcortex_visualization import plot_subcortical_data

# Load your insula neuron analysis results
insula_df = pd.read_excel('251637_INS_results.xlsx')

# Aggregate by region
region_data = insula_df.groupby('Soma_Region').agg({
    'Total_Length': 'mean'
}).reset_index()
region_data.columns = ['region', 'value']
region_data['Hemisphere'] = 'L'  # or extract from data

# Plot
plot_subcortical_data(
    subcortex_data=region_data,
    atlas='monkey_atlas',
    hemisphere='L',
    cmap='viridis',
    fill_title='Mean Axon Length (µm)'
)
```

---

## 🔧 Troubleshooting

### Issue: `niimath` not found
```bash
# Download niimath binary
wget https://github.com/rordenlab/niimath/releases/latest/download/niimath_linux.zip
unzip niimath_linux.zip
sudo mv niimath /usr/local/bin/
```

### Issue: Black/empty plots
- Check that region names in data match SVG titles exactly
- Verify lookup table CSV format
- Ensure SVG titles use format: `region_face_hemisphere`

### Issue: Mesh download fails
- Verify `niimath` is in PATH
- Check input NIfTI file is valid
- Ensure output directory exists

---

## 📚 References

1. **Original Package**: https://github.com/anniegbryant/subcortex_visualization
2. **Surf Ice**: https://github.com/neurolabusc/surf-ice
3. **niimath**: https://github.com/rordenlab/niimath
4. **Detailed Pipeline Guide**: See `CUSTOM_SEGMENTATION_GUIDE.md` in this folder

---

## 📁 File Checklist

Before installing your custom atlas, ensure you have:

- [ ] `monkey_atlas_L.svg` - Left hemisphere outlines
- [ ] `monkey_atlas_R.svg` - Right hemisphere outlines
- [ ] `monkey_atlas_both.svg` - Combined view
- [ ] `monkey_atlas_L_ordering.csv` - Left lookup table
- [ ] `monkey_atlas_R_ordering.csv` - Right lookup table
- [ ] `monkey_atlas_both_ordering.csv` - Combined lookup table
- [ ] All SVG region titles follow format: `regionname_face_hemisphere`
- [ ] All CSV files have correct column headers

---

## 📝 Citation

If you use this package for monkey atlas visualization:

```
Bryant, Annie G. (2026). Subcortex visualization: A toolbox for custom 
data visualization in the subcortex and cerebellum. bioRxiv, 2026-01. 
doi:10.64898/2026.01.23.699785
```

---

**Last Updated**: 2026-03-20
**Maintainer**: Projectome Analysis Team
