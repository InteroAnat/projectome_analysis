# Deep Research: subcortex_visualization Package

## 📚 Overview

**subcortex_visualization** is a Python/R toolbox for creating 2D vector graphics visualizations of subcortical and cerebellar brain atlases. It was developed by Annie G. Bryant (2026) to fill a gap in neuroimaging visualization - specifically the lack of tools to visualize subcortical atlases in 2D similar to the popular `ggseg` package.

**Repository**: https://github.com/anniegbryant/subcortex_visualization  
**Citation**: Bryant, A.G. (2026). Subcortex visualization: A toolbox for custom data visualization in the subcortex and cerebellum. *bioRxiv*, doi:10.64898/2026.01.23.699785

---

## 🧠 Core Principles

### 1. **3D-to-2D Transformation Pipeline**

The package follows a clear transformation workflow:

```
Volumetric NIfTI (3D) → Triangulated Mesh (3D) → Vector Graphic (2D)
     (.nii.gz)             (.mz3 format)            (.svg format)
```

**Key Insight**: The 2D representations are NOT simple projections - they are manually traced vector outlines from optimal 3D viewing angles.

### 2. **Two-Stage Visualization Process**

| Stage | Tool | Purpose | Output |
|-------|------|---------|--------|
| **Stage 1** | niimath + Surf Ice | Generate and view 3D meshes | .mz3 mesh files, PNG screenshots |
| **Stage 2** | Inkscape | Trace 2D outlines from screenshots | .svg vector graphics |

### 3. **Data Structure Requirements**

The package requires three coordinated components:

```
Atlas Package Structure:
├── {atlas}_L.svg              # Left hemisphere vector graphic
├── {atlas}_R.svg              # Right hemisphere vector graphic
├── {atlas}_both.svg           # Combined view
├── {atlas}_L_ordering.csv     # Left region ordering/plotting info
├── {atlas}_R_ordering.csv     # Right region ordering/plotting info
└── {atlas}_both_ordering.csv  # Combined ordering info
```

**CSV Format** (ordering files):
```csv
region,face,plot_order,Hemisphere
insula,lateral,1,L
putamen,lateral,2,L
caudate,medial,3,L
...
```

**SVG Title Format** (critical for matching):
```
{region}_{face}_{hemisphere}

Examples:
- insula_lateral_L
- putamen_medial_R
- thalamus_lateral_L
```

---

## 🔧 Technical Architecture

### Core Modules

```python
subcortex_visualization/
├── __init__.py              # Package initialization
├── plotting.py              # Main visualization functions
├── segmentation.py          # Atlas application to functional data
├── utils.py                 # Utility functions
├── atlases/                 # NIfTI atlas volumes (.nii.gz)
│   ├── aseg_subcortex.nii.gz
│   ├── Melbourne_S1_subcortex.nii.gz
│   └── ...
└── data/                    # SVG vector graphics and ordering files
    ├── aseg_L.svg
    ├── aseg_L_ordering.csv
    └── ...
```

### Key Functions

#### `plot_subcortical_data()` - Main Visualization Function

```python
from subcortex_visualization import plot_subcortical_data

plot_subcortical_data(
    subcortex_data=None,       # DataFrame with ['region', 'value', 'Hemisphere']
    atlas='aseg',              # Atlas name
    value_column='value',      # Column to visualize
    hemisphere='L',            # 'L', 'R', or 'both'
    cmap='viridis',            # Colormap
    line_color='black',        # Outline color
    line_thickness=1.5,        # Outline thickness
    fill_title="values",       # Legend title
    vmin=None, vmax=None,      # Color range
    midpoint=None,             # For diverging colormaps
    show_legend=True,
    ax=None                    # Matplotlib axes object
)
```

#### `apply_atlas_to_data()` - Functional Data Extraction

```python
from subcortex_visualization import apply_atlas_to_data

results = apply_atlas_to_data(
    functional_map="brain_data.nii.gz",  # fMRI/PET/etc.
    atlas="aseg",                        # Atlas name(s)
    func_name="My Analysis"
)
# Returns DataFrame with mean signal per region
```

---

## 📊 Built-in Atlases (Human)

| Atlas | Type | Regions (approx.) | Reference |
|-------|------|-------------------|-----------|
| **aseg** | FreeSurfer subcortex | 7-14 | Fischl et al. |
| **Melbourne_S1-S4** | Subcortical hierarchy | 16-128 | Tian et al. (2020) |
| **AICHA** | Subcortical | ~20 | Joliot et al. (2015) |
| **Brainnetome** | Subcortical | ~20 | Fan et al. (2016) |
| **Thalamus_Nuclei_HCP** | Thalamic nuclei | 26 | Najdenovska et al. (2018) |
| **SUIT** | Cerebellar lobules | 17 | Diedrichsen (2006) |

---

## 🐵 Monkey Atlas Adaptation (Your Fork)

Your fork extends this package for **macaque monkey brain atlases**, specifically the **ARM atlas** (A multifaceted Representation of the Monkey brain).

### Monkey-Specific Modifications

**File**: `monkey_atlas_guide/volume_to_mesh_mz3_MONKEY.py`

Key adaptations for monkey atlases:

1. **Multi-hierarchy support**: ARM has 6 hierarchy levels (5D NIfTI: x,y,z,1,6)
2. **Non-sequential indices**: Region IDs like 1258, 1758, 1786 (not 1,2,3...)
3. **Sequential color mapping**: Maps arbitrary region IDs to sequential colors

### Hierarchy Levels (ARM Atlas)

| Level | Regions | Use Case |
|-------|---------|----------|
| 0 | 66 | Coarsest overview |
| 1-2 | 70 | Very coarse grouping |
| 3 | 72 | Medium granularity |
| 4-5 | 72-74 | Fine granularity |
| **6** | **72** | **Standard detailed level** |

---

## 🎯 Application to Your Projectome Data

### Use Case 1: Visualize Soma Distribution by Region

```python
import pandas as pd
from subcortex_visualization import plot_subcortical_data
import matplotlib.pyplot as plt

# Load your neuron data
neuron_df = pd.read_excel('251637_INS.xlsx')

# Aggregate neuron counts by soma region
region_counts = neuron_df.groupby('Soma_Region').size().reset_index()
region_counts.columns = ['region', 'value']
region_counts['Hemisphere'] = 'L'  # or extract from data

# Create visualization
plot_subcortical_data(
    subcortex_data=region_counts,
    atlas='monkey_atlas',  # Your custom atlas
    hemisphere='L',
    cmap='plasma',
    fill_title='Neuron Count',
    line_thickness=1.0
)
```

### Use Case 2: Projection Strength Heatmap

```python
# Aggregate projection metrics
projection_data = neuron_df.groupby('Soma_Region').agg({
    'Total_Projection_Length': 'mean',
    'Terminal_Count': 'sum'
}).reset_index()

projection_data.columns = ['region', 'value', 'terminals']
projection_data['Hemisphere'] = 'L'

# Plot with diverging colormap (centered at mean)
plot_subcortical_data(
    subcortex_data=projection_data,
    atlas='monkey_atlas',
    hemisphere='L',
    cmap='RdBu_r',  # Red-Blue diverging
    midpoint=projection_data['value'].mean(),
    fill_title='Mean Projection Length (μm)'
)
```

### Use Case 3: Multi-metric Comparison

```python
# Create subplots for different metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = [
    ('Total_Length', 'Total Axon Length'),
    ('Terminal_Count', 'Terminal Count'),
    ('Branch_Points', 'Branch Points'),
    ('Soma_Size', 'Soma Size')
]

for ax, (col, title) in zip(axes.flat, metrics):
    data = neuron_df.groupby('Soma_Region')[col].mean().reset_index()
    data.columns = ['region', 'value']
    data['Hemisphere'] = 'L'
    
    plot_subcortical_data(
        subcortex_data=data,
        atlas='monkey_atlas',
        hemisphere='L',
        ax=ax,
        show_legend=False,
        fill_title=title
    )

plt.tight_layout()
plt.savefig('multi_metric_comparison.png', dpi=300)
```

---

## 🔬 Creating Custom Monkey Atlas (Full Pipeline)

### Step 1: Generate 3D Mesh from ARM Atlas

```bash
python main_scripts/mesh_from_atlas.py \
    --atlas /path/to/ARM_in_NMT_v2.1_sym.nii.gz \
    --level 6 \
    --output_dir ./meshes \
    --output_name ARM_mesh.mz3 \
    --min-voxels 100
```

### Step 2: Visualize in Surf Ice

```bash
# Launch Surf Ice
surfice

# Open mesh and adjust view
# Save screenshots: lateral and medial views
```

### Step 3: Create SVG in Inkscape

1. Import screenshot
2. Trace region outlines (Freehand Lines tool, smoothing=20-25)
3. **Label each path**: `Object > Object Properties > Title`
   - Format: `{region}_{face}_{hemisphere}`
   - Example: `insula_lateral_L`, `claustrum_medial_R`
4. Save: `monkey_atlas_L.svg`, `monkey_atlas_R.svg`

### Step 4: Create Ordering CSVs

```csv
# monkey_atlas_L_ordering.csv
region,face,plot_order,Hemisphere
insula,lateral,1,L
claustrum,lateral,2,L
putamen,lateral,3,L
...
```

### Step 5: Install Custom Atlas

```bash
# Copy to package data directory
cp monkey_atlas_*.svg \
   subcortex_visualization/subcortex_visualization/data/

cp monkey_atlas_*_ordering.csv \
   subcortex_visualization/subcortex_visualization/data/

# Reinstall
cd subcortex_visualization
pip install .
```

### Step 6: Use in Analysis

```python
from subcortex_visualization import plot_subcortical_data

plot_subcortical_data(
    subcortex_data=my_data,
    atlas='monkey_atlas',  # Matches filename prefix
    hemisphere='L',
    ...
)
```

---

## 💡 Key Advantages for Your Project

| Feature | Benefit for Projectome Analysis |
|---------|--------------------------------|
| **2D Visualization** | Easy publication-ready figures |
| **Region-level Aggregation** | Summarize 1000s of neurons by region |
| **Flexible Data Input** | Works with any tabular metric |
| **Multiple Atlases** | Compare across different parcellations |
| **Custom Atlas Support** | Can create macaque-specific atlases |
| **Hierarchical Levels** | View data at different granularities |

---

## ⚠️ Limitations & Considerations

1. **Manual SVG Creation**: Requires manual tracing in Inkscape - time-consuming for many regions
2. **2D Simplification**: Loses 3D spatial information
3. **Atlas Dependency**: Requires accurate registration to atlas space
4. **Region Name Matching**: Data region names must exactly match SVG titles
5. **Hemisphere Handling**: Currently assumes lateralized data

---

## 🔗 Integration with Your Existing Pipeline

```
Your Current Workflow:
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ SWC Files   │ → │ IONData      │ → │ Region      │
│ (Neurons)   │   │ (Analysis)   │   │ Analysis    │
└─────────────┘   └──────────────┘   └─────────────┘
                                              ↓
┌─────────────────────────────────────────────┐
│ subcortex_visualization                     │
│ - Aggregate by Soma_Region                  │
│ - Match to atlas regions                    │
│ - Generate 2D visualization                 │
└─────────────────────────────────────────────┘
                                              ↓
                                        ┌─────────────┐
                                        │ Publication │
                                        │ Figures     │
                                        └─────────────┘
```

---

## 📖 Recommended Workflow for Your Data

1. **Preprocessing**:
   - Ensure neuron data has `Soma_Region` column
   - Standardize region names to match atlas
   - Aggregate metrics by region

2. **Visualization**:
   - Use `plot_subcortical_data()` for quick exploration
   - Create multi-panel figures for different metrics
   - Use diverging colormaps for signed data

3. **Publication**:
   - Save as SVG for vector quality
   - Consistent color scales across figures
   - Include region labels in legends

---

## 📚 Additional Resources

- **Main Package**: https://github.com/anniegbryant/subcortex_visualization
- **Documentation**: https://anniegbryant.github.io/subcortex_visualization/
- **niimath**: https://github.com/rordenlab/niimath
- **Surf Ice**: https://github.com/neurolabusc/surf-ice
- **Inkscape**: https://inkscape.org/

---

*Research compiled: 2026-03-20*  
*Package Version: 1.0+*  
*Your Fork: monkey-atlas-guide branch*
