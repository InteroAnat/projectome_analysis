# Brain Visualization Pipeline (IDE Version)

IDE-friendly pipeline scripts for generating brain region meshes and neuron visualization. **No CLI arguments needed** - just edit parameters and run.

## Scripts

### 1. `step3.2.pipeline_simple.py` - Quick Start (Recommended)

Simplest pipeline for most use cases. Edit parameters at the top, then run in your IDE.

**Usage:**
1. Open the script in your IDE
2. Edit the `PARAMETERS` section
3. Run the script
4. PNG saved to `OUTPUT_DIR`

**Key Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `REGIONS` | Dictionary of regions to visualize | See examples in file |
| `REGION_COLORS` | Colors for each region | See examples in file |
| `SOMA_SIZE` | Neuron marker size | `50` |
| `LEGEND_MARKERSCALE` | Legend marker size | `2.5` |
| `OUTPUT_NAME` | Output filename | `"brain_viz.png"` |
| `PLOT_TITLE` | Plot title | `"3D Brain Projectome"` |
| `SHOW_PLOT` | Show plot window | `False` |

---

### 2. `step3.2.pipeline_mesh_viz.py` - Full Featured

Advanced pipeline with more customization options including per-type soma sizes.

**Additional Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `SOMA_SIZES_BY_TYPE` | Custom sizes per neuron type | `{}` |

---

## Customization Guide

### Define Regions

Edit the `REGIONS` dictionary with ARM abbreviations from `ARM_key_all.txt`:

```python
REGIONS = {
    # Example: Striatum
    "striatum": ["Cpu"],
    
    # Example: Thalamic nuclei
    "thalamus_l": ["CL_VPL", "CL_VPM"],
    "thalamus_r": ["CR_VPL", "CR_VPM"],
    
    # Example: Cortical areas
    "motor_cortex": ["M1"],
    "somatosensory_cortex": ["S1"],
}
```

### Set Region Colors

```python
REGION_COLORS = {
    "striatum": "#E74C3C",
    "thalamus_l": "#3498DB",
    "thalamus_r": "#2980B9",
    "motor_cortex": "#2ECC71",
    "somatosensory_cortex": "#9B59B6",
}
```

### Change Soma (Neuron) Size

**Uniform size for all neurons:**
```python
SOMA_SIZE = 50   # Default size
SOMA_SIZE = 80   # Larger neurons
SOMA_SIZE = 30   # Smaller neurons
```

**Different sizes per neuron type:**
```python
SOMA_SIZES_BY_TYPE = {
    'PT': 80,    # Pyramidal tract - larger
    'CT': 60,    # Corticothalamic - medium
    'ITs': 40,   # Intratelencephalic - smaller
}
```

### Change Legend Marker Size

```python
LEGEND_MARKERSCALE = 1.0  # Default size
LEGEND_MARKERSCALE = 2.0  # 2x larger
LEGEND_MARKERSCALE = 2.5  # 2.5x larger (recommended)
LEGEND_MARKERSCALE = 3.0  # 3x larger
```

### Change Soma Colors

```python
SOMA_COLORS = {
    'PT': '#d62728',   # Red
    'CT': '#2ca02c',   # Green
    'ITs': '#9467bd',  # Purple
    'ITc': '#e377c2',  # Pink
    'ITi': '#17becf',  # Cyan
    'Unclassified': 'gray'
}
```

### Change View Angle

```python
VIEW_ELEV = 30   # Camera elevation (vertical)
VIEW_AZIM = 60   # Camera azimuth (horizontal rotation)
```

### Show Plot Window

```python
SHOW_PLOT = True   # Show matplotlib window after generation
SHOW_PLOT = False  # Just save PNG (default)
```

---

## Output

- **PNG file** saved to `OUTPUT_DIR` with 300 DPI
- **High-quality** rendering with tight bounding box
- **Enlarged legend markers** (2.5x by default)

---

## Finding Region Abbreviations

Region abbreviations are defined in `ARM_key_all.txt`. Example format:
```
Index    Abbreviation    Full_Name    Last_Level
1    Cpu    Caudate Putamen    6
2    CL_Ig    Insula Granular Left    6
3    CR_Ig    Insula Granular Right    6
```

Use the `Abbreviation` column values in your `REGIONS` dictionary.

---

## Troubleshooting

### "Missing required files" error
Check that these paths are correct:
- `NMT_DIR`: Path to NMT template directory
- `ARM_ATLAS`: Path to ARM atlas NIfTI file
- `ARM_KEY`: Path to ARM key text file
- `NEURON_TABLE`: Path to neuron table (Excel or CSV)

### No regions extracted
- Check region abbreviations in `ARM_key_all.txt`
- Some regions may not exist at level 6
- Try different abbreviations

### Legend markers too small/large
Adjust `LEGEND_MARKERSCALE` in the parameters section.

### Neurons not visible
Increase `SOMA_SIZE` parameter.
