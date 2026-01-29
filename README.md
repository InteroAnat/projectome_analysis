# Projectome Data Analysis

A comprehensive toolkit for analyzing macaque brain neuron morphology data, including visualization, clustering, and distance analysis workflows.

## Overview

This repository contains tools for processing and analyzing fMOST (fluorescence Micro-Optical Sectioning Tomography) neuron data, with a focus on:

- **Neuron Visualization**: High and low-resolution visualization of neuron morphology
- **Clustering Analysis**: FNT (Functional Neuroanatomy Toolbox) distance-based clustering
- **Region Analysis**: Anatomical region-based neuron classification
- **Data Conversion**: SWC to FNT format conversion and processing

## Quick Start

### Prerequisites

1. Clone the repository:
   ```bash
   git clone https://github.com/InteroAnat/projectome_analysis.git
   cd projectome_analysis
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f deb_fmost.yml
   conda activate deb_fmost
   ```

3. Install additional dependencies (if needed):
   ```bash
   pip install paramiko nibabel tifffile matplotlib numpy
   ```

4. Ensure `neuron-vis` module is available:
   The code depends on `IONData` from the `neuron-vis/neuronVis` package.
   Make sure this directory exists and is in the Python path.

## Main Components

### 1. Visual Toolkit (`main_scripts/Visual_toolkit.py`)

A unified tool for retrieving and visualizing Macaque brain data from mixed sources.

**Features:**
- **High Resolution (0.65µm)**: Block-based data acquisition via HTTP
- **Low Resolution (5.0µm)**: Slice-based data acquisition via SSH
- **SWC Overlay**: Overlay neuron traces on anatomical images
- **Export Formats**: NIfTI (.nii.gz) and TIFF (.tif) support

**Usage:**
```python
# Run from project root directory:
# cd /path/to/projectome_analysis
# python -c "
import sys
sys.path.insert(0, 'main_scripts')
from Visual_toolkit import Visual_toolkit

toolkit = Visual_toolkit('251637')

# Get high-resolution soma block
volume, origin, resolution = toolkit.get_high_res_block(
    center_um=[18000, 18000, 1000], 
    grid_radius=2
)

# Get low-resolution wide field
volume, origin, resolution = toolkit.get_low_res_widefield(
    center_um=[18000, 18000, 1000],
    width_um=8000,
    height_um=8000,
    depth_um=30
)

toolkit.close()
# "
```

Or directly inside main_scripts/:
```python
from Visual_toolkit import Visual_toolkit
toolkit = Visual_toolkit('251637')
# ... use toolkit ...
toolkit.close()
```

### 2. Visual Toolkit GUI (`main_scripts/Visual_toolkit_gui.py`)

Interactive GUI for the Visual Toolkit.

**Launch:**
```bash
python main_scripts/Visual_toolkit_gui.py
```

**Features:**
- Auto-fill soma coordinates from neuron trees
- Interactive parameter adjustment
- Threaded processing with progress indicators
- Separate or combined high/low resolution processing

### 3. FNT Distance Analysis Tools

Tools for converting SWC files to FNT format and calculating distance matrices.

| Tool | Description |
|------|-------------|
| `convert_swc_to_fnt_decimate.py` | Convert SWC to FNT with decimation (in root dir) |
| `join_fnt_decimate_files.py` | Join multiple FNT files (in root dir) |
| `fnt_distance_workflow.py` | Complete workflow from SWC to distance matrix (in root dir) |
| `fnt_tools_adapter.py` | Adapter for FNT tool integration (in root dir) |

**Complete Workflow:**
```bash
python fnt_distance_workflow.py \
    --input_dir /path/to/swc/files \
    --output_dir /path/to/output \
    --decimate_distance 5000 \
    --decimate_angle 5000
```

Note: Run from project root directory where the script is located.

### 4. Clustering Analysis (`main_scripts/fnt_dist_clustering.py`)

Distance-based clustering of neurons using FNT distance matrices.

### 5. Region Analysis (`main_scripts/region_analysis.py`)

Anatomical region-based analysis of neuron projections.

## Directory Structure

```
projectome_analysis/
├── main_scripts/              # Core analysis scripts
│   ├── Visual_toolkit.py      # Main visualization toolkit
│   ├── Visual_toolkit_gui.py  # GUI for visualization
│   ├── fnt_dist_clustering.py # Clustering algorithms
│   ├── fnt_tools.py           # FNT utility functions
│   ├── monkey_936.py          # Macaque 936 region analysis
│   ├── region_analysis.py     # Region-based analysis
│   ├── tiff_ds2m.py           # TIFF processing
│   ├── volume2.py             # Volume processing
│   └── subsidary_functions/   # Helper functions
│
├── fnt_dist_on_cluster/       # HPC cluster job scripts
├── insula_macaque_results/    # Analysis results
├── soma_detection_unet/       # U-Net soma detection
├── atlas/                     # Atlas data
├── literature/                # Reference papers
├── resource/                  # Output data (gitignored)
├── processed_neurons/         # Processed neuron files (gitignored)
├── deb_fmost.yml              # Conda environment
└── README.md                  # This file
```

## Configuration

### SSH Configuration (for Low-Res Data)

Edit `main_scripts/Visual_toolkit.py` to configure SSH access:
```python
SSH_HOST = "your.server.ip"
SSH_PORT = 22
SSH_USER = "username"
SSH_PASS = "password"  # Consider using environment variables
SSH_REMOTE_BASE = "/path/to/resampled/data"
```

### HTTP Configuration (for High-Res Data)

Default configuration:
```python
HTTP_HOST = 'http://bap.cebsit.ac.cn'
HTTP_PATH = 'monkeydata'
```

## Common Workflows

### 1. Visualize a Single Neuron

```python
# Add paths and import (run from project root)
import sys
sys.path.insert(0, 'main_scripts')
sys.path.insert(0, 'neuron-vis/neuronVis')

from Visual_toolkit import Visual_toolkit
import IONData as IT

toolkit = Visual_toolkit('251637')
ion = IT.IONData()

# Load neuron
tree = ion.getRawNeuronTreeByID('251637', '003.swc')
soma_xyz = [tree.root.x, tree.root.y, tree.root.z]

# Get and plot wide field context
volume, origin, resolution = toolkit.get_low_res_widefield(
    soma_xyz, width_um=8000, height_um=8000, depth_um=30
)
toolkit.plot_widefield_context(volume, origin, resolution, 
                               soma_xyz, '003.swc', swc_tree=tree)
toolkit.close()
```

### 2. Batch Process Neurons for Clustering

```bash
# Convert all SWC files
python convert_swc_to_fnt_decimate.py \
    --input_dir processed_neurons/251637 \
    --output_dir fnt_output/251637 \
    --workers 8

# Join FNT files
python join_fnt_decimate_files.py \
    --input_dir fnt_output/251637 \
    --output_file joined_fnt/251637-004-merge.fnt

# Calculate distance matrix (on cluster)
sbatch fnt_dist_on_cluster/fnt_dist.slurm
```

### 3. Run Clustering Analysis

```python
import sys
sys.path.insert(0, 'main_scripts')
from fnt_dist_clustering import run_clustering

run_clustering(
    distance_matrix_file='dist.txt',
    n_clusters=5,
    output_dir='clustering_results/'
)
```

Note: The `run_clustering` function may need to be called from within the script or imported depending on the current implementation. Check `fnt_dist_clustering.py` for the latest API.

## Output Files

### Visualization Outputs
- `.nii.gz` - 3D volume files (NIfTI format)
- `.tif` - 2D maximum intensity projection images
- `_Plot.png` - Annotated visualization plots

### Analysis Outputs
- `.fnt` - FNT format neuron files
- `_clusters.csv` - Cluster assignment results
- `dist.txt` - Distance matrices

## Notes

- **Data directories** (`resource/`, `processed_neurons/`) are gitignored due to large file sizes
- **Cache files** (`__pycache__/`, `.ipynb_checkpoints/`) are excluded from version control
- **SSH credentials** should be moved to environment variables for security

## Troubleshooting

### Issue: SSH connection fails
**Solution:** Check network connectivity and SSH credentials in `Visual_toolkit.py`

### Issue: HTTP blocks not downloading
**Solution:** Verify the server URL and check if the sample ID exists

### Issue: FNT tools not found
**Solution:** Install FNT toolkit and ensure it's in your PATH

### Issue: Memory errors with large volumes
**Solution:** Reduce grid_radius or field-of-view dimensions

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

[Add your license information here]

## Contact

[Add contact information here]

---

**Last Updated:** January 2026
