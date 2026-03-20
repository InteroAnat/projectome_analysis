# Bulk Visual Data Output Structure

This document illustrates the folder and file structure created by `bulk_visual_data.py`.

## Overview

```
PARENT_OUTPUT_DIR/                          # e.g., W:\fMOST
└── 251637/                                 # Sample ID
    └── cube_data_251637_INS_20260320/      # Timestamped batch folder
        ├── Region_DI/                      # Soma region group
        │   ├── HighRes/
        │   │   ├── Data/                   # NIfTI volumes (.nii.gz)
        │   │   │   ├── 251637_003.swc_DI_SomaBlock.nii.gz
        │   │   │   ├── 251637_004.swc_DI_SomaBlock.nii.gz
        │   │   │   └── ...
        │   │   └── Plots/                  # High-res MIP plots (.png)
        │   │       ├── 251637_003.swc_DI_SomaBlock_Plot.png
        │   │       ├── 251637_004.swc_DI_SomaBlock_Plot.png
        │   │       └── ...
        │   └── LowRes/
        │       ├── Data/                   # Widefield NIfTI volumes
        │       │   ├── 251637_003.swc_DI_WideField.nii.gz
        │       │   └── ...
        │       └── Plots/                  # Widefield composite plots
        │           ├── 251637_003.swc_DI_WideField_Plot.png
        │           └── ...
        ├── Region_Gu/                      # Another soma region
        │   ├── HighRes/
        │   │   ├── Data/
        │   │   └── Plots/
        │   └── LowRes/
        │       ├── Data/
        │       └── Plots/
        └── Region_.../                     # Other regions
```

## File Naming Convention

### Data Files (NIfTI)
```
{sample_id}_{neuron_id}_{soma_region}_{suffix}.nii.gz
```
- Example: `251637_003.swc_DI_SomaBlock.nii.gz`

### Plot Files (PNG)
```
{sample_id}_{neuron_id}_{soma_region}_{suffix}_Plot.png
```
- Example: `251637_003.swc_DI_SomaBlock_Plot.png`

## Plot Title Format

The plot titles include coordinates for traceability:

```
251637 | 003.swc | SomaBlock
Region: DI | XYZ: (13965, 34488, 20977)
FOV: 234x234 µm | Depth: 270 µm
```

## Shared Cache Structure (GUI + Bulk)

Both GUI and bulk script share the same cache:

```
project_root/                               # D:\projectome_analysis
└── resource/
    └── cubes/
        └── 251637/                         # Sample ID
            ├── high_res_http/              # HTTP downloaded blocks (0.65µm)
            │   ├── 77/
            │   │   ├── 59_147_77.tif
            │   │   ├── 60_147_77.tif
            │   │   └── ...
            │   ├── 78/
            │   └── ...
            └── low_res_ssh/                # SSH downloaded slices (5µm)
                ├── 251637_00001_CH1_resample.tif
                ├── 251637_00002_CH1_resample.tif
                └── ...
```

## How to Generate This Structure

### Method 1: Using `tree` Command (Windows/Linux)

**Windows (PowerShell):**
```powershell
# Install tree if not available
# Using built-in: Get-ChildItem -Recurse | Select-Object FullName

# Or use tree.com if available
tree /F W:\fMOST\251637\cube_data_251637_INS_20260320 > structure.txt
```

**Linux/Mac:**
```bash
# Install tree if needed
sudo apt-get install tree  # Ubuntu/Debian
brew install tree          # macOS

# Generate structure
tree -L 4 /mnt/d/projectome_analysis/resource/cubes/251637

# Save to file
tree -L 4 /mnt/d/projectome_analysis/resource/cubes/251637 > structure.txt

# Show only directories
tree -d -L 4 W:/fMOST/251637/cube_data_251637_INS_20260320
```

### Method 2: Using Python

```python
import os

def print_tree(path, prefix="", max_depth=4, current_depth=0):
    """Print directory tree structure."""
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return
    
    # Separate dirs and files
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
    
    # Print directories
    for i, name in enumerate(dirs):
        is_last = (i == len(dirs) - 1) and not files
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{name}/")
        
        extension = "    " if is_last else "│   "
        print_tree(
            os.path.join(path, name),
            prefix + extension,
            max_depth,
            current_depth + 1
        )
    
    # Print files (limited)
    for i, name in enumerate(files[:5]):  # Limit to 5 files per dir
        is_last = (i == min(4, len(files) - 1))
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{name}")
    
    if len(files) > 5:
        print(f"{prefix}└── ... and {len(files) - 5} more files")

# Usage
print_tree("W:\\fMOST\\251637\\cube_data_251637_INS_20260320")
```

### Method 3: VS Code Extension

Install **"File Tree Generator"** extension in VS Code:
1. Right-click any folder
2. Select "Generate File Tree"
3. Copy the markdown or ASCII output

### Method 4: Online Tools

- **ASCII Tree Generator**: https://ascii-tree-generator.com/
- **Tree.nathanfriend.io**: https://tree.nathanfriend.io/

Paste your folder paths and get formatted output.

## Example Actual Output

```
W:\fMOST\251637\cube_data_251637_INS_20250320\
├── Region_DI\
│   ├── HighRes\
│   │   ├── Data\
│   │   │   ├── 251637_003.swc_DI_SomaBlock.nii.gz
│   │   │   ├── 251637_004.swc_DI_SomaBlock.nii.gz
│   │   │   ├── 251637_007.swc_DI_SomaBlock.nii.gz
│   │   │   └── ... (12 files total)
│   │   └── Plots\
│   │       ├── 251637_003.swc_DI_SomaBlock_Plot.png
│   │       ├── 251637_004.swc_DI_SomaBlock_Plot.png
│   │       └── ... (12 files total)
│   └── LowRes\
│       ├── Data\
│       │   └── ... (12 files)
│       └── Plots\
│           └── ... (12 files)
├── Region_Gu\
│   ├── HighRes\
│   │   ├── Data\
│   │   │   └── ... (8 files)
│   │   └── Plots\
│   │       └── ... (8 files)
│   └── LowRes\
│       ├── Data\
│       └── Plots\
└── Region_Unknown\
    └── ...
```
