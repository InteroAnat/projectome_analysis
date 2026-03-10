# Visual Toolkit / GUI - Workflow Flowchart

## Overview

The Visual Toolkit provides a hybrid-resolution visualization system for macaque brain data. It downloads high-resolution (0.65µm) and low-resolution (5.0µm) brain images around neuron somas and generates publication-quality plots.

```mermaid
flowchart TB
    subgraph Input["📁 Input"]
        A1["Sample ID<br/>e.g., 251637"]
        A2["Neuron ID<br/>e.g., 003.swc"]
        A3["Soma Coordinates<br/>(X, Y, Z) µm"]
    end

    subgraph GUI["🖥️ Visual Toolkit GUI<br/>Visual_toolkit_gui.py"]
        direction TB
        B1["Launch GUI<br/>NeuroVisGUI"] --> B2["Auto-Load Default Neuron<br/>auto_load_default_neuron()"]
        B2 --> B3["Load & Auto-Fill Soma<br/>load_and_autofill()"]
        B3 --> B4["Extract Soma from SWC<br/>tree.root.x/y/z"]
        B4 --> B5["User Configuration"]
        B5 --> B6["Grid Radius<br/>High-Res"]
        B5 --> B7["FOV Dimensions<br/>Width/Height/Depth"]
        B5 --> B8["Output Directory"]
    end

    subgraph Actions["⚡ Action Buttons"]
        C1["Soma Plot<br/>(High-Res)"] 
        C2["Wide-field Plot<br/>(Low-Res)"]
        C3["Plot Both<br/>(Sequential)"]
    end

    subgraph HighRes["🔍 High-Res Pipeline<br/>0.65µm Resolution"]
        direction TB
        D1["get_high_res_block()"] --> D2["Calculate Block Indices<br/>center_um → block_idx"]
        D2 --> D3["Download HTTP Blocks<br/>_download_http_block()"]
        D3 --> D4["URL: bap.cebsit.ac.cn<br/>Format: {x}_{y}_{z}.tif"]
        D4 --> D5["Assemble 3D Volume<br/>Grid-based stitching"]
        D5 --> D6["export_data()<br/>NIfTI/TIFF"]
        D5 --> D7["plot_soma_block()<br/>MIP Visualization"]
    end

    subgraph LowRes["🌐 Low-Res Pipeline<br/>5.0µm Resolution"]
        direction TB
        E1["get_low_res_widefield()"] --> E2["SSH Connection<br/>_init_ssh()"]
        E2 --> E3["Download TIFF Slices<br/>_download_ssh_slice()"]
        E3 --> E4["Server: 172.20.10.250<br/>Path: resample_5um"]
        E4 --> E5["Crop & Assemble<br/>Center-based FOV"]
        E5 --> E6["Load SWC Tree<br/>IONData.getRawNeuronTreeByID()"]
        E5 --> E7["export_data()<br/>NIfTI/TIFF"]
        E6 --> E8["plot_widefield_context()<br/>MIP + SWC Overlay"]
    end

    subgraph Output["📤 Output Files"]
        F1["{sample}_{neuron}_SomaBlock.nii.gz<br/>High-Res Volume"]
        F2["{sample}_{neuron}_SomaBlock_Plot.png<br/>MIP Grayscale"]
        F3["{sample}_{neuron}_WideField.nii.gz<br/>Low-Res Volume"]
        F4["{sample}_{neuron}_WideField_Plot.png<br/>Green Signal + SWC"]
    end

    %% Connections
    A1 --> GUI
    A2 --> GUI
    GUI --> Actions
    C1 --> HighRes
    C2 --> LowRes
    C3 --> HighRes
    HighRes --> LowRes
    HighRes --> Output
    LowRes --> Output

    %% Styling
    style Input fill:#e1f5fe
    style GUI fill:#fff3e0
    style Actions fill:#fff9c4
    style HighRes fill:#e8f5e9
    style LowRes fill:#fce4ec
    style Output fill:#f3e5f5
```

---

## Component Details

### 1. Visual Toolkit GUI (`Visual_toolkit_gui.py`)

| Component | Method | Purpose |
|-----------|--------|---------|
| Auto-Load | `auto_load_default_neuron()` | Loads default neuron (003.swc) on startup |
| Load Neuron | `load_and_autofill()` | Fetches SWC and extracts soma coordinates |
| Validation | `validate_inputs()` | Ensures numeric coordinates |
| Output Prep | `prepare_output_dir()` | Creates output directory |
| Threading | `threading.Thread()` | Non-blocking operations |

### 2. Visual Toolkit Backend (`Visual_toolkit.py`)

#### High-Resolution (HTTP Source)

| Method | Purpose | Parameters |
|--------|---------|------------|
| `get_high_res_block()` | Main entry point | `center_um`, `grid_radius` |
| `_download_http_block()` | Downloads individual blocks | `idx_x`, `idx_y`, `idx_z` |
| `plot_soma_block()` | MIP grayscale visualization | `volume_3d`, `soma_coords` |

**Configuration:**
- Host: `http://bap.cebsit.ac.cn/monkeydata`
- Block Size: 360×360×90 pixels
- Resolution: 0.65×0.65×3.0 µm
- Grid Radius: 1-3 (default: 1)

#### Low-Resolution (SSH Source)

| Method | Purpose | Parameters |
|--------|---------|------------|
| `get_low_res_widefield()` | Main entry point | `center_um`, `width_um`, `height_um`, `depth_um` |
| `_init_ssh()` | Establishes SSH connection | - |
| `_download_ssh_slice()` | Downloads TIFF slices | `z_index` |
| `plot_widefield_context()` | MIP with SWC overlay | `volume_3d`, `swc_tree` |

**Configuration:**
- Host: `172.20.10.250:20007`
- Base Path: `/home/binbin/share/251637CH1_projection/.../resample_5um`
- Resolution: 5.0×5.0×3.0 µm
- Default FOV: 8000×8000×30 µm

### 3. Data Flow

```
User Input (Sample/Neuron ID)
         │
         ▼
┌─────────────────────┐
│  Auto-Load Neuron   │
│  Load SWC from DB   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Extract Soma XYZ   │ ← tree.root.x/y/z
└─────────────────────┘
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
┌─────────────────────┐            ┌─────────────────────┐
│   HIGH-RES MODE     │            │    LOW-RES MODE     │
│   (Soma Plot)       │            │  (Wide-field Plot)  │
├─────────────────────┤            ├─────────────────────┤
│ • Grid Radius: 1-3  │            │ • FOV: 8000×8000 µm │
│ • HTTP Download     │            │ • SSH Download      │
│ • 0.65 µm/pixel     │            │ • 5.0 µm/pixel      │
│ • 3D Block Assembly │            │ • Z-slice Stack     │
└─────────────────────┘            └─────────────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────┐            ┌─────────────────────┐
│   MIP Grayscale     │            │   MIP Composite     │
│   + Soma Marker     │            │   + SWC Overlay     │
└─────────────────────┘            └─────────────────────┘
```

---

## Execution Flow

### Quick Start

```bash
# Launch GUI
python Visual_toolkit_gui.py

# Or use backend directly
python Visual_toolkit.py
```

### GUI Workflow

1. **Startup** (Automatic)
   - GUI loads with default values
   - Auto-loads neuron 003.swc
   - Extracts and fills soma coordinates

2. **User Actions**
   - Adjust grid radius (1-3) for high-res
   - Set FOV dimensions for low-res
   - Choose output directory
   - Click action button

3. **Processing**
   - High-Res: Downloads 3×3×3 to 5×5×5 blocks via HTTP
   - Low-Res: Downloads Z-slice stack via SSH
   - Generates MIP visualizations
   - Exports NIfTI/TIFF volumes

---

## Integration with Region Analysis

```mermaid
flowchart LR
    subgraph RegionAnalysis["Region Analysis Output"]
        A1["ACC_df.xlsx"]
        A2["INS_df.xlsx"]
    end
    
    subgraph Tables["Recording Sheets"]
        B1["NeuronID<br/>Soma_Region<br/>Terminal_Regions"]
    end
    
    subgraph VisualToolkit["Visual Toolkit"]
        C1["Select Neuron<br/>from Table"]
        C2["Download &<br/>Visualize"]
    end
    
    RegionAnalysis --> Tables
    Tables --> VisualToolkit
```

The DataFrames (ACC_df, INS_df) generated by region analysis serve as "recording sheets" - users can identify neurons of interest from the tables and then use Visual Toolkit to download and inspect specific neurons in 3D or large 2D images.

---

## File Structure

```
main_scripts/
├── Visual_toolkit_gui.py       # GUI application
├── Visual_toolkit.py           # Backend toolkit
├── VISUAL_TOOLKIT_GUIDE.md     # Detailed guide
│
└── resource/
    ├── cubes/                  # Cache directory
    │   └── 251637/
    │       ├── high_res_http/  # HTTP downloaded blocks
    │       └── low_res_ssh/    # SSH downloaded slices
    └── segmented_cubes/        # Output directory
        └── 251637/
            ├── 251637_003.swc_SomaBlock.nii.gz
            ├── 251637_003.swc_SomaBlock_Plot.png
            ├── 251637_003.swc_WideField.nii.gz
            └── 251637_003.swc_WideField_Plot.png
```
