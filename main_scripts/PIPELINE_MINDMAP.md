# Projectome Analysis Pipeline Mindmap

## Quick Overview (Corrected Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Projectome Analysis Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT                    PROCESSING                    OUTPUT              │
│                                                                              │
│   Sample ID                STEP 1: Region Analysis       Region Stats        │
│   (e.g., "251637")         ────────────────────────      ───────────         │
│        │                   PopulationRegionAnalysis      • neuron_tables     │
│        │                   • Cortex/Subcortex hierarchy      (.xlsx)         │
│        ▼                   • ARM/CHARM/SARM Atlas        • projection        │
│   ┌─────────┐              • Laterality Analysis           strength          │
│   │Neuron   │                      │                      • soma_region       │
│   │IDs from │◄─────────────────────┤                      • terminals         │
│   │getNeuron│                      │                      • laterality        │
│   │ListBy  │                      ▼                      • etc.              │
│   │Region()│              Neuron Tables ─────────────────────────►            │
│   └─────────┘              (.xlsx output)              (to Step 2 & 3)       │
│                                   │                                          │
│                                   ▼                                          │
│                            STEP 2: FNT Pipeline                              │
│                            ─────────────────────                             │
│                            • fnt-from-swc                                    │
│                            • fnt-decimate                                    │
│                            • fnt-join                                        │
│                            • fnt-dist              FNT Distance Matrix       │
│                            • IONData (SWC fetch)     (*_dist.txt)            │
│                                    │                                         │
│                                    ▼                                         │
│                            STEP 3: Bulk Visualization                     Plots
│                            ─────────────────────────                    & NIfTI
│                            • Visual_toolkit                      ───────────
│                            • IONData                     • HighRes Soma Block
│                            • High/Low Res Blocks         • LowRes WideField  │
│                                                                              │
│                            STEP 4: Additional Rendering                      │
│                            ─────────────────────────────                     │
│                            • Brain Mesh Render (brain_viz.py)                │
│                            • NeuronView Render (GL-based)                    │
│                            • Clustered Heatmap (Gou 2025 Fig2A)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Corrected Data Flow

```mermaid
flowchart TB
    %% Define styles with proper contrast
    classDef default fill:#fff,stroke:#333,color:#000
    classDef titleStyle fill:#424242,stroke:#212121,color:#fff,stroke-width:3px
    classDef inputStyle fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef step1Style fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef step2Style fill:#fff3e0,stroke:#ef6c00,color:#000
    classDef step3Style fill:#fce4ec,stroke:#c2185b,color:#000
    classDef step4Style fill:#e0f2f1,stroke:#00695c,color:#000
    classDef outputStyle fill:#ffebee,stroke:#c62828,color:#000
    classDef gap padding:30px,margin:30px

    subgraph Input["📥 INPUT"]
        direction TB
        SID[/"Sample ID<br/>e.g., 251637"/]
        GL[getNeuronListByRegion<br/>function]
        NIDS[Neuron IDs list]
    end

    subgraph Step1["🔵 STEP 1: Region Analysis"]
        direction TB
        S1[step1.run_region_analysis.py]
        S1_deps[Dependencies:<br/>PopulationRegionAnalysis, Cortex/Subcortex<br/>Hierarchy, ARM/CHARM/SARM Atlas,<br/>Laterality Analysis]
        S1_out[/"Output: Neuron Tables (.xlsx)<br/>projection_strength, soma_region,<br/>terminal_regions, laterality"/]
    end

    subgraph Step2["🟢 STEP 2: FNT Distance Pipeline"]
        direction TB
        S2[step2.fnt-dist_pipeline.py]
        S2_deps[Dependencies:<br/>fnt-from-swc, fnt-decimate,<br/>fnt-join, fnt-dist, IONData]
        S2_out[/"Output: FNT Files + Distance Matrix<br/>(*_joined.fnt, *_dist.txt)"/]
    end

    subgraph Step3["🟠 STEP 3: Bulk Visualization"]
        direction TB
        S3[step3.1.bulk_visual_data.py]
        S3_deps[Dependencies:<br/>Visual_toolkit, IONData,<br/>High/Low Res Blocks]
        S3_out[/"Output: Plots & NIfTI<br/>HighRes Soma Block, LowRes WideField"/]
    end
    
    subgraph Step4["🟣 STEP 4: Statistical Analysis"]
        direction TB
        S4[R based analysis]

    end 

    subgraph Step5["🔷 STEP 5: Additional Rendering"]
        direction TB
        S5A[step3.2.run_brain_viz_meshRender.py<br/>BrainViz + RegionExtractor]
        S5B[step3.3.neuronviewRender.py<br/>neuronVis + RenderGL]
        S5C[step3.4.region_flatmap.viz.py<br/>RegionFlatmap]
        S5_out[/"Output:<br/>• brain_viz_*<br/>• neuronview renders<br/>"/]
    end

    %% Main vertical flow - sequential
    SID --> GL --> NIDS --> S1 --> S1_deps --> S1_out
    
    %% Branching from S1_out to Step2 and Step3
   

    S2 --> S2_deps --> S2_out
    

    S3 --> S3_deps --> S3_out

    S1_out --> Step2
    S1_out --> Step3

    S1_out --> Step4

    Step2 ~~~ Step3 ~~~ Step4

    S4 --> Step5
    S3_out --> Step5
    
    S5A --> S5_out
    S5B --> S5_out
    

    %% Apply dark background to subgraph titles
    style Input fill:#1565c0,stroke:#0d47a1,color:#fff
    style Step1 fill:#2e7d32,stroke:#1b5e20,color:#fff
    style Step2 fill:#ef6c00,stroke:#e65100,color:#fff
    style Step3 fill:#6a1b9a,stroke:#4a148c,color:#fff
    style Step4 fill:#c62828,stroke:#b71c1c,color:#fff
    style Step5 fill:#00695c,stroke:#004d40,color:#fff
```

---

## Step 1: Getting Neuron IDs

```python
# Method 1: Get neurons by region names
from region_analysis import getNeuronListByRegion

neuron_ids = getNeuronListByRegion(
    sample_id='251637',
    region_names=['motor', 'premotor'],
    return_ids_only=True,
    verbose=False
)

# Method 2: Use specific neuron IDs
neuron_ids = ['001.swc', '002.swc', '003.swc']

# Then run Step 1
from step1.run_region_analysis import main
main(neuron_ids=neuron_ids, sample_id='251637')
```

---

## Key Data Files

| Category | Files | Location |
|----------|-------|----------|
| **Step 1 Input** | Sample ID, Neuron IDs | From getNeuronListByRegion |
| **Step 1 Output** | `251637_INS.xlsx`, `251637_ACC.xlsx`, etc. | `neuron_tables/` |
| **Atlas Keys** | `ARM_key_all.txt`, `CHARM_key_table_v2.csv`, `SARM_key_table_v2.csv` | `atlas/` |
| **Step 2 Output** | `*_joined.fnt`, `*_dist.txt` | `processed_neurons/{sample_id}/fnt_processed/` |
| **Step 3 Output** | `.png` plots, `.nii.gz` volumes | Configurable output dir |
| **Step 5 Output** | Brain viz PNGs, heatmaps | `brain_viz_output/`, `output/` |
| **Cache** | Cube data | `resource/cubes/{sample_id}/` |

---

## Supporting Analysis Tools

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  FNTCubeVis.py  │fnt_dist_cluster │ brain_mesh_viz  │  neuro_tracer   │   fnt_tools     │
│   FNT 3D Viz    │ Distance Analysis│ Brain Surface   │ Neuron Tracing  │  SWC/FNT Utils  │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

---

## Step 1: Region Analysis - Detailed Dependencies

```mermaid
flowchart TD
    subgraph Input["Step 1 Input"]
        I1[Sample ID: 251637]
        I2[Neuron IDs from getNeuronListByRegion]
    end

    subgraph Processing["Step 1 Processing"]
        P1[PopulationRegionAnalysis]
        P2[Atlas Data:
        ARM_in_NMT_v2.1_sym.nii.gz
        CHARM_key_table_v2.csv
        SARM_key_table_v2.csv]
        P3[Analysis Modules:
        hierarchy.py
        laterality.py
        neuron_analysis.py
        population.py]
    end

    subgraph Output["Step 1 Output"]
        O1[Neuron Tables .xlsx
        - projection_strength
        - soma_region
        - terminal_regions
        - laterality
        - projection_length_by_region]
    end

    I1 --> P1
    I2 --> P1
    P2 --> P1
    P3 --> P1
    P1 --> O1
```

---

## Step 2: FNT Distance Pipeline - Dependencies

```mermaid
flowchart TD
    subgraph Input["Step 2 Input"]
        I1[Neuron Tables from Step 1
        e.g., 251637_INS.xlsx]
        I2[Sample ID]
    end

    subgraph Processing["Step 2 Processing"]
        P1[Load Neuron IDs from table]
        P2[Process each neuron:
        preprocess_swc_coordinates
        swc_to_fnt
        decimate_fnt
        update_fnt_neuron_name]
        P3[Join FNT files:
        fnt-join.exe]
        P4[Compute distances:
        fnt-dist.exe]
    end

    subgraph Output["Step 2 Output"]
        O1[*_joined.fnt]
        O2[*_dist.txt
        Distance Matrix]
    end

    I1 --> P1
    I2 --> P2
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> O1
    P4 --> O2
```

---

## Step 3: Bulk Visualization - Dependencies

```mermaid
flowchart TD
    subgraph Input["Step 3 Input"]
        I1[Neuron Tables from Step 1
        e.g., 251637_INS.xlsx]
        I2[Sample ID]
    end

    subgraph Processing["Step 3 Processing"]
        P1[Visual_toolkit]
        P2[IONData for SWC fetch]
        P3[High Res: get_high_res_block
        plot_soma_block]
        P4[Low Res: get_low_res_widefield
        plot_widefield_context]
    end

    subgraph Output["Step 3 Output"]
        O1[HighRes Plots
        SomaBlock.png]
        O2[LowRes Plots
        WideField.png]
        O3[NIfTI volumes
        .nii.gz]
    end

    I1 --> P1
    I2 --> P2
    P2 --> P3
    P2 --> P4
    P1 --> P3
    P1 --> P4
    P3 --> O1
    P3 --> O3
    P4 --> O2
    P4 --> O3
```

---

## Step 5: Additional Rendering

### 5.1 Brain Mesh Render (step3.2.run_brain_viz_meshRender.py)

```mermaid
flowchart TD
    subgraph Input["Step 5.1 Input"]
        I1[Neuron Tables from Step 1]
        I2[ARM Atlas
        ARM_in_NMT_v2.1_sym.nii.gz]
        I3[NMT Brain Template]
    end

    subgraph Processing["Step 5.1 Processing"]
        P1[BrainViz Class]
        P2[RegionExtractor
        extract_regions from atlas]
        P3[Mesh Generation
        add_mesh_from_array]
        P4[Neuron Loading
        load_neurons with type_col]
    end

    subgraph Output["Step 5.1 Output"]
        O1[brain_viz_recreated.png]
        O2[brain_viz_generated.png]
        O3[brain_viz_dynamic.png]
    end

    I1 --> P4
    I2 --> P2
    I3 --> P1
    P1 --> P3
    P2 --> P3
    P3 --> P1
    P4 --> P1
    P1 --> O1
    P1 --> O2
    P1 --> O3
```

### 5.2 NeuronView Render (step3.3.neuronviewRender.py)

```mermaid
flowchart TD
    subgraph Input["Step 5.2 Input"]
        I1[Sample ID: 251637]
        I2[Neuron ID Lists
        bg_neurons by type]
        I3[Region .obj Files]
    end

    subgraph Processing["Step 5.2 Processing"]
        P1[neuronVis Class]
        P2[RenderGL / RenderMacaqueGL]
        P3[addNeuronByID with colors
        ITs/ITi/CT/PT types]
        P4[addRegion from .obj]
        P5[setView / animation]
    end

    subgraph Output["Step 5.2 Output"]
        O1[Interactive 3D Window]
        O2[savepng output]
    end

    I1 --> P3
    I2 --> P3
    I3 --> P4
    P1 --> P2
    P3 --> P1
    P4 --> P1
    P2 --> P5
    P5 --> O1
    P5 --> O2
```

### 5.3 Clustered Heatmap (visualize_clustered_heatmap.py)

```mermaid
flowchart TD
    subgraph Input["Step 5.3 Input"]
        I1[Clustered Neuron Table
        251637_results_clustered_k9_spearman_penalty.xlsx]
    end

    subgraph Processing["Step 5.3 Processing"]
        P1[Load projection matrix]
        P2[Log transform
        log10(proj + 0.001)]
        P3[Sort by Morph_Cluster
        and Neuron_Type]
        P4[Matplotlib gridspec
        • Cluster color bar
        • Type color bar
        • Main heatmap]
    end

    subgraph Output["Step 5.3 Output"]
        O1[fig2a_style_clustered_heatmap.png]
        O2[fig2a_style_clustered_heatmap.pdf]
        O3[Cluster statistics]
    end

    I1 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> O1
    P4 --> O2
    P4 --> O3
```

---

## Generic Pipeline Variant

> **Note:** `fnt-dist_pipeline_generic.py` - Works with **any** neuron table (not restricted to ACC/INS)

```mermaid
flowchart LR
    A[Neuron Table<br/>from Step 1<br/>or manual] --> B[fnt-dist_pipeline_generic.py]
    B --> C{get_paths<br/>sample_id}
    C --> D[processed_neurons/sample_id/]
    D --> E[FNT Processing]
    E --> F[*_joined.fnt<br/>*_dist.txt]
```

---

## File Locations

| File | Path |
|------|------|
| Step 1 Script | `main_scripts/step1.run_region_analysis.py` |
| Step 2 Script | `main_scripts/step2.fnt-dist_pipeline.py` |
| Step 3.1 Script | `main_scripts/step3.1.bulk_visual_data.py` |
| Step 3.2 Script | `main_scripts/step3.2.run_brain_viz_meshRender.py` |
| Step 3.3 Script | `main_scripts/step3.3.neuronviewRender.py` |
| Clustered Heatmap | `main_scripts/visualize_clustered_heatmap.py` |
| Generic Pipeline | `main_scripts/fnt-dist_pipeline_generic.py` |
| getNeuronListByRegion | `main_scripts/region_analysis/getNeuronListByRegion.py` |
| Region Analysis Module | `main_scripts/region_analysis/` |
| Step 1 Output Tables | `main_scripts/neuron_tables/` |
| Step 2 Output | `processed_neurons/{sample_id}/fnt_processed/` |
| Cache | `resource/cubes/{sample_id}/` |

---

## Usage Summary

```python
# === STEP 1: Region Analysis ===
from region_analysis import getNeuronListByRegion
from step1.run_region_analysis import main

# Get neuron IDs by region
neuron_ids = getNeuronListByRegion('251637', ['insula'], return_ids_only=True)

# Run region analysis → outputs neuron_tables/251637_*.xlsx
main(neuron_ids=neuron_ids, sample_id='251637')

# === STEP 2: FNT Distance Pipeline ===
# Edit NEURON_TABLE_FILE in step2.fnt-dist_pipeline.py
# Run → outputs *_joined.fnt, *_dist.txt

# === STEP 3: Bulk Visualization ===
# Edit INPUT_FILE in step3.1.bulk_visual_data.py
# Run → outputs plots and NIfTI files

# === STEP 5: Additional Rendering ===
# 5.1 Brain Mesh Render
python main_scripts/step3.2.run_brain_viz_meshRender.py [1-6]

# 5.2 NeuronView Render
python main_scripts/step3.3.neuronviewRender.py

# 5.3 Clustered Heatmap
python main_scripts/visualize_clustered_heatmap.py
```

---

*Generated: 2026-03-26*
*View this file on GitHub for interactive Mermaid diagrams*
