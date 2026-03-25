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
│                            • High/Low Res Blocks         • LowRes WideField
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Corrected Data Flow

```mermaid
flowchart TB
    subgraph Input["📥 INPUT"]
        SID[Sample ID<br/>e.g., 251637]
        GL[getNeuronListByRegion<br/>function]
        NIDS[Neuron IDs list]
    end

    subgraph Step1["🔵 STEP 1: Region Analysis"]
        S1[step1.run_region_analysis.py]
        S1_deps[Dependencies:
        PopulationRegionAnalysis
        Cortex/Subcortex Hierarchy
        ARM/CHARM/SARM Atlas
        Laterality Analysis]
        S1_out["Output: Neuron Tables
        (.xlsx with projection_strength,
        soma_region, terminal_regions,
        laterality, etc.)"]
    end

    subgraph Step2["🟢 STEP 2: FNT Distance Pipeline"]
        S2[step2.fnt-dist_pipeline.py]
        S2_deps[Dependencies:
        fnt-from-swc
        fnt-decimate
        fnt-join
        fnt-dist
        IONData]
        S2_out["Output: FNT Files +
        Distance Matrix
        (*_joined.fnt,
        *_dist.txt)"]
    end

    subgraph Step3["🟠 STEP 3: Bulk Visualization"]
        S3[step3.bulk_visual_data.py]
        S3_deps[Dependencies:
        Visual_toolkit
        IONData
        High/Low Res Blocks]
        S3_out["Output: Plots & NIfTI
        (HighRes Soma Block,
        LowRes WideField)"]
    end

    SID --> GL
    GL --> NIDS
    NIDS --> S1
    S1 --> S1_deps
    S1_deps --> S1_out
    
    S1_out -.->|neuron_tables| S2
    S2 --> S2_deps
    S2_deps --> S2_out
    
    S1_out -.->|neuron_tables| S3
    S3 --> S3_deps
    S3_deps --> S3_out
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
        I1[Sample ID: "251637"]
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

## Generic Pipeline Variant

> **Note:** `fnt-dist_pipeline_generic.py` - Works with **any** neuron table (not restricted to ACC/INS)

```mermaid
flowchart LR
    A[Neuron Table<br/>from Step 1<br/>or manual] --> B[fnt-dist_pipeline_generic.py]
    B --> C{get_paths<br/>sample_id}
    C --> D[processed_neurons/{sample_id}/]
    D --> E[FNT Processing]
    E --> F[*_joined.fnt<br/>*_dist.txt]
```

---

## File Locations

| File | Path |
|------|------|
| Step 1 Script | `main_scripts/step1.run_region_analysis.py` |
| Step 2 Script | `main_scripts/step2.fnt-dist_pipeline.py` |
| Step 3 Script | `main_scripts/step3.bulk_visual_data.py` |
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
# Edit INPUT_FILE in step3.bulk_visual_data.py
# Run → outputs plots and NIfTI files
```

---

*Generated: 2026-03-24*
*View this file on GitHub for interactive Mermaid diagrams*
