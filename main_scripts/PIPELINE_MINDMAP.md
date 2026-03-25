# Projectome Analysis Pipeline Mindmap

## Quick Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Projectome Analysis Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT                    PROCESSING                    OUTPUT              │
│                                                                              │
│   Neuron Tables            STEP 1: Region Analysis       Region Stats        │
│   (.xlsx/.csv)             ────────────────────────      ───────────         │
│        │                   PopulationRegionAnalysis      • projection        │
│        │                   • Cortex/Subcortex hierarchy    strength          │
│        ▼                   • ARM/CHARM/SARM Atlas        • soma_region       │
│   ┌─────────┐              • Laterality Analysis         • terminals         │
│   │251637_  │                      │                      │                 │
│   │ INS.xlsx│                      ▼                      ▼                 │
│   │ ACC.xlsx│              neuron_ids ─────────────────────────►            │
│   │ M1.xlsx │                      │                      │                 │
│   └─────────┘                      ▼                      │                 │
│                            STEP 2: FNT Pipeline           │                 │
│                            ─────────────────────          │                 │
│                            • fnt-from-swc                 │                 │
│                            • fnt-decimate                 ▼                 │
│                            • fnt-join              FNT Distance Matrix      │
│                            • fnt-dist                (*_dist.txt)           │
│                            • IONData (SWC fetch)                            │
│                                    │                      │                 │
│                                    ▼                      │                 │
│                            neuron_tables ────────────────►│                 │
│                                    │                      │                 │
│                                    ▼                      ▼                 │
│                            STEP 3: Bulk Visualization  Plots & NIfTI        │
│                            ─────────────────────────   ─────────────        │
│                            • Visual_toolkit            • HighRes Soma       │
│                            • IONData                   • LowRes WideField   │
│                            • High/Low Res Blocks                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Mermaid Diagram (for GitHub/GitLab rendering)

```mermaid
flowchart TB
    subgraph Input["📁 Input Data"]
        NT[Neuron Tables<br/>251637_INS.xlsx<br/>251637_ACC.xlsx<br/>251637_M1.xlsx]
    end

    subgraph Step1["🔵 STEP 1: Region Analysis"]
        S1[step1.run_region_analysis.py]
        S1_deps[Dependencies:
        PopulationRegionAnalysis
        Cortex/Subcortex Hierarchy
        ARM/CHARM/SARM Atlas
        Laterality Analysis]
    end

    subgraph Step2["🟢 STEP 2: FNT Distance Pipeline"]
        S2[step2.fnt-dist_pipeline.py]
        S2_deps[Dependencies:
        fnt-from-swc
        fnt-decimate
        fnt-join
        fnt-dist
        IONData]
    end

    subgraph Step3["🟠 STEP 3: Bulk Visualization"]
        S3[step3.bulk_visual_data.py]
        S3_deps[Dependencies:
        Visual_toolkit
        IONData
        High/Low Res Blocks]
    end

    subgraph Output["📤 Output Data"]
        O1[Region Analysis Results:
        projection_strength
        soma_region
        terminal_regions]
        
        O2[FNT Files + Distance Matrix:
        *_joined.fnt
        *_dist.txt]
        
        O3[Plots & NIfTI:
        HighRes Soma Block
        LowRes WideField]
    end

    NT --> S1
    S1 --> O1
    
    S1 -.->|neuron_ids| S2
    S2 --> O2
    
    S2 -.->|neuron_tables| S3
    S3 --> O3
    
    S1 -.-> S1_deps
    S2 -.-> S2_deps
    S3 -.-> S3_deps
```

---

## Key Data Files

| Category | Files |
|----------|-------|
| **Neuron Tables** | `251637_INS.xlsx`, `251637_ACC.xlsx`, `251637_M1.xlsx` |
| **Atlas Keys** | `ARM_key_all.txt`, `CHARM_key_table_v2.csv`, `SARM_key_table_v2.csv` |
| **FNT Outputs** | `*_joined.fnt`, `*_dist.txt` |
| **Cache** | `resource/cubes/` |

---

## Supporting Analysis Tools

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  FNTCubeVis.py  │fnt_dist_cluster │ brain_mesh_viz  │  neuro_tracer   │   fnt_tools     │
│   FNT 3D Viz    │ Distance Analysis│ Brain Surface   │ Neuron Tracing  │  SWC/FNT Utils  │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

---

## Step 1: Region Analysis - Dependencies

```mermaid
flowchart TD
    S1[step1.run_region_analysis.py] --> A[PopulationRegionAnalysis]
    S1 --> B[region_analysis module]
    B --> B1[classifier.py]
    B --> B2[hierarchy.py]
    B --> B3[laterality.py]
    B --> B4[neuron_analysis.py]
    B --> B5[population.py]
    S1 --> C[ARM/CHARM/SARM Atlas]
```

---

## Step 2: FNT Distance Pipeline - Dependencies

```mermaid
flowchart TD
    S2[step2.fnt-dist_pipeline.py] --> A[fnt-from-swc]
    S2 --> B[fnt-decimate]
    S2 --> C[fnt-join]
    S2 --> D[fnt-dist]
    S2 --> E[IONData]
    S2 --> F[fnt_tools.py]
```

---

## Step 3: Bulk Visualization - Dependencies

```mermaid
flowchart TD
    S3[step3.bulk_visual_data.py] --> A[Visual_toolkit]
    S3 --> B[IONData]
    S3 --> C[get_high_res_block]
    S3 --> D[get_low_res_widefield]
    S3 --> E[plot_soma_block]
    S3 --> F[plot_widefield_context]
```

---

## Generic Pipeline Variant

> **Note:** `fnt-dist_pipeline_generic.py` - Works with **any** neuron table (not restricted to ACC/INS)

```
Any Neuron Table (.xlsx/.csv)
    ↓
fnt-dist_pipeline_generic.py
    ↓
get_paths(sample_id) → processed_neurons/{sample_id}/
    ↓
FNT Processing → *_joined.fnt, *_dist.txt
```

---

## File Locations

| File | Path |
|------|------|
| Step 1 Script | `main_scripts/step1.run_region_analysis.py` |
| Step 2 Script | `main_scripts/step2.fnt-dist_pipeline.py` |
| Step 3 Script | `main_scripts/step3.bulk_visual_data.py` |
| Generic Pipeline | `main_scripts/fnt-dist_pipeline_generic.py` |
| Region Analysis Module | `main_scripts/region_analysis/` |
| Neuron Tables | `main_scripts/neuron_tables/` |
| Output Figures | `figures_charts/processing_pipeline_mindmap.png` |

---

*Generated: 2026-03-24*
*View this file on GitHub for interactive Mermaid diagrams*
