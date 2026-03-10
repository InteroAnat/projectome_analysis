# Complete Flowchart Summary

This document provides a quick reference to all the flowcharts created for the projectome analysis pipeline.

---

## Central Architecture: Region Analysis as Backbone

```mermaid
flowchart TB
    subgraph RawData["📦 Raw Data"]
        R1["fMOST Images<br/>Sample: 251637"]
        R2["562 SWC Files"]
        R3["ARM Atlas<br/>NMT v2.1 sym"]
    end

    subgraph Backbone["🔷 BACKBONE: Region Analysis<br/>region_analysis.py"]
        direction TB
        B1["Load & Process SWCs"] --> B2["Atlas Mapping"]
        B2 --> B3["Calculate Branch Lengths"]
        B3 --> B4["Identify Soma & Terminals"]
        B4 --> B5["Classify Projection Subtypes<br/>PT / CT / ITs / ITc / ITi"]
        B5 --> B6["Generate Recording Tables<br/>ACC_df / INS_df"]
        
        B7["Population Statistics"] 
        B8["Outlier Detection"]
        B9["Single Neuron Inspection"]
    end

    subgraph FNTClustering["🎯 FNT-Dist Clustering<br/>(Morphological Analysis)"]
        C1["Read SWC Files"] 
        C2["Convert to FNT"]
        C3["Compute Distance Matrix"]
        C4["Ward Clustering"]
        C5["Morphological Clusters"]
        
        C6["Uses Subtype Info<br/>from Backbone Tables<br/>(optional penalty)"]
    end

    subgraph VisualToolkit["🔍 Visual Toolkit<br/>(Illustrative Visualization)"]
        V1["Read Recording Tables"]
        V2["Select Neuron by<br/>Subtype/Region/Target"]
        V3["Auto-Fill Soma XYZ"]
        V4["High-Res Download"]
        V5["Low-Res Download"]
        V6["MIP Illustrations"]
    end

    subgraph Outputs["📤 Final Outputs"]
        O1["Projection Subtypes"]
        O2["Morphological Clusters"]
        O3["Illustrative Visualizations"]
        O4["Statistical Reports"]
    end

    RawData --> Backbone
    
    Backbone --> FNTClustering
    Backbone --> VisualToolkit
    
    Backbone -.->|Provides subtype metadata| C6
    Backbone -.->|Provides recording tables| V1
    
    FNTClustering --> O2
    VisualToolkit --> O3
    Backbone --> O1
    Backbone --> O4

    style Backbone fill:#e3f2fd,stroke:#1565c0,stroke-width:4px
    style FNTClustering fill:#fff3e0,stroke:#ef6c00
    style VisualToolkit fill:#f3e5f5,stroke:#6a1b9a
```

**Key Point:** `region_analysis.py` is the **backbone** that:
- **Feeds** FNT-Dist Clustering with subtype metadata (for supervised penalty)
- **Provides** Visual Toolkit with recording tables (for neuron selection)
- **Generates** the primary projection subtypes used throughout the pipeline

---

## 1. FNT-Dist Clustering Flowchart

**File:** `fnt_clustering_flowchart.md`

**Purpose:** Morphological clustering of neurons based on FNT (Feature Neuron Tree) distance. Receives subtype information from the backbone (region_analysis) for optional supervised penalty.

```mermaid
flowchart TB
    subgraph Input["📁 Input Data"]
        A1["SWC Files"]
        A2["Neuron Metadata<br/>from Backbone Tables<br/>(ACC_df / INS_df)"]
    end

    subgraph Pipeline["🔧 FNT Processing Pipeline"]
        B1["Load Neuron Lists"] --> B2["SWC → FNT Conversion<br/>fnt-from-swc"]
        B2 --> B3["Decimate FNT<br/>fnt-decimate -d 5000 -a 5000"] 
        B3 --> B4["Join FNT Files<br/>fnt-join.exe"]
        B4 --> B5["Compute Distance Matrix<br/>fnt-dist.exe"]
    end

    subgraph Clustering["🎯 Clustering Analysis<br/>(Morphological Subtypes)"]
        C1["Load Distance Matrix"] --> C2["Spearman/Log1p Transform"]
        C2 --> C3["Apply Penalty<br/>(uses subtype info<br/>from Backbone)"] 
        C3 --> C4["Ward Linkage<br/>Hierarchical clustering"]
        C4 --> C5["C-Index Optimization<br/>k=2..65"] 
        C5 --> C6["Assign Morphological<br/>Clusters"]
    end

    subgraph Results["📈 Output Results"]
        D1["Morphological_Clusters_k{N}.xlsx"]
        D2["Heatmaps & Dendrograms"]
        D3["Note: Independent of<br/>projection subtypes<br/>(PT/CT/IT from Backbone)"]
    end

    Input --> Pipeline --> Clustering --> Results
```

**Key Points:**
- **Morphological clustering** based on tree structure similarity
- Uses **FNT tools** (fnt-from-swc, fnt-decimate, fnt-join, fnt-dist)
- **Spearman correlation** or **Log1p magnitude** transforms
- **C-index optimization** for optimal cluster count
- **Optional supervised penalty** uses subtypes from backbone (region_analysis)
- Output: **Morphological subtypes** (distinct from projection subtypes PT/CT/IT)

---

## 2. Visual Toolkit Flowchart

**File:** `visual_toolkit_flowchart.md`

**Purpose:** Download and visualize neurons in two different resolutions - using recording tables from the backbone (region_analysis) for neuron selection.

```mermaid
flowchart TB
    subgraph Input["📁 Input"]
        A1["Sample ID<br/>e.g., 251637"]
        A2["Recording Tables<br/>from Backbone<br/>ACC_df / INS_df"]
        A3["Soma Coordinates<br/>auto-extracted from SWC"]
    end

    subgraph Selection["🎯 Neuron Selection<br/>(via Backbone Tables)"]
        S1["Review ACC_df / INS_df"] --> S2["Filter by Subtype<br/>PT / CT / ITs / ITc / ITi"]
        S2 --> S3["Filter by Soma Region"] 
        S3 --> S4["Filter by Terminal Targets"]
        S4 --> S5["Select NeuronID"]
    end

    subgraph GUI["🖥️ Visual Toolkit GUI"]
        B1["Launch GUI"] --> B2["Load Selected Neuron<br/>from Backbone Table"]
        B2 --> B3["Extract Soma XYZ<br/>from tree.root"]
        B3 --> B4["Configure Parameters"]
    end

    subgraph HighRes["🔍 High-Resolution Visualization<br/>0.65µm via HTTP"]
        C1["Download 3D Blocks<br/>bap.cebsit.ac.cn"] --> C2["Assemble Volume<br/>Grid stitching"]
        C2 --> C3["Maximum Intensity<br/>Projection (MIP)"]
        C3 --> C4["Grayscale Rendering<br/>+ Soma Marker"]
        C4 --> C5["Illustrative Output<br/>Soma Context"]
    end

    subgraph LowRes["🌐 Low-Resolution Visualization<br/>5.0µm via SSH"]
        D1["Download Z-Slices<br/>172.20.10.250"] --> D2["Assemble Stack<br/>Wide FOV"]
        D2 --> D3["Maximum Intensity<br/>Projection (MIP)"]
        D3 --> D4["Green Signal on<br/>Dark Background"]
        D4 --> D5["SWC Overlay<br/>Neuron Trace"]
        D5 --> D6["Illustrative Output<br/>Projection Context"]
    end

    subgraph Output["📤 Output Files"]
        E1["{neuron}_SomaBlock.nii.gz<br/>High-res Volume"]
        E2["{neuron}_SomaBlock_Plot.png<br/>Soma Visualization"]
        E3["{neuron}_WideField.nii.gz<br/>Low-res Volume"]
        E4["{neuron}_WideField_Plot.png<br/>Wide-field Illustration"]
    end

    Input --> Selection --> GUI
    GUI --> HighRes --> Output
    GUI --> LowRes --> Output
```

**Key Points:**
- **Relies on backbone tables** (ACC_df/INS_df) for neuron selection
- Can filter by **projection subtypes** (PT/CT/IT) from region_analysis
- **Dual-resolution visualization**: High-res (0.65µm) for soma detail, Low-res (5.0µm) for projection context
- **Illustrative rendering**: MIP projections with anatomical context
- **SWC overlay**: Neuron traces on low-res wide field

---

## 3. Region Analysis (Backbone)

**File:** `region_analysis_flowchart.md`

**Purpose:** **BACKBONE MODULE** - Analyzes neuron projections in atlas space, classifies into projection subtypes (PT, CT, ITs, ITc, ITi), and generates recording tables used by both FNT clustering and Visual Toolkit.

```mermaid
flowchart TB
    subgraph Input["📁 Input Data"]
        A1["SWC Files<br/>562 neurons"]
        A2["ARM Atlas<br/>NMT v2.1 sym"]
        A3["Atlas Table<br/>ARM_key_all.txt"]
        A4["Template Image<br/>NMT_v2.1_sym_SS.nii.gz"]
    end

    subgraph PerNeuron["🔬 Per-Neuron Analysis<br/>region_analysis_per_neuron"]
        B1["Load SWC<br/>neuro_tracer.process()"] --> B2["Transform to NII Space"]
        B2 --> B3["Calculate Branch Lengths<br/>per atlas region"]
        B3 --> B4["Map to ARM Atlas<br/>Level 6 parcellation"]
        B4 --> B5["Identify Soma Region<br/>(CL/CR/SL/SR)"]
        B4 --> B6["Identify Terminal Regions"]
        B5 --> B7["Detect Outliers<br/>(Unknown regions)"]
        B6 --> B7
    end

    subgraph Classification["🎯 Projection Subtype Classification<br/>NeuronClassifier"]
        C1["Analyze Terminal List<br/>+ Soma Side"] --> C2{"Hierarchical Rules"}
        C2 -->|Has Brainstem/<br/>Hypothalamus| C3["PT<br/>Pyramidal Tract"]
        C2 -->|Has Thalamus<br/>(not PT)| C4["CT<br/>Corticothalamic"]
        C2 -->|Has Striatum<br/>(not PT/CT)| C5["ITs<br/>Intratelencephalic"]
        C2 -->|Has Contralateral<br/>Cortex (not PT/CT/ITs)| C6["ITc<br/>Interhemispheric"]
        C2 -->|Has Ipsilateral<br/>Cortex only| C7["ITi<br/>Ipsilateral"]
        C2 -->|None match| C8["Unclassified"]
    end

    subgraph Population["📊 Population Analysis<br/>PopulationRegionAnalysis"]
        D1["Batch Process<br/>562 neurons"] --> D2["Collect All Metrics"]
        D2 --> D3["Subtype Classification<br/>(PT/CT/ITs/ITc/ITi)"]
    end

    subgraph BackboneOutput["📤 Backbone Output Tables<br/>(Used by Other Modules)"]
        E1["ACC_df.xlsx<br/>Cingulate neurons"] 
        E2["INS_df.xlsx<br/>Insula neurons"]
        E3["Columns:<br/>NeuronID, Neuron_Type (subtypes),<br/>Soma_Region, Terminal_Regions,<br/>Total_Length, Outlier_Count..."]
    end

    subgraph Downstream1["⬇️ Feeds FNT-Dist Clustering"]
        F1["Subtype metadata<br/>for supervised penalty"]
    end

    subgraph Downstream2["⬇️ Feeds Visual Toolkit"]
        G1["Recording tables<br/>for neuron selection"]
    end

    Input --> PerNeuron --> Classification --> Population --> BackboneOutput
    
    BackboneOutput -.->|subtype info| Downstream1
    BackboneOutput -.->|recording tables| Downstream2

    style BackboneOutput fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    style Downstream1 fill:#fff3e0,stroke:#ef6c00
    style Downstream2 fill:#f3e5f5,stroke:#6a1b9a
```

**Key Points (Backbone Role):**
- **Central hub** that processes all neuron data
- **Generates projection subtypes** (PT/CT/ITs/ITc/ITi) used across pipeline
- **Produces recording tables** (ACC_df/INS_df) for Visual Toolkit selection
- **Provides subtype metadata** for FNT-Dist supervised penalty
- **ARM Atlas** with cortical laterality (CL/CR/SL/SR)
- **Outlier detection** for quality control

---

## 4. Analysis Notebook

**File:** `analysis_notebook_flowchart.md`

**Purpose:** Main analysis pipeline that orchestrates the **backbone** (region_analysis) to process all neurons and generate recording tables.

```mermaid
flowchart TB
    subgraph Setup["⚙️ Setup & Configuration"]
        A1["Import Modules<br/>neuro_tracer, region_analysis"]
        A2["Load ARM Atlas<br/>ARM_in_NMT_v2.1_sym.nii.gz"]
        A3["Load Atlas Key<br/>ARM_key_all.txt"]
        A4["Load Template<br/>NMT_v2.1_sym_SS.nii.gz"]
        A5["Atlas Level 6<br/>(5D: x,y,z,0,level-1)"]
    end

    subgraph Processing["🔬 Batch Processing<br/>936_251637_analysis_doc.ipynb"]
        B1["Initialize BACKBONE<br/>PopulationRegionAnalysis"] --> B2["Process 562 Neurons"]
        B2 --> B3["For Each Neuron:"]
        B3 --> B4["BACKBONE: Load SWC"]
        B4 --> B5["BACKBONE: Build Branch Topology"]
        B5 --> B6["BACKBONE: Calculate Metrics"]
        B6 --> B7["BACKBONE: Classify Subtype<br/>PT/CT/ITs/ITc/ITi"]
        B7 --> B8["Store in DataFrame"]
        B8 --> B9{"More neurons?"}
        B9 -->|Yes| B3
        B9 -->|No| B10["Complete"]
    end

    subgraph Visualization["📈 Population Visualizations"]
        C1["Type Distribution<br/>Pie Chart"]
        C2["Soma Distribution<br/>Bar Chart"]
        C3["Terminal Distribution<br/>Bar Chart"]
        C4["Projection Count<br/>Histogram + Boxplot"]
    end

    subgraph Tables["📊 Generate Recording Tables<br/>(Backbone Output)"]
        D1["Extract ACC<br/>(Cingulate neurons)"]
        D2["Extract INS<br/>(Insula neurons)"]
        D3["Save ACC_df.xlsx"]
        D4["Save INS_df.xlsx"]
    end

    subgraph VTK["🔍 Visual Toolkit Integration"]
        E1["Review Backbone Tables"] --> E2["Identify Neurons<br/>of Interest"]
        E2 --> E3["Select by:<br/>• Subtype (PT/CT/IT)<br/>• Soma Region<br/>• Terminal Targets"]
        E3 --> E4["Extract NeuronID"]
        E4 --> E5["Launch Visual Toolkit<br/>with selected neuron"]
    end

    Setup --> Processing --> Visualization
    Processing --> Tables --> VTK

    style Processing fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Tables fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

**Key Points:**
- **Entry point** for the analysis pipeline
- **Orchestrates the backbone** (region_analysis) to process 562 neurons
- Uses **Level 6** ARM atlas (most detailed parcellation)
- Generates **recording sheets** (ACC_df, INS_df) - backbone output used by other modules
- Tables feed into **Visual Toolkit** for targeted inspection

---

## 5. Integrated Workflow with Backbone

**File:** `integrated_workflow_flowchart.md`

**Purpose:** Shows the **central role of region_analysis as backbone** - feeding both FNT-Dist Clustering and Visual Toolkit.

```mermaid
flowchart TB
    subgraph RawData["📦 Raw Data"]
        R1["fMOST Images<br/>Sample: 251637"]
        R2["562 SWC Files<br/>processed_neurons/"]
        R3["ARM Atlas<br/>NMT v2.1 sym space"]
        R4["Atlas Key Table"]
    end

    subgraph Backbone["🔷 BACKBONE<br/>region_analysis.py"]
        B1["Load & Process SWCs"] --> B2["Atlas Mapping<br/>Branch Lengths"]
        B2 --> B3["Soma & Terminal ID"] --> B4["Classification"]
        B4 --> B5["Projection Subtypes<br/>PT / CT / ITs / ITc / ITi"]
        B5 --> B6["Recording Tables<br/>ACC_df / INS_df"]
    end

    subgraph FNT["🎯 FNT-Dist Clustering"]
        F1["SWC → FNT"] --> F2["Distance Matrix"]
        F2 --> F3["Ward Clustering"]
        F3 --> F4["Morphological<br/>Clusters"]
    end

    subgraph VTK["🔍 Visual Toolkit"]
        V1["Select from Tables"] --> V2["Auto-Fill Soma"]
        V2 --> V3["High-Res DL"] --> V4["MIP Soma View"]
        V2 --> V5["Low-Res DL"] --> V6["MIP + SWC Overlay"]
    end

    subgraph Outputs["📤 Outputs"]
        O1["Projection Subtypes<br/>(from Backbone)"]
        O2["Morphological Clusters<br/>(from FNT)"]
        O3["Illustrative Views<br/>(from VTK)"]
    end

    RawData --> Backbone
    
    Backbone -.->|subtype metadata| FNT
    Backbone -.->|recording tables| VTK
    
    RawData -.->|SWC files| FNT
    
    Backbone --> O1
    FNT --> O2
    VTK --> O3

    style Backbone fill:#e3f2fd,stroke:#1565c0,stroke-width:5px
    style FNT fill:#fff3e0,stroke:#ef6c00
    style VTK fill:#f3e5f5,stroke:#6a1b9a
```

---

## Backbone Communication Flow

```mermaid
sequenceDiagram
    participant User
    participant Notebook as Analysis Notebook
    participant Backbone as Region Analysis<br/>(Backbone)
    participant FNT as FNT-Dist Clustering
    participant VTK as Visual Toolkit
    participant Files as Output Files

    %% Setup Backbone
    User->>Notebook: Run 936_251637_analysis.ipynb
    Notebook->>Backbone: Initialize PopulationRegionAnalysis
    
    %% Backbone Processing
    loop Process 562 neurons
        Backbone->>Backbone: Load SWC
        Backbone->>Backbone: Atlas mapping
        Backbone->>Backbone: Calculate lengths
        Backbone->>Backbone: Classify subtype (PT/CT/IT)
    end
    
    %% Backbone Output
    Backbone->>Files: Save ACC_df.xlsx
    Backbone->>Files: Save INS_df.xlsx
    Note over Files: Recording Tables with Subtypes
    
    %% FNT-Dist uses Backbone
    User->>FNT: Run fnt_dist_clustering.py
    FNT->>Files: Read ACC_df/INS_df
    Files-->>FNT: Subtype metadata
    FNT->>FNT: Apply penalty using subtypes
    FNT->>FNT: Ward clustering
    FNT->>Files: Save morphological clusters
    
    %% Visual Toolkit uses Backbone
    User->>VTK: Launch Visual_toolkit_gui.py
    VTK->>Files: Load ACC_df/INS_df
    Files-->>VTK: Recording tables
    VTK->>VTK: Display table for selection
    User->>VTK: Select neuron (e.g., PT from CL)
    VTK->>VTK: Download & visualize
    VTK->>Files: Save illustrations
```

---

## Key Relationships & Data Flow

```
                    ┌─────────────────────────────────────┐
                    │          Raw Data                   │
                    │  (562 SWCs + ARM Atlas + fMOST)    │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     🔷 BACKBONE: region_analysis    │
                    │  ┌─────────────────────────────┐   │
                    │  │ • Process all 562 neurons   │   │
                    │  │ • Atlas mapping             │   │
                    │  │ • Projection subtypes       │   │
                    │  │   (PT/CT/ITs/ITc/ITi)       │   │
                    │  │ • Recording tables          │   │
                    │  │   (ACC_df/INS_df)           │   │
                    │  └─────────────────────────────┘   │
                    └────────────┬────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ FNT-Dist        │  │ Visual Toolkit  │  │ Direct Outputs  │
   │ Clustering      │  │                 │  │                 │
   ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
   │ • Uses subtype  │  │ • Uses recording│  │ • Subtypes      │
   │   info from     │  │   tables from   │  │ • Heatmaps      │
   │   backbone for  │  │   backbone for  │  │ • Statistics    │
   │   penalty       │  │   selection     │  │                 │
   │ • Morphological │  │ • Dual-res      │  │                 │
   │   clusters      │  │   illustration  │  │                 │
   └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Important Distinctions

### Projection Subtypes (from Backbone) vs Morphological Clusters

| Aspect | Projection Subtypes | Morphological Clusters |
|--------|---------------------|------------------------|
| **Source** | `region_analysis.py` (**Backbone**) | `fnt_dist_clustering.py` |
| **Input from Backbone** | Direct processing | Uses tables for penalty |
| **Basis** | Target regions (where axons go) | Tree structure (shape similarity) |
| **Types** | PT, CT, ITs, ITc, ITi | Cluster 1, 2, 3... (data-driven) |
| **Method** | Hierarchical rules | Hierarchical clustering + C-index |
| **Output Location** | ACC_df/INS_df columns | Separate .xlsx file |

### Backbone Outputs & Consumers

| Backbone Output | Consumer Module | Usage |
|-----------------|-----------------|-------|
| `ACC_df.xlsx` | Visual Toolkit | Neuron selection by subtype/region |
| `ACC_df.xlsx` | FNT-Dist | Subtype metadata for penalty |
| `INS_df.xlsx` | Visual Toolkit | Neuron selection by subtype/region |
| `INS_df.xlsx` | FNT-Dist | Subtype metadata for penalty |
| Projection subtypes | All downstream | PT/CT/IT classification |

---

## File Locations

| Flowchart | File |
|-----------|------|
| FNT-Dist Clustering | `fnt_clustering_flowchart.md` |
| Visual Toolkit | `visual_toolkit_flowchart.md` |
| Region Analysis | `region_analysis_flowchart.md` |
| Analysis Notebook | `analysis_notebook_flowchart.md` |
| Integrated Workflow | `integrated_workflow_flowchart.md` |
| This Summary | `all_flowcharts_summary.md` |

---

## Usage Workflows

### Workflow 1: Backbone-First (Standard)
```
936_251637_analysis_doc.ipynb
    → region_analysis.py (BACKBONE)
        → ACC_df.xlsx / INS_df.xlsx
            → Used by Visual Toolkit
            → Used by FNT-Dist (optional penalty)
```

### Workflow 2: Visual Inspection (Uses Backbone Output)
```
ACC_df/INS_df (from BACKBONE)
    → Visual_toolkit_gui.py
        → Select by subtype (PT/CT/IT)
        → Download & illustrate
```

### Workflow 3: Morphological Clustering (Optionally Uses Backbone)
```
SWC files + ACC_df/INS_df (subtype info from BACKBONE)
    → fnt-dist_pipeline.py
        → fnt_dist_clustering.py
            → Apply supervised penalty (using backbone subtypes)
            → Generate morphological clusters
```
