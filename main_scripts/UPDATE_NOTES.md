# Projectome Analysis - Update Notes

## Overview

This document tracks the evolution of the projectome analysis pipeline, from initial monolithic scripts to the current modular architecture.

---

## 1. Evolution of Region Analysis

### Phase 1: Initial Single-File Implementation

**Files:** `region_analysis.py` (monolithic, ~2000+ lines)

**Characteristics:**
- Single Python file containing all analysis logic
- Direct SWC processing with inline coordinate transformations
- Hardcoded ARM atlas level mappings
- Basic classification rules (PT/CT/IT) embedded in main class
- Simple Excel output with limited columns

**Limitations:**
- Difficult to maintain and extend
- No hierarchy support beyond L1-L2 from ARM key
- Limited laterality handling (no ipsi/contra separation)
- No modular visualization components

---

### Phase 2: Modular Package Structure

**Current Structure:** `region_analysis/` package (~5000 lines)

```
region_analysis/
├── __init__.py                 # Package exports
├── population.py               # Main orchestrator (PopulationRegionAnalysis)
├── neuron_analysis.py          # Per-neuron analysis (RegionAnalysisPerNeuron)
├── hierarchy.py                # Hierarchy resolution & aggregation
├── hierarchy_table.py          # CHARM/SARM CSV parsing
├── classifier.py               # Neuron classification (PT/CT/ITs/ITc/ITi)
├── laterality.py               # Ipsilateral/contralateral parsing
├── plotting.py                 # Visualization functions
├── output_manager.py           # File organization
└── utils.py                    # Helper utilities
```

**Key Improvements:**

| Feature | Phase 1 | Phase 2 (Current) |
|---------|---------|-------------------|
| **Hierarchy Support** | ARM key only (L1-L2) | CHARM/SARM CSV (L1-L6) with index-based lookup |
| **Laterality** | Basic side detection | Full ipsi/contra classification with split sheets |
| **Modularity** | Single file | 11 modules with clear separation |
| **Output Format** | Single Excel | Multi-sheet with separate ipsi/contra projections |
| **Hierarchy Columns** | None | Region_Projection_Length_L1-L6 |
| **Prefix Handling** | None | CL_/CR_/SL_/SR_ stripped for display, kept for classification |

---

### Phase 3: Latest Updates (Current)

**Recent Changes:**

1. **Index-Based Hierarchy Lookup**
   - ARM indices (1001, 1004, etc.) now resolve correctly to hierarchy paths
   - Dual table system (cortex + subcortical) prevents region name conflicts

2. **Level Mapping Fix**
   - CSV has Level_0 as root → adjusted indexing to skip Level_0
   - Level 3 now correctly returns Level 3 regions (not Level 2)

3. **Laterality-Split Output**
   - Separate ipsi/contra sheets for projection length and strength
   - Prefixes stripped only after classification (prevents mixing SL_/SR_)

4. **Code Cleanup**
   - Removed ~200 lines of debug prints
   - Simplified hierarchy loading
   - Removed unused soma hierarchy columns

---

## 2. Analysis Pipeline Structure

### Simplified Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA                                       │
│  (fMOST Images + SWC Files + ARM Atlas NMT v2.1)                        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  🔷 BACKBONE: Region Analysis (region_analysis/)                        │
│                                                                          │
│  • Load & Process SWCs    →  neuro_tracer.process()                     │
│  • Atlas Mapping          →  ARM Level 6 parcellation                   │
│  • Calculate Branch Lengths per region                                  │
│  • Identify Soma & Terminals (CL/CR/SL/SR)                              │
│  • Classify Projection Subtypes:                                        │
│    - PT (Pyramidal Tract)                                               │
│    - CT (Corticothalamic)                                               │
│    - ITs (Intratelencephalic - Striatum)                                │
│    - ITc (Interhemispheric - Contra Cortex)                             │
│    - ITi (Ipsilateral - Ipsi Cortex only)                               │
│  • Generate Recording Tables: ACC_df.xlsx / INS_df.xlsx                 │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ├─────────────────────────────┬──────────────────────────────┐
           │                             │                              │
           ▼                             ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ FNT-Dist Clustering │    │   Visual Toolkit    │    │   Direct Outputs    │
│                     │    │                     │    │                     │
│ • Morphological     │    │ • Select neurons    │    │ • Subtype stats     │
│   clustering based  │    │   from backbone     │    │ • Projection        │
│   on tree structure │    │   tables            │    │   matrices          │
│ • Uses subtype info │    │ • Dual-resolution   │    │ • Laterality split  │
│   from backbone     │    │   visualization:    │    │   sheets            │
│   (optional penalty)│    │   - High-res soma   │    │                     │
│ • Output:           │    │   - Low-res context │    │                     │
│   Morphological     │    │ • MIP + SWC overlay │    │                     │
│   subtypes          │    │                     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### Module Details

#### A. Region Analysis (Backbone)
**Entry Point:** `PopulationRegionAnalysis.process()`

**Key Features:**
- **Multi-level hierarchy support:** CHARM (cortical) + SARM (subcortical)
- **Laterality classification:** Automatic ipsi/contra based on soma location
- **Output sheets:** Summary | Projection_Length | Projection_Strength | Laterality | Outliers | Ipsi_Len | Contra_Len | Ipsi_Str | Contra_Str

#### B. FNT-Dist Clustering
**Entry Point:** `fnt_dist_clustering.py`

**Pipeline:** SWC → FNT conversion → Decimate → Join → Distance matrix → Ward clustering → C-index optimization

**Relationship to Backbone:** Optionally uses subtype metadata from ACC_df/INS_df for supervised penalty

#### C. Visual Toolkit
**Entry Point:** `Visual_toolkit_gui.py`

**Features:**
- Reads recording tables (ACC_df/INS_df) from backbone
- Filter neurons by subtype (PT/CT/IT), soma region, terminal targets
- High-resolution download (0.65µm) for soma detail
- Low-resolution download (5.0µm) for projection context

---

## 3. TODO / To Update

### High Priority

- [ ] **Include Hierarchy in Final Table**
  - Add `Region_Projection_Length_L1` through `L6` columns to final output
  - Ensure hierarchy columns are properly populated with aggregated data
  - Verify level mapping correctness (L3 should show L3 regions, not L2)

- [ ] **Hierarchy Validation**
  - Add validation to ensure all regions map correctly to their target levels
  - Log unmapped regions for debugging

### Medium Priority

- [ ] **Documentation**
  - Update docstrings for all public methods
  - Add usage examples for common workflows

- [ ] **Testing**
  - Add unit tests for hierarchy resolution
  - Add integration tests for full pipeline

### Low Priority

- [ ] **Performance**
  - Optimize DataFrame operations for large neuron counts
  - Cache hierarchy lookups

---

## 4. Usage Quick Reference

### Basic Usage

```python
import region_analysis as ra
import nibabel as nib
import pandas as pd

# Load atlas
atlas_nii = nib.load('ARM_in_NMT_v2.1_sym.nii.gz')
atlas_data = atlas_nii.get_fdata()
table = pd.read_csv('ARM_key_all.txt', delimiter='\t')

# Initialize backbone
pop = ra.PopulationRegionAnalysis(
    '251637', 
    atlas_data, 
    table,
    cortex_hierarchy_csv='CHARM_key_table_v2.csv',
    subcortical_hierarchy_csv='SARM_key_table_v2.csv'
)

# Process all neurons
pop.process(level=6)

# Export results
pop.save_all()
```

### Output Structure

```
output/
├── tables/
│   └── {sample_id}_results.xlsx
│       ├── Summary                 # Basic stats
│       ├── Projection_Length       # Full projection matrix
│       ├── Projection_Strength     # log10(length + 1)
│       ├── Projection_Length_L3    # Level 3 hierarchy
│       ├── Projection_Strength_L3  # Level 3 strength
│       ├── Laterality              # Ipsilateral/contralateral counts
│       ├── Projection_Length_ipsi  # Ipsi-only regions (no prefix)
│       ├── Projection_Length_contra # Contra-only regions (no prefix)
│       ├── Projection_Strength_ipsi
│       ├── Projection_Strength_contra
│       ├── Terminal_Sites          # Long-format terminals
│       └── Outliers                # Unknown region coordinates
├── reports/
│   ├── analysis_summary.txt
│   ├── terminal_report.txt
│   └── projection_sites_report.txt
└── plots/
    ├── type_distribution.png
    ├── soma_distribution.png
    ├── terminal_distribution.png
    └── projection_sites_count.png
```

---

## 5. File Locations

| Component | Path |
|-----------|------|
| Region Analysis Package | `main_scripts/region_analysis/` |
| Analysis Notebook | `main_scripts/936_251637_analysis_doc.ipynb` |
| FNT-Dist Clustering | `main_scripts/fnt_dist_clustering.py` |
| Visual Toolkit | `main_scripts/Visual_toolkit_gui.py` |
| CHARM Hierarchy | `atlas/CHARM_key_table_v2.csv` |
| SARM Hierarchy | `atlas/SARM_key_table_v2.csv` |
| ARM Key | `atlas/NMT_v2.1_sym/tables_ARM/ARM_key_table.csv` |
| Workflow Docs | `figures_charts/workflows/` |

---

*Last Updated: March 2026*
