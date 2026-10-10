# Projectome Data Analysis

A comprehensive toolkit for analyzing macaque brain neuron morphology data, including visualization, clustering, and distance analysis workflows.

Current insula evolution work: [dated evidence index](group_analysis/evolution_20261008/README.md) and [terminal-first goal](group_analysis/evolution_20261008/terminal_projection_goal_20261008.md). Descriptive mapping and software verification remain separate from anatomical acceptance.

Current projection-map review: [second inspection and endpoint-marker sheets](notes/projection_map_review_round2/README.md), with [step-by-step reproduction](notes/projection_map_review_round2/reproduction.md) and dependency-aware local archive records. Original numerical maps and source-bound historical figures retain their established paths.

Current soma-origin maps: [clearly titled source maps, all-462 evidence views and July/September per-monkey count reconciliation](notes/projection_maps_by_origin/README.md), with [reproduction scripts and procedures](notes/projection_maps_by_origin/reproduction.md). The 301 named ARM-insula assignments are a subset of the 462 selected neurons.

Review the work in development order using the [step-by-step insula audit](docs/insula_pipeline_development_audit_20261010.md), with evidence links, expected checks and separate computational/anatomical decisions.

The [region/table audit](notes/region_analysis_review_20261009/README.md) records validated repairs, corrected diagnostic workbooks and scientific dependencies. Use the [measurement guide](docs/region_analysis_terminology.md), [primary ARM map index](group_analysis/evolution_20261008/arm_mapping_20261009/README.md), [six-level projection tables](notes/region_analysis_review_20261009/hierarchy_tables_20261009/README.md), [projection-profile clustering](notes/region_analysis_review_20261009/clustering_20261009/README.md) and [MSTIM integration record](notes/region_analysis_review_20261009/mstim_integration_20261009/README.md). Earlier display variants remain archived.

## Overview

This repository contains tools for processing and analyzing fMOST (fluorescence Micro-Optical Sectioning Tomography) neuron data, with a focus on:

- **Neuron Visualization**: High and low-resolution visualization of neuron morphology
- **Clustering Analysis**: Fast Neurite Tracer (FNT) morphology dissimilarities with identity and stability diagnostics
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

5. Atlas mesh extraction (step 3.4) also needs:
   - Local clone `subcortex_visualization/` (upstream: [anniegbryant/subcortex_visualization](https://github.com/anniegbryant/subcortex_visualization))
   - Monkey renderer `subcortex_visualization/monkey_atlas_guide/volume_to_mesh_mz3_MONKEY.py`
   - `niimath` on PATH (mesh generation; [rordenlab/niimath](https://github.com/rordenlab/niimath/releases))

## Pipeline Overview

The analysis workflow consists of 5 main steps:

```
┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
│  INPUT  │───▶│  STEP 1     │───▶│  STEP 2     │───▶│  STEP 3     │───▶│  STEP 5: Additional │
│ Sample  │    │  Region     │    │  FNT        │    │  Bulk Viz   │    │  Rendering          │
│   ID    │    │  Analysis   │    │  Pipeline   │    │             │    │                     │
└─────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────────────┘
                      │                                    │                      │
                      ▼                                    ▼                      ▼
               ┌─────────────┐                      ┌─────────────┐    ┌─────────────────────┐
               │ Neuron      │                      │ Plots &     │    │ • Brain Mesh Render │
               │ Tables      │                      │ NIfTI       │    │ • NeuronView Render │
               │ (.xlsx)     │                      │             │    │ • Atlas Mesh (.mz3) │
               └─────────────┘                      └─────────────┘    │ • Clustered Heatmap │
                                                                       └─────────────────────┘
```

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `step1.run_region_analysis.py` | Population region analysis, hierarchy, laterality | `neuron_tables/*.xlsx` |
| 2 | `step2.fnt-dist_pipeline.py` | SWC→FNT conversion, decimation, distance matrix | `*_joined.fnt`, `*_dist.txt` |
| 3 | `step3.1.bulk_visual_data.py` | High/low resolution visualization | `.png`, `.nii.gz` |
| 4 | R analysis scripts | Statistical analysis, clustering | `heatmap_LR_combined_split.pdf` |
| 5.1 | `step3.2.run_brain_viz_meshRender.py` | 3D brain mesh rendering with regions | `brain_viz_*.png` |
| 5.2 | `step3.3.neuronviewRender.py` | Interactive 3D neuron visualization | Interactive GL window |
| 5.3 | `step3.4.mesh_extraction.py` | ARM/NMT volume → Surf Ice `.mz3` (monkey atlas mesh) | `mesh_outputs/*.mz3` |
| 5.4 | `visualize_clustered_heatmap.py` | Gou 2025 Fig2A style clustered heatmap | `fig2a_style_heatmap.png` |

See `main_scripts/PIPELINE_MINDMAP.md` for detailed flowcharts and dependencies.

---

## Main Components

### 1. Visual Toolkit (`main_scripts/Visual_toolkit.py`)

A unified tool for retrieving and visualizing Macaque brain data from mixed sources.

**Features:**
- **High Resolution**: Own-sample HTTP cubes; nominal XYZ spacing `[0.65, 0.65, 3]` µm
- **Low Resolution**: Own-sample copied overview sections or explicitly configured SSH series; nominal XYZ spacing `[5, 5, 3]` µm
- **SWC Overlay**: Native microscopy traces with explicit crop geometry and missing-source provenance
- **Export Formats**: XYZ NIfTI volumes with micron units and YX TIFF planes, each with a JSON sidecar

Crop centers come from validated own-sample **raw SWC roots** in declared native microscopy µm. Portal/template NMT soma coordinates belong to a different frame. The acquisition API returns `(volume_zyx, origin_xyz_um, spacing_xyz_um)`. Spacing and axes are repository declarations; physical calibration, laterality and anatomical acceptance require independent evidence. Partial acquisition requires explicit `allow_partial=True` and valid central source data.

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
# Replace these example coordinates with the validated own-sample raw SWC root.
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

**Native review batches on the current branch:**

`group_analysis/scripts/visual_review_20261002.py` builds the candidate manifest, native panels, gallery and editable correction table. Selection includes INS/PrCO, atlas Unknown and potential nearby regions while retaining portal, historical, harmonized and coordinate-inference origins. Copied-overview availability, accessible soma components and production coverage are separate states. Canonical cohorts and human labels retain their own review gates.

Run one production batch at a time and resume from the recorded package state:

```bash
python -B group_analysis/scripts/visual_review_20261002.py manifest
python -B group_analysis/scripts/visual_review_20261002.py render
python -B group_analysis/scripts/visual_review_20261002.py render-soma
python -B group_analysis/scripts/visual_review_20261002.py gallery
python -B group_analysis/scripts/visual_review_20261002.py verify
```

`render` uses copied overview sources; `render-soma` extends soma-detail production into overview-pending samples. Both accept repeatable `--sample` selectors. Human region/subregion/layer and reviewer columns are merged by UID, with conflicts preserved for resolution. The repaired derived-context sampler has source-grid, missing-center, artifact-preservation and bounded-download regressions; one real-source pilot passed pixel/affine/hash readback with partial field coverage (249 of 324 tiles loaded). Derived fields use nominal 5.2 x 5.2 x 3 um point decimation and retain full-section locators as unavailable components. See the [repair validation](notes/bulk_visual_review_20261002/derived_context_repair_validation_20261003.json). Soma-detail completion is a milestone; all-nine regional-context production and final verification remain part of the active goal. See the [living project note](docs/project_note.md), [visual methods](notes/bulk_visual_review_20261002/visual_methods.md), and [current review findings](notes/bulk_visual_review_20261002/CURSOR_STAGE2_REVIEW_ADDENDUM.md).

After soma production is terminal, regional contexts for the five monkeys without copied overview sections use a separate resumable command:

```bash
python -B group_analysis/scripts/regional_context_batch_20261003.py --cache-gib 300 --min-free-gib 50 --max-requests 60000 --max-seconds 28800
```

The driver prioritizes insula candidates, keeps every selected UID, and records per-identity source failures and resource stops. It uses a declared 2.8 mm fallback when a 4 mm field exceeds the per-context bounds. Cached complete fields and partial fields with observed HTTP-404 gaps can resume; resource/transient partials are retried. `--retry-missing` rechecks previously missing source tiles. A live process lock prevents overlapping regional runs. Coverage panels distinguish acquired zero intensity from unavailable source pixels; the actual FOV, spacing and acquired fraction remain in the review artifacts. The [99-test snapshot](notes/bulk_visual_review_20261002/regional_driver_validation_20261003.json) and [real saved-pilot resume check](notes/bulk_visual_review_20261002/independent_regional_resume_validation_20261003.json) validate these software contracts; Cursor regional production is active for 252790; remaining monkeys and final verification stay in scope.

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

Exploratory clustering of neurons using validated FNT dissimilarities. The
default is raw scores with average linkage and no biological-type penalty.
Joined-FNT markers define the neuron order. Missing pairs, missing annotations,
invalid distances and unsupported Ward geometry fail explicitly.

`--mode spearman-profile` compares distance-to-cohort rank profiles;
`--mode log1p` compresses score magnitude. `--supervised-penalty` is an explicit
type-guided sensitivity mode whose type agreement cannot independently validate
type enrichment. Inspect candidate-k curves, clusterwise stability and animal
sensitivity before accepting any taxonomy. See
[the current clustering review](notes/clustering_review_20261002/README.md).

### 4. Region Analysis (`main_scripts/region_analysis/`)

Anatomical region-based analysis of neuron projections.

**Key Scripts:**
- `step1.run_region_analysis.py` - Main region analysis pipeline
- `region_analysis/getNeuronListByRegion.py` - Query neurons by region
- `region_analysis/population.py` - Population-level analysis
- `region_analysis/hierarchy.py` - Cortex/Subcortex hierarchy
- `region_analysis/laterality.py` - Laterality analysis

### 5. Additional Rendering Tools

#### 5.1 Brain Mesh Render (`step3.2.run_brain_viz_meshRender.py`)
3D brain surface visualization with region meshes.

**Features:**
- Extract region meshes from ARM atlas
- Visualize neurons by type (ITs, ITi, CT, PT)
- Multiple rendering examples (1-6)

```bash
python main_scripts/step3.2.run_brain_viz_meshRender.py 2
```

#### 5.2 NeuronView Render (`step3.3.neuronviewRender.py`)
Interactive OpenGL-based neuron visualization.

**Features:**
- Interactive 3D navigation
- Type-based neuron coloring
- Region overlay from .obj files

```bash
python main_scripts/step3.3.neuronviewRender.py
```

#### 5.3 Atlas Mesh Extraction (`step3.4.mesh_extraction.py`)
Convert ARM atlas regions or the NMT brainmask to a color-coded `.mz3` mesh for [Surf Ice](https://github.com/neurolabusc/surf-ice).

This is the macaque fork of Bryant’s `subcortex_visualization` custom-segmentation renderer. The mesh engine is `volume_to_mesh_mz3_MONKEY.py` (ARM hierarchy levels, non-sequential region IDs). Wrappers: `monkey_atlas_mesh.py`, `mesh_from_atlas.py`.

**Features:**
- Atlas mode: extract regions by name substring, ARM ID, or abbreviation at a chosen hierarchy level (1–6)
- Template mode: whole-brain mesh from `NMT_v2.1_sym_brainmask.nii.gz`
- Output for Surf Ice / Inkscape 2D tracing (Bryant pipeline)

```bash
python main_scripts/step3.4.mesh_extraction.py
```

Edit `MODE`, `ATLAS_REGION_NAMES` / `ATLAS_REGION_IDS`, and `HIERARCHY_LEVEL` at the top of the script before running. Requires `niimath` on PATH.

#### 5.4 Clustered Heatmap (`visualize_clustered_heatmap.py`)
Generate publication-quality heatmaps (Gou et al. 2025 style).

**Features:**
- Morphological cluster visualization
- Neuron type color bars
- Log-transformed projection strength

```bash
python main_scripts/visualize_clustered_heatmap.py
```

## Directory Structure

```
projectome_analysis/
├── main_scripts/                  # Core analysis scripts
│   ├── step1.run_region_analysis.py      # Step 1: Region analysis
│   ├── step2.fnt-dist_pipeline.py        # Step 2: FNT distance pipeline
│   ├── step3.1.bulk_visual_data.py       # Step 3: Bulk visualization
│   ├── step3.2.run_brain_viz_meshRender.py  # Step 5.1: Brain mesh render
│   ├── step3.3.neuronviewRender.py       # Step 5.2: NeuronView render
│   ├── step3.4.mesh_extraction.py        # Step 5.3: ARM/NMT → .mz3 mesh
│   ├── monkey_atlas_mesh.py       # Import wrapper for monkey mesh renderer
│   ├── mesh_from_atlas.py         # CLI wrapper around volume_to_mesh_mz3_MONKEY
│   ├── visualize_clustered_heatmap.py    # Step 5.4: Clustered heatmap
│   ├── Visual_toolkit.py          # Main visualization toolkit
│   ├── Visual_toolkit_gui.py      # GUI for visualization
│   ├── brain_viz.py               # Brain visualization class
│   ├── fnt_dist_clustering.py     # Clustering algorithms
│   ├── fnt_tools.py               # FNT utility functions
│   ├── region_analysis/           # Region analysis modules
│   │   ├── getNeuronListByRegion.py
│   │   ├── population.py
│   │   ├── hierarchy.py
│   │   └── laterality.py
│   └── subsidary_functions/       # Helper functions
│
├── R_analysis/                # R-based statistical analysis
│   └── scripts/               # R analysis scripts
├── fnt_dist_on_cluster/       # HPC cluster job scripts
├── atlas/                     # Atlas data (ARM, CHARM, SARM)
├── subcortex_visualization/   # Bryant toolbox + monkey_atlas_guide renderer
│   └── monkey_atlas_guide/volume_to_mesh_mz3_MONKEY.py
├── mesh_outputs/              # .mz3 meshes from step 3.4 (often gitignored)
├── literature/                # Reference papers
├── resource/                  # Data cache (gitignored)
├── processed_neurons/         # Processed neuron files (gitignored)
├── brain_viz_output/          # Brain visualization outputs
├── figures_charts/            # Generated figures
├── deb_fmost.yml              # Conda environment
├── main_scripts/PIPELINE_MINDMAP.md  # Detailed pipeline documentation
└── README.md                  # This file
```

## Configuration

### Low-resolution source configuration

Pass the requested sample's copied directory explicitly, or use its existing own-sample directory mapping:
```python
toolkit = Visual_toolkit(sample_id, low_res_dir=own_sample_directory)
```

For an own-sample SSH series, pass `low_res_ssh_base` and configure the connection separately. The legacy 251637 directory belongs to 251637. An absent copied overview is recorded as a missing component while its raw SWC and high-resolution source routes are evaluated independently.

Optional SSH reads its password from `PROJECTOME_SSH_PASSWORD`; keep credentials outside source files. [Validation requirements](requirements-validation.txt) pin the observed Python packages for the dated audit, while atlas/source files and external tools remain separately provisioned.

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

```powershell
# Historical 306-neuron matrix: review only; it does not cover the 353/420 cohorts.
python -B main_scripts/fnt_dist_clustering.py `
    --dist-file group_analysis/fnt/multi_monkey_INS_dist.txt `
    --joined-fnt group_analysis/fnt/multi_monkey_INS_joined.fnt `
    --type-file group_analysis/combined/multi_monkey_INS_combined.xlsx `
    --sheet Summary --mode raw --linkage average `
    --max-k 20 --repeats 100 --seed 42 --no-plots `
    --output-dir output/clustering_review
```

Use the projectome Python environment. Add `--k 9` only for an explicit
exploratory cut; otherwise the silhouette suggestion is recorded as exploratory.
Output includes every candidate partition, requested/realized cluster count,
silhouette/C-index, subset ARI, clusterwise Jaccard, animal-removal sensitivity,
input/source hashes and software versions. Singleton stability is unassessed.
These diagnostics do not establish biological subtypes or new-animal prediction.
The historical R clustering scripts retain their old settings and are separate
analysis paths; they were not rerun by this review.

## Output Files

### Visualization Outputs
- `.nii.gz` - Persisted XYZ volumes with micron units; source arrays are ZYX
- `.tif` - YX planes or caller-prepared projections
- `.json` sidecars - Crop origin, spacing, raw root when supplied, matching acquisition and explicit missing coverage
- `_Plot.png` - Annotated visualization plots

### Analysis Outputs
- `.fnt` - FNT format neuron files
- `cluster_assignments.csv` / `candidate_cluster_assignments.csv` - Exploratory selected/all-candidate assignments
- `clustering_diagnostics.csv`, `cluster_stability_k*.csv`, `group_holdout_diagnostics.csv` - Separation and stability diagnostics
- `clustering_metadata.json` - Cohort identity, settings, hashes, software versions and interpretation limits
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

**Last Updated:** October 9, 2026

## Changelog

### August 2026
- **Documented Step 5.3: Atlas mesh extraction**
  - `step3.4.mesh_extraction.py` — ARM/NMT volume → Surf Ice `.mz3`
  - Engine: `subcortex_visualization/monkey_atlas_guide/volume_to_mesh_mz3_MONKEY.py` (fork of [anniegbryant/subcortex_visualization](https://github.com/anniegbryant/subcortex_visualization))
  - Clustered heatmap renumbered to Step 5.4

### March 2026
- **Added Step 5: Additional Rendering**
  - `step3.2.run_brain_viz_meshRender.py` - 3D brain mesh visualization
  - `step3.3.neuronviewRender.py` - Interactive OpenGL neuron rendering
  - `visualize_clustered_heatmap.py` - Gou 2025 Fig2A style heatmaps
- **Restructured pipeline scripts** with step numbering (step1, step2, step3.1, etc.)
- **Added PIPELINE_MINDMAP.md** with detailed flowcharts and Mermaid diagrams
- **Enhanced region analysis** with modular architecture in `region_analysis/`
