# Handoff: Liu et al. 2026 — AIC VEN Atlas → fMOST/fMRI Integration

> **Written**: 2026-07-07 | **For**: Cursor (projectome_analysis)
> **Paper**: Liu R-F, Huang M, Shen Y, Shao M, Jing J, Xu N, et al. "An atlas of primate insular cortex reveals a signal-processing strategy in von Economo neurons." *Nature Cell Biology*, 2026 Jul 2. DOI: `10.1038/s41556-026-02009-4`

---

## 1. Why This Paper Matters for Our Project

We have:
- **fMOST whole-brain morphologies** (SWC, thousands of neurons, macaque `251637`) — projection-based classification (PT/CT/ITs/ITc/ITi)
- **7T fMRI insula activation maps** (cm043/cm044, QST cooling paradigm) — BOLD response in AIC
- **Missing**: cell-type identity of insula neurons in our fMOST data

This paper provides:
- **Complete molecular atlas of macaque AIC**: 78 cell types from >150K scRNA-seq + 578 Patch-seq neurons
- **Two VEN subtypes** with marker genes: VEN-L (DSG2+/HAPLN4+) & VEN-S (POC5+/COL24A1+)
- **231 fully reconstructed 3D morphologies** on NeuroMorpho.Org (ASC format)
- **Morphology→transcriptome correlation framework** (Fig 7, Spearman-based phenotype-genotype coupling)

**Integration goal**: Identify VEN candidates in our fMOST insula data by morphological similarity scoring, then correlate with fMRI activation patterns.

---

## 2. Key Findings (What Cursor Needs to Know)

### 2.1 VEN Classification
| Subtype | Marker | Morphology | Transcriptomic Affinity |
|---------|--------|-----------|------------------------|
| VEN-L | DSG2, HAPLN4, TOX2 | Thick basal dendrite >500μm, symmetric spindle | L5 ET (extratelencephalic) |
| VEN-S | POC5, COL24A1, ATP8A2 | Shorter basal dendrite, rapid bifurcation | L5/6 CT (corticothalamic) |

### 2.2 Signal-Processing Strategy
- Axon originates from **basal dendrite** (not soma) — unique among cortical neurons
- AIS farther from soma, shorter → lower Na⁺ current → broader AP
- VENs are **functionally isolated** in local circuits: zero VEN↔VEN / VEN↔PC / VEN↔IN connections found (24/24 connections were PC↔PC)
- VENs respond **more efficiently to deep-layer inputs** (larger EPSP amplitude) but receive fewer total inputs
- **Interpretation**: VENs are long-range projection specialists, not local processors

### 2.3 Metabolism
- AIC is glycolysis-dominant (TCA cycle ↓) — may influence BOLD signal interpretation

---

## 3. Data Repositories (Ready to Download)

| Data Type | Repository | Accession | Format | Size |
|-----------|-----------|-----------|--------|------|
| **scRNA-seq** | GEO | `GSE319557` | h5ad / MTX+TSV | 23 GB / 1.5 GB |
| **Patch-seq** | GEO | `GSE319369` | FASTQ | — |
| **Patch-seq ephys** | DANDI | `dandiset/001746` | NWB | — |
| **Na⁺ currents** | DANDI | `dandiset/001750` | NWB | — |
| **Multi-patch (8ch)** | DANDI | `dandiset/001751` | NWB | — |
| **PSCs** | DANDI | `dandiset/001752` | NWB | — |
| **Morphologies** | NeuroMorpho.Org | `10.13021/y4be-0p18` | ASC (zipped) | 13.8 MB |
| **Metabolomics** | MetaboLights | `MTBLS13927` | — | — |
| **Raw seq** | SRA | `PRJNA1423731` | FASTQ | — |
| **Analysis code** | GitHub | `RFLiu2021/AIC_proj` | Python/MATLAB | — |
| **Processed data** | Zenodo | `10.5281/zenodo.17799559` | misc | — |
| **PatchAnaLab (ephys)** | Zenodo | `10.5281/zenodo.19995355` | MATLAB | — |

### Quick Download (Morphologies Only — Most Relevant)

```bash
# NeuroMorpho.Org — 231 reconstructed neurons (13.8 MB)
curl -L -o liu2026_morph.zip "https://neuromorpho.org/dableFiles/liu_s/Supplementary/liu_s.zip"
unzip liu2026_morph.zip

# Directory structure:
# NeuroMorph_upload260215/
#   PatchClamp_morph/
#     VENL/         28 neurons (ASC)
#     VENS/         24 neurons (ASC)
#     PC-L5_ET/     13 neurons (ASC)
#   Patchseq_morph/
#     Excitatory/  150 neurons (ASC)
#     Inhibitory/   16 neurons (ASC)
#   metadata.csv
```

---

## 4. Morphology Format: ASC vs Our SWC

### ASC (Neurolucida ASCII — Liu et al.)
```
("Cell Body"
  (CellBody)
  (x y z radius)
  ...
)
( (Color Cyan) (Dendrite)
  (x y z radius)       ← root
  (x y z radius)       ← child
  |                     ← bifurcation
  (x y z radius)       ← branch 1
  (x y z radius)       ← branch 2
)
```
- Hierarchical bracket tree with `|` bifurcation markers
- Manual/semi-auto tracing from confocal stacks
- **Precision**: sub-micron, human-verified

### SWC (Our fMOST Data)
```
node_id  type  x  y  z  radius  parent_id
1        1     1234.5  5678.9  100.0  2.5  -1   ← soma
2        3     1235.0  5679.2  101.0  1.8   1   ← basal dendrite
3        3     1236.0  5680.0  102.0  1.2   2
```
- Flat table, parent_id defines topology
- Auto/semi-auto tracing (APP2/Vaa3D) from fMOST voxel data
- **Precision**: 0.65µm resolution, but automated tracing errors (broken branches, false bifurcations, fiber crossing artifacts)

### Conversion
ASC → SWC is standard (many tools exist: `neurom`, `NeuroMorpho.Org API`, custom Python with `neurom`). SWC → ASC is less common but possible.

---

## 5. Integration Strategy: Morphological Bridging

### Pipeline

```
Liu ASC (231 cells, AIC)              Our SWC (insular neurons from 251637)
       ↓                                        ↓
  L-Measure feature extraction              L-Measure feature extraction
  (40 features: Sholl, branch,               (same 40 features)
   taper, tortuosity, etc.)                         ↓
       ↓                                        z-score normalize
  z-score normalize                                    ↓
       ↓                                     ┌──────────────────┐
       └───── dCor / Spearman ρ ────────────→│ similarity matrix │
                                             └──────────────────┘
                                                      ↓
                                          Identify fMOST insula neurons
                                          morphologically closest to VEN-L / VEN-S
                                                      ↓
                                          Map those candidates' projections
                                          (from our region_analysis pipeline)
                                                      ↓
                                          Correlate with fMRI activation maps
```

### L-Measure Feature Set (Sci Unit)

```python
from neurom import load_neuron
from neurom.features import get

features = {
    'soma_surface':         get('soma_surface_area', neuron),
    'soma_radius':          get('soma_radii', neuron),
    'n_stems':              get('number_of_stems', neuron),
    'n_bifs':               get('number_of_bifurcations', neuron),
    'n_branches':           get('number_of_sections', neuron),
    'n_tips':               get('number_of_tips', neuron),
    'total_length':         get('total_length', neuron),
    'total_surface':        get('total_surface_area', neuron),
    'total_volume':         get('total_volume', neuron),
    'max_branch_order':     get('max_branch_order', neuron),
    'max_euclidean_dist':   get('max_euclidean_distance', neuron),
    'max_path_length':      get('max_path_length', neuron),
    'taper_rate':           get('taper_rates', neuron),
    # Sholl profile — 10 radial bins
    'sholl':                get('sholl_frequency', neuron, step_size=10),
    # Per-compartment (basal / apical / axon)
    'basal_length':         get('total_length', neuron, neurite_type=neurite_type.basal_dendrite),
    'basal_n_bifs':         get('number_of_bifurcations', neuron, neurite_type=neurite_type.basal_dendrite),
    'apical_length':        get('total_length', neuron, neurite_type=neurite_type.apical_dendrite),
}
```

### Similarity Metric

Use **distance correlation** (`dcor` package) instead of Spearman:
- `dCor(X, Y) = 0 ⇔ X ⊥ Y` (independent)
- Captures non-monotonic relationships (e.g., U-shaped Sholl profiles)
- Our pipeline already has `USE_SPEARMAN=True` mode; replace with:

```python
from dcor import distance_correlation
dissim = 1 - distance_correlation(feature_i, feature_j)
```

---

## 6. File Paths & Locations

| Resource | Path |
|----------|------|
| This handoff | `knowledge/phd/liu2026_aic_ven_handoff.md` |
| fMOST data (low-res, SSH) | `ssh://172.20.10.250:20007/home/binbin/share/251637CH1_projection/` |
| fMOST data (high-res, HTTP) | `http://bap.cebsit.ac.cn/monkeydata/251637/cube/` |
| Analysis pipeline | `D:\projectome_analysis\` (Windows PC, `ssh binbi@10.102.8.206`) |
| region_analysis output | `neuron_tables/251637_results.xlsx` |
| FNT distance matrix | `ins_dist.txt` (computed via SLURM HPC) |
| Clustering | `fnt_dist_clustering.py` |
| Liu morph data (post-download) | Suggest: `D:\projectome_analysis\external\liu2026_morph\` |

---

## 7. Next Steps (Priority-Ordered)

### 🔴 Immediate (This Week)
1. **Download Liu morphology zip** → store in projectome workspace
2. **Convert ASC → SWC** (use `neurom` Python package: `neurom.load_neuron('file.ASC')` then export SWC)
3. **Run L-Measure on Liu VEN-L/VEN-S** → establish reference feature vectors

### 🟡 Medium-Term
4. **Extract insula-projecting neurons** from our fMOST pipeline (filter by Soma_Region = insula/insula-related ARM regions)
5. **Run L-Measure on our insula SWC set** with same feature set
6. **Compute dCor similarity matrix** → rank our neurons by VEN-L/VEN-S similarity
7. **Validate**: do the top-ranked candidates' projection patterns make sense for VEN (long-range, subcortical targets)?

### 🟢 Long-Term (Full Integration)
8. **Register VEN candidates to MRI space** → overlay with BOLD activation from cm043/cm044
9. **Test hypothesis**: VEN-dense insula subregions show distinct BOLD response to QST cooling
10. **Add VEN marker genes** to spatial transcriptomics validation (if available)

---

## 8. Key Caveats for Cursor

- **ASC ≠ SWC**: Must convert before our pipeline can ingest. Use `neurom` (Python) or `vaa3d` plugin.
- **Coordinate systems**: Liu morphologies are in slice-local coordinates (no atlas registration). Cross-comparison relies on scale-invariant features (Sholl, angles, ratios), not absolute positions.
- **VEN isolation in local circuits** (paper's key finding): This means our pipeline's projection-based clustering might naturally separate VENs from other IT/ET populations.
- **Taper rate differences**: VEN axons originate from basal dendrites with short AIS — this should produce a distinct taper signature that L-Measure can capture.
- **Python environment**: `neurom`, `dcor`, `scipy`, `pandas`. Already in our conda env `projectome`.

---

## 9. Quick Reference: Marker Genes for Validation

```
VEN-L: DSG2, HAPLN4, TOX2, TRPC4, CAMK1G, TAGLN2
VEN-S: POC5, COL24A1, ATP8A2, SEMA3E
```

If we ever get spatial transcriptomics or ISH data in macaque insula, these are the probes to use.
