# Liu et al. 2026 — AIC VEN atlas (external reference)

**Paper:** Liu R-F, Huang M, Shen Y, Shao M, Jing J, Xu N, et al. *An atlas of primate insular cortex reveals a signal-processing strategy in von Economo neurons.* **Nature Cell Biology**, 2026 Jul 2.  
**DOI:** [10.1038/s41556-026-02009-4](https://doi.org/10.1038/s41556-026-02009-4)

**Local PDF (canonical for Methods / layer text):** [`../Liu et al. - 2026 - An atlas of primate insular cortex reveals a signal-processing strategy in von Economo neurons.pdf`](../Liu%20et%20al.%20-%202026%20-%20An%20atlas%20of%20primate%20insular%20cortex%20reveals%20a%20signal-processing%20strategy%20in%20von%20Economo%20neurons.pdf) — use this file, not web snippets, for laminar context (VENs in **layer 5b** AIC; normalized soma depth in Patch-seq morph database; `metadata.csv` has transcriptomic class only).

Registered in [`../ven_validation/registry.tsv`](../ven_validation/registry.tsv) as `liu2026_aic_ven` (**additional** morphology/transcriptome reference).

Full agent handoff: [`liu2026_aic_ven_handoff.md`](liu2026_aic_ven_handoff.md)

## Why add this dataset

Our fMOST pipeline gives **projection-based** cell classes (PT/CT/IT) but not molecular identity. Liu et al. provide:

- **VEN-L** (DSG2+, HAPLN4+): thick basal dendrite >500 µm, L5 ET affinity — **28 ASC morphologies**
- **VEN-S** (POC5+, COL24A1+): shorter basal dendrite, L5/6 CT affinity — **24 ASC morphologies**
- **231 total** reconstructed neurons on NeuroMorpho.Org (Patch-clamp + Patch-seq subsets)
- scRNA-seq atlas (78 cell types, >150K cells) — GEO **`GSE319557`**

Use case: score our insula SWC neurons by morphological similarity to VEN-L/VEN-S, then validate projection patterns against expected ET/CT targets.

## Quick start — morphology only

```powershell
cd D:\projectome_analysis\external\liu2026
.\scripts\download_morph.ps1
```

Expected layout after unzip:

```
morph/
  NeuroMorph_upload260215/
    PatchClamp_morph/
      VENL/          # 28 × .ASC
      VENS/          # 24 × .ASC
      PC-L5_ET/      # 13 × .ASC
    Patchseq_morph/
      Excitatory/    # 150 × .ASC
      Inhibitory/    # 16 × .ASC
    metadata.csv
```

`morph/` is gitignored (binary data). Only scripts and docs are tracked.

## ASC → SWC (next step)

Liu data are Neurolucida **ASC**; our pipeline uses **SWC**. Convert before FNT or L-Measure:

```python
# neurom example (after: uv pip install neurom)
from neurom import load_neuron
from neurom.io import write_data
from pathlib import Path

asc_path = Path("morph/NeuroMorph_upload260215/PatchClamp_morph/VENL")
for asc in asc_path.glob("*.ASC"):
    nrn = load_neuron(asc)
    write_data(nrn, asc.with_suffix(".swc"))
```

Coordinate systems differ — compare **scale-invariant** features (Sholl, branch order, taper), not absolute xyz.

## Key marker genes (validation probes)

```
VEN-L: DSG2, HAPLN4, TOX2, TRPC4, CAMK1G, TAGLN2
VEN-S: POC5, COL24A1, ATP8A2, SEMA3E
```

## Other repositories (not downloaded by default)

| Data | Accession |
|------|-----------|
| scRNA-seq | GEO `GSE319557` (23 GB h5ad) |
| Patch-seq | GEO `GSE319369` |
| Ephys | DANDI `001746`, `001750`, `001751`, `001752` |
| Analysis code | GitHub `RFLiu2021/AIC_proj` |
| Processed | Zenodo `10.5281/zenodo.17799559` |

## Caveats

- VENs are **functionally isolated** in local circuits (paper: 0/24 VEN↔VEN connections) — projection clustering may already separate them from local IT populations.
- AIC is **glycolysis-dominant** — may affect BOLD interpretation when linking to fMRI later.
- This reference **does not replace** internal 251637 manual screening; it adds a quantitative morphological anchor.
