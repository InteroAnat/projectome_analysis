# VEN validation — external references

Cross-dataset checks for von Economo neuron (VEN) candidates in our fMOST projectome data (`251637` insula, planned multi-monkey expansion).

## Strategy

| Layer | What we have | What external refs add |
|-------|--------------|------------------------|
| **Projection** | `step1` region analysis → PT/CT/IT subtypes | Liu 2026: VEN-L ≈ L5 ET, VEN-S ≈ L5/6 CT — projection sanity check on top-ranked morph candidates |
| **Morphology** | Manual spindle screening + bulk visual (`step3.1`) | Liu 2026: 52 labeled VEN-L/VEN-S ASC reconstructions for L-Measure / dCor similarity |
| **Transcriptome** | None in fMOST | Liu 2026: marker genes (DSG2/HAPLN4 vs POC5/COL24A1) for future ISH / spatial validation |
| **fMRI** | cm043/cm044 QST cooling (QST BIDS repo) | Long-term: register VEN-dense subregions ↔ BOLD (see Liu handoff §7) |

Primary validation stays **internal** (251637 candidate tables + projection patterns). External sets are **additional** morphological anchors — not ground truth for every insula neuron.

## Registry

Machine-readable list: [`registry.tsv`](registry.tsv)

| ref_id | role | status |
|--------|------|--------|
| `internal_251637_von_economo` | primary | active |
| `evrard2012`, `nimchinsky1999` | historical | literature only |
| `liu2026_aic_ven` | **additional** | run `external/liu2026/scripts/download_morph.ps1` → [`../liu2026/`](../liu2026/) |

## Morphology bridge (Liu → fMOST) — **NeuroM**

Methods: [`docs/METHODS.md`](docs/METHODS.md)  
(Keydians: `projectome_analysis_doc/liu2026_fmost_ven_morph_bridge_methods.md`)

```
Liu ASC (VEN-L/VEN-S)          fMOST SWC (insula subset)
        ↓                                ↓
   MorphIO load + crop            MorphIO load + crop
   (drop axon, R=800 µm)          (drop axon, R=800 µm)
        ↓                                ↓
   NeuroM features                same NeuroM feature set
        └──── z-score vs Liu class centroid (Euclidean) ────┘
                      ↓
           rank fMOST neurons vs VEN-L / VEN-S / PC
                      ↓
           cross-check projection labels (step1)
```

Liu ASC: expert Neurolucida labels (`complete_morph_labels=True`). fMOST SWC without type-4: `_infer_polarity_tags()` auto-assigns apical (type 4) before NeuroM features. Smoke script calls `features_table()` for all six confirmed IDs.  
Scale: local FOV only — Liu coords are slice-local µm, not registered to ARM/NMT.

## Related paths

| Resource | Path |
|----------|------|
| Internal VEN candidates (user-confirmed 2026-07-09) | `neuron_tables/251637_von_economo_user_confirmed.xlsx` — **007, 026, 028, 036, 055, 056** |
| Canonical projection metadata | `group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx` (`Summary` sheet) |
| Legacy spreadsheet | `neuron_tables/251637_von_economo_candidates.xlsx` |
| Bridge scripts | `external/ven_validation/scripts/` (`asc_to_swc.py`, `morph_features.py`, `run_morph_bridge_smoke.py`) |
| Smoke outputs | `external/ven_validation/outputs/` |
| Liu ASC / converted SWC | `external/liu2026/morph/` (gitignored), `external/liu2026/swc/` |
| Bulk visual | `main_scripts/step3.1.bulk_visual_data.py` |
| Liu 2026 handoff | `external/liu2026/` |

## Dependencies

Installed in `projectome` env (see `requirements-optional.txt`):

- **`neurom`** (+ **MorphIO**) — ASC/SWC load, crop, morphometrics (**required** for scoring)
- `dcor` — optional future Sholl distance-correlation

```powershell
conda activate projectome
python -m pip install "neurom>=3.2.0"
```

```powershell
$py = "$env:USERPROFILE\miniconda3\envs\projectome\python.exe"
cd D:\projectome_analysis\external\ven_validation\scripts
& $py run_morph_bridge_smoke.py
& $py make_comparison_gallery.py
start ..\outputs\VEN_MORPH_SHOWCASE.html
```
