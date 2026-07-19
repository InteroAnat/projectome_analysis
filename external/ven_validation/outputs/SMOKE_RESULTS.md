# Morph bridge smoke — NeuroM (2026-07-09)

**Branch:** `test/liu-ven-fmost-bridge`  
**Backend:** `neurom-4.0.5` + MorphIO (ASC/SWC load, crop, features)  
**User VENs (251637):** 007, 026, 028, 036, 055, 056  

Methods: [`../docs/METHODS.md`](../docs/METHODS.md)

## Showcase

Open **`VEN_MORPH_SHOWCASE.html`**:

| Figure | Content |
|--------|---------|
| `gallery/00_methods_metrics.png` | NeuroM preprocess + scored metrics |
| `gallery/01_liu_VENL_full_gallery.png` | **All** 28 Liu VEN-L |
| `gallery/01_liu_VENS_full_gallery.png` | **All** 24 Liu VEN-S |
| `gallery/02_pairwise_fMOST_vs_Liu.png` | 6 fMOST VENs vs Liu exemplars |
| `gallery/03_metric_bars.png` | basal_max, bipolar symmetry, d_z |

## Scores (NeuroM + VEN polarity: longest stem = basal)

| Neuron | Nearest | prefer_VENL | basal_max | apical_max | path_sym | Neuron_Type |
|--------|---------|-------------|-----------|------------|----------|-------------|
| 007 | VENL | yes | ~885 | ~879 | 0.99 | ITs |
| 036 | VENL | yes | ~922 | ~850 | 0.92 | ITi |
| 056 | VENL | yes | ~951 | ~836 | 0.88 | ITi |
| 026 | VENL | yes | ~1029 | ~305 | 0.30 | ITi |
| 028 | VENL | yes | ~989 | ~316 | 0.32 | ITi |
| 055 | VENL | yes | ~870 | ~308 | 0.35 | ITi |

All six: `basal_gt_500=True`, nearest VEN-L (055 close to VEN-S on d_z but still prefer_VENL).

## Caveats

- Polarity: Liu `complete_morph_labels=True` (expert ASC); fMOST `False` → `_infer_polarity_tags()` auto type-4
- Absolute d_z large (modality gap); use ranks
- MorphIO may warn about soma shape on some ASC; suppressed in batch
- Projection classes joined from `group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx` (`Summary`; `Neuron_Type`, `Soma_Region_Refined`)

## Re-run

```powershell
$py = "$env:USERPROFILE\miniconda3\envs\projectome\python.exe"
cd D:\projectome_analysis\external\ven_validation\scripts
& $py run_morph_bridge_smoke.py
& $py make_comparison_gallery.py
start ..\outputs\VEN_MORPH_SHOWCASE.html
```
