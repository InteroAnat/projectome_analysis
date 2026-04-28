# combined_primary_v2 outputs

**Pipeline:** `group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd` (canonical source).
**Launcher:** `Rscript v2_combined_primary_pipeline.R` (calls `rmarkdown::render`).
**Input workbook:** `d:/projectome_analysis/group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx`
**Anchor doc:** [`notes/LR_insula_analysis_review.md`](../../../../notes/LR_insula_analysis_review.md) (biological framing, sampling-design audit, full reference list).

> ⚠️ **Interim cohort.** This run analyses the partial dataset currently available (4 macaques, 306 neurons after harmonization). Numbers will change as more samples are added. Always cite this README's cohort table next to any figure quoted in a manuscript draft.

## Cohort (this run)

| Sub-region | n_L | n_R | total |
|---|---:|---:|---:|
| IAL | 21 | 117 | 138 |
| IAPM | 16 | 0 | 16 |
| IDD5 | 34 | 36 | 70 |
| IDM | 47 | 32 | 79 |
| IDV | 3 | 0 | 3 |
| **TOTAL** | **121** | **185** | **306** |

**Inferential anchor:** `IDD5_plus_IDM_balanced` (only L/R-balanced anatomical stratum). All BH-corrected laterality claims are anchored here.

## Reproduce

```bash
Rscript group_analysis/R_analysis/v2_combined_primary_pipeline.R
# or knit v2_combined_primary_pipeline.Rmd in RStudio
```

## Output layout

- `figures/` — PNG panels F1–F10 + F9b (SQ5 heatmaps) + F6_supplement
- `figures/FIGURE_CAPTIONS.md` — publication-style captions + this-run numbers (rewritten each render)
- `flatmap_overlays/` — `python group_analysis/scripts/13_flatmap_context_strip.py` outputs
- `spec/strata_spec.csv` — pre-registered inferential strata
- `spec/metric_spec.csv` — metric definitions (`frac_projecting`, `mean_prop`, `cliffs_delta`, `log_odds_presence`, `regression_slope`)
- `spec/strata_and_metrics_spec.csv` — merged long table
- `spec/stat_registry.csv` — `figure_file → primary_stats_csv → analysis_summary` (one row per figure)
- `stats/` — numeric outputs (every figure's source CSVs; see Figure index)
- `stats/99_figure_QC_report.csv` — registry-driven alignment QC (runs at end of pipeline)
- `stats/99_orphan_stats_files.csv` — auxiliary CSVs not linked in registry
- `tables/manuscript_quoted_numbers.xlsx` — figure registry + headline numbers + provenance (fill `manuscript_quote_ref` / `manuscript_section` here)

## Figure index

| Fig | Question (anchor: IDD5+IDM balanced unless noted) | Primary stats CSV |
|-----|---|---|
| F1A | Functional-domain composition (Gou 6 panels) by sub-region; L+R combined | `01_subregion_x_gou_panels.csv` |
| F1B | F1A as L vs R Cliff's d, BH within row (inferential rows only) | `01_subregion_x_gou_panels.csv` |
| F2  | Sub-region × key target heatmap (L+R combined; hybrid L3/L6) | `02_subregion_x_key_targets.csv` |
| F2B | F2 as L vs R Cliff's d, BH within row | `02_subregion_x_key_targets.csv` |
| F3A | Intra-insula source × target mean prop (L+R) | `03_intra_insula_meanprop.csv` |
| F3B | Intra-insula prevalence (% projecting) | `03_intra_insula_prevalence.csv` |
| F3C | Intra-insula L vs R Cliff's d, BH within source row | `03_intra_insula_full.csv` |
| F4  | Interoceptive prop ~ Soma_NII_Y per side (per target OLS) — descriptive due to sub-region/side confound | `04_interoceptive_gradient_regressions.csv` |
| F5  | Per-neuron Ibias = (contra-ipsi)/(contra+ipsi) on axon length | `05_per_neuron_ibias.csv` |
| F6 LI | Composition LI per target × stratum (`bilateral_receiving` panel preferred) | `06b_projection_laterality_index_all_targets.csv` |
| F6 supplement | Top BH-significant rank-based receivers across balanced strata | `06_asymmetric_receivers_BH.csv` |
| F7  | Headline endpoints (Ig, Cl, LPal, VPal, PAG-proxy, caudal_OFC) across 4 strata | `07_hierarchy_sensitivity_per_target.csv` |
| F8  | Leave-one-monkey-out for headline targets (IDD5+IDM stratum primary) | `09_loso_sensitivity.csv` |
| F9  | Pairwise FNT vs Bray-Curtis (hex bin) — positive control | `08_mantel_replication.csv` |
| F9b | F9 as parallel SQ5-style heatmaps (FNT01 ‖ Proj ‖ |Δrank|) | `08_mantel_replication.csv` |
| F10 | Interoceptive targets vs soma AP octile heatmap | `10_ap_octile_interoceptive_profile.csv` |
| P13 | Multi-monkey somata on insula flatmap (Python overlay) | `00_context_per_subregion_n.csv` |

## Headline findings (this run, BH within stratum)

1. **Dysgranular → granular Ig L > R** — F3C IDD5 row Cliff *d* = +0.72 (BH < 0.001); F6 supplement q ≈ 8 × 10⁻¹⁰ in IDD5_balanced.
2. **Dysgranular → ventral / lateral pallidum + claustrum L > R** — F2B IDD5 LPal Cliff *d* = +0.36 (BH < 0.01); F6 supplement q ≈ 0.01.
3. **Insula is dominantly intra-insula** — F3A/B 96.4% of 306 neurons project intra-insula; mean intra fraction = 58% of axonal budget.
4. **Caudal-OFC R > L is a sampling artefact** — F7 R>L*** in `all_combined` only; null in IDD5/IDM balanced strata.

Autonomic Gou panel (F1) and PAG / VPal endpoints (F7) are firmly null in inferential strata — Oppenheimer 1992 cardiac sub-model is not testable at NMT v2.1 L6 brainstem aggregation.

## QC checklist (P15)

- `99_figure_QC_report.csv` verifies each registry figure exists, is non-empty, and has a readable primary stats CSV; `bh_summary` is filled when BH columns are present in the CSV.
- F3C significance stars are pulled from `03_intra_insula_full.csv` at build time; the registry still points at `03_intra_insula_meanprop.csv` as the mean-prop layer.
- `99_orphan_stats_files.csv` lists auxiliary CSVs (gou domain map, scatter points, BH-significant subset) not referenced by `stat_registry.csv`.

