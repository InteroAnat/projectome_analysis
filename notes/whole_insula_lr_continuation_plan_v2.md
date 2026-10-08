# Whole-insula L+R continuation plan v2

**Date:** 2026-04-28
**Supersedes:** `C:\Users\binbi\.cursor\plans\whole_insula_lr_continuation_5085d785.plan.md` v1
**Anchor doc:** `notes/LR_insula_analysis_review.md` §11A–C
**Input data:** `group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx`

## What changed from v1

1. Adopt the **harmonized combined table** (atlas → 251637-manual mapping applied; see `Mapping_Rule` sheet). This grows the IDD5 balanced stratum from 33:30 to **34:36**, and IDM from 44:32 to **47:32**. Headline IDD5+IDM balanced stratum is now **n_L = 81, n_R = 68** (was 77:62).
2. Replace the v1 single-bar functional-domain panel with **per-subregion × {Gou-panel, key-target}** heatmaps (sub-region on rows, L+R combined; L vs R as a paired secondary panel).
3. Adopt **Gou's `vote_connections` + `plot_subreg_conn_hm`** recipe for the intra-insula heatmap (Ward+barjoseph order, k=2 cut-line, `thr=3` edges, `≥10`-source-row inferential filter).
4. Reformulate the interoceptive-gradient endpoint as a **fitted regression** (slope vs soma AP/ML) per side, with L-R contrast — Gou's `linear_reg_scatter!` idiom — instead of imposing axis order.
5. Add a **continuous per-neuron Ibias** (Gou's actual hemispheric bias on axon length, applied to projection-strength-weighted contra fraction in our case).
6. Fix **BH semantics in the receiver-rank table** (BH within (stratum × family), not min across families).
7. **Hierarchy sensitivity at target level** (Ig, Cl, LPal, VPal, autonomic proxies) — not panel-level only.
8. **LOSO sensitivity** sweep — drop-one-monkey for headline endpoints.
9. **Mantel replication aligned with SQ4** — use `multi_monkey_INS_dist.txt` directly as a distance, normalized Bray on projection.
10. **Flatmap overlays**: use existing Julia flatmap PNGs as base, overlay sub-region labels (harmonized scheme) and L-R effect colors via Python.
11. **Stat registry**: every quoted manuscript number maps to one row of `stat_registry.csv`.

## Background grounding (hypotheses being tested)

| Pillar | Hypothesis | Source | Test endpoint |
|---|---|---|---|
| 1 | Sub-region × target profile differs (agranular ≠ dysgranular ≠ granular) | Mesulam-Mufson 1982; Evrard 2014 | Per-subregion × Gou-panel heatmap; per-subregion × key-target heatmap |
| 2 | Insula is dominantly self-projecting | Gehrlach 2020; Gogolla 2017 | Intra-insula prevalence + mean-strength heatmaps |
| 3 | Granular Ig as primary interoceptive cortex (target-side) | Craig 2002; Evrard 2019 | Intra-insula `IDD5+IDM → Ig (target)` panel; gradient regression |
| 4 | Left-side dominant approach/integration limb | Craig 2005, 2011 | L>R in Pal/Cl/Ig in IDD5 balanced stratum |
| 5 | Asymmetric reward-prediction-error coding via insula → VPal | Hoy 2022/2023 | L>R in VPal/LPal in dysgranular |
| 6 | EPIC predictive-coding: dysgranular feedback to Ig | Barrett & Simmons 2015 | Direction + slope of dysgranular → Ig in regression |
| 7 | Oppenheimer's cardiac-autonomic submodel | Oppenheimer 1992 | Brainstem autonomic-target panel; expected to be null at L6 |

## Output organisation

```
group_analysis/R_analysis/outputs/combined_primary_v2/
├── README.md                      (figure-to-stat map + reproduction notes)
├── spec/
│   ├── strata_and_metrics_spec.csv
│   └── stat_registry.csv          (manuscript number → figure → source CSV row)
├── stats/
│   ├── 00_context_per_subregion_n.csv
│   ├── 00_context_label_provenance.csv
│   ├── 01_subregion_x_gou_panels.csv
│   ├── 02_subregion_x_key_targets.csv
│   ├── 03_intra_insula_prevalence.csv
│   ├── 03_intra_insula_meanprop.csv
│   ├── 04_interoceptive_gradient_regressions.csv
│   ├── 05_per_neuron_ibias.csv
│   ├── 06_asymmetric_receivers_BH.csv
│   ├── 07_hierarchy_sensitivity_per_target.csv
│   ├── 08_mantel_replication.csv
│   └── 09_loso_sensitivity.csv
├── figures/
│   ├── F1_subregion_x_gou_panels.png/.svg
│   ├── F2_subregion_x_key_targets.png/.svg
│   ├── F3_intra_insula_heatmap.png/.svg
│   ├── F4_interoceptive_gradient.png/.svg
│   ├── F5_per_neuron_ibias.png/.svg
│   ├── F6_asymmetric_receivers.png/.svg
│   ├── F7_hierarchy_sensitivity.png/.svg
│   └── F8_loso_stability.png/.svg
├── flatmap_overlays/
│   ├── F9_flatmap_per_subregion_signal.png    (annotates existing Julia flatmap)
│   └── F10_flatmap_LR_difference.png
└── tables/
    └── manuscript_quoted_numbers.xlsx
```

## Strata and metrics specification

**Inferential strata (in priority order):**

| Stratum ID | n_L | n_R | Use for |
|---|---|---|---|
| `IDD5_plus_IDM_balanced` | 81 | 68 | Primary inferential layer (laterality claims) |
| `IDD5_balanced` | 34 | 36 | Stricter L/R-balanced check (most robust) |
| `IDM_balanced` | 47 | 32 | Sub-region replication of dysgranular L/R |
| `all_combined` | 121 | 185 | Sensitivity / overall composition |
| `IAL_combined` | 21 | 117 | Imbalanced — descriptive only, sensitivity check for v3 caudal-OFC artefact |

**Descriptive-only strata (n<10 or unilateral, no inferential testing):**

- `IAPM_combined` (16 L, 0 R)
- `IDV_combined` (3 L, 0 R)

**BH families:** within each (stratum × analysis-type) — never across families.

**Sample-size thresholds:**

- Cell shown but greyed if `n_source < 10` (Gou rule).
- Edge shown only if `n_neurons ≥ 3` per edge in connectivity heatmaps (Gou `thr=3`).
- BH testing only on cells with `n ≥ 10`.

**Effect-size measures:**

- Continuous: Cliff's delta (rank-based, robust).
- Presence: log-odds ratio + Fisher's exact.
- Magnitude: Wilcoxon mean-rank.

## Execution sequence

P1 → P3 (scaffold + spec + context), then P4–P12 (figures + stats), then P13 (flatmap), then P14–P16 (registry + QC + README), then P17 (journal).

Each figure step writes its CSV, then renders the PNG/SVG, then runs an alignment QC (label match, n match, BH stars match) before moving on.
