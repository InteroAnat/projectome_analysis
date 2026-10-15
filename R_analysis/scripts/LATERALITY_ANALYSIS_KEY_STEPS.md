# Laterality Analysis: Key Steps and Knowledge

This note summarizes the core workflow and statistical reasoning used in `laterality_by_questions.Rmd`.

**v2 (imbalance-robust):** [`laterality_ver2.Rmd`](laterality_ver2.Rmd) mirrors the same SQ1–SQ6 structure, writes **`laterality_ver2_results.xlsx`**, and adds (1) **PERMANOVA with `Neuron_Type` / `Soma_Region` covariates**, (2) **Monte Carlo PERMANOVA** on stratified equal-L/R subsets within each neuron type (`BALANCE_REPS`, `RUN_BALANCE_PERMANOVA`), and (3) **`lm(log1p(total) ~ Neuron_Type + Soma_Side)`** with ANOVA + `drop1` tables. Use v1 for the unchanged baseline; use v2 when reporting robustness to L/R count imbalance and composition. See Section 11 below.

## 1) Data Preparation

- Load ipsilateral and contralateral projection matrices from INS workbook sheets.
- Merge neuron metadata (side, type, region, cluster annotations).
- Standardize `Soma_Region` labels by trimming side prefixes (`CL_`, `CR_`) so regions are comparable across hemispheres.
- Build matrix forms:
  - `mat_ipsi`: region x neuron (ipsilateral strengths)
  - `mat_contra`: region x neuron (contralateral strengths)
- Keep `meta` with neuron-level fields (`NeuronID`, `Soma_Side`, `Neuron_Type`, `Soma_Region`).

## 2) Sparsity Diagnostics (important context)

- Report sparsity for:
  - ipsi matrix
  - contra matrix
  - total matrix (ipsi + contra, after safe alignment on shared rows/columns)
- Metrics:
  - zero fraction / nonzero fraction
  - nonzero targets per neuron (mean/median)
  - neurons per region (mean/median)
- Why it matters:
  - explains ordination shape (bands/arcs in low-dimensional plots)
  - affects distance behavior and interpretation.

## 3) SQ1: Count-Based Composition Tests (unified block)

### SQ1a: `Neuron_Type x Soma_Side`

- Test: chi-square independence (`chisq.test`), Fisher fallback for small expected counts.
- Visuals:
  - raw side counts
  - within-side type proportions
  - 100% stacked side composition.

### SQ1b: `Neuron_Type x Soma_Region`

- Test: chi-square independence, Fisher fallback if needed.
- Interpretation target: whether composition differs across regions.

### SQ1c: `Soma_Region x Soma_Side`

- Test: chi-square independence, Fisher fallback if needed.
- Interpretation target: whether side sampling is region-balanced.

### Auto-significance labeling

- All key SQ1 outputs include:
  - `significant` (p < 0.05)
  - `sig_label` (`***`, `**`, `*`, `ns`).

## 4) SQ2: Projection-Strength Counterparts

- Build neuron-level strength table:
  - `ipsi`, `contra`, `total = ipsi + contra`
  - with side/type/region metadata.

### SQ2a: Side differences by type

- Test: Wilcoxon `total ~ Soma_Side` within each `Neuron_Type`.
- BH correction across types.
- Plot: distribution/boxplot by type and side.

### SQ2b: Region-aware side differences

- Restrict to bilateral regions (regions with both L and R present).
- Tests:
  - within region (all types pooled)
  - within region x type strata
- BH correction across tested strata.

### Auto-significance

- Wilcoxon result tables include:
  - adjusted p (`p_wilcox_BH`)
  - `significant`
  - `sig_label`.

## 5) SQ3: Multivariate Pattern Differences

Goal: test side differences in multivariate projection profiles.

### Why ipsi-only in SQ3

- Keeps side comparison in a consistent reference frame.
- Avoids side-reference coupling from pooling ipsi+contra in one matrix.

### Multi-distance sensitivity analysis

- Distances tested:
  - `bray_raw` (raw strengths)
  - `bray_prop` (per-neuron normalized profiles)
  - `jaccard_pa` (presence/absence)
  - `aitchison_clr` (CLR + Euclidean, compositional view)
- For each distance:
  - PERMANOVA (`adonis2`)
  - dispersion check (`betadisper` + `permutest`, euclidified when needed).

### Interpretation pattern

- PERMANOVA significant + betadisper non-significant:
  - supports location difference (group centroids) rather than spread artifact.
- `bray_raw` vs `bray_prop`:
  - raw captures "where + how much"
  - normalized captures mostly "where".

### Ordination panels

- 2x2 PCoA panels for the four distances to check visual concordance.

## 6) SQ4: Morphology-Projection Concordance

- Compare FNT pairwise distances vs projection pairwise distances.
- Inference uses Mantel test p-value.
- Spearman upper-triangle correlation is descriptive only (not independent pairs).

## 7) SQ5: Heatmap Concordance View

- Side-by-side distance heatmaps with identical ordering.
- Additional `|z_FNT - z_Proj|` panel highlights pairwise mismatches.

## 8) SQ6: Type-Stratified PERMANOVA

- Test side effect within each neuron type (where data are sufficient).
- BH correction across types.
- Includes raw and adjusted significance labels.

## 9) Reporting / Export Design

Export workbook includes:

- SQ1/SQ2/SQ3/SQ4/SQ6 test tables
- region-level and multi-distance sensitivity tables
- sparsity tables (`MATRIX_sparsity`, `SQ3_sparsity`)
- significance flags and star labels for quick scanning.

## 10) Practical Knowledge Checklist

- Always inspect both:
  - p-value (significance)
  - effect size (`R2`, group differences, magnitude).
- Use betadisper alongside PERMANOVA to avoid false interpretation from heterogeneity.
- Keep count questions (composition) separate from strength questions (continuous output).
- Validate robustness by comparing multiple distances, not only one metric.

## 11) laterality_ver2 (imbalance-robust extensions)

**File:** `laterality_ver2.Rmd` (same folder as v1). **Output:** `laterality_ver2_results.xlsx` (does not overwrite `laterality_by_questions_results.xlsx`).

**Config (top of Rmd):**

- `OUTPUT_XLSX` – result workbook name.
- `BALANCE_REPS`, `BALANCE_MIN_PER_SIDE` – Monte Carlo replication count and minimum L/R count per type stratum.
- `RUN_BALANCE_PERMANOVA` – set `FALSE` to skip the balance loop (faster knit).
- `RUN_BALANCE_BRAY_PROP` – optional second balance analysis on profile-normalized Bray–Curtis.

**What v2 adds:**

1. **SQ3 adjusted PERMANOVA:** `adonis2(~ Neuron_Type + Soma_Side)` and, when estimable, `~ Neuron_Type + Soma_Region + Soma_Side` (`by = "margin"`), for each distance in the existing multi-distance block. Unadjusted group tests and `betadisper` stay as in v1.
2. **SQ3b balance MC:** Within each `Neuron_Type`, draw `min(n_L, n_R)` neurons per side, pool, `adonis2(~ Soma_Side)` on Bray–Curtis of the subset; summarize median/mean `R2` and p, and fraction of reps with p < 0.05.
3. **SQ2c:** `lm(log1p(total) ~ Neuron_Type + Soma_Side)` – Type I ANOVA (side after type) and `drop1(., test = "F")`; exported as `SQ2_lm_anova`, `SQ2_lm_drop1`, `SQ2_lm_Soma_Side_row`.

**Export naming:** v2 uses `SQ3_PERMANOVA_bray_raw` (single-distance slice) instead of the broken v1 `ado` object; full multi-distance table remains `SQ3_multi_distance`.

