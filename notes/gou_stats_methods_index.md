# Gou Codebase Statistics Methods Index

## Purpose

This note is a method-centric index of statistics used in Gou et al. Julia code, with practical mapping to the current projectome pipeline.

---

## 1) Core hypothesis-test families found

## 1.1 Mann-Whitney U test (two-sided)

- API: `HypothesisTests.MannWhitneyUTest(...)`
- P-value: `HypothesisTests.pvalue(t)`
- Role: dominant two-group nonparametric test across many figure analyses

Representative locations:
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl`
  - laterality blocks and many subtype/group comparisons
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/zz2fig-itcs.jl`
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/zz2fig-ptct.jl`
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/zz2fig-compare.jl`

Recorded method tags in outputs:
- `:MannWhitneyUTestTwoSided`

---

## 1.2 Chi-square test

- API: `HypothesisTests.ChisqTest(...)`
- Role: binary/contingency comparisons and pairwise categorical assessments

Representative locations:
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl`
- `references/analysis-code_gou_etal_2025/monkeyrec/2-analysis.jl`

---

## 1.3 Fisher exact test

- API: `HypothesisTests.FisherExactTest(...)`
- Role: enrichment / small-count contingency contexts

Representative locations:
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl`
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/zz2fig-longproj.jl`

---

## 1.4 Kolmogorov–Smirnov tests

- APIs:
  - `HypothesisTests.ApproximateTwoSampleKSTest(...)`
  - `HypothesisTests.ExactOneSampleKSTest(...)`
- Role: distribution-shape comparisons in depth/histogram style analyses

Representative locations:
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl`
- `references/analysis-code_gou_etal_2025/monkeyrec/2-analysis.jl`

---

## 1.5 Correlation tests

- API: `HypothesisTests.CorrelationTest(...)`
- Role: pairwise association checks in selected figure blocks

Representative location:
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl`

---

## 1.6 ANOVA / regression family

- `HypothesisTests.OneWayANOVATest(...)`
- `glm(...)` and `lm(...)` models in some analysis modules

Representative locations:
- `references/analysis-code_gou_etal_2025/monkeyrec/2-draw.jl` (One-way ANOVA)
- `references/analysis-code_gou_etal_2025/monkeyrec/gapr/src/3-mus-hy.jl` (GLM/LM)

---

## 2) Gou laterality-specific metric and stats

## 2.1 Per-neuron LI-like index

In ITC/bias blocks:

\[
\mathrm{symm} = \frac{\mathrm{contra} - \mathrm{ipsi}}{\mathrm{contra} + \mathrm{ipsi}}
\]

Code path:
- `zz1data.calc_ipsicontra_len(...)` computes per-neuron ipsi/contra lengths
- `plot_itc_ipsicontra_len(...)` defines and visualizes `symm(r)`

Files:
- `references/analysis-code_gou_etal_2025/monkeyrec/zz1data/src/zz1data.jl`
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl`

## 2.2 Typical test on laterality distributions

- Mann-Whitney U two-sided
- ECDF / density style displays
- direct p-value reporting

---

## 3) Comparison with current projectome pipeline

Current file:
- `R_analysis/scripts/laterality_v1_v2_INS_HE_merged_v3.Rmd`

What your pipeline adds beyond Gou LI blocks:
- PERMANOVA (`adonis2`) with dispersion checks (`betadisper`)
- Mantel concordance (Spearman)
- explicit BH/FDR correction across families
- balanced Monte Carlo robustness analysis

What now aligns with Gou style:
- SQ3e per-neuron LI block:
  \[
  LI_j = \frac{contra_j - ipsi_j}{contra_j + ipsi_j + \epsilon}
  \]
- with Mann-Whitney/Wilcoxon-type inference and type-stratified reporting

---

## 4) Practical takeaways

1. Gou codebase primarily uses nonparametric two-sample testing (MWU) plus contingency tests.
2. Your pipeline is statistically broader and more explicitly corrected for multiplicity.
3. Keeping both SQ3d (region-level LI) and SQ3e (per-neuron LI, Gou-aligned) gives complementary biological interpretation.
