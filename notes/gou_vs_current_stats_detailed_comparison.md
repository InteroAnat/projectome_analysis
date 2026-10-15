# Gou et al. vs Current Pipeline: Detailed Statistics Comparison

## Scope

This document compares:
- **Gou et al. laterality-related stats** in Julia code
- **Current `laterality_v1_v2_INS_HE_merged_v3.Rmd` stats framework**
- practical implications for interpretation and reporting

Primary code references:
- `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl`
- `references/analysis-code_gou_etal_2025/monkeyrec/zz1data/src/zz1data.jl`
- `R_analysis/scripts/laterality_v1_v2_INS_HE_merged_v3.Rmd`

---

## A. What Gou et al. use for laterality-related analyses

## A1) Per-neuron hemispheric bias index (core laterality metric)

Gou computes per-neuron bias as:

\[
\text{symm} = \frac{\text{contra} - \text{ipsi}}{\text{contra} + \text{ipsi}}
\]

In code:
- `plot_itc_ipsicontra_len(...)` defines
  - `symm(r) = (r.arborcontra - r.arboripsi) / (r.arborcontra + r.arboripsi)`
- upstream `calc_ipsicontra_len(...)` accumulates per-neuron ipsi/contra arbor lengths.

Implication:
- statistical unit is the **neuron**
- index is based on **raw arbor length totals** (not projection-profile normalization)

## A2) Main hypothesis test type in laterality plots

Across many laterality-related comparisons, Gou heavily uses:
- `HypothesisTests.MannWhitneyUTest(...)`
- p-value via `HypothesisTests.pvalue(...)`

Method tags in code often record:
- `:MannWhitneyUTestTwoSided`

## A3) Other tests present in Gou codebase (broader than core LI)

Also used in related analyses:
- `HypothesisTests.ChisqTest(...)`
- `HypothesisTests.FisherExactTest(...)`
- `HypothesisTests.ApproximateTwoSampleKSTest(...)`
- `HypothesisTests.CorrelationTest(...)`

Important caveat:
- these are not all for the same endpoint; they appear across multiple figure functions.

## A4) Multiple-testing behavior in Gou code

In the inspected laterality blocks:
- no clear global BH/FDR framework consistently applied for per-neuron LI tests
- many tests appear as pairwise/two-group comparisons with per-test p-values

---

## B. What your current pipeline uses

File: `R_analysis/scripts/laterality_v1_v2_INS_HE_merged_v3.Rmd`

## B1) SQ1 (counts/composition)
- Chi-square (`chisq.test`)
- Fisher fallback (`fisher.test`)
- pair-level proportion tests with BH correction in relevant tables

## B2) SQ2 (univariate strength)
- Wilcoxon rank-sum by type/region
- BH corrections for families
- adjusted linear model:
  - `lm(log1p(total) ~ Neuron_Type + Soma_Side)`
  - `anova(...)` and `drop1(..., test="F")`

## B3) SQ3 (multivariate projection pattern)
- PERMANOVA via `adonis2`
- dispersion check via `betadisper` + `permutest`
- multiple distance definitions (raw/proportion/Jaccard/Aitchison)

## B4) SQ3b (imbalance robustness)
- balanced Monte Carlo resampling within type
- repeated PERMANOVA summaries

## B5) SQ3c (stratified multivariate)
- class/target strata
- PERMANOVA + betadisper
- BH adjustment across stratum families

## B6) SQ3d (current region-level LI endpoint)
- group-mean LI per target region:
  \[
  LI_{r} = \frac{\mu_{L,r}-\mu_{R,r}}{\mu_{L,r}+\mu_{R,r}+\epsilon}
  \]
- neuron-level values within region compared by Wilcoxon
- BH correction

## B7) SQ3e (newly added Gou-style per-neuron LI block)
- per-neuron LI:
  \[
  LI_j = \frac{contra_j - ipsi_j}{contra_j + ipsi_j + \epsilon}
  \]
- overall and type-stratified Mann-Whitney/Wilcoxon-style tests
- BH adjustment in type-stratified table

## B8) SQ4/SQ5 (morphology–projection concordance)
- SQ4 Mantel (Spearman) with projection endpoint now pattern-focused (`bray_prop` logic)
- SQ5 matrix visualization with rank-based mismatch panel (`|rank(FNT01)-rank(Proj)|`)

## B9) SQ6 (type-stratified multivariate side effects)
- PERMANOVA by type
- BH adjustment across types

---

## C. Direct comparison: Gou vs current

| Dimension | Gou laterality blocks | Your current pipeline |
|---|---|---|
| Main LI unit | Per-neuron | SQ3d: per-target-region; SQ3e: per-neuron |
| Main LI formula | \((contra-ipsi)/(contra+ipsi)\) | SQ3d region-level mean ratio; SQ3e matches Gou style |
| Dominant test family | Mann-Whitney U (two-group nonparametric) | Mixed: chi/fisher, Wilcoxon, LM/ANOVA, PERMANOVA, Mantel |
| Multivariate distance tests | Not central in Gou LI block | Core in SQ3/SQ3b/SQ3c/SQ6 |
| Dispersion confound checks | Not emphasized in LI snippets | Explicit (`betadisper`) |
| Multiple testing control | Not consistently visible | Explicit BH across multiple families |
| Concordance matrix tests | Not the same framework | SQ4 Mantel + SQ5 mismatch maps |

---

## D. Interpretation impact

## D1) Strength of Gou-style per-neuron LI
- intuitive neuron-level asymmetry endpoint
- easy to compare distributions across groups
- aligns with single-neuron framing

## D2) Strength of your broader framework
- covers count, univariate, multivariate, concordance, and robustness analyses
- explicit correction for multiplicity and dispersion caveats
- stronger inferential control for exploratory multi-test setting

## D3) Why keeping both SQ3d + SQ3e is valuable
- SQ3d: target-region asymmetry map (where asymmetry lands)
- SQ3e: neuron-level asymmetry distribution (how asymmetric neurons are)
- together provide complementary biological views

---

## E. Recommended reporting structure (for manuscripts/slides)

1. **Primary laterality endpoint (neuron-level):** SQ3e per-neuron LI (Gou-aligned)
2. **Regional localization endpoint:** SQ3d region-level LI with BH correction
3. **Pattern-level side effects:** SQ3/SQ3c PERMANOVA + betadisper
4. **Morphology–projection concordance:** SQ4 Mantel (Spearman), SQ5 rank mismatch map
5. **Imbalance robustness:** SQ3b summaries + n_L/n_R transparency in all LI outputs

---

## F. Practical caveats

- Gou-style LI can still be sensitive when one side has small \(n\); report group sizes and avoid over-claiming borderline p-values.
- Region-level LI can show \(LI=\pm1\) in one-sided receiving targets; this is expected and should be interpreted with receipt-class annotation.
- For sparse/non-Gaussian distances, rank-based diagnostics are preferred over z-score mismatch maps.

---

## Bottom line

Your current pipeline is statistically broader and more controlled than the direct Gou LI block.
Adding SQ3e preserves comparability with Gou while retaining your stronger multilevel inference stack.

---

## G. Pipeline-Wide Hybrid Decision Map

This section defines the hybrid policy now applied in
`laterality_v1_v2_INS_HE_merged_hybrid.Rmd`.

| SQ block | Primary inferential unit | Test strategy | Multiple-testing family policy |
|---|---|---|---|
| SQ1 (global tables) | contingency table | chi-square with Fisher fallback | single global table tests; no family correction (`adjust_method = none`) |
| SQ1 (type/region pair bars) | repeated L vs R proportion tests | Fisher/prop fallback per pair | BH within each pairwise family (`sq1_type_pair_lr`, `sq1_region_pair_lr`) |
| SQ2 type-wise | repeated L vs R univariate totals | Wilcoxon rank-sum | BH across neuron types (`sq2_type_lr`) |
| SQ2 region-wise | repeated L vs R univariate totals in bilateral regions | Wilcoxon rank-sum | BH within declared region family (`sq2_region_lr_overall`, `sq2_region_type_lr`) |
| SQ3 multi-distance | repeated side PERMANOVA over distance metrics | adonis2 + betadisper caveat | BH across distance-metric family (`sq3_multidistance_permanova`) |
| SQ3c class/target strata | repeated stratified side PERMANOVA | adonis2 + betadisper caveat | BH per stratum family (`sq3c_class_permanova`, `sq3c_target_permanova`) |
| SQ3d region LI | per-target region LI + Wilcoxon by stratum | Wilcoxon on region vectors | BH within each stratum-specific target family (`sq3d_<stratum>_targets`) |
| SQ3e neuron LI | per-neuron LI distributions | MWU/Wilcoxon | overall single test = none; type-stratified BH (`sq3e_type_stratified`) |
| SQ4 | single matrix concordance endpoint | Mantel Spearman | single test (`sq4_single_mantel`, no family adjustment) |
| SQ6 | type-stratified multivariate side effect | PERMANOVA | BH across neuron types (`sq6_type_stratified_permanova`) |

### Unified reporting fields

To reduce table inconsistency and interpretation errors, all updated inferential outputs include:

- `p_raw`
- `p_adj` (aliases retained: `p_BH`, `q_BH` for compatibility)
- `adjust_method`
- `family_id`
- `significant`
- `sig_label`
