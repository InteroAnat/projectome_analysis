# Laterality V3 Handoff

This note is for resuming work on the insula laterality analysis without rereading the full chat history.

## Primary working file

- `R_analysis/scripts/laterality_v1_v2_INS_HE_merged_v3.Rmd`

This file is a full duplicate of the earlier merged notebook, created so the newer global-laterality revisions can be developed without overwriting the prior workflow.

## Current scope

The current notebook is framed around **global hemispheric asymmetry**:

- compare **left-source neurons vs right-source neurons**
- do **not** interpret SQ3d as single-neuron ipsi-vs-contra laterality
- keep region/family follow-up analyses as stratified views of that global asymmetry endpoint

## Important methodological decisions already in `v3`

### 1) Multiple-testing / stars policy

- Region-wise significance labels are intended to use **BH-adjusted p-values**, not raw p-values.
- The goal was to reduce false positives from many per-region tests.

### 2) SQ3d endpoint

- SQ3d was simplified to:
  - `all_neurons` whole-brain stratum
  - bilateral soma-region strata only
- Earlier broader family/strata LI outputs were intentionally reduced.

### 3) Normalization choice

- SQ3d uses **per-neuron proportion normalization** before LI:
  - for neuron `j`, `p_ij = x_ij / sum_k x_kj`
- This means LI is based on **relative targeting composition**, not absolute total output.
- A methodological comment explaining this is already present in the SQ3d code block of `laterality_v1_v2_INS_HE_merged_v3.Rmd`.

### 4) Interpretation of the normalization

- Benefit:
  - reduces dominance by high-output neurons
  - reduces magnitude-driven shifts when L and R groups differ in overall output scale
- Trade-off:
  - de-emphasizes absolute projection magnitude
  - LI should be described as **pattern/composition asymmetry**, not absolute output asymmetry

### 5) Weighted formula discussion

- The user proposed an additional weighted LI formula to compensate for left/right sample-size imbalance.
- Current recommendation from the prior discussion:
  - keep the existing proportion-normalized LI as the **primary** endpoint
  - if needed, compare the user-weighted LI as a **sensitivity analysis**, not the sole primary result
- In the prior exchange, weighted and unweighted LI appeared very similar, likely because the weighted version behaved close to a monotonic rescaling under the present imbalance.

## Key analysis blocks in `v3`

- `SQ1`: composition/count analyses
- `SQ2`: projection-strength analyses plus AP/ML descriptive plots
- `SQ3`: multivariate side-difference analyses
- `SQ3b`: equal-n Monte Carlo PERMANOVA
- `SQ3c`: region-stratified multivariate robustness
- `SQ3d`: projection laterality index by target region
- `SQ4` to `SQ6`: concordance / heatmap / type-stratified analyses

## Notable SQ3c result remembered from the prior chat

The previous run reported:

- coarse IA/ID class-stratified multivariate tests were significant after BH
- targeted `IDD5` / `IDM` tests were not significant after BH

Interpretation used in the prior discussion:

- evidence is stronger for coarse class-level L/R projection-pattern differences
- evidence is weaker for fine single-region claims

## Expected figure outputs from `v3`

Figures are written under:

- `R_analysis/scripts/figures/LR_analysis`

Notable outputs include:

- `sq1_composition.png`
- `sq1_ap_count_density_by_type.png`
- `sq1_ap_count_density_by_region.png`
- `sq2_univariate_strength.png`
- `sq2_region_mean_strength.png`
- `sq2_ap_ml_strength_density.png`
- `sq3c_region_stratified_multivariate.png`
- `sq3d_projection_laterality_index.png`
- `sq3d_projection_laterality_index_bars.png`
- `sq3d_li_inspect_one_side_or_extreme.png`
- `sq3d_li_inspect_bilateral_receiving.png`

## Relevant input paths in `v3`

- workbook: `R_analysis/tables/251637_INS_HE_inferred.xlsx`
- clustering RDS: `R_analysis/scripts/data_output/clustering/clustering_results.rds`
- raw matrix xlsx: `R_analysis/scripts/data_output/clustering/raw_matrix_神经元矩阵.xlsx`

## State of the last unfinished request

The last explicit request in the earlier chat was:

1. add the proportion-normalization explanation as a methodological comment in the Rmd
2. produce a handoff document for a new chat

Status now:

- item 1 is already present in `laterality_v1_v2_INS_HE_merged_v3.Rmd`
- this file completes item 2

## Best next steps

1. Re-run `prep` and the `SQ3d` chunk in `laterality_v1_v2_INS_HE_merged_v3.Rmd` to confirm the current LI tables/plots match the intended BH-filtered policy.
2. Decide whether to add the user-weighted LI as a **side-by-side sensitivity block** in SQ3d.
3. If weighted LI is kept, report it as a robustness comparison rather than a replacement unless the user explicitly wants to redefine the main endpoint.
4. Inspect whether the manuscript wording consistently says **global hemispheric asymmetry** rather than neuron-level laterality.

## Good resume prompt for a new chat

Use this if starting fresh:

> Continue work from `R_analysis/scripts/laterality_v1_v2_INS_HE_merged_v3.Rmd` and `R_analysis/scripts/LATERALITY_V3_HANDOFF.md`. The notebook now treats SQ3d as global hemispheric asymmetry using per-neuron proportion-normalized projection profiles, with BH-adjusted region-wise inference. First verify the current SQ3d outputs and then help compare the existing LI against an optional weighted sensitivity version without replacing the primary endpoint unless justified.

## Hybrid refactor branch file (new)

A separate refactor target was created to avoid changing `v3` in place:

- `R_analysis/scripts/laterality_v1_v2_INS_HE_merged_hybrid.Rmd`

Current hybrid-specific upgrades:

1. Added shared inferential metadata helper (`add_family_inference`) to standardize:
   - `p_raw`, `p_adj`, `adjust_method`, `family_id`, `significant`, `sig_label`
2. Applied family-scoped BH correction tags across core SQ blocks (`SQ1`-`SQ6`) while keeping single global tests unadjusted (`adjust_method = none`).
3. Standardized stars/labels to adjusted values where comparisons are repeated within a family.
4. Updated SQ5 description text to match rank-based mismatch panel (`|Δrank|`) instead of legacy z-score wording.

If continuing methodological work, prefer editing `..._hybrid.Rmd` first, then selectively port stable changes back to `v3` only after review.
