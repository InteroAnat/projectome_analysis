# Figure captions — whole-insula combined primary (v2)

**Pipeline:** [`v2_combined_primary_pipeline.Rmd`](../../../v2_combined_primary_pipeline.Rmd). **Stats:** [`../stats/`](../stats/). **Spec:** [`../spec/`](../spec/). **Anchor doc:** [`notes/LR_insula_analysis_review.md`](../../../../../notes/LR_insula_analysis_review.md).

**Cohort (this run).** N = **306** neurons (L=121, R=185) across 4 macaques after harmonization; primary inferential stratum is `IDD5_plus_IDM_balanced` (n_L = 81, n_R = 68).

**`p_combo` policy.** Extra-insula targets at **L3** (Gou panel definitions); intra-insula leaves at **L6** (`Ial`, `Iai`, `Iapl`, `Iam/Iapm`, `Ia/Id`, `Ig`, `Pi`, `Ri`); row-normalized per neuron. Column suffix `@L3` or `@L6` audits the hierarchy.

> **Mantel / F9 (this run).** Neurons in test *n* = 306; Mantel correlation ≈ 0.360; permutation *p* = 0.001. Upper-triangle pair count (F9 hex / F9b heatmaps) = 46665. Full row: `stats/08_mantel_replication.csv`.

**LI summary.** 15 targets reach BH q < 0.05 in F6 across all strata (`bilateral_receiving` panel only is interpretable as laterality; `one_side_or_extreme` panel is QC).

> ⚠️ **Interim cohort.** This run uses the partial dataset currently available. Numbers will change as more samples are added. CSVs in `../stats/` are the canonical record.

---

## F1A — `F1A_subregion_x_gou_panels_combined.png`

**Question.** How does each insula sub-region's axonal budget split across the six Gou et al. 2025 functional domains?
**What is shown.** Mean summed target proportion (L+R combined) per (sub-region × Gou domain). The `All_insula` row pools all 306 neurons. Each cell is annotated with the value and `n_total`.
**Anchor stats.** `../stats/01_subregion_x_gou_panels.csv` (one row per `Region × panel`); column-to-domain provenance in `../stats/01_gou_domain_column_map.csv`.
**How to interpret.** F1A is descriptive structure. Dysgranular IDD5/IDM are sensory-dominant (Ia/Id and Ig in p_combo at L6); IAL (anterior agranular) loads Emotional/Reward via caudal_OFC. Read F1B for the L vs R contrast.

## F1B — `F1B_subregion_x_gou_panels_LRcontrast.png`

**Question.** Within each sub-region (BH per row), which functional domains are biased L vs R?
**What is shown.** Cliff's *d* (L vs R) per (sub-region × domain) for **inferential rows only** (n_total ≥ 10; n_L, n_R ≥ 3). IAPM (L-only) and IDV (n=3) are auto-excluded. Stars = BH within row.
**Anchor stats.** Same CSV as F1A.
**How to interpret.** The dramatic All_insula `Emotional/Reward` Cliff *d* ≈ -0.35*** is driven by R-IAL caudal-OFC bias (anatomy × side confound; see F7). IAL row's `Sensory` Cliff *d* ≈ +0.38* is the LeftIAL-from-252385 Ial-loading effect; not a primary laterality finding.

## F2 — `F2_subregion_x_key_targets.png`

**Question.** Which curated extra-insula and intra-insula targets does each sub-region project to (L+R combined)?
**What is shown.** Mean proportion per (sub-region × target). Labels are `region@L3` for extra-insula, `region@L6` for intra-insula leaves. Cells with mean prop < 0.005 are blank to reduce visual noise.
**Anchor stats.** `../stats/02_subregion_x_key_targets.csv` (`frac_projecting`, `mean_prop`, Cliff *d*, Wilcoxon, Fisher; BH within row).
**How to interpret.** Confirms canonical macaque insula anatomy: IAL → caudal_OFC ≈ 0.64; IAPM → caudal_OFC ≈ 0.35; IDD5 → Ig@L6 ≈ 0.28.

## F2B — `F2B_subregion_x_key_targets_LRcontrast.png`

**Question.** Per row (sub-region), which key targets show L vs R asymmetry?
**What is shown.** Cliff's *d* (L vs R) for inferential rows only (IAPM L-only and IDV n=3 excluded). Stars = BH within row.
**Anchor stats.** Same CSV as F2.
**How to interpret.** Headline: IDD5 Ig@L6 Cliff *d* = +0.72*** (left dysgranular preferentially feeds Ig); IDD5 LPal@L3 *d* = +0.36** (left dysgranular preferentially routes through ventral pallidum). IAL VPal *d* = +0.21** is small and direction-consistent.

## F3A — `F3A_intra_insula_combined.png`

**Question.** What is the intra-insula source × target projection structure (L+R)?
**What is shown.** Mean prop on the L6 insula sub-set (5 source rows × 8 target columns; Ward.D2 column order on prevalence). Dotted outline = same-subregion self-edge; cells annotated with mean prop and number of source neurons projecting.
**Anchor stats.** `../stats/03_intra_insula_meanprop.csv` and `../stats/03_intra_insula_full.csv`.
**How to interpret.** IDM → Ia/Id ≈ 0.67 and IAL → Ial ≈ 0.31 are the dominant self / near-self projections. IDD5 → Ig ≈ 0.28 is the substrate of the L > R laterality finding (split by side in F3C).

## F3B — `F3B_intra_insula_prevalence.png`

**Question.** What fraction of each source sub-region projects to each insula target?
**What is shown.** Prevalence (% projecting; n with prop > 0 / n_total). Edges with < 3 projecting neurons are blanked. Cells annotated with percent and `n_with_proj / n_total`.
**Anchor stats.** `../stats/03_intra_insula_prevalence.csv`.
**How to interpret.** 96% of dysgranular neurons project intra-insula at L6 — the structural fact that the macaque insula is dominantly self-projecting.

## F3C — `F3C_intra_insula_LRcontrast.png`

**Question.** Which intra-insula edges differ L vs R within source row, BH-corrected?
**What is shown.** Cliff's *d* L vs R per (source × target) for **inferential rows only** (n_total ≥ 10). Grey row = excluded (IAPM L-only). Stars = BH within source row.
**Anchor stats.** `../stats/03_intra_insula_full.csv`.
**How to interpret.** Two BH-significant cells: **IDD5 → Ig +0.72*** (left dysgranular preferentially feeds primary interoceptive cortex; EPIC + Craig 2005 left-integration limb) and **IDD5 → Ia/Id -0.46*** (right dysgranular re-enters its own Ia/Id more heavily).

## F4 — `F4_interoceptive_gradient.png`

**Question.** Does projection to interoceptive targets (Ig, Pi, Ial, Iam/Iapm, Ia/Id) track soma AP per side?
**What is shown.** OLS `prop ~ Soma_NII_Y` per side and target; one facet per target.
**Anchor stats.** `../stats/04_interoceptive_gradient_regressions.csv` (slope, SE, p, BH, R² per side × target); raw points in `04_interoceptive_gradient_scatter_points.csv`.
**Caveat.** L neurons cluster posterior, R neurons anterior (sub-region/side confound across the whole cohort). Use F3C (within-IDD5) or F10 (octile) for laterality reads. F4 is descriptive.

## F5 — `F5_per_neuron_ibias.png`

**Question.** Per-neuron continuous hemispheric bias on axon length (Gou-style).
**What is shown.** Ibias = (contra − ipsi)/(contra + ipsi) on total axon length, vs Soma_NII_Y, faceted by sub-region. -1 = purely ipsilateral.
**Anchor stats.** `../stats/05_per_neuron_ibias.csv` (per-neuron) + `05_per_neuron_ibias_summary.csv` (per region × side).
**Biological note.** Insula has very sparse contralateral projection (~98% of insula neurons are purely ipsilateral). This is a structural property, not a null result; insula is not a major callosal source (compare Gou's PFC ITc neurons).

## F6 LI — `F6_projection_LI_bilateral_receiving.png`

**Question.** Per stratum, which targets show the largest BH-significant L vs R group-mean composition difference (LI), restricted to bilaterally-received targets?
**What is shown.** LI = (mean_L − mean_R) / (mean_L + mean_R + ε) per target × stratum; one facet per stratum. Bars labelled with `nL`, `nR`, BH q.
**Anchor stats.** `../stats/06b_projection_laterality_index_all_targets.csv`. BH-significant subset: `06b_projection_laterality_index_BH_significant.csv`.
**How to interpret.** `Ig@L6 LI = +0.68` in `all_neurons` (BH < 10⁻⁹) is the headline. `caudal_OFC@L3 LI = -0.48` is the sampling artefact (collapses inside dysgranular strata). `Ia/Id@L6 LI = -0.39` in IDD5 is the right-dysgranular self-loading.

## F6 (one-side / extreme) — `F6_projection_LI_one_side_or_extreme.png`

**Question.** QC view: targets where one side has a zero group-mean (LI hits ±1 mechanically).
**What is shown.** Same axes as the bilateral panel, but for `receipt_class == one_side_only_or_extreme`. Bars labelled with `nL`, `nR`, BH q.
**How to interpret.** **Do not interpret as laterality.** These bars (e.g. `belt@L3` LI = +1.0; `HF@L3` LI = +1.0) reflect absent projection on one side, not a magnitude difference. Published as QC only.

## F6 (violin) — `F6_projection_LI_violin_by_stratum_family.png`

**Question.** What does the LI distribution look like across strata families?
**What is shown.** Violins of LI for `all_neurons` vs `bilateral_soma_subregions` (per-region strata combined). Box overlay shows median + IQR.
**How to interpret.** `bilateral_soma_subregions` is bimodal (peaks at ±1) due to one-side-only targets — a visual reminder why `receipt_class` matters.

## F6 supplement — `F6_supplement_rank_based_receivers.png`

**Question.** Cross-checking F6 with rank-based tests: top BH-significant asymmetric receivers across balanced strata.
**What is shown.** Top 18 (or fewer) targets in any balanced stratum at BH q < 0.10, ranked by `best_q`; bars labelled with q, best stratum, family (intra/extra insula), n_L, n_R.
**Anchor stats.** `../stats/06_asymmetric_receivers_BH.csv`.
**How to interpret.** Three bars in this run: `Ig@L6` L>R (q ≈ 8 × 10⁻¹⁰, intra-insula), `Ia/Id@L6` R>L (q ≈ 8 × 10⁻⁴, intra-insula), `LPal@L3` L>R (q ≈ 0.01, extra-insula). Sparse because this is the partial cohort.

## F7 — `F7_hierarchy_sensitivity.png`

**Question.** Across 4 strata simultaneously, where does each headline endpoint (Ig, Cl, LPal, VPal, PAG-proxy = VMid, caudal_OFC) test L vs R?
**What is shown.** Cliff's *d* per (target × stratum); panels = strata. Bar label = layer (L3/L6) and BH stars (`*` <.05, `**` <.01, `***` <.001).
**Anchor stats.** `../stats/07_hierarchy_sensitivity_per_target.csv`.
**How to interpret.** `Ig@L6` is L>R*** in 3 of 4 strata (the only signal that survives every stratum). `LPal@L3` is L>R*** in IDD5_balanced. `caudal_OFC@L3` is R>L*** only in `all_combined` (sampling artefact). `PAG` (VMid proxy) is null — autonomic sub-model not testable at this resolution.

## F8 — `F8_loso_sensitivity.png`

**Question.** Leave-one-monkey-out: do the headline endpoints survive removing each animal?
**What is shown.** Cliff's *d* per (target × dropped sample) in IDD5+IDM stratum; `__none__` = full data. q labels = BH p_mag<0.05 within (stratum × dropped).
**Anchor stats.** `../stats/09_loso_sensitivity.csv` (includes `n_L`, `n_R` per drop).
**Caveat.** Dropping 251637 collapses the dysgranular cohort (no IDD5/IDM neurons left). Read the `__none__`, `252383`, `252384`, `252385` columns as the real check; the 251637 column is a structural artefact.

## F9 — `F9_FNT_vs_projection_pairwise.png`

**Question.** Positive control: are FNT morphological distances correlated with Bray-Curtis projection dissimilarity (hybrid `p_combo`)?
**What is shown.** Hex-bin scatter of upper-triangle pair distances. Subtitle reports n neurons, pair count, Spearman ρ on displayed pairs, Mantel p (999 permutations).
**Anchor stats.** `../stats/08_mantel_replication.csv`.
**Note.** This Mantel ρ ≈ 0.36 uses the hybrid `p_combo` profile and is lower than the §11A.5 ρ ≈ 0.676 (which used the full L6 profile); both are highly significant.

## F9b — `F9b_FNT_vs_projection_heatmaps_SQ5.png`

**Question.** SQ5-style triple heatmap view of the FNT vs Projection comparison.
**What is shown.** Three aligned heatmaps in the same hclust order (combined-distance): FNT01 ‖ Projection Bray ‖ |Δrank|. Side annotation track on the left. Column title repeats n and pair count.
**Anchor stats.** Same as F9.
**How to interpret.** Block structure indicates morphology-projection coupling clusters; |Δrank| highlights pairs that are similar by FNT but dissimilar by projection (or vice versa).

## F10 — `F10_interoceptive_AP_octile_heatmap.png`

**Question.** Sub-region-agnostic AP gradient: how do interoceptive target proportions track soma Y in 8 octiles?
**What is shown.** Mean L6 proportion per (octile × target) for {Ig, Pi, Ial, Iam/Iapm, Ia/Id}; rows sorted by median Y (posterior bottom).
**Anchor stats.** `../stats/10_ap_octile_interoceptive_profile.csv`.
**How to interpret.** Posterior octiles (Y ~170–185) are Ia/Id-dominant with secondary Ig (up to 0.31); transition octile (Y ~210) loses Ia/Id and gains Ial; anterior octiles (Y ~218) are Ial-dominant. The macaque counterpart of Gehrlach 2020 mouse posterior-to-anterior intra-insula flow.

## P13 flatmap — `../flatmap_overlays/P13_flatmap_with_LR_context_strip.png`

**Question.** Where do the 306 somata sit on the insula flatmap, by monkey × side?
**What is shown.** Combined flatmap with monkey-coloured + side-shaped (circle = L, triangle = R) markers; `n` strip below shows per-sub-region L/R counts (harmonized scheme).
**Anchor stats.** `../stats/00_context_per_subregion_n.csv`.

