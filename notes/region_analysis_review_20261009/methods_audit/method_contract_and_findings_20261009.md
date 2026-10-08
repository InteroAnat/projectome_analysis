# Methods and statistics audit of R consumers

Date: 2026-10-09. Read-only production review. This audit does not accept anatomical labels, registration, terminal fields, neuron subclasses or existing inferential claims. No historical report, workbook, source graph or production analysis was changed. The [numerical receipt](independent_consumer_measurements_20261009.json), [per-neuron diagnostic table](historical306_logstrength_vs_legacy_voxellength.csv) and [reproduction script](reproduce_consumer_measurements.py) bind the checked files and calculations.

## Measurement contract from the primary Methods

[Gou et al. 2025](https://doi.org/10.1016/j.cell.2025.06.005), STAR Methods e5–e7, distinguishes segmented terminal **arbors**, pooled total **neurite length**, and binary neuron-by-target **arbor presence**. The fMRI comparison uses pooled neurite length, whereas co-innervation uses arbor presence. Neither measure is a count of bare graph leaves. The original parsed text, attachment `8BNNG4KV`, lines 435–438, 444–446 and 463–477, and inspected deposited code are bound in the [full Methods review](../../../group_analysis/evolution_20261008/references/fmost_full_methods_20261009.md). Temporary full texts were read back and their existing hashes checked; none were copied into the repository.

The author's `zz1data.jl:237–257` separately accumulates whole-axon `lenipsi/lencontra` and segmented-arbor `arboripsi/arborcontra`. `zz2fig/src/2-plot.jl:4048–4053` excludes zero arbor totals and computes `(arborcontra-arboripsi)/(arborcontra+arboripsi)`. Positive means **contralateral**, not right hemisphere. A local all-compartment, terminal-target-conditioned voxel-length index is not a direct replication of that arbor index, even when its algebraic form matches.

The current legacy tables store per-region `Projection_Strength = round(log10(length+1),4)` (`population.py:663–670`, split implementation at 777–849). Their underlying `length` is an atlas-index-space legacy reconstruction measure; the core audit identifies whole-edge/proximal-label assignment, all-compartment inclusion and terminal-containing-target filtering. Calling it raw here means **before logarithmic transformation**, not calibrated native axon length or verified terminal-arbor length. It must not be conflated with the separate physical-affine, child-type-2 all-axon mapping kernel.

For the new descriptive maps, count, neuron occupancy and axon skeleton length remain separate observables. Candidate endpoints require non-root type-2 leaves of the complete graph; image review and biological terminal acceptance remain unresolved. Endpoint maps condition on neurons with at least one eligible leaf anywhere, including outside the reference FOV. All-outside neurons remain in that conditional denominator; no-leaf reconstructions are unassessed proxy contributions. Axon-length maps average selected neurons within animal, then animals equally. Neither family provides a t map, synaptic count or reviewed terminal field.

## Independently reproduced findings

### M01 — Transformed-strength composition is mislabeled as axonal budget

**Verified measurement-label defect.** `v2_combined_primary_pipeline.R:126–150` row-normalizes the `Projection_Strength` sheets; the same consumer pattern occurs in `combined_lr_primary_analysis.R:47–94`, `functional_hubs_L6.R:86–99`, `functional_hubs_analysis.R:86–107` and the Gou-category/laterality notebooks. A share of sums of logarithms is not a share of axon length. This transformation may be an explicitly chosen descriptive profile; it is not the raw-length budget.

For the exact **306 saved historical UIDs**, independently calculated mean intra-insula share is **0.5820449585087877** using normalized log strengths, versus **0.6399457333830875** using the corresponding legacy untransformed regional lengths. The maximum discrepancy between workbook strengths and `log10(length+1)` is 0.0000499784, consistent with four-decimal rounding. The 58% figure is reproduced by the transformed measure.

Exact overclaims: historical `outputs/combined_primary_v2/README.md:70`; canonical `v2_combined_primary_pipeline.Rmd:2076,2410`; living `notes/LR_insula_analysis_review.md:416,725–731`. They call this 58% an axonal/output budget. The 64% diagnostic must **not** silently replace it as a biological result: the legacy length semantics, compartment inclusion, atlas labels, sampling and registration remain unaccepted.

**Focused proposed correction:** retain values and column compatibility, but label current-source plots, metric specifications and generated captions as “normalized log-strength share” or “composition of log10(legacy regional voxel length + 1).” Add explicit measurement metadata and a living-report audit notice that historical 58% describes this transformed composition. Preserve the existing output directory and its bytes. A raw-length composition, if subsequently requested, is a separately named outcome with its own provenance and validated target partition; do not reuse old p-values for it.

### M02 — The gradient specification claims permutation p-values but stores OLS t p-values

**Verified method-label defect.** `v2_combined_primary_pipeline.R:170` and canonical Rmd line 410 specify “OLS slope … with permutation p.” Actual R lines 704–708 fit `lm` and extract `Pr(>|t|)`; Rmd uses the same implementation. The saved `spec/metric_spec.csv` repeats the permutation claim.

Independent SciPy OLS on the exact historical 306 IDs reproduces every inspected Ig slope and saved raw p-value to floating-point precision: left p = 5.721502546586582e-6, right p = 4.967409844337769e-14, pooled p = 5.553733980202841e-19. These are **receipts for the existing OLS calculation**, not new inferential evidence. There is no permutation calculation in that block. BH is applied within target across the three side analyses, not across all targets jointly (`R:721–724`).

**Focused proposed correction:** change the current-source metric description to “OLS slope with ordinary parametric t-test p; descriptive and unadjusted for animal dependence.” Keep historical numerical output unchanged and annotate its method classification in the living report. Do not implement a new permutation design or choose animal exchangeability assumptions in a labeling fix.

### M03 — Hybrid L3/L6 features overlap spatially

**Verified feature-construction property with interpretation consequences.** R lines 135–150 retain all 65 actual L3 columns and add eight L6 insular columns. Removing exact L6 target names does not remove their differently named L3 ancestors. Independent ARM volume sampling proves all Ial/Iai/Iapl/Iam-Iapm voxels lie inside the retained L3 `caudal_OFC`, and all Ia/Id, Ig and Ri voxels lie inside retained `floor_of_ls`. Cortical Pi also lies inside `floor_of_ls`. Full voxel counts are in the receipt. The normalized hybrid vector is therefore a multiscale overlapping feature profile, not a disjoint anatomical budget. No geometric anatomy acceptance follows from this array cross-tab.

Stripping the namespace also merges cortical `CL/CR_Pi` with subcortical `SL/SR_Pi` (pineal); 524 finest-level pineal voxels map to retained L3 EpiThal, whereas 3,492 cortical Pi voxels map to floor_of_ls. This proves label ambiguity, not that a particular neuron's Pi value contains pineal projection. The core reviewer is handling the producer collision; an already merged historical table cannot be disambiguated by its stripped name alone.

**Proposed correction:** label the current hybrid as an overlapping multiscale transformed-strength profile; do not call its domain sums anatomical mass. A future quantitative budget must use a verified nonoverlapping partition and retained cortical/subcortical namespaces. Changing the partition is an analysis change requiring new versioned outputs, not a silent table repair.

### M04 — Historical R outputs use a different cohort from the current workbook

**Verified version/cohort mismatch.** Current harmonized Summary, L3 strength and L6 strength have **353 common unique IDs**, all with valid final L/R. The saved Ibias/context tables use **306 IDs**, with 47 current IDs absent and no saved-only IDs. The saved README declares its historical 306 cohort; this is not evidence that its calculations used 353. Context values, full current-only identities and input hashes are retained in the receipt.

The current canonical notebook contains hard-coded 306/four-animal narrative and generated README text, so a future run on 353 can regenerate stale cohort assertions. “Figure QC passed” here only confirms that files and linked table dimensions exist; it does not establish fresh input/output lineage, semantic correctness or scientific acceptance.

**Proposed correction:** preserve historical outputs as a named historical run; add an external audit classification and bind future input hashes/run IDs. Derive future cohort text from the chosen, explicitly accepted manifest, rather than replacing 306 with another hard-coded number. The newer coarse human/atlas/candidate manifests remain separate evidence strata and must not be silently substituted into this fine-parcel analysis.

### M05 — Neuron-level tests do not establish animal-population laterality

**Verified design limitation and labeling overreach, not a claim that every calculated p-value is arithmetically wrong.** Current v2 Wilcoxon/Fisher and OLS blocks use neurons as observations, and Mantel permutes neuron labels without animal restrictions (`R:87–97,704–708,1075–1077,1335`). The “balanced” strata are only region filters (`R:154–160,195–204`); there is no balancing operation in them. Current IDD5+IDM contains 153 neurons: **139 from 251637**, plus 14 from three other exact samples. Rough left/right counts are not independent bilateral animal replication.

LOSO drops `SampleID` (`R:1239–1257`), although the figure/registry calls it leave-one-monkey-out (`1278,1484`). That name requires a verified one-sample-to-one-animal mapping for the specific run; the newer pipeline correctly uses explicit registry AnimalID. BH corrects declared testing multiplicity, not within-animal dependence, source/target sampling confounds, atlas uncertainty or registration. The deprecated multi-monkey script's SampleID-restricted PERMANOVA does not repair the unstratified primary consumer, and restricting permutations alone is not a general animal-population model.

The historical notes' “statistical unit is a brain region” statement for SQ3d (`notes/hemispheric_asymmetry_methods_comparison.md:62,74`) is also incorrect for its rank-sum test: each target is a hypothesis/summary location; the input observations are neurons. The ratio of group means is a target-level effect summary, not an independent region replicate. Positive `(contra-ipsi)/(contra+ipsi)` is contralateral dominance; calling it right-lateralized (`same note:76`) is incorrect without a fixed left-soma condition.

**Proposed correction:** retain historical p-values as conditional neuron-sample exploratory results with explicit dependence warnings, and remove claims of independent replication or stronger “multilevel” inference based solely on more tests/BH. Select animal-level, hierarchical or restricted inference only after the biological design and replication are agreed; this audit computes none.

### M06 — Tutorial contralateral panels deliberately reuse ipsilateral data

**Verified example/data-label hazard, not an observed main-pipeline map error.** Three copies of `Projectome_Analysis_Tutorial.Rmd` under `R_analysis/`, `R_analysis/scripts/` and `R_analysis/scripts/Projectome_Tutorials/` pass `mat_strength_ipsi_t` into the panel named CONTRA at lines 381–382 and 702–703. The inline comment acknowledges an example; the rendered panel title still describes it as contralateral. Do not use these example panels as biological evidence. A future tutorial fix should use aligned real contra data or label both as demonstration-only; no main saved figures were traced to this example in this audit.

## Checked coverage and limits

All **63 R/Rmd files** in `R_analysis` and `group_analysis/R_analysis` were read in a complete static scan and hashed, including duplicate tutorials and archived consumers; per-file line counts, test/join tokens and coverage status are in the JSON. Focused contextual review covered current v2 R/Rmd, combined-primary and deprecated multi-monkey consumers, functional-hub and intra-insula scripts, Gou-category extraction/mapping, FNT clustering, bilateral merging, laterality hybrid/v3 notebooks, tutorial examples, living Methods notes, parsed full-paper reviews, relevant author-code laterality and arbor definitions, and current map measurement contracts. This is not a claim that every R notebook was executed or every generated workbook was independently reconstructed. Rscript was unavailable on PATH; numerical receipts use independent Pandas/NumPy/SciPy calculations. Core and inventory specialists own the producer/table audits.

Actual FNT input checks passed: 306 contiguous zero-based indices, 306 unique joined names, and exactly one score for every unordered pair including diagonal (46,971 rows). No missing-pair zero-imputation defect was observed in that actual file. Older fallback truncation/natural-sort alignment and supervised type penalties remain different methods that require explicit provenance; they are not evidence that current FNT registration is accepted. Regional category assignments are local harmonizations/many-to-one summaries, not measured autonomic function or a literal replication of every Gou Table S2 association.

No review can establish “no errors” from static scanning alone. The defensible result is the enumerated verified defects, passed checks and remaining gates: legacy measurement/namespace repair, immutable input/output lineage, native/atlas correspondence, source-label/cohort acceptance, terminal/arbor image evidence and an agreed animal-level inferential design. Existing descriptive software-verified candidate maps remain interpretable within their explicit proxy and coordinate limitations.
