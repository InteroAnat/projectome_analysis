# Reconstructed axon end-branch profile sensitivity

This is a source-relative **graph-length sensitivity**, not evidence for biological terminal arbors or cell types. It uses the same selected462 ledger, the actual exclusive ARM L6 volume and 342 official bilateral pairs. Original atlas/source classifications remain unchanged. No extra maps, whole matrices, workbooks or figure family were generated.

The [completed receipt](profile_sensitivity_provenance.json), SHA256 `2d208bf218173e01abecf7039941a16646f7243634bb7550b1fd31f3ebb2a962`, binds the completed end-branch producer, [all462 independent readback](../independent_end_branch_readback.json), every consumed saved table, prior source-relative partitions, original manifest, whole-axon source summary, producer/analysis code and versions. All named output hashes were read back successfully. Thirteen focused synthetic/gating/identity/NA/conservation/coverage/ARI checks passed, including during the real execution.

There are **429 graph-computable neurons**, **33 unassessed NA rows**, and **428 positive mapped profiles** across eight animals. One computable neuron has length only in background/outside targets and is excluded from compositional clustering. Total reconstructed end-branch length is **36,651.459961mm** in the declared reference template; **30,946.627569mm** lies in mapped L6 targets. Among429 computable neurons, median mapped fraction is **0.88259** and median end-branch/whole-axon template-length fraction is **0.50299** (5th–95th percentile **0.31461–0.95003**). Those fractions join already saved whole-axon measurements by exact identity; whole-axon measurements were not recomputed. Missing ends and zero denominators remain NA.

The primary representation divides each positive mapped profile by its mapped total, takes square roots, and uses Hellinger distance with average linkage. Official same-domain bilateral targets are permuted into ipsilateral/contralateral features using each pinned source hemisphere. Fits remain neuron-weighted. Candidate k2–8 were evaluated with seed42, 100 subsets at0.8, within-animal stratification, equal-animal caps and remaining-cohort animal holdouts. Background/outside lengths and absolute graph totals remain visible in the QC ledger instead of entering the mapped compositional denominator.

**No candidate cut meets the declared balanced-size criteria.** k2 is retained only as the highest-silhouette diagnostic display, with **2 versus426** neurons and silhouette **0.12166**. The two small-partition identities are `251637::434.swc` and `251637::463.swc`, both animal936 and Henry visual cases; they are diagnostic members, not a new cell class. k2 within-animal subset mean ARI is **0.81606**, but its5th percentile is **−0.00393**. Equal-animal-cap subsets contain88 neurons and have mean ARI **0.06977**; removing animal936 gives remaining-cohort ARI **0**. Thus a high fit-specific agreement for many resamples would not establish a replicated partition. All260 Henry visual cases come from animal936;227 have mapped end-branch profiles. No redundant Henry-only fit was added.

On common exact identities, k2 agreement with prior source-relative **whole-axon length** is ARI **−0.0014063** over428 neurons. Agreement with prior **candidate-end counts** is ARI **−0.00313965** over425 neurons; three neurons have mapped end-branch profiles but no mapped candidate-end profile. The latter difference can occur because a distal chain traverses mapped tissue while its endpoint lies elsewhere. These comparisons show sensitivity to the measured observable, not accepted biological classes, inferential significance, or predictive performance.

Essential outputs:

- [All462 eligibility, absolute lengths, coverage and assignments](all462_end_branch_eligibility_and_assignments.csv): unassessed/profile-ineligible assignments remain blank. Original graph end/edge counts, total/mapped/background/outside length, whole-axon fractions and source/QC fields are retained.
- [Candidate-cut diagnostics](candidate_k_diagnostics.csv) and [stacked resampling/holdout diagnostics](stacked_resampling_and_holdout_diagnostics.csv).
- [Common-UID k2 comparisons](common_UID_k2_partition_comparisons.csv).
- [Official bilateral feature dictionary](official_ARM6_bilateral_feature_dictionary.csv), [diagnostic-cut source/animal composition](display_cut_source_animal_composition.csv), and [top20 relative targets per diagnostic cluster](display_cut_top20_relative_targets.csv). The top20 table is a compact signature, not a complete matrix or anatomical reassignment.

The producer used the completed enriched `neuron_summary.csv` as its manifest. Its hash differs from the older original CSV because it includes measured evidence. The adapter accepts only the original manifest or its exactly hash-bound completed summary, then independently requires exact ordered identities and identical reconstruction hashes, coordinate frame/scale/reference, ARM indices/full names/side, original labels and evidence/source-group fields. It does not accept a merely matching set of bare neuron numbers.

Reproduction requires a new output destination. The script refuses an existing destination and incomplete/mismatched producer or independent-review receipts:

```powershell
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -B `
  'notes/region_analysis_review_20261009/axon_end_branches_20261009/analyse_end_branch_profiles.py' `
  --run 'notes/region_analysis_review_20261009/axon_end_branches_20261009/selected462_ARM' `
  --review 'notes/region_analysis_review_20261009/axon_end_branches_20261009/independent_end_branch_readback.json' `
  --prior-relative 'notes/region_analysis_review_20261009/clustering_20261009/ARM6_source_relative_sensitivity' `
  --output '<fresh-directory-under-axon_end_branches_20261009>'
```

Unfinished unbranched shafts can qualify under this graph rule. Template length is not calibrated native tissue length. Reconstruction completeness, source/export origin and registration remain limitations; biological/anatomical acceptance is false. Profile normalization can amplify short or poorly covered traces. The software result completes a bounded sensitivity analysis and does not replace the image-based terminal-field review.
