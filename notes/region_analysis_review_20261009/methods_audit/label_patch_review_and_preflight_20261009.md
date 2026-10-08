# Factual label corrections and consumer preflight

The proposed v2 patch is staged only. It corrects scientific labels and specifications in seven active R/Rmd consumers and appends a historical-results warning to the living report. It does not change numeric operations, statistical tests, selection thresholds, matrix column identifiers, output filenames, cohort membership, or historical outputs. Use `proposed_methods_label_corrections_v2.patch`; the earlier unversioned proposal is retained as development history.

Patch SHA256: `dda80bc8f4429c82b6edf441987e9a84a16f5b31deb6bb5e507e5e7a6662cb04`.

Validation: `git apply --check` passed against the recorded originals; four focused Python unittest checks passed. The tests compare R code tokens outside comments and string literals, output references and quoted stratum identifiers, specification corrections, append-only preservation, and source/proposal hashes. Rscript was unavailable, so no R execution or figure regeneration is claimed. The changed strings require scientific review; token preservation alone cannot establish that every caption is correct.

## Verified meanings

- The historical 58% result is a mean share of row-normalized `log10(legacy_length + 1)` values. It is not a raw axonal-length budget. Independently matching the historical 306 UIDs gives 0.5820449585087877 for that transformed share and 0.6399457333830875 for an analogous untransformed legacy-length share. The second number is a diagnostic, not a replacement biological result: the legacy length measurement itself uses atlas-index distances, all compartments, proximal-node whole-edge assignment and terminal-containing-target selection.
- The gradient code extracts the ordinary OLS coefficient t-test p-value. The specification's permutation-p claim is incorrect. The patch names the existing calculation without changing it or asserting animal-independent inference.
- L3 ancestors and selected L6 descendants overlap spatially. Their normalized feature shares are not an exclusive anatomical partition or conserved length budget.
- The `_balanced` strata are region restrictions with L/R sampling. They do not establish animal balancing. The LOSO code removes a `SampleID`, so its display is changed to leave-one-SampleID-out. A sample-to-animal registry would be required to assert the animal as the omitted unit.

Independent numerical receipts and source anchors are in `independent_consumer_measurements_20261009.json` and `method_contract_and_findings_20261009.md`.

## Required preflight before a corrected R rerun

These are recommendations for a separate, coordinated implementation. They are not implemented by the label patch.

1. Bind an explicit expected cohort UID set and source hashes. Check UID uniqueness and exact set equality in Summary and every consumed ipsi/contra strength/length sheet; check composite UID against SampleID/NeuronID and metadata agreement. Record any intended exclusions explicitly. An intersection must not silently choose the analysis cohort. The current nine-sheet harmonized workbook actually has the same 353 unique UIDs in every sheet, with no observed duplicates or set differences. Historical outputs use 306 UIDs; that historical cohort is not evidence that a new 353-neuron rerun has occurred.
2. Bind the matrix measurement, units, atlas version, reference and target namespace through sidecar provenance before numeric conversion. The seven active consumers treat every column outside NeuronUID/SampleID/NeuronID/Neuron_Type as a numeric target and replace resulting NA with zero; none has an exact-UID-set guard. Require finite nonnegative target values and fail on unexpected metadata/non-numeric target cells. The planned exporter adds no arbitrary matrix metadata, so this is a missing safety contract rather than an observed metadata-column failure in the current workbook.
3. Use a lossless target dictionary. The planned collision-safe split exports distinguish C_PM/S_PM, C_CM/S_CM, C_RM/S_RM, C_R/S_R and C_Pi/S_Pi; other target names remain unchanged. Absolute Projection_Length_all keeps CL_/CR_/SL_/SR_ and the existing index-length units. Update all INS lists, hub maps, gradient targets, atlas keys and laterality queries together. Only C_Pi can correspond to the cortical INS target; S_Pi must never be included as INS. Plain Pi in historical merged exports cannot retrospectively resolve the two components. Plain CM queries also cannot silently stand in for S_CM.
4. Report target coverage and expected target omissions explicitly. The current `intersect` pattern would silently discard C_Pi when the old INS list still requests Pi; the independent synthetic namespace oracle reproduces this. Check target namespace and exact mapping before any intersection or row normalization.
5. Choose and bind a canonical R entrypoint before rerunning. The canonical Rmd and standalone R currently implement materially different F6 selection rules. Save generated/source hashes and run into a fresh directory after reconciliation. Do not change p-values or select a new inferential design as part of a label correction.

## Separate unresolved entrypoint defect

`v2_combined_primary_pipeline.Rmd:1508` computes `pmin(p_pres_BH, p_mag_BH, na.rm = TRUE)` and lines 1518-1521 select the resulting value below 0.10. `v2_combined_primary_pipeline.R:1100` onward retains separate presence and magnitude test families and selects q below 0.05. Applying these existing selectors to the saved F6 table yields six target IDs for the Rmd selector, versus two presence rows and three magnitude rows for the R selector. No new inferential p-values were computed. The exact target lists and source hashes are saved in `schema_and_entrypoint_readback_20261009.json`. This requires an explicit design choice; the factual-label patch deliberately preserves both existing implementations.

## Scope and acceptance

The staged warning classifies old results as historical and unsuitable for direct biological budget, accepted terminal-field or animal-independent claims. The patch does not establish anatomical INS membership, registration acceptance, genuine terminal arbors, synapses, a valid inferential model, or compatibility with the upcoming collision-safe exports. Candidate graph leaves, axon trajectories and author-defined terminal arbors remain distinct measurements.
