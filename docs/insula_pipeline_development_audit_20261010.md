# Insula projectome: a step-by-step development audit

Prepared by Codex, 2026-10-10, Asia/Shanghai. This is a guided review of the saved development history, not a new scientific analysis.

**Start with the evidence snapshot at `1eba0fd5e5aca1b24ed959664e553af8b44cc85e` on `codex/insula-pipeline-evolution-20261008`.** The branch began from `78008cbaab71bf737dc7129c72e5fe284c3ba40f`. This guide is an additive documentation follow-up. Its chronology follows substantive development commits; historical source dates and processing dates are kept separate.

For each step, open the evidence, perform the listed comparisons, then write your decision. Checking a box means you completed that review action. It does not automatically accept anatomy, registration, a terminal field or a biological hypothesis. Use **agrees for the stated measure / discrepancy to investigate / unassessed / deferred**. Keep computational and anatomical decisions separate.

Use the current evidence linked below. Earlier READMEs contain dated checkpoints and sometimes describe a limitation subsequently repaired; read their update notices and later receipts before treating them as current. Test totals and map counts have different scopes and must not be added into an overall scientific acceptance score.

## The development route

| Review step | Development checkpoint | Main question |
|---|---|---|
| 1 | Legacy history; source-preserving baseline, `c6bea20` | Which inputs and earlier results are actually preserved? |
| 2 | Literature/registration basis, consolidated through `6baef3d`, refreshed in `1d0307f` | What do the primary methods support, and what evidence is unavailable? |
| 3 | Initial inventory and explicit 251637 discovery/refinement | Which datasets and spatial candidates exist, under which rule? |
| 4 | Source/atlas classification and current ARM labels, `a3fa188` | What constitutes an INS-supported neuron versus a candidate? |
| 5 | Region/software/table repairs, `c6bea20`, `a3fa188` | Which defects were demonstrated and how were they repaired? |
| 6 | Initial maps, official ARM display and consolidation, `a3fa188`, `adf64fe` | What does each map measure, and what is the background? |
| 7 | Six actual hierarchy volumes, `d95b9be`; clustering, `3671700` | Do regional tables agree, and how exploratory are the clusters? |
| 8 | Native field review, `1d0307f`; graph census/expanded review, `2e629b7` | How much terminal evidence was actually inspected? |
| 9 | End-branch maps and bounded projectome delivery, `9b46f7c` | Does distal graph geometry agree without being promoted to an arbor? |
| 10 | Legacy overview reconciliation, `2075426` | Do animal identities, counts and dataset claims reconcile? |
| 11 | Second map inspection/marker displays, `7d3fefc` | Are map values, visible dots and reproduction procedures trustworthy? |
| 12 | Origin/evidence maps, historical counts and 250432/115 investigation, `4e51d91` | Were new neurons retained and uncertain origins represented honestly? |
| 13 | Legacy-strength receiving-parcel maps, also `4e51d91` | Does the new display preserve the legacy measure at six levels? |
| 14 | Public publication and verification, `1eba0fd` | What was published, and what remains local or unresolved? |

The two parts of step 12–13 share a commit but represent successive user-directed work. Publication-status commits between substantive changes are recorded in the [publication history](../notes/region_analysis_review_20261009/publication_20261009.md). CM032/CM033 history is an optional appendix; their integration remains deferred.

## 1. Establish the preserved baseline

Open the [living project note](project_note.md), [region/table review](../notes/region_analysis_review_20261009/README.md) and [source/table census](../notes/region_analysis_review_20261009/table_audit/README.md).

- [ ] Identify the original source workbooks, human annotations and SWCs separately from historical derivatives, staging products and current maps. Verify that corrected diagnostic outputs have separate paths.
- [ ] Inspect the baseline commit and change sequence without switching the dirty checkout to an older revision. The existing Cursor journal and tracked Python cache changes were preserved outside the scientific publication.
- [ ] Note that an older filename, modification time or matching count does not establish its original generation date, units or exact lineage.

Read-only history commands, from `D:\projectome_analysis`:

```powershell
git -c safe.directory=D:/projectome_analysis show --stat c6bea20
git -c safe.directory=D:/projectome_analysis log --reverse --format="%h %ad %s" --date=iso-strict 78008cb..1eba0fd
git -c safe.directory=D:/projectome_analysis status --short -uno
```

**Expected evidence:** the table census assessed 2,475 files; source preservation and software contracts have specific receipts. Historical cohorts and original human records were not silently overwritten. This is bounded validation, not proof that every possible error has been excluded.

**Your decision:** source preservation ______; historical results suitable for reuse ______; unresolved lineage/units ______.

## 2. Audit the scientific and registration basis

Open the [primary-methods basis](../group_analysis/evolution_20261008/references/scientific_basis_20261009.md), [local Gao/Gou review](../notes/region_analysis_review_20261009/literature_method_update_20261009/gao_local_methods.md), [Liu review](../notes/region_analysis_review_20261009/literature_method_update_20261009/liu_local_methods.md), and the [new target-strength methods ledger](../notes/projection_maps_by_origin/legacy_target_strength_methods.md).

- [ ] For each methodological claim, locate the exact paper, DOI, PDF page and supporting method. Keep author methods, our adaptation and unavailable implementation details separate.
- [ ] Verify the working model: native brain images are registered to a template, with neuron coordinates transformed separately. The original individual fMOST-to-NMT transforms, export-origin receipts and landmark evidence are unavailable here.
- [ ] Distinguish regional axonal length, segmented arbor morphology, image-derived varicosities, bulk fluorescence density and synaptic evidence. They are different measurements.

**Expected evidence:** Gao2022 supports input-profile-based target subdivision; Gao2023 supports regional axonal length; Gou2025 requires additional arbor segmentation and native registration context. The Liu papers have distinct scopes; the exact Sang Liu Methods were not located in the recorded search. Yan2022 is a serial-two-photon bulk-tracing comparator, not an implemented single-neuron fMOST detector. Its atlas labels were not transferred to our maps.

**Your decision:** literature support for each retained measure ______; acceptable adaptations ______; anatomical/registration dependencies still open ______.

## 3. Recover the inventory and the actual 251637 selection rule

Open the [dated complete inventory](../group_analysis/evolution_20261008/inventory/live_inventory_20261008/README.md), [explicit geometry rule](../group_analysis/evolution_20261008/validation/discovery_geometry_rule_20261009.md), [refinement readback](../group_analysis/evolution_20261008/reference/refinement_folded_20261009/independent_readback_20261009.json), and [corrected review-priority rule](../group_analysis/evolution_20261008/classification/coarse_insula_review_20261009/distance_priority_v2_20261009/README.md).

- [ ] Separate the 47 portal sample IDs plus two tracker-only identities from the eight animals in the current map selection. Channel/sample variants are not extra independent monkeys.
- [ ] Trace the spatial rule: 260 manually labelled 251637 reference somata; `abs(X-128)` hemisphere folding; separate 0.005/0.995 axis quantiles; 0.25 mm/index conversion; inclusive 2 mm padding for the folded discovery screen.
- [ ] Check that discovery and refinement have explicit geometry/reference parameters. Discovery's 179 candidates and refinement's 164 retained identities are different selection contracts, not contradictory anatomical counts.
- [ ] In a nearest-anchor example, confirm that the queried neuron cannot anchor itself. Use another Henry/original-portal INS identity in the same exact sample. Adjacent numeric IDs are retrieval cues only.

**Expected evidence:** the inventory contains 8,746 neuron identities. Independent discovery/refinement checks agree on 1,223 historical identities. The axis-wise box is a spatial candidacy screen, not a probability of insular membership. Missing coordinates and incomplete screens retain unknown status or lower bounds.

**Your decision:** inventory scope ______; rule reproduced ______; source assumptions acceptable for candidate retrieval ______.

## 4. Review soma identity, ARM location and human evidence

Open the [soma-audit delivery](../group_analysis/evolution_20261008/atlas_locations/soma_audit_20261009/delivery_readback_20261009.md), [coarse review](../group_analysis/evolution_20261008/classification/coarse_insula_review_20261009/README.md), [52 unassigned-source cases](../group_analysis/evolution_20261008/classification/background_neuron_audit_20261009/README.md), and [current evidence definitions](../notes/projection_maps_by_origin/README.md).

- [ ] Follow one neuron by full `SampleID::NeuronID`, from source hash and original graph to unique soma/root, coordinate encoding, saved voxel and official ARM label. Bare neuron numbers repeat across samples.
- [ ] Confirm that `XYZ/250` gives atlas-index coordinates here; applying the reference affine gives declared NMT millimetres. These are not interchangeable numeric coordinates. Compare current rounding with the alternative export-origin policy without automatically relabelling the neuron.
- [ ] Keep direct ARM assignment, Henry coarse INS evidence and coordinate candidacy as distinct fields. Henry's 260 unique INS neurons belong to **251637 / animal 936**, not 251736. His wide-field checks do not independently establish fine ARM parcels.

**Current evidence partition, to verify against the ledger:**

| Category | Neurons |
|---|---:|
| ARM insula and Henry coarse INS evidence | 212 |
| ARM insula without a Henry note | 89 |
| Henry INS with conflicting ARM precentral operular assignment | 35 |
| Henry INS with unassigned ARM label 0 | 13 |
| Other named-neighbor candidates | 74 |
| Other unassigned candidates | 39 |
| Total selected | 462 |

Thus 301 have named ARM-insula origins, 48 more have Henry INS evidence outside those labels, and 113 remain other candidates. The union of ARM/Henry support is 349, with overlap accounted for. There are 52 label-0 sources: 13 Henry-supported plus 39 other candidates. **Atlas label 0 does not diagnose native white matter.** The 1,912 own-source roots in the broader inventory and the selected 462 are different scopes; earlier byte-collision flags were subsequently reconciled by full numeric graphs.

**Your decision:** source identity/lookup agrees ______; coarse INS evidence retained ______; fine anatomical acceptance still requires ______.

## 5. Audit verified software and historical table defects

Open the [table findings and independent source/cell evidence](../notes/region_analysis_review_20261009/table_audit/README.md), [code review](../notes/region_analysis_review_20261009/code_audit/region_analysis_code_review_20261009.md), [isolated correction provenance](../notes/region_analysis_review_20261009/table_audit/corrected_diagnostics_20261009/correction_provenance.json), and [live software validation](../notes/region_analysis_review_20261009/validated_repairs/live_suite.json).

- [ ] Inspect neuron `251637::523.swc`: retained M1 length 1074.521 implies `round(log10(1074.521+1),4)=3.0316`, rather than the historical 4.8942. Check that the repair aggregates justified lengths before taking the logarithm.
- [ ] Check the unknown-side diagnostic: historical length total 1164.580 is routed to unknown, with ipsi/contra zero. The fresh current-rule total 1156.749 has a different unresolved lineage and is not substituted into the historical correction.
- [ ] Check cortical/subcortical namespace collisions, composite identities, lossless reload/export, missing values, sparse hierarchy handling and no-overwrite behavior. Preserve demonstrated defects separately from hypothetical risks.

**Expected evidence:** focused regressions and independent cell readback support the specific repairs. The older 353-neuron canonical workbook passed its paired strength checks and did not contain 523; do not describe the historical defect as corruption of that entire cohort. Historical units are absent from 74/100 unique Summary-containing workbook contents. Missing units cannot be supplied merely by assuming current source encoding.

**Your decision:** verified defects resolved within their stated scope ______; historical results requiring caution ______.

## 6. Follow the transition from initial maps to official ARM displays

Open the [earlier mapping workflow](../group_analysis/evolution_20261008/reports/mapping_workflow_20261009.md), [current primary ARM maps](../group_analysis/evolution_20261008/arm_mapping_20261009/README.md), and [measurement terminology](region_analysis_terminology.md).

- [ ] Recognize the earlier 260-neuron one-animal pilot and MIP views as historical products. Current multi-monkey displays use matched single slices; a MIP does not validate source anatomy.
- [ ] In a current sheet, confirm that the background is the pinned skull-stripped symmetric **NMT v2.1 T1-weighted MRI**, not an atlas-label image or fMOST fluorescence. MRI and overlay must use the same plane and affine. The intensity window controls brightness, not anatomy.
- [ ] Check full official ARM source labels. Historical labels such as HumanINS or G are evidence/selection provenance, not fine anatomical parcels for current titles. Exact key spellings, including `operular`, are preserved.

**Keep these map families distinct:**

| Measure | What contributes | Denominator/scale |
|---|---|---|
| Whole-axon length | Child-type-2 edges split along their trajectories | Template mm; within-animal mean over selected neurons, then equal contributing-animal mean |
| Candidate axon-end abundance/density | Original non-root type-2 full-graph leaves | Count, or count/mm³, per end-eligible neuron; equal contributing-animal mean |
| Candidate end occupancy | At least one eligible original end in a voxel | Fraction of eligible neurons; several ends in one voxel do not multiply occupancy |
| Legacy projection strength | All-compartment proximal-edge lengths retained by a target-region graph-leaf gate | Per-neuron `round(log10(retained voxel length+1),4)`; a separate regional measure |

**Expected evidence:** the current selection is 462, represented by disjoint historical preparation runs of 436 and 26. There are 429 endpoint-eligible neurons and 33 unassessed endpoint rows. Whole-axon and end-conditioned denominators differ. Source/animal groups without observations are not imputed as zero animals. No t-map was generated.

**Your decision:** metric understood ______; MRI/display alignment agrees ______; map family suitable for the intended question ______.

## 7. Check the six-level regional tables and exploratory profiles

Open the [six-level hierarchy export](../notes/region_analysis_review_20261009/hierarchy_tables_20261009/README.md), [actual target dictionary](../notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/targets.csv), and [clustering review](../notes/region_analysis_review_20261009/clustering_20261009/README.md).

- [ ] Follow a neuron at two target levels. Confirm both levels come from their actual ARM volumes, with full names, target statuses, volumes and cortical/subcortical domains. Keep four key-range conflict entries visible.
- [ ] Check a positive target, an omitted measured-zero target and an endpoint-ineligible neuron. Sparse omitted endpoint values mean zero only for eligible neurons; ineligible values remain NA. Outside-reference and label-0 evidence are retained separately.
- [ ] Inspect the independent matrix readback and reconciliation with 154 saved count/length maps. Never sum ancestors and descendants into one anatomical budget.
- [ ] For clustering, compare absolute target-side and source-relative features, animal holdout, unequal sampling and sparse outliers. A silhouette-selected cut is not a demonstrated cell type.

**Expected evidence:** saved checks cover 2,331,252 matrix cells. There are 428 mapped-positive whole-axon profiles and 425 mapped-positive endpoint profiles. The absolute axon split largely follows hemisphere; source-relative analysis exposes animal dependence. Henry-only evidence comes from one animal. These are neuron-profile analyses; they are not Gao2022's target-location subdivision method.

**Your decision:** tables agree for the declared measures ______; cluster conclusions support only ______; further replication needed ______.

## 8. Audit what was actually assessed at terminal locations

Open the [native field pilot](../notes/region_analysis_review_20261009/terminal_field_assessment_20261009/README.md), [whole-selected graph census](../notes/region_analysis_review_20261009/terminal_field_census_20261009/README.md), and [expanded native review](../notes/region_analysis_review_20261009/terminal_review_expansion_20261009/README.md).

- [ ] In an original graph, verify that SWC type **2 means axon**, and that an ending has no children in the complete graph. Cropping at a target boundary must not create a new ending.
- [ ] Compare branching sections with endings, simple endings and internal passage examples. A connected ARM-labelled section is a graph representation, not a segmented biological arbor; passage does not exclude en passant boutons.
- [ ] Open representative native panels and examine neighboring depth planes for continuation, overlap or truncation. Separate a covered cache filename from actually reviewed image pixels.

**Expected evidence:** all 462 graphs contain 75,625 original axon endings and 47,044 connected sections. Native correspondence is available for 201 pairs; 9,209 endings have a matching cached-cube filename. Actual combined review inspected **11 individual leaves in 11 neurons across seven animals**, plus two internal passage locations. This purposive sample does not validate all endings or estimate terminal-field prevalence. Source soma membership is not accepted from target morphology.

**Your decision:** local image correspondence ______; ending completeness/field status ______; unassessed scope ______.

## 9. Review the end-branch extension and its independent check

Open the [end-branch method and outputs](../notes/region_analysis_review_20261009/axon_end_branches_20261009/README.md), [independent graph/volume readback](../notes/region_analysis_review_20261009/axon_end_branches_20261009/independent_end_branch_readback.json), and [bounded projectome delivery](../notes/region_analysis_review_20261009/projectome_delivery_20261009/README.md).

- [ ] Trace one original axon leaf upstream to the first full-graph branch point, root or non-axon parent; check the flagged final compartment-transition edge. Confirm selection is made before regional cropping.
- [ ] Check that selected edges are allocated along their trajectories and not deposited as whole lengths at the ending voxel. Compare an end-branch length to its whole-axon length using matching eligibility.
- [ ] Read the voxel-face rounding repair and real-source impact audit before assuming that every repaired kernel required old maps to be regenerated.

**Expected evidence:** 67 end-branch NIfTIs and six-level matrices use the same 462 neurons, with 429 eligible and 33 NA. Independent selected-edge, regional-cell and full-voxel agreement passed. The real-source rounding-impact audit found no old whole-axon allocation change. These distal chains remain graph proxies; even a long unfinished shaft can contribute. The end-branch clustering sensitivity did not establish biological types.

**Your decision:** end-branch arithmetic agrees ______; value as a terminal-review descriptor ______; anatomical acceptance unresolved because ______.

## 10. Reconcile the legacy per-monkey overview

Open the [legacy overview reconciliation](../group_analysis/evolution_20261008/inventory/legacy_overview_20261009/README.md), [eight-monkey table](../group_analysis/evolution_20261008/inventory/legacy_overview_20261009/per_monkey_insula_inventory.csv), and [independent overview readback](../group_analysis/evolution_20261008/inventory/legacy_overview_20261009/independent_readback.json).

- [ ] Match each AnimalID to its exact fMOST SampleID using the original Monkey Data overview and established progress-table producer. Investigate stale zeros and historical count claims rather than filling them by assumption.
- [ ] Check verified copied nominal 5 µm series for animals 936, 945, 948, 331 and 631. For 797, 605 and 900, no copy is verified at the scoped root; availability elsewhere remains unknown.
- [ ] Keep atlas counts, Henry counts, overlapping spatial screens and historical review claims separate. Examine the 15 source discrepancies and the duplicate Henry annotation for 114.

**Expected evidence:** eight source identities and 72 count cells were independently checked. Henry's 261 INS annotation rows represent 260 unique neurons; 114's fine-label conflict remains open. Review priority puts unverified-copy animals 797, 605 and 900 first. Nominal 5 µm sampling does not establish isotropic sampling or optical resolution.

**Your decision:** monkey identity/inventory agrees ______; dataset status correctly scoped ______; highest-priority unresolved animal ______.

## 11. Inspect the second map review and the visible markers

Open the [second inspection](../notes/projection_map_review_round2/README.md), [full numerical receipt](../notes/projection_map_review_round2/reproduction_checked/inspection/inspection_receipt.json), [marker page 1](../notes/projection_map_review_round2/reproduction_checked/endpoint_markers/space-NMTv2p1_desc-candidateAxonEndVoxelMarkers_page-01.png), and [reproduction procedure](../notes/projection_map_review_round2/reproduction.md).

- [ ] Check one count/density pair using the 0.015625 mm³ reference voxel volume. For occupancy, confirm the [0,1] bound and distinction between multiple ends and one occupied voxel.
- [ ] Choose a visible dot and read the display contract: it represents an occupied candidate-end voxel centre, not an individual neuron, arbor or synapse. Fixed dot area is a readability choice; colour carries the quantitative density.
- [ ] Compare the saved MRI and overlay slices at X56/Y200/Z87. Inspect the common log-density colour scale, its display cap and the unsmoothed quantitative NIfTI.

**Expected evidence:** an orthogonal inspector checked 452 existing NIfTIs and original full-graph endpoints. No numerical correction was needed. Four new marker sheets were reviewed. Dependency-aware sorting preserved referenced historical files; only explicit task-owned drafts/previews were archived with old/new mappings and byte checks.

**Your decision:** quantitative values and visible marker semantics agree ______; display clarity ______; reproduction/archive references resolve ______.

## 12. Reconcile historical counts, source-origin views and the two requested cases

Open the [origin/count report](../notes/projection_maps_by_origin/README.md), [per-monkey history table](../group_analysis/evolution_20261008/soma_origin_maps_20261010/count_reconciliation/per_monkey_counts_history_and_evidence.csv), [distribution figure](../group_analysis/evolution_20261008/soma_origin_maps_20261010/count_reconciliation/per_monkey_history_and_evidence.png), and [source investigations](../group_analysis/evolution_20261008/soma_origin_maps_20261010/source_investigation/source_investigation_provenance.json).

- [ ] Compare identities, not just totals: older unharmonized selection 306; dated July tracker 353; September staging 420; current selection 462. Every local older 353 identity is retained, with 109 additions. September retains 412 current identities plus eight unregistered 250432 entries; 50 further current identities yield 462.
- [ ] Explain the apparent 300-versus-400 discrepancy: **301 is named ARM-insula origin assignment within the 462 selection**, not the whole selection. Of those 301, 258 were retained from the older 353 and 43 were added. Matching July counts do not prove identical July workbook bytes.
- [ ] For 250432, check all 234 local reconstructions, its 128-versus-234 metadata discrepancy, eight direct ARM-insula roots and 21 other spatial candidates. No verified monkey ID or explicit INS injection record was recovered; keep all 29 at dataset level.
- [ ] For the sole added 251637 selection identity, `115.swc`, confirm current label 0, no Henry note and 31 candidate ends. Alternative-origin assignment and distance to another Henry anchor, 114, remain sensitivity/retrieval evidence. It was newly included, not demonstrated to be newly traced or newly confirmed INS.

**Selected-neuron count checkpoints for the eight registered animals:**

| Animal / dataset | July | September registered entries | Current | ARM insula | Henry INS outside ARM insula | Other candidates |
|---|---:|---:|---:|---:|---:|---:|
| 331 / 252385 | 47 | 50 | 55 | 18 | 0 | 37 |
| 605 / 252383 | 5 | 5 | 21 | 6 | 0 | 15 |
| 631 / 252714 | 0 | 16 | 16 | 4 | 0 | 12 |
| 797 / 252790 | 0 | 38 | 52 | 38 | 0 | 14 |
| 900 / 252718 | 13 | 13 | 14 | 4 | 0 | 10 |
| 936 / 251637 | 260 | 260 | 261 | 212 | 48 | 1 |
| 945 / 252527 | 23 | 25 | 25 | 14 | 0 | 11 |
| 948 / 252384 | 5 | 5 | 18 | 5 | 0 | 13 |
| Total registered entries | 353 | 412 | 462 | 301 | 48 | 113 |

The separate eight September 250432 records explain the staging total of 420; they are not extra current registered monkeys.

**Expected display evidence:** nine named-origin cards plus six evidence-category views are alternate views of this one selection, with 30 additive NIfTIs. Solid diamonds and yellow arrows identify actual displayed source somata. The [independent source/voxel readback](../notes/projection_maps_by_origin/independent_readback.json) checks all 462 sources, 75,625 ends and 30 maps. [Scale documentation](../notes/projection_maps_by_origin/map_scale.md) separates raw soma counts, endpoint density, arbitrary marker size and MRI brightness.

**Your decision:** count history reconciled ______; source uncertainty retained ______; 250432/115 next anatomical evidence needed ______.

## 13. Audit the latest receiving-parcel strength maps

Open the [legacy-strength methods and reproduction guide](../notes/projection_maps_by_origin/legacy_target_strength_methods.md), [map index](../group_analysis/evolution_20261008/arm_target_strength_20261010/map_index.csv), [source-by-target table](../group_analysis/evolution_20261008/arm_target_strength_20261010/source_target_strength.csv), [representative six-level figure](../group_analysis/evolution_20261008/arm_target_strength_20261010/figures/from-ARM6-728_target-ARM1to6_legacy_strength.png), and [independent readback](../notes/projection_maps_by_origin/legacy_parcel_independent_readback.json).

- [ ] Follow a raw retained length through `round(log10(L+1),4)`, its within-monkey per-neuron mean, and the equal-contributing-monkey mean. The implementation preserves the legacy all-compartment proximal-edge/graph-leaf gate. **Mean of per-neuron logarithms is not log of mean length.**
- [ ] Choose one receiving ARM parcel in the NIfTI. All its voxels hold the regional mean; they do not represent independently measured local density. Label-0 background is NaN, measured positive-label zeros remain zero, and zero parcels are transparent only in the figure.
- [ ] Compare source origins and target levels separately. Nine level-6 source origins times six actual target hierarchy grids gives 54 maps, not 54 cohorts. The source names are 41/42/43/228/229/541/542/728/729 in the official key; all 301 named-origin neurons are represented. The other 161 sources remain in the complete 462-neuron measurement ledger, with uncertain fine origins preserved.
- [ ] Verify the common display scale 0–2.5667130303030303, with no additional log, percentile clipping, smoothing or MIP. Target-level graph-leaf gates are freshly recalculated; coarse/fine retained lengths need not add as historical parent pooling would.

**Expected evidence:** 54 NIfTIs, nine figures, six levels of named-origin neuron-by-target length/strength matrices, all-462 sparse measures and per-animal/source-by-target summaries. Independent checks passed on 462 hashes, 18 representative neurons through the separate legacy API, and every map voxel. Four focused legacy tests, six source-evidence tests and nine publication-tool tests passed within their respective scopes.

**Methodological decision:** Gao2022's neuron-by-target-cube axonal-length clustering explains the requested input-based target subdivision. These legacy-strength ARM parcel maps are the first descriptive view. They do not implement that clustering or establish new subdivisions. Legacy retained reconstruction strength remains unit-dependent; it is not accepted terminal-arbor strength.

**Your decision:** legacy formula preserved ______; receiving-parcel display interpretable ______; alternative axon/arbor measure or target-profile analysis needed ______.

## 14. Verify publication and write your overall audit conclusion

Open the [publication log](../notes/region_analysis_review_20261009/publication_20261009.md), [original exact-index receipt](../notes/region_analysis_review_20261009/publication_index_receipt.json), and the [public feature branch](https://github.com/InteroAnat/projectome_analysis/tree/codex/insula-pipeline-evolution-20261008).

- [ ] Confirm that the substantive origin/legacy-map commit is `4e51d91`, followed by publication records through `1eba0fd`. Automatic public-payload review was initially rejected, then the user explicitly authorized publication and the push was verified.
- [ ] Confirm that code, reviewed figures, tables and provenance are published. Raw SWCs, source PDFs, NIfTIs and private archives remain local, with paths/hashes/reproduction procedures. Public presence is not anatomical acceptance.
- [ ] Read a receipt as a dated snapshot: the 1,393-file exact-index check binds the prepared content snapshot, not every later documentation append. Compare the current feature HEAD with the live remote when conducting your audit; later legitimate documentation commits may advance it beyond this guide's `1eba0fd` evidence snapshot.

For this development delivery, remote master remained `f096e8165da18f84e91ea69d6e5cbfd58f248e75`; unrelated work was preserved. Use `git ls-remote` for a fresh remote check rather than a stale tracking ref.

**Write your conclusion in four separate sentences:**

1. Software/source consistency I have verified: ______.
2. Anatomical classifications or terminal observations I accept, with case-specific evidence: ______.
3. Exploratory findings I consider worth following up: ______.
4. Unresolved dependencies and the next evidence I need: ______.

There is no overall “all errors excluded” checkbox. Unavailable transforms, incomplete tracing/terminal review, uncertain fine origins, unequal animal sampling and unimplemented target-profile subdivision remain explicit limits.

## Optional historical appendix: CM032 and CM033

Regional MSTIM comparisons entered the branch at `534344d`; CM032 warp checks followed at `df53bbe`, with additional source/bridge investigations preserved in later records. Open the [integration history](../notes/region_analysis_review_20261009/mstim_integration_20261009/README.md), [CM032 site-figure review](../notes/region_analysis_review_20261009/mstim_source_update_20261009/cm032_site_figure_review.md), and [updated CM033 lineage audit](../notes/region_analysis_review_20261009/mstim_source_update_20261009/cm033_updated_debfmri_audit.md).

Review image-space definitions, accepted orientation, statistical contrast, estimation-mask coverage, registration lineage and stimulation-location evidence independently. Existing CM033 alignment work is preserved; an older rejection or a later candidate bridge is not the complete alignment history. The user subsequently asked to finish the projectome phase first. Integration remains deferred and is not required to mark the projectome audit actions above as reviewed. Projection geometry and stimulation responses support different biological interpretations.

## Optional live reproduction after the documentary review

The audit above can be completed by reading saved evidence. To obtain a new computational receipt, use the detailed [map reproduction procedure](../notes/projection_map_review_round2/reproduction.md), [origin-map procedure](../notes/projection_maps_by_origin/reproduction.md) or [legacy receiving-parcel procedure](../notes/projection_maps_by_origin/legacy_target_strength_methods.md). Follow the listed script dependencies and exact input hashes. Use fresh output and receipt paths; do not overwrite bound deliveries. Numerical agreement on a repeated run still leaves anatomical acceptance as a separate decision.
