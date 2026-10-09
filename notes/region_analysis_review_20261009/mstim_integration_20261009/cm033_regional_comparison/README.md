# CM033 provisional MSTIM–projectome summaries

Processing date: 2026-10-09. This single run summarizes the saved CM033 signed contrast, descriptive T values and actual SPM coverage in all six distinct ARM hierarchy volumes. It reuses existing maps and fits no registration, GLM or association. Software reproducibility does not establish anatomical registration, physical laterality or stimulation coordinates.

Saved outputs:

- [Signed response by ARM level](cm033_signed_response_by_ARM_level.csv): 1,676 rows, with 19/65/143/251/513/685 rows at L1–L6. There are 1,299 covered and 377 uncovered rows.
- [Projectome measures in the same coverage](cm033_projectome_common_coverage.csv): 25,140 rows across 15 mapped source groups. Each measure has 5,655 NA rows, corresponding to the 377 uncovered targets across all 15 groups.
- [QC image](cm033_alignment_and_response_qc.png): actual XY/XZ/YZ slices at X=56, Y=200, Z=87. Titles and colorbar were inspected and fit; no display correction was needed. The external anatomy contour broadly follows the template outline at these cuts. Fine/internal alignment and physical laterality remain unaccepted.
- [Integration provenance](integration_provenance.json): complete source hashes, saved-map hashes, producer version, constraints and output hashes. Receipt SHA256: `b60a759f01057ca977ff363714cbe2bba7beac98d07fc74f33c215d05487b52a`.
- [Exact source-identity reconciliation and saved-image review](mapped_source_identity_reconciliation.json): all 410 mapped UIDs and all 82 common source metadata fields match exactly across endpoint, whole-axon and end-branch ledgers. Endpoint and end-branch eligible UID sets are exactly the same 378 identities. The additional 26 combined-ledger identities are all unresolved ARM0 source QC.

The original endpoint and whole-axon runs contain 436 selected identities; 403 are endpoint-eligible. The unchanged end-branch run contains the combined 462 identities, of which 429 are eligible. These different global counts do not change the named-source comparison: all three measures use the same 410 mapped source identities in 15 groups and 40 animal/source strata. Conditional eligibility is 378 identities in those strata. The remaining 52 combined-ledger identities have unresolved source labels and retain their QC evidence rather than receiving invented source parcels. No new cohort was selected.

Saved maps already average neurons within each animal/source group and then weight contributing animals equally. The summary does not divide those map values again. Whole-axon length uses selected-neuron denominators; candidate-end counts and reconstructed end-branch length use their eligible-neuron denominators. Group-specific selected/eligible/contributing-animal counts remain explicit in the table. Lengths are reference-template millimetres, candidate ends are graph proxies, and the additive count sum is not regional neuron frequency or voxel occupancy.

The source is the historical 2016-01-20 `glm-mstim-pos1-short-tr3-smooth2mm` model, runs 57/58/60/61/64, TR 3 s, contrast 1 (`stimulus_gt_baseline`, “stimulus > implicit baseline - All Sessions”). The actual SPM model records session-specific global scaling to 100, scaling factors 0.00026903194–0.00027627997 and residual df 1,364. Fitted contrast units are not established percent signal change. Regional median T values are descriptive and are not regional t statistics.

The [genuine frozen CM033 chain](../../cm033_samecontrast_bridge_candidate_20261009/statistic_chain/chain_provenance.json) and [independent exact reapplication](../../cm033_samecontrast_bridge_candidate_20261009/statistic_chain/independent_statistic_chain_readback.json) bind every source and transform. All three native and NMT arrays reproduced exactly. The original SPM mask contains 68,267 voxels; its propagated nearest-neighbour validity mask contains 4,259,693 NMT voxels. Outside that validity mask, zero is fill rather than evidence of no response; nonzero interpolated boundary values are also invalid and excluded. Actual coverage extending into ARM0/background is retained as measured source-mask geometry, without substituting a brain mask. No response-value support threshold was used.

The exact supplied NMTv2.0 and pinned NMTv2.1 reference grids and scaled arrays match. This is an equality result for the bound files, not general interchangeability of atlas versions. All six actual ARM volumes are reported separately, without pooling hierarchy levels. Zero labels and key-level conflicts—including 45/545 at L4/L5—remain explicit image-footprint/QC categories. These values do not resolve atlas metadata conflicts or establish anatomical density.

The chain remains provisional: physical left/right and original historical estimation-payload identity are unproven; the explicit coordinate recoding is not an accepted physical rotation. This regional run performed no new GLM estimation, smoothing or normalization fit. There is no inferential claim from the interpolated T image, association test, accepted fine parcel, terminal arbor/bouton/synapse, or causal/monosynaptic interpretation. The unavailable original fMOST transform remains a limitation, not a data-recovery prerequisite. CM032 sources and saved tables remain unchanged.

The producer is `group_analysis/scripts/summarize_mstim_arm.py`, SHA256 `b023fcd7fc3cab2b42c002528b676197ef6b56556a688637391247136761c48b`. Its additive subject/end-branch interface preserves CM032 default numeric behavior; historical artifacts retain their original producer hashes. Receipt path transport changes filesystem I/O only and preserves the genuine original source dictionaries and receipts.

Run from the project root using the projectome Python environment. The original destination was `cm033_regional_comparison`; the command below requires a fresh reproduction destination:

```powershell
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -B group_analysis/scripts/summarize_mstim_arm.py `
  --subject cm033 `
  --warp-receipt notes/region_analysis_review_20261009/cm033_samecontrast_bridge_candidate_20261009/statistic_chain/chain_provenance.json `
  --warp-review notes/region_analysis_review_20261009/cm033_samecontrast_bridge_candidate_20261009/statistic_chain/independent_statistic_chain_readback.json `
  --statistic notes/region_analysis_review_20261009/cm033_samecontrast_bridge_candidate_20261009/statistic_chain/nmt/sub-cm033_ses-20160120_space-NMT_desc-provisionalBridge_spmT.nii.gz `
  --normalized-anatomy 'C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/cm033_Dsource_20261009/norm_reference/derivatives/norm/sub-cm033/ses-20160120_ana/cm033_20160120_Dsrc_taskA_ana_NMT_Warped.nii.gz' `
  --endpoint-run group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints `
  --endpoint-review group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints_review/endpoint_run_readback.json `
  --axon-run group_analysis/evolution_20261008/arm_mapping_20261009/main/axons `
  --axon-review group_analysis/evolution_20261008/arm_mapping_20261009/main/axons_review/projection_run_readback.json `
  --end-branch-run notes/region_analysis_review_20261009/axon_end_branches_20261009/selected462_ARM `
  --end-branch-review notes/region_analysis_review_20261009/axon_end_branches_20261009/independent_end_branch_readback.json `
  --label-map group_analysis/evolution_20261008/projection_inputs/arm_labels_20261009/main/label_map.csv `
  --output notes/region_analysis_review_20261009/mstim_integration_20261009/cm033_regional_comparison_reproduction_20261009
```
