# MSTIM integration in the pinned NMT/ARM reference

CM032 has a reproducible **provisional regional comparison** and a source-backed coarse **left Ial / left vAIC pos1** characterization for MSTIM38–41. Its exact NMT electrode-tip coordinate was not recovered and is not required for this coarse description; anatomical registration remains provisional. The [site-note and actual FLASH figure review](../mstim_source_update_20261009/cm032_site_figure_review.md) separates human/session labels from image-only observations.

CM033 was checked against the **updated** DEB fMRI working records and the actual Oct9 audit. The scan16/39 header repair passes independent source readback. Current EPI normalization is absent, historical GLM affines differ from the current reference, and the all-TR3 fit mixes left/right sites; no compatible current NMT statistic was identified. The [updated CM033 source/model audit](../mstim_source_update_20261009/cm033_updated_debfmri_audit.md) identifies separate native models without treating code publication as anatomical acceptance. External fMRI projects were read only.

An [additive source refresh](../mstim_source_update_20261009/cm033_source_refresh.md) records changed external notes/manifest and the intended existing converted inputs in `D:/OPTO_fMRI_CM`. The absent old configured BIDS route does not mean CM033 data are unavailable. A separately corrected reference is optional corroboration; a trustworthy source/model/reference binding and anatomical QC still need to be established before spatial comparison.

- [Exact CM032 contrast and coverage warp reproduction](cm032_warp_reproduction/warp_reproduction.json)
- [Additive exact CM032 T-image and coverage warp check](../goal_completion_20261009/cm032_t_warp_recheck/t_warp_recheck.json)
- [Signed fMRI response and coverage at ARM L1–L6](cm032_regional_comparison/cm032_signed_response_by_ARM_level.csv)
- [Projection and response summaries within identical coverage](cm032_regional_comparison/cm032_projectome_common_coverage.csv)
- [Regional integration parameters, hashes and limitations](cm032_regional_comparison/integration_provenance.json)
- [Readable alignment/response QC](cm032_readable_qc/cm032_alignment_and_response_qc.png) and [display correction receipt](cm032_readable_qc/display_provenance.json)
- [Ancillary source and reference readback](ancillary_source_readback.json)
- [Independent readback of all 1,676 regional and 25,140 joint rows](independent_cm032_regional_readback_20261009.json)

The saved CM032 signed contrast reproduced **exactly voxel for voxel**, with maximum absolute difference zero, using ANTs 2.5.1 and the existing CMT/NMT affine and warp pairs. Its normalization moving image was already in EPI physical space: the first step is identity. Presence of a converted inverse anat-to-EPI file does not show that it was applied. This verified chain supersedes the generic inverse-affine description in the earlier [readiness snapshot](../../../group_analysis/evolution_20261008/crossmodal/input_readiness.json); that original snapshot is preserved.

The actual `SPM.mat` contrast is stimulus versus implicit baseline, averaging four run-specific stimulus coefficients with weights 0.25 (runs 38–41, TR 2 s). SPM used session-specific global scaling with target 100. These are **globally scaled fitted contrast units**; percent signal change has not been established. An additive check now independently reproduces the existing T-image warp: all 15,974,400 T voxels and the 5,015,104-voxel coverage mask agree exactly with their saved counterparts, with maximum T difference zero. This supersedes the historical ancillary receipt's unperformed-T-check status without rewriting it. The T image remains descriptive here; its median within a region is not a new regional t statistic. No threshold, association p-value or independent-voxel inference is supplied. Numerical warp agreement does not accept anatomical registration or stimulation-site localization.

The native fMRI sampling is 0.75 × 0.75 × 2 mm. Its 0.25 mm NMT output is an interpolation grid, not improved acquisition resolution. The actual skull-stripped fMRI NMT-v2.0-named reference and projectome NMT-v2.1 reference have identical grids and scaled voxel values; their compressed file hashes differ. This comparison applies to those exact references, not all atlas versions. All target definitions use the same pinned ARM image and key, including their flagged L4/L5 key conflicts.

The actual SPM estimation mask was transformed with nearest-neighbour interpolation. Its 5,015,104 covered NMT voxels define identical support for signed response, candidate-end counts and axon length. Zero measured responses remain zero; uncovered regions remain missing. There are 1,676 regional rows across six separately sampled ARM levels and 25,140 source-target rows for all 15 named source groups. L3 is the declared primary descriptive comparison; the other hierarchy levels remain available for scale sensitivity and must not be summed together. Atlas-background source groups remain in the full neuron ledger, outside comparisons requiring a named source parcel.

Projection columns retain the saved equal-animal mean candidate-end counts per eligible neuron and axon length per selected neuron, integrated only within fMRI coverage. They do not sum voxel occupancy to estimate regional neuron frequency. They also do not reweight poorly sampled source groups into comparable biological populations. The eight projectome animals are not paired to CM032 by verified identity. The user and acquisition-note/ontology chain support a coarse left Ial / left vAIC pos1 site; the reviewed FLASH screenshots do not delineate Ial boundaries or calibrate an NMT electrode tip. Use that coarse description, with its attribution, for exploratory context. Precise source-to-tip distances, spatially accepted matching and laterality conclusions remain unsupported.

The [verified primary Methods review](../../../group_analysis/evolution_20261008/references/scientific_basis_20261009.md) records exact Zotero citation details and source anchors: Jung 2021 supports NMT hierarchy/reference definitions; Gou 2025 motivates comparing anatomical extent with cortical fMRI while distinguishing axon arbors; Yang 2025 supports modality-specific interpretation and independent stimulation validation. Our coverage-aware regional summaries are an explicit adaptation. Anatomical projections and MSTIM responses measure different processes, and spatial agreement would not establish monosynaptic causation.

Reproduce from the project root into fresh destinations using [the warp reproducer](reproduce_cm032_warp.py) and `group_analysis/scripts/summarize_mstim_arm.py --help`. The exact original warp commands are in its receipt. Regional input arguments and hashes are recoverable from the integration receipt. [The display-only correction](render_cm032_qc.py) wraps a clipped colorbar label without recomputing or changing the regional tables. PSC in that compact label means percent signal change. Sources, original maps and the original QC image are preserved; the readable image is the current display.

```powershell
python -X utf8 -B notes/region_analysis_review_20261009/mstim_integration_20261009/reproduce_cm032_warp.py --output <fresh-warp-directory>
python -X utf8 -B group_analysis/scripts/summarize_mstim_arm.py `
  --warp-receipt <fresh-warp-directory>/warp_reproduction.json `
  --statistic D:/multimodal_fmri/derivatives/glm/sub-cm032/ses-20160309/glm-mstim-pos1/maps-NMT/spmT_0001_in_NMT.nii.gz `
  --normalized-anatomy D:/multimodal_fmri/derivatives/norm/sub-cm032/ses-20160309_ana/cm032_20160309_ana_NMT_Warped.nii.gz `
  --endpoint-run group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints `
  --endpoint-review group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints_review/endpoint_run_readback.json `
  --axon-run group_analysis/evolution_20261008/arm_mapping_20261009/main/axons `
  --axon-review group_analysis/evolution_20261008/arm_mapping_20261009/main/axons_review/projection_run_readback.json `
  --label-map group_analysis/evolution_20261008/projection_inputs/arm_labels_20261009/main/label_map.csv `
  --output <fresh-regional-directory>
```
