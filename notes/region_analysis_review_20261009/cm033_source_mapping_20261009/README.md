# CM033 local-source mapping and fresh normalization snapshot — 2026-10-09

**Actual progress:** the 19 selected series have a documented local-D source mapping; a new fine anatomy→CMT→NMT fit and forward/inverse technical QC are saved. The all-run spatial rebuild is active. **The historical left-only GLM still lacks a verified bridge to that new anatomical frame.** No old contrast or T image was flipped, warped or assigned a transform by this audit.

Evidence: [19-source mapping](selected19_source_mapping.csv), [all 37 conversion records](all37_conversion_source_mapping.csv), [mapping/normalization snapshot](mapping_and_normalization_snapshot.json), [stored-array diagnostic](storage_order_intensity_correspondence.json), [full-array correspondence](full_array_correspondence.json), and [final process/source-frame snapshot](final_snapshot_status.json). All writes are confined to this folder; external data and jobs were read only.

## Mapping evidence

The conversion workbook's reconstruction-2 functional rows give the sorted task-specific scanner order. The original MATLAB `construct_GLM_scaninfo_033.m:14–26` resolves scanner ID to its position in the task list; `concatenate_ids.m:8–9` sorts and deduplicates that list. Conversion CSV scanner/reconstruction pairs, acquisition rows, source JSON, TR/frame headers and content fingerprints provide additional evidence. The saved `first_level_GLM_cm033.m` lists MSTIM IDs only through 55; it demonstrates the indexing method and earlier sequence, not the complete later cohort. Later IDs rely on the workbook sequence, with scanner 57 independently anchored by exact voxel correspondence.

| Task | Scanner IDs | Legacy task-specific indices |
|---|---|---|
| OPTO | 16, 17, 18, 39, 45, 46, 47 | 01–07 |
| Selected MSTIM | 23, 25, 27, 28, 53, 54, 55, 57, 58, 60, 61, 64 | 01–12 |

All 19 independent scanner/task/index/source-path assignments agree with the new external staging manifest. The full 37-record conversion ledger includes 36 unprefixed local series and the explicit legacy-resliced fallback for scanner 23; its extra 18 records do not enlarge the selected 19-run cohort. Scanner23's unprefixed reconstruction-2 source is absent; the surviving `rsub-CM033_task-MSTIM_run-01_EPI.nii` is explicitly resliced. Reconstruction1/poorEPI is not substituted.

Every scaled voxel value matches exactly between preserved scanner-ID sources and local legacy files for **16→OPTO01, 39→OPTO04, 47→OPTO07 and 57→MSTIM08**, when comparing the stored arrays with X/Y order reversed and Z unchanged. Maximum difference is zero across the complete four arrays. This is a comparison-only storage-order relationship, **not proof of physical left/right or permission to flip another image**. All source bytes remain untouched. Scanner23 is close in the same storage order (sample correlation 0.9999807) but differs at 14,938,927 values, maximum absolute difference 15,747.534; it remains a resliced source, not an original-equivalent series. Other selected mappings are record/header corroborated, without an independent full-array scanner-ID anchor in this audit.

Scans57/58 contain 160 actual frames each, whereas the acquisition workbook records NR384. The preserved scanner57 source also has 160 frames. Actual frames are retained; missing frames are not invented. All 19 source TR headers match the curated table. The new spatial route explicitly disables fresh slice-timing correction because local timing numbers are unverified; it must not be described as GLM-ready preprocessing.

## Current normalization and live state

The [new external reconstruction note](C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/docs/CM033_D_SOURCE_SPATIAL_REBUILD_20261009.md) supersedes the earlier “no fresh normalization” state. Its completed fit uses the frozen task-A candidate (`SHA256 f0a95a6b7e666c3b58a5e80fd179178bdca1da6e9b8405cbba730c26cb49c576`), fine ANTs settings, seed20261009, eight threads and the supplied 0.25-mm CMT/NMT grids. All three input hashes match. The candidate anatomy retains stored axis signs, replaces dimension-valued affine scaling with pixdims and uses an explicitly arbitrary centred computational origin. Its acquisition/study identity and physical laterality remain unconfirmed.

Independent checks of saved outputs establish:

- Both forward images are finite and match their exact bound reference grids.
- Both nonlinear Jacobian images are finite with zero nonpositive voxels; minima are 0.249633 (CMT) and 0.288322 (NMT). These exclude the affine component and do not constitute anatomical acceptance.
- Ten normalization output files plus their manifest, input hashes, QC images and supporting records are hash-bound in the 74-source snapshot. Fresh forward/inverse technical QC is saved; the earlier premature QC assertion remains historical and is superseded by the successful current QC receipt.

At the final snapshot, WSL Python **PID405, PPID317**, was executing `run_spatial_rebuild.py` (elapsed256 s); the parent bash was PID317. The source manifest exists, but `spatial_rebuild_completed.json` does not and no fresh `SPM.mat` was found in the rebuild. These are time-bounded process/output observations, not a claim that a wait completed. No process was controlled, stopped or restarted.

Root actually inspected five hash-bound figures: CMT forward three-plane/colour and 18-section checkerboards; NMT forward three-plane/colour and 18-section checkerboards; and NMT inverse three-plane/colour. Broad envelope, central fissures and ventricular-region correspondence are plausible, with residual cortical/peripheral, inferior/cerebellar/brainstem and contrast differences. These support coarse candidate review, not fine Ial borders, physical laterality or the historical GLM bridge. NMT review paths:

- [Forward slices](C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/cm033_Dsource_20261009/normalization_qc/NMT_forward_checkerboard_slices.png)
- [Forward colour/checkerboard](C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/cm033_Dsource_20261009/normalization_qc/NMT_forward_checkerboard.png)
- [Inverse colour/checkerboard](C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/cm033_Dsource_20261009/normalization_qc/NMT_inverse_checkerboard.png)

## Left-only GLM and the remaining integration step

The actual historical `glm-mstim-pos1-short-tr3-smooth2mm/SPM.mat` has **1,472 inputs: scanner57:160, 58:160, 60:384, 61:384 and64:384**, TR3 s. This is the separately identified left-only model; the mixed-site all-TR3 model is not substituted. Its contrast, T and estimation mask exist. All 1,472 cached SPM matrices agree with the old contrast affine after explicit one-based→zero-based conversion.

However, all five current smoothed-input NIfTI headers use `(-0.75,-0.75,+2)` mm diagonal, while cached SPM input matrices and produced contrast use `(+0.75,-0.75,-2)` mm at the old frame. The newly normalized task-A anatomy is a third, arbitrary-centred frame. Current input filenames/header labels therefore cannot establish an old-GLM→new-anatomy transform. Saved source correspondence is not automatically a coordinate bridge between resampled derivatives.

Safe next steps in fresh derivatives are:

1. Let the existing isolated spatial job finish; independently check its full-series geometry, EPI→anatomy fit, source hashes and functional-chain QC without duplicating it.
2. For reuse of the historical left-only result, construct a reference with verified source payload lineage in the **cached SPM fit coordinates**, then explicitly fit and inspect its bridge to the frozen task-A anatomy. The mean/reference must not silently use the source files' changed current headers. If payload lineage cannot be established, fit a new left-only model after temporal/data-quality prerequisites are explicit; the spatial-only job is insufficient by itself.
3. Once the source frame and bridge are verified, compose that bridge with the pinned native→CMT→NMT chain. Warp the signed contrast/T with declared interpolation and the actual SPM estimation mask with nearest neighbour into fresh projectome derivatives; independently reproduce grid, finite values and coverage before provisional regional integration.

This is a concrete available route toward CM033 descriptive integration, with one unresolved source-frame bridge rather than a request for another dataset. Neither a network raw source nor a separately corrected reference is required. The original fMOST transform is confirmed unavailable and remains a limitation, not a recovery gate. Scientific/anatomical acceptance remains false.
