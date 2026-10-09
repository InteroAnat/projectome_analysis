# CM033 current-payload reference and integration bridge follow-up

Audit date: 2026-10-09. External projects and the active process were read only. Earlier receipts remain unchanged. No registration, statistical fitting, contrast warping, or anatomical acceptance occurred here.

## Completed alignment handoff: current readback (2026-10-09)

The user identified the completed conversation **Verify cm033 alignment and normalise**. Its DEB report and delivery records supersede this folder's earlier running-process snapshot: all 19 runs and 7,376 frames passed the spatial/frame checks, the fresh EPI→native→CMT→NMT chain completed, and every run's EPI and NMT mean boundary montage was visually reviewed. The completed isolated reconstruction used adaptive SPM reorientation; it produced 3D normalization checks, not a new GLM or normalized 4D BOLD.

The [completed alignment handoff readback](completed_alignment_handoff_readback_20261009.json) freshly verifies 25 source bindings and reconciles actual identities, frame totals and recorded review scope. All 19 external bindings already used by the projectome bridge and statistic chain match, including the installed reference mean, native anatomy, AFNI matrix and four ANTs transforms. The [current CM033 regional comparison](../mstim_integration_20261009/cm033_regional_comparison/README.md) already reuses that completed spatial chain. No new registration, resampling or visual inspection was performed in this readback.

Completion and broad spatial correspondence are established at that scope. The historical GLM's original payload identity remains unproven; its separately documented provisional bridge and actual model mask retain that limitation. Physical laterality, fine INS localization, source slice-gap assumptions and unverified slice timing remain qualified as recorded in the external report. Earlier receipts and images remain unchanged.

## Historical preparation and running-process snapshot

- [Prepared mean NIfTI](sub-cm033_ses-20160120_desc-currentPayload1472VolumeMean_space-cachedSPM_bold.nii.gz): SHA256 `b964c0424e872e397e273851809c19d7ac5875c866736ff4250faffbae5157d8`. It averages all 1,472 **current nibabel-scaled** input volumes from scanner runs 57/58/60/61/64, using float64 accumulation and float32 storage. This is a spatial reference, not a new GLM or activation estimate. Its array was neither reversed nor spatially interpolated. Saved-array readback was exact.
- [SPM/current-input receipt](spm_bridge_input_audit.json): SHA256 `f43482ca6a68914e0b0cb60e3af5d4fbe331ae47505d34ba907351f40cf56ab8`. It binds SPM, five complete current input files and payload streams, cached/current read descriptors, every input volume index/offset, maps, source records, and mean output. All current scaled voxels were finite. All 1,472 cached SPM matrices have exactly the same zero-based geometry; the cached estimation geometry also matches con/T/mask geometry.
- [Bounded historical-reference and selected-source receipt](historical_reference_search.json): SHA256 `2ec32964ae75d79d159b75062c1645756028eb43cc2b2988931cd270ab153896`. It binds the historical candidates and verifies hashes, actual TR3 and actual 160/160/384/384/384 frame counts for the same five D-source scanner identities in the current rebuild. Scanner identity/count agreement does not mean the old smoothed arrays equal freshly aligned arrays.
- [Statistical mask-scope check](statistics_mask_scope.json): the actual estimation mask contains **68,267** positive voxels. Contrast and T are finite throughout this mask. Contrast has **226,645** nonfinite voxels, all outside it; T is finite everywhere. A future warp must explicitly handle outside-mask contrast values in an isolated work copy and keep mask eligibility separate.

At the final receipt snapshot, WSL PID **405**, PPID 317, elapsed **2,062 seconds**, remained live with command `/home/binbin/miniconda3/envs/deb_fmri_pipeline/bin/python -u alignment_review/cm033_Dsource_20261009/run_spatial_rebuild.py`. `spatial_rebuild_completed.json`, `full_series_qc/full_series_checks.json`, and `functional_normalization_qc/chain_checks.json` remained absent. The saved bounded log tail records ongoing AFNI alignment. This is a verified process snapshot, not an assumed wait based on a log. No process-control action was taken.

## Exact coordinate contract and its limits

SPM stores one-based matrices. Conversion here is `A_cached = M_SPM @ T(1,1,1)`, without changing array indices. The cached zero-based affine is:

```text
 0.75   0      0            -48
 0     -0.75   0             -5.022499084472656
 0      0     -1.9999997616   63.351417541503906
 0      0      0              1
```

Every current smoothed input instead has negative X and positive Z diagonal entries, with the same origin. Thus the explicit cached-world to current-world coordinate change, `B = A_current @ inverse(A_cached)`, is identical across all five:

```text
-1   0   0   -96
 0   1   0     0
 0   0  -1   126.70283508300781
 0   0   0     1
```

This equation maps world-coordinate descriptions of **the same array index**. It is not a physical laterality determination, an image registration, or an instruction to reverse an array. The prepared NIfTI uses cached SPM geometry so that a subsequently fitted bridge can be applied consistently to the saved con/T/estimation mask.

The first audit failed closed on an old/current scaling mismatch. Current files retain cached dimensions, datatype, byte offset, frame ordering and byte layout, but have intensity slope **1**, whereas cached SPM private.dat/pinfo slopes are nonunit (run57 `0.00027215422790475744`). The September7 v1.0.0 session receipt lists these exact inputs and a passed geometry gate. The actual historical `D:/multimodal_fmri/code/tools/preproc/nii_reorientation.py` implementation loads **scaled** arrays and writes replacement NIfTIs; it does not merely modify header bytes. Its session receipt does **not** record per-file voxel equality or before/after payload hashes. A separate smoke function checks first/last frames of small copied examples; this does not validate all five installed historical input payloads. Therefore cached scaling must never be reapplied to today's already-scaled stored floats, and original July estimation payload identity remains unproven.

The bounded search found an archived old-frame refmean and a saved SPM geometry gate linking it to run57/all21 same-grid images under `_archive/cm033_redo_20260725_115901`. However, its values differ from today's refmean (maximum absolute difference 1,228,345.375). All five archived smoothed first frames also differ from current values (maximum absolute differences 319,818.96875–340,194.125, correlations approximately 0.9977–0.9980). The `.nii.gz` archived refmean retains an oblique frame. No found candidate establishes exact current-estimation payload/reference correspondence. The archive is supporting historical evidence, not an interchangeable moving reference for this model.

## Concrete next integration step

The five selected local source records are MSTIM legacy runs **08/09/10/11/12** ↔ scanner **57/58/60/61/64**, all TR3, with the same actual frame counts used by the saved left-only model. Current D-source hashes were independently checked against the rebuild source manifest. The rebuild is spatial-only: unverified slice timing is explicitly unapplied, so it is not evidence that fresh GLM preprocessing is ready.

Preferred spatial bridge after rebuild completion:

1. Pin completion, full-series and all19 functional-chain QC, the final new refmean, EPI-to-taskA matrix, and frozen taskA input. The expected final common refmean is `C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/cm033_Dsource_20261009/bids/sub-cm033/ses-20160120/derivatives/preproc/anat/sub-cm033_ses-20160120_desc-refmean_epi.nii`. Verify its actual geometry and reference-scan39 lineage; do not treat a partially present file as final.
2. In a **fresh projectome derivative**, fit the prepared cached-SPM mean to that final common refmean with a separately saved and reviewed spatial transform. This is a same-contrast bridge between two preprocessing references. Bind both source images and all fit parameters. The new rebuild EPI-to-anatomy transform cannot replace this step: identical scanner membership does not establish identical alignment/motion/reference coordinates. Alternatively, independently fit the prepared mean directly to frozen taskA anatomy, with its own coarse image QC; this bypasses waiting for the common-reference bridge but retains the same historical-payload uncertainty.
3. Apply that fitted bridge to **saved** `maps/con_0001.nii`, `maps/spmT_0001.nii`, and **actual estimation** `maps/mask.nii`, whose current hashes/geometry are recorded in the first receipt. Contrast is finite inside the mask but nonfinite outside: create a separately hashed finite work copy with an explicit outside-mask fill policy before continuous interpolation, preserving every within-mask value and all source bytes. Use continuous-image interpolation for con/T and nearest-neighbor for the estimation mask; exclude unestimated support from regional summaries and document boundary interpolation. Do not substitute the separate requested-mask file for the actual estimation mask. Retain the stimulus-versus-implicit-baseline contrast and its original global-scaling units; it is not PSC.
4. For the common-reference route, use the checked final `desc-epiToAnatAffine.aff12.1D` to move the bridged outputs into the frozen taskA anatomy. The next fixed/native image is `anatomy_candidates/sub-cm033_srcdate-unknown_desc-taskAHeaderRepair_brain.nii`, not the old native norm moving image. Its computational center and physical laterality remain unaccepted.
5. Use the existing fresh anatomical forward chain from `norm_reference/derivatives/norm/sub-cm033/ses-20160120_ana`: ANTs arguments in order `NMT_1Warp`, `NMT_0GenericAffine`, `CMT_1Warp`, `CMT_0GenericAffine`, all with prefix `cm033_20160120_Dsrc_taskA_ana_`, reference `D:/OPTO_fMRI_CM/Templates/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_SS.nii.gz`. Independently validate transform hashes, coded grids, finite voxels, transformed estimation-mask support, and saved forward/inverse coarse overlays. The already reviewed broad anatomical correspondence supports candidate QC only.

The bridge fit and con/T/mask warp are still **unexecuted**. A provisional descriptive CM033 integration may retain the recorded current-payload/historical-identity limitation after a satisfactory independent bridge and coarse image review. Grid correspondence, positive Jacobians, and header axis codes do not establish physical laterality, accepted Ial borders, or statistical validity. Original fMOST transform unavailability remains a reported limitation; neither a network raw-data recovery nor a new corrected-reference acquisition is required for these available local spatial steps.

## Reproducibility

The two saved scripts refuse existing receipt outputs, and the first also refuses an existing mean output. They read external sources and write only this folder. To repeat, copy the scripts into a new derivative directory and run with the installed projectome Python; never rerun against these frozen output names:

```powershell
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -B audit_spm_bridge_inputs.py
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -B audit_reference_candidates.py
```

The first script's structural/volume/offset checks are strict; scaling mismatch is explicitly recorded while mean generation uses the current nibabel proxy's scaling. No SPM data interpretation is silently replaced.
