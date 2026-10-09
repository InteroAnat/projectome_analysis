# CM033 direct bridge candidate: rejected

Codex ran one isolated public-DEB rigid fit of the prepared current-payload mean in cached-SPM coordinates to the frozen task-A anatomical candidate. The fit completed as software. Its independently calculated rotation was **33.3054408 degrees**, exceeding the unchanged **30-degree review gate**. Codex and the root reviewer inspected the forward and inverse panels and found broad contour/slab mismatch. This candidate is rejected for propagation to contrast, T or model-mask images. No second fit has been started.

The primary readable panels are [mean on anatomy](coarse_qc/readable/cm033_cachedSPM_mean_on_taskA_candidate.png) and [anatomical support on mean](coarse_qc/readable/cm033_taskA_support_on_cachedSPM_mean_candidate.png). Original images, matrices, logs, producer provenance and the earlier display variants remain preserved. The support contour is derived from positive values in the frozen brain candidate; it is display support, not an accepted tissue segmentation. Display directions follow image headers and do not establish physical laterality.

[Saved readback](candidate_saved_readback.json) confirms finite forward/inverse images, exact requested grids, binary nearest-neighbour inverse support and near-identity matrix/inverse multiplication (maximum absolute error 2.28e-5 from text precision). These checks do not rescue the rejected registration. [Rejected-candidate disposition](rejected_candidate_review.json) binds the actual readable panels and essential artifacts without changing the original fit receipt.

The public method is `init_epi_anat_wf` in `fmri_pipeline/workflows/nipype_anat_epi.py`, lines 95–135: AFNI `shift_rotate`, `-cost lpa`, centre-of-mass initialization and `wsinc5` interpolation. The public rotation validator is in `fmri_pipeline/preproc/run_anat_epi_bids.py`, lines 172–190. The complete command and actual AFNI output are preserved in [fit_execution_log.txt](fit_execution_log.txt). AFNI's saved matrix maps **output/base DICOM coordinates to input/source DICOM coordinates**, as documented by the pinned local [AFNI help](afni_3dAllineate_help.txt), lines 228–242 and 716. The filename `epiToAnatAffine` describes image-resampling use; it must not be interpreted as a forward source-point matrix. No ITK conversion or statistics warp was performed.

# Explicit coordinate initialization: evidence only

[coordinate_initialization_evidence.json](coordinate_initialization_evidence.json) freshly rechecked all five model-input hashes and actual coded-mm affines. Their current coordinate description is the same across runs 57, 58, 60, 61 and 64. For the unchanged array index, the explicit cached-world to current-world coordinate recoding is:

```text
B = A_current @ inverse(A_cached)
  = [-1  0  0  -96
      0  1  0    0
      0  0 -1  126.70283508300781
      0  0  0    1]
```

This is a coordinate recoding, with no array reversal or interpolation. It is supported by the actual current inputs and cached-SPM coordinate matrices. It does **not** prove physical orientation or original estimation-payload identity. Cached SPM slopes differ from the current scaled float payloads; the prepared mean represents the current payload, not a recovered July estimation payload. A later fit cannot resolve that historical identity question.

The actual new installed reference mean remains finite, shape 128 × 128 × 18, with unchanged SHA256 `0d610e2c7e35723cbe55044d7a2f9161260cb0eb2cb6c5bce82d09c2ab887877`. Its zero-based affine is:

```text
[-0.7500000596  0             0  46.7922134399
  0             0.7500000596  0 -43.1798744202
  0             0             2 -12.2274446487
  0             0             0   1]
```

It is the declared scan-39 reference, not the mean of the five historical model inputs. Its grid differs from their current grid; attaching this affine to old arrays would invent correspondence. The current-input header axis codes are LPS, while this reference has LAS header codes. Such header facts are not independent physical orientation evidence. A separately fitted bridge and actual image inspection would still be necessary.

The external `spatial_rebuild_completed.json` now exists. It records recovery of the saved public SPM job through Windows, 19 installed runs, the same post-SPM gate and candidate anatomical status. The live scan-39 reference agrees with the SPM one-based matrix after the explicit index-origin conversion. Full-series and functional-chain QC receipts also now exist and are hash-bound in the initialization evidence. This supersedes the earlier snapshot in which WSL failed to launch Windows MATLAB. The new marker still declares that slice timing was not reconstructed and these spatial candidates are not GLM-ready.

The configured orientation policy is `signs: null, origin_mm: null`. Public `orientation_policy.py`, lines 31–43 and 67–101, preserves unambiguous cardinal signs through explicit working-copy preparation; it does not independently identify physical left/right. A proposed next route would use a separately saved, explicitly B-recoded working copy of the current-payload mean, then a fresh same-contrast fit to the actual installed reference with the unchanged public rotation gate. Any approved downstream propagation must retain B and the fitted bridge as separate named steps with correct pullback composition. **No such working copy or second fit has been created here.**

Reproduce the read-only geometry check in a fresh output directory using [audit_coordinate_initialization.py](audit_coordinate_initialization.py). The script fails if its receipt already exists, never edits external inputs and does not run registration. The actual-fit driver likewise refuses an existing fit folder; the rejected candidate must remain immutable.
