# CM033 updated deb_fmri source and integration audit

Agent: Codex specialist audit | Date: 2026-10-09 (Asia/Shanghai)

**Updated-source conclusion:** the September 26–29 CM033 mechanical/header repair is real and must be recognized. However, the actual **2026-10-09** source-linked audit still explicitly marks CM033 anatomical orientation and normalization as unaccepted. The configured native BIDS session and EPI-based normalization output are absent; the existing native GLMs are historical and have no saved NMT statistic maps in the checked model trees. No currently compatible, accepted CM033 NMT statistic was identified for the projectome integration. This conclusion is based on the updated checkout and today's receipts, not merely the earlier September rejection.

All external projects were read-only. This audit wrote only this note and [the compact evidence JSON](cm033_updated_debfmri_evidence.json). No registration/model fitting, preprocessing, source correction, publication or new cohort was performed. The user-confirmed unavailable fMOST transform is now an acknowledged limitation, **not a recovery gate**; it does not prevent declared template-space descriptive work. It is separate from the unresolved CM033 fMRI coordinate/registration lineage discussed here.

Evidence JSON SHA256: `658c537c9150844b553eca6c7267e8a81338fdc4e46c2ca9faecf4863e833849`. It records 90 exact source/code/receipt/map/figure bindings. Every binding was rehashed at the end with no source change. Its explicit path, Git, SPM and geometry records support the findings below.

## Which checkout is current

| Location | Actual readback | Appropriate interpretation |
|---|---|---|
| `D:/deb_fmri` | HEAD `175015f…`, recorded 2026-09-10 | Older code checkout. It is insufficient to assess later CM033 policy-handoff repairs. |
| `C:/Users/laika_yan/Documents/ChatGPT/deb_fmri` | Loose working tree on unborn `master`; current source/audit files exist, including the Oct9 CM033 audit | Current local working/audit context. There is no commit to identify all these loose bytes; use exact file hashes. |
| `C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/publish_checkout` | Clean tracked tree on `main`, HEAD and cached `origin/main` both `73544f327c0f03b3c70700939a517049dcc1e0e7` (2026-09-29) | Separate publication checkout. September source publication is not October real-data/anatomical acceptance. This audit checked local Git/cached remote refs, not a fresh network remote. |
| `D:/multimodal_fmri` | Data/results tree referenced by both current session YAMLs and the current audit | Its **results** remain relevant, but historical outputs and status flags must be checked against actual current configuration/source bindings. Updating code does not replace those images. |

The loose and clean checkouts are not interchangeable byte snapshots. For `fmri_pipeline/preproc/nii_reorientation.py`, their current raw SHA256 values are respectively `0216729a2f017812cb9471cb1969eeec2316d6af8527a43f6315463a38138071` and `1363a2a585081e681ff0c2db6d7d69e93f41f26510062091a911f5c750394861`. Their LF-normalized hashes also differ. The saved scan16/39 repair receipt records an earlier tested writer hash `16d9a1b884630caf928bb3114bfb70ece279403fbce87a4b7b605ef602adc2f3`. Its saved real-output validation remains useful, but it must not be described as an exact-hash replay of either current source file. No new recipe execution was inferred from source publication or helper renaming.

## Verified updated scan16/39 repair

Primary records are `alignment_review/cm033_overlay_recheck_20260928/RECHECK_REPORT.md`, its `recheck_manifest.json`, and `alignment_review/cm033_sform_handoff_repair_20260929/{verification.json,before.json}` under the C: working tree. The published [orientation-repair note](C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/alignment_review/publish_checkout/docs/CM033_ORIENTATION_REPAIR_20260926.md) also explicitly separates array correspondence from physical orientation.

The Sept28 copied-source replay used scan16 (TR6 s, 96 frames, native 19 slices) and scan39 (TR3 s, 384 frames, native 18 slices), with method-derived STC patterns `alt+z` and `alt+z2`. Explicit historical matched references enabled mixed-grid alignment to the 18-slice scan39 grid. Default volreg correctly rejected the mixed input grids, while the default fixed-sign path correctly rejected unknown origin. Candidate brain anatomy and candidate `[+1,-1,+1]` EPI signs were explicitly unvalidated. They do not establish the user's expected `[-1,-1,+1]` physical interpretation.

The Sept28 recipe's `run_spm_session(...matched_references...)` interface failure was subsequently repaired. The saved Sept29 handoff receipt passes; this audit independently reread all five installed image payloads against `before.json` and the complete 96-/384-frame SPM sidecars:

- Both aligned BOLD payloads, three anatomical/reference/mask payloads, datatype and intensity scaling match the pre-repair snapshots.
- All five affine differences are exactly zero.
- SPM sidecars are exactly 4 × 4 × 96 and 4 × 4 × 384, with maximum difference zero from the recorded SPM matrix across every frame.

This establishes a successful real-data header-policy handoff for those copied products. It does not establish anatomical left/right, sulcal/AIC correspondence, a full-session GLM replay, or a new accepted normalization. Historical voxel-flip agreement is evidence of array correspondence only.

## Today's CM033 status, independently bound

The controlling current record is [CM033_ALIGNMENT_NORMALIZATION_AUDIT_20261009.md](C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/docs/CM033_ALIGNMENT_NORMALIZATION_AUDIT_20261009.md), with `alignment_review/cm033_alignment_norm_audit_procdate-20261009/audit_manifest.json` and `run_coverage.tsv`.

| Current observation | Result and boundary |
|---|---|
| Saved included aligned runs | 19/19 present and on the refmean grid; 19/19 SPM sidecars agree with their NIfTI geometry in the saved all-run audit. This is surviving-derivative coverage, not fresh preprocessing. |
| Numeric TR | 15/19 match the curated scanner table. Scans16/17/18/23 still store TR1 instead of6 in the historical production copies. |
| Explicit mm/sec units | 13/19; scans16/17/18/23/45/46 lack explicit units in production. |
| Isolated timing fixes | Six copies exist, preserving inherited spatial registration. This audit verified all six original/corrected hashes against the timing manifest. No corrected copy was installed into production or treated as a new fitted model. |
| Configured raw/native BIDS session | `D:/multimodal_fmri/sub-cm033/ses-20160120` is absent; the current audit records 0/19 native inputs at configured paths. Preserved isolated probes exist, so this does not mean every original is lost. |
| Configured normalization | Current JSON requests `epi_based`, `derivatives/norm/sub-cm033/ses-20160120_epi`; that directory is absent. |
| Surviving normalization | `ses-20160120_ana` exists with old CMT/NMT products, but the current audit found no input/transform provenance manifest binding them to the current moving image. Presence/convergence does not certify that chain. |
| Current acceptance | `UNACCEPTED: trusted physical orientation and corrected-source identity pending`. All nine core sources bound by the Oct9 audit still match its recorded hashes in this fresh readback. |

The current audit also rechecks the allegedly corrected historical FLASH file: voxel spacing is about 0.75 × 0.75 × 0.7547 mm, while its coded sform diagonal is `(128,-128,53)` mm and qform code is zero. It cannot supply a valid physical reference unchanged. A matching corrected reference or an independently trusted scanner-coordinate/landmark derivation remains missing. No origin/sign was borrowed from CM032.

## Direct inspection of current figures

I opened the actual Oct9 scan16 and scan39 brain-boundary PNGs and the Oct9 NMT historical checkerboard. Their exact hashes and source JSON/NIfTI paths are recorded under `Oct9_figures_inspected` in the evidence JSON:

| Oct9 PNG | SHA256 | Direct observation |
|---|---|---|
| `figures/sub-cm033_ses-20160120_run-16_desc-brainBoundarySampled_qc.png` | `fef6cf2b7a4fada7c9d669e64cc3b0b90c0d097f132d593214647fa1d63edd76` | Central axial brain contours broadly track measured EPI support. Inferior axial and peripheral coronal/sagittal contours have holes or discrepancies. Explicitly labelled unaccepted/inherited registration. |
| `figures/sub-cm033_ses-20160120_run-39_desc-brainBoundarySampled_qc.png` | `e7c92db98f4960df8c1bf6fa627443f58812151613ff4272806cb6c74b90ad53` | Similar central support and inferior/peripheral limitations. The upper title is visibly clipped, a readability issue rather than a numerical defect. |
| `figures/sub-cm033_ses-20160120_space-NMT_desc-historicalNormCheckerboard_qc.png` | `ae6bc9d30f35ad34bd44a5db05f5a2ea4234626dd7244e19cae1451f4d2f789f` | Outer support broadly overlaps the template, while internal structures and tissue contrast visibly differ. Its header explicitly says historical normalization and unvalidated physical LR. |

The scan plates average only five evenly spaced frames; they do not establish full-series motion or distortion QC. Projected anatomical boundaries are not independently measured EPI tissue boundaries. Neither these pictures nor Dice ≈0.95 on positive NMT support establishes sulcal/AIC precision or physical laterality. The Sept28 matched-header run39 plate was also opened as historical comparison, without replacing the Oct9 observations.

## Actual model/contrast/site inventory

All 14 checked production GLM directories have native contrast/T/mask files and their corresponding configs. No NMT statistic map was found in these model trees; `NMT_on_ana.nii` is a display overlay, not a warped GLM statistic. The evidence JSON retains the full 14-model inventory. Four actual `SPM.mat` files were read directly; their saved input paths confirm the run IDs below, not merely configuration intent.

| Native model under `D:/multimodal_fmri/derivatives/glm/sub-cm033/ses-20160120/` | Actual fitted stimulus contrast | Locked site grouping | Integration use |
|---|---|---|---|
| `glm-mstim-pos1-short-tr3-smooth2mm` | TR 3 s; scans 57/58/60/61/64, five stimulus coefficients of 0.2; stimulus > implicit baseline | Left AIC only | Best existing left-only native MSTIM candidate for a future independently bound spatial chain. `maps/con_0001.nii`, `maps/spmT_0001.nii`, `maps/mask.nii`, root `SPM.mat` and `contrasts.json` exist. Not currently atlas-ready. |
| `glm-mstim-tr2-smooth2mm` | TR 2 s; scan 25, coefficient 1 | Right AIC only | Separate native right-site acquisition/model; do not average indiscriminately with TR6/left-site models. |
| `glm-mstim-tr6-smooth2mm` | TR 6 s; scan 23, coefficient 1 | Right AIC only | Separate native right-site model on a 19-slice grid. Historical TR metadata needs reconciliation. |
| `glm-mstim-all-tr3-smooth2mm` | TR 3 s; scans 27/28/53/54/55/57/58, seven coefficients of 1/7 | **Right 27/28 plus left 53/54/55/57/58** | Already pooled across sites. It cannot be labelled a single right- or left-site response or retrospectively split into two fitted contrasts from `con_0001` alone. |

The `-tapas` variants are distinct existing fits, not extra animals. OPTO models are separate from MSTIM: right scans 16/17/18 at TR 6 s, left scans 39/47 at TR 3 s and left scans 45/46 at TR 1 s. The locked ontology is `D:/multimodal_fmri/docs/provenance/stim_sites_cm033.yaml`: scans before 39 are right AIC; 39 onward are left AIC. `acq-pos1` alone must not define hemisphere, and CM032/CM033 must not be pooled into one first-level GLM.

An additional direct compatibility problem prevents silent use of old transforms. The existing 18-slice MSTIM contrast affine has diagonal approximately **(+0.75,-0.75,-2)** and origin **(-48,-5.0225,63.3514)**. Today's production refmean/brain-in-EPI uses approximately **(-0.75,-0.75,+2)** with that same origin. Matching shape/spacing is therefore insufficient: these are different physical grids. No voxel flip, header rewrite or transform substitution was guessed. The actual model-affine records are in `selected_actual_SPM_metadata` in the evidence JSON.

## Reproducible next steps and current-versus-stale reconciliation

The appropriate present action is **retain CM033 as unassessed for spatial projectome comparison**, while recognizing the completed updated software/mechanical repairs and preserving its useful native models. There is no accepted ready-to-run CM033 NMT input pair to add by simply changing a filename in the CM032 integration command.

1. Use the explicit clean publication revision or a deliberately hash-pinned current loose source snapshot; do not identify current code as `D:/deb_fmri` or conflate the two C: trees.
2. Resolve the CM033 fMRI reference/physical-grid lineage with a trusted corrected same-acquisition reference or independently confirmed scanner-to-NIfTI landmarks. Reconcile the existing model affine against its exact fitted input/moving images. This is a CM033 lineage requirement, not a request to recover the unavailable fMOST transform.
3. Locate the intended native source tree or explicitly select a documented bounded cohort from surviving inputs, keeping task, site and TR groups distinct. The existing two-/three-run probes do not establish all19 production inputs.
4. Once a legitimate source/transform chain is available, reuse appropriate existing transforms only if exact native, moving/reference, transform order, interpolation and hashes are proven. If not, any new fitting belongs to a separately authorized fresh derivative workflow, not this audit.
5. For each eligible site-specific model, warp its signed `con_0001` with Linear interpolation and actual SPM `mask.nii` with NearestNeighbor into the pinned projectome NMT reference, writing a fresh directory and complete source/transform/coverage receipt. Compare whole saved volumes/grid where a historical counterpart exists; do not derive coverage from zero-valued statistics.
6. Integrate one explicitly selected site-specific contrast using all six **separate actual ARM** levels, retaining native sampling/contrast units and unpaired-animal/site uncertainty. Reuse the current CM032 regional readback/coverage principles; do not treat interpolated voxels as independent replicates or regional T medians as new t statistics.

Earlier blanket statements that CM033 lacked a working header repair are superseded by the updated scan16/39 handoff evidence. Earlier statements implying historical sforms establish accepted orientation are also incorrect. Today's explicit unaccepted-reference/no-current-EPI-normalization status remains supported by fresh paths, hashes and visual inspection. All native models/old transforms remain preserved; no claim of accepted anatomy, stimulation-site location, cross-animal identity or causal association is made.
