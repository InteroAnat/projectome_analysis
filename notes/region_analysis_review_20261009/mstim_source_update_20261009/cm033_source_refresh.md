# CM033 source-scope refresh — 2026-10-09

**Local CM033 sources are available. No new accepted normalization, anatomical orientation, or site-specific model repair is demonstrated.** This additive, time-bounded read-only refresh supersedes the earlier input-prerequisite wording while preserving the [original audit](cm033_updated_debfmri_audit.md) and its [90-binding historical receipt](cm033_updated_debfmri_evidence.json).

The [refresh evidence JSON](cm033_source_refresh.json) records exact previous/current hashes in `changed_bindings`: exactly three external documentation/manifest files changed; the other 87 original bindings, including audited images, code, configurations, figures, SPM models and contrasts, match their historical hashes.

## What changed

The current [dated source audit](C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/docs/CM033_ALIGNMENT_NORMALIZATION_AUDIT_20261009.md:9) now explicitly designates `D:/OPTO_fMRI_CM` as the intended input. Its lines 9–11, 67 and 77–84 remove the old Z:/Bruker-original route and mandatory separately corrected NIfTI as prerequisites. A corrected same-scan image is optional corroboration. Codex should complete source mapping, geometry and reproducibility checks using available local records; Binbin reviews the resulting anatomical figures. The [project note](C:/Users/laika_yan/Documents/ChatGPT/deb_fmri/docs/project_note.md:1486) records this clarification, not a new imaging fit or scientific acceptance.

| Changed record under the updated C: checkout | Previous SHA256 | Current SHA256 |
|---|---|---|
| alignment_review/cm033_alignment_norm_audit_procdate-20261009/audit_manifest.json | b2c5b17d37d356b92e204459b5ea47e1e2f01bad214c8e5bbf0935363c572dd9 | 7127132e55ed07adcd9bb5104345c49593810651a1b78a4e91496f14e856291c |
| docs/CM033_ALIGNMENT_NORMALIZATION_AUDIT_20261009.md | 5bdb9cea24d6064b24a785942b06f5684c89e0d8209e099558afce6d528047e6 | 0191ff4aa18c015854de938e973a9a630b862653b15155d6d4d93b54f62b949b |
| docs/project_note.md | 506572563438976d98275ae4e3a9f6df909a26a71ee026ba68e9e314e65e1382 | 9acf7eac1bdefbbb610faa3d2f27e3f81d17cb6823fb07ed599c9938be92109e |

## Verified actual sources

Direct inventory of `D:/OPTO_fMRI_CM/BIDS_data/CM033_BIDS/sub-CM033/func` finds **36 plain unprefixed EPI NIfTIs: 29 MSTIM and 7 OPTO**, with 36 matching JSON files. Headers were opened for all 36; the receipt binds each binary header fingerprint and full JSON hash. This inventory does not claim full-payload hashes for all 36. Two representative functional NIfTIs and the malformed corrected-brain image have full-file hashes in the receipt. The anatomical folder contains 66 NIfTIs, including multiple processing variants; this is an inventory count, not 66 independent acquisitions.

Available records were actually opened and hashed:

- [Conversion CSV](D:/OPTO_fMRI_CM/cm033.csv): scanner/reconstruction identifiers exist, but task fields are blank. It does not independently assign legacy task-specific run names to scanner IDs.
- [CM033 conversion workbook](<D:/OPTO_fMRI_CM/Info_tables&sheets/CM033.xlsx>): explicitly distinguishes reconstruction 1/poorEPI and reconstruction 2/functional EPI, with OPTO/MSTIM task labels. Selected exact row values are retained in the receipt.
- [Acquisition workbook](<D:/OPTO_fMRI_CM/Info_tables&sheets/opto-fMRI_sessions.xlsx>): the `cm033.zJ1` sheet records 2016-01-20, scanner IDs, segmentation, TR, frame counts and stimulation comments. For example, scanner 16 has SEG=8, TR=750 ms and NR=96; the first legacy OPTO NIfTI has 96 frames and volume TR=6 s. This is useful corroboration, not sufficient identity proof by itself. The workbook's scanner-39/47 row explicitly states left AIC; image laterality must still be checked independently.

No `method`, `acqp`, `visu_pars` or `2dseq` files were found under the designated D: tree. Their absence does not make the local converted images unavailable or justify requiring another dataset. Legacy run numbering must be mapped with metadata/content checks; this refresh does not infer that mapping from filenames alone.

## What remains unchanged

The current manifest acceptance is **“UNACCEPTED: local source mapping, image geometry and anatomical QC pending.”** Its scope still says no new registration or normalization fit. The configured `D:/multimodal_fmri/sub-cm033/ses-20160120` and requested `ses-20160120_epi` normalization directory remain absent on direct path checks. Historical `ses-20160120_ana` products exist. Thus the old configured `0/19` statistic describes obsolete configured paths; it must not be reported as no local CM033 data.

The four sampled current native MSTIM contrast files remain on an affine with diagonal approximately `(+0.75,-0.75,-2)` mm, while the production refmean uses `(-0.75,-0.75,+2)` mm at the same origin. Independent current header comparison rejects grid identity for all four. A shape match does not resolve this coordinate mismatch; historical anatomy transforms must not simply be attached to these contrasts. The old malformed corrected-brain file still has sform diagonal `(128,-128,53)` mm despite submillimetre pixdims. Its name does not establish valid geometry, and its failure does not make a separately corrected file mandatory.

The bound SPM/model/configuration sources are unchanged. The `all-tr3` contrast still mixes right-AIC runs 27/28 with left-AIC runs 53/54/55/57/58. The left-only short-TR3 candidate still uses runs 57/58/60/61/64 and remains native-only, without a verified source-to-NMT chain. No eligible accepted CM033 NMT statistic map is identified by this refresh. Real scan16/39 mechanical header synchronization remains distinct from anatomical acceptance.

The next available technical step is explicit local-image-to-scanner/reconstruction mapping, then source geometry/timing checks and isolated pipeline reconstruction with fresh provenance before anatomical figure review. This refresh performed no fitting, copied no external sources, and modified no external files. The unavailable fMOST transform remains an acknowledged limitation, not a recovery gate. Concurrent future changes in the external checkout require another dated additive refresh rather than retroactively changing this historical snapshot.
