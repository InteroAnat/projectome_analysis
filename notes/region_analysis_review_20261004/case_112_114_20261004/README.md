# 251637 neurons 112 and 114: targeted laterality check

Agent: Codex. Updated: 2026-10-08 (Asia/Shanghai). Branch: `codex/fmost-methods-review-20261002`.

**Current hemisphere status: 112.swc and 114.swc are confirmed anatomical R by the user's direct NeuronView inspection.** The user reported on 2026-10-08: "confirmed, those two are indeed on the right, as I checked using neuronview". This resolves the two-neuron hemisphere question and establishes that their supplied manual L prefixes are incorrect. Codex records the user's inspection; it did not independently inspect these neurons in NeuronView.

**114's current working assignment is R-IDM, layer 3, provisionally selected by the user on 2026-10-08** ("make it IDM first,"). The original duplicate remains preserved for provenance and later anatomical validation. The original Henry annotations and historical isolated Unknown/Unclassified outputs are preserved. No data table, production code, source image or registration transform was changed by this confirmation. It does not establish a global acquisition orientation or fine parcel boundaries. The machine-readable provenance is [human_laterality_confirmation_20261008.json](human_laterality_confirmation_20261008.json); pre-edit note copies are in `confirmation_20261008_rollback/`.

## Current reviewed assignments (2026-10-08)

Agent: Codex. Date: 2026-10-08 (Asia/Shanghai).

| Neuron | Working label | Layer | Basis |
| --- | --- | --- | --- |
| 112.swc | R-IDM | 3 | User-confirmed R; original IDM retained |
| 114.swc | R-IDM | 3 | User-confirmed R; IDM provisionally selected by user |

[Assignment provenance](idm_assignment_20261008/reviewed_soma_assignments.json) preserves the original labels and the user's instruction. The existing analysis interface applied these assignments to the actual cached atlas-space graphs. [Saved readback](idm_assignment_20261008/assignment_readback.json) verifies both neurons as R-IDM / R / ITi with zero hemisphere conflicts; [fresh JSON results](idm_assignment_20261008/reviewed_neuron_results.json) contain recomputed hierarchy and laterality. The original Henry workbook and source SWCs are hash-unchanged. Existing workbooks and bulk images remain untouched. Note edits have exact pre-edit copies in `idm_assignment_20261008/note_rollback/`.

## Historical evidence before the human confirmation (2026-10-04)

Agent: Codex. Date: 2026-10-04 (Asia/Shanghai). Branch: `codex/fmost-methods-review-20261002`.

**The combined evidence favors R hemisphere for both neurons and localized errors in their manual L prefixes.** This is a working conclusion anchored to the supplied atlas and Henry's other annotations. The original annotations remain intact; absolute acquisition orientation and the actual registration transforms were not independently recovered.

| Neuron | Original Henry annotation | Fresh atlas root X, voxels | Official mask | Own native full-section soma |
| --- | --- | --- | --- | --- |
| 112.swc | L-IDM, layer 3; Excel row 113 | 211.5384 | R | Viewer-left, section 9118 |
| 114.swc | L-IDD5 and L-IDM, both layer 3; Excel rows 115/116 | 209.5244 | R | Viewer-left, section 8580 |

## What was checked

- Fresh server atlas SWCs are numerically identical to the cached atlas-space graphs. The separate FNT NMT copies also agree. Mirrored FNT comparison copies have different X coordinates and were not substituted into region lookup.
- Own native raw SWCs and atlas SWCs have identical node IDs, node types, radii and parent links: 7,474 nodes for 112 and 11,744 for 114. They are the corresponding graphs in different coordinate frames.
- All 296 annotated rows in the original Henry sheet were compared against actual cached atlas roots: 295 distinct neurons; 293 agree in hemisphere. Only 112 and 114 conflict (three conflicting rows because 114 occurs twice). No source SWC was missing.
- Nearby annotated 110, 111, 113 and 116 all have manual R labels and nearby R atlas roots. Same-sample native controls were actually inspected: 111 (Henry R-IDD5 / atlas R) is on viewer-left; 422 (Henry L-IDD5 / atlas L) is on viewer-right. Both 112/114 occupy the same displayed hemisphere as the R control.
- The lab Henry workbook on `X:/fMOST/251637/` is byte-identical to the repository copy. The contradictory prefixes and duplicate 114 therefore exist in the original supplied sheet, not just in a later table export.
- The original March bulk high-resolution and 8-mm wide-field plots for both neurons were found on the lab drive and actually inspected. Newly generated high-resolution, 4-mm wide-field and full-section views were also inspected, using the public toolkit plotting methods. The two regenerated soma NIfTIs match every voxel of the original March soma data. Header spacing/sign corrections remain a separate implementation difference.
- The current portal metadata does not supply injection coordinates or an independent orientation declaration. No source-provided raw-to-atlas transformation record was recovered in the checked paths.

## Original duplicate and provisional parcel decision

114 has two region entries, IDD5 and IDM. The duplicate handling in the existing Henry-layer merge is already documented: it prefers the leaf matching the harmonized label. That is a downstream selection policy, not a second anatomical assessment. There is no Henry row for 115, and the duplicate occurs between 113 and 116; a row-identity typo is possible but unproven. Do not silently move the second 114 annotation to 115.

Following the 2026-10-08 user confirmation, the reviewed hemisphere is **R for 112 and 114**. Keep 112's IDM/layer 3 as supplied. Following the subsequent user instruction on the same date, 114 uses IDM/layer 3 provisionally. Its original IDD5-versus-IDM duplicate remains a source ambiguity for later architecture review; it no longer blocks use of the user-selected working label. The October 4 isolated reruns still retain the original L labels with an explicit conflict and resolved side Unknown; they have not been regenerated and accepted workbooks have not been replaced.

## Evidence and reversibility

`source_readback.json` binds thirty source workbooks and bounded portal/raw/atlas requests. `annotation_and_identity_readback.json` preserves exact Excel row numbers, neighboring labels, duplicate rows and graph variants. `native_visual_provenance.json` binds current public-toolkit acquisitions and outputs. `visual_and_original_source_readback.json` records ten inspected original/generated images, complete original soma-voxel comparisons and unchanged workbooks. `same_sample_control_visual_readback.json` records two inspected controls with independently re-read Henry/atlas labels.

The first isolated helper invocation used the wrong export keyword and wrote no images; its failure ledger was retained. The helper was corrected to the existing `acquisition_metadata` interface and the successful rerun has zero producer errors. No production code, original images or annotations were changed during this targeted check. New evidence remains in this dated case folder.
