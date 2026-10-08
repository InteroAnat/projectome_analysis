# The 52 selected neurons with ARM background soma locations

The primary result is [selected_52_background_cases_with_case_overlays.csv](selected_52_background_cases_with_case_overlays.csv): all 52 existing selected neurons whose own NMT-space SWC roots land on ARM level6 index0 under the current coordinate policy. The ledger preserves exact sample/channel, neuron filename/UID, animal, hemisphere mask, original labels, evidence category and source hashes. No selection, map or anatomical label changed.

| Existing evidence | Neurons | Meaning |
|---|---:|---|
| Henry coarse INS visual review | 13 | Human coarse-INS evidence from confirmed historical wide-field checks; retain this decision despite ARM0. |
| Main folded-box candidates | 13 | Spatial candidates from the original251637 folded screen, without new anatomical acceptance. |
| Preview origin-sensitive candidates | 15 | Alternative coordinate-origin ARM lookup reaches INS; sensitivity only. |
| Preview nearby candidates | 11 | Near another established same-sample INS anchor; retrieval proximity only. |

The first13 are sample251637 neurons112,114,422,423,438,439,440,444,458,459,460,461,469. Their original Henry fine annotations and workbook rows are retained verbatim, including both conflicting114 rows. Coarse visual evidence does not prove the fine parcel. Later hash-bound review retains112 as R-IDM and114 as provisional R-IDM; original L-prefixed Henry labels remain separately visible. The other39 remain potential cases. Original portal region metadata is missing for all52, rather than an explicit INS assignment.

The52 comprise14 sample251637/animal936,15 sample252383/animal605,8 sample252384/animal948,3 sample252385/animal331,1 sample252718/animal900 and11 sample252790/animal797;33 mask-left and19 mask-right.

## Tissue and coordinate-origin sensitivity

ARM background means no assigned ARM parcel at that voxel. Tissue was measured separately from the pinned NMT segmentation, using its embedded official class table, not inferred from ARM0.

| Tissue policy | GM | Subcortical GM | WM | CSF |
|---|---:|---:|---:|---:|
| Current `np.rint(XYZ/250)` | 0 | 0 | 49 | 3 |
| Literal edge-origin sensitivity `ceil(XYZ/250)-1` | 31 | 2 | 19 | 0 |

Thirty roots change from WM to GM/scGM and three from CSF to GM;19 remain WM. Under edge-origin sensitivity27 obtain INS labels, two claustrum, one secondary somatosensory cortex, two gustatory cortex, one parietal operculum, and19 remain background. These are separate measurements on the SAME pinned ARM6, not accepted replacements. Henry neurons440,459,469 remain WM/background in both policies; their human coarse-INS evidence is retained, and the disagreement remains unresolved.

Half-open `floor(XYZ/250+0.5)` changes one selected voxel but zero labels relative to the current policy. This is an observed result for these52, not generic equivalence of rounding policies. XYZ is declared voxel-index-encoded micrometres; divide by250 before lookup, without inverse-applying the world affine. The pinned NMTv2.1 symmetric reference is256×312×200 at0.25mm; all reference/ARM/key/mask/segmentation hashes and affine are in the receipts. Coordinate origin and original raw-fMOST-to-NMT transform provenance remain unresolved. Neither origin sensitivity nor these tissue values prove registration error or native WM anatomy.

## Source and visual evidence

All52 selected atlas SWC hashes were freshly reverified and roots reread. Saved nearest-OTHER-anchor distances were independently recomputed using stored portal NMT-index coordinates times0.25mm, with exact same-sample/channel identities, excluding self. Only original portal INS or exact Henry coarse-INS annotations qualify as anchors; alternative-origin candidates do not. Numeric-neighbor IDs remain retrieval cues only. Prior full graph-equivalence evidence is reused after hash revalidation; it is software evidence, not anatomical acceptance.

The immutable primary8746-row audit predates a recorded native-case fallback. Therefore [selected_52_background_cases.csv](selected_52_background_cases.csv) preserves its original fields, and the recommended ledger adds explicit `Effective*`, `NativeCaseOverlay*` and `AdditionalCaseVisualAssets` columns. Existing hash-bound overlays supply native SWCs for251637/112,114,422: effective native/atlas pairs are41/52, with all41 native hashes freshly verified. This does not claim that native and atlas XYZ should match.

Existing visual assets are recorded for40/52 cases:38 bulk-manifest cases plus112/114's reviewed case assets. Twenty-seven have a recorded derived context status (three derived_highres,24 derived_partial);25 lack that bulk context-status field. The latter includes112/114, whose separately linked full native locators, wide-field and soma views are preserved. These coverage fields are not quality acceptance. Asset links/hashes and prior observations are carried from bound records; no images were opened or downloaded in this audit, and historical X: paths were not reprobed. CH1 fluorescence is not cytoarchitecture. See the linked [prior review protocol](../coarse_insula_review_20261009/review_protocol_20261009.md) for context limits.

## Supplementary wider scope

The earlier broad request is retained only as supplementary saved-data counts and queue. The complete8746 identity audit has1912 available own roots:1617 positive ARM6 locations and295 ARM0. Of these295,52 are selected and243 unselected. The other6834 identities lack local atlas SWCs and remain unassessed, not background;337 lack portal coordinates.

[sample_coverage.csv](sample_coverage.csv) gives all47 exact sample/channel denominators. [supplementary_unselected_background_queue.csv](supplementary_unselected_background_queue.csv) carries all243 unselected background rows and their existing source/evidence/distance fields. No unselected background row has Henry coarse-INS or original portal INS evidence. Seven are within2mm of another established same-sample INS anchor, all sample250432 with no animal mapping. All140 registry-mapped unselected background roots are beyond2mm. Eighty-four sample251730 rows have coordinates but no same-sample established INS anchor, so distance remains unknown. Broader rows reuse the prior hash-bound lookup; this audit's fresh SWC work is limited to the selected52. No background neuron was automatically proposed for maps.

The original folded screen uses q0.005–q0.995 bounds from251637, folded X=abs(X_mm−32), NII bounds×0.25mm and2mm padding. Flags are carried from their bound provenance; these are candidate screens, not anatomical boundaries.

## Reproduction and verification

Run `python -B audit_selected_background.py` then `python -B apply_existing_case_evidence.py` in a fresh copy at the same project-relative depth. Both refuse output overwrite. Source root checks are in [source_root_readback.csv](source_root_readback.csv); [independent_receipt.json](independent_receipt.json) binds the fresh roots, prior inputs and first saved outputs; [delivery_receipt_v2.json](delivery_receipt_v2.json) binds the additive case overlay. [final_independent_saved_readback.json](final_independent_saved_readback.json) independently verifies exact saved identities, unchanged original fields and output/source hashes. No code or tests outside this audit folder were added. No network, canonical/source writes, map recomputation, or new scientific acceptance occurred.
