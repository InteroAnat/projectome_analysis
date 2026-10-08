# Coarse insula review: native evidence and warped soma

2026-10-09. The user clarified that Henry's labels came from visual checks of wide-field bulk outputs, and prioritizes coarse INS membership and unresolved Unknown/PrCO-like neurons near INS. This protocol proposes review records only. No coordinates, source annotations, canonical cohort, transforms or endpoint maps were changed.

## Existing human evidence

Fresh read-only inspection of [the Henry workbook](../../../../R_analysis/tables/somainfo_Henry_2026.04.03.xlsx), sheet `251637 mostly insula`, found **296 annotated rows / 295 distinct neurons**, with columns `Folder, SWC, Area, Layer`. IAL, IDD5, IDM, IDV and IAPM account for **261 rows / 260 distinct neurons**, including both original 114 entries. These are source human INS annotations; the user's current clarification establishes their wide-field review basis. The other 35 neurons have F1, 24c or 3b annotations; the sheet name does not make them INS.

Among the 261 INS annotation rows, `Folder` contains 212 original INS-folder rows, **35 `Region_CR_PrCO` rows**, and **14 `Region_Unknown_0` rows**. Excel row 4, for example, is neuron 003, folder `Region_CR_PrCO`, human Area `R-IAL`; rows 5–7 similarly annotate 004–006 as R-IAL. Old folder/automatic PrCO or Unknown labels therefore cannot override existing human INS determinations. Preserve both channels. Do not count 261 rows as 261 neurons.

Workbook SHA256: `2665d1f3cdd3fc19e0204a8dcc5bb45199c2f4db18d08e39273de4bf789bcae6`. The lab source is recorded as `X:/fMOST/251637/Area and Layers by Henry_2026.04.03.xlsx`; the October 4 readback established byte identity with the repository copy. This protocol did not reread that lab path.

**Existing identity-linked human wide-field evidence can support coarse INS acceptance without requiring every fine parcel or cortical layer to be independently established.** Preserve fine labels but record their status separately. For 112/114, the [reviewed decisions](../../../../notes/region_analysis_review_20261004/case_112_114_20261004/idm_assignment_20261008/reviewed_soma_assignments.json) and [user NeuronView confirmation](../../../../notes/region_analysis_review_20261004/case_112_114_20261004/human_laterality_confirmation_20261008.json) establish R hemisphere; 112 retains IDM and 114 uses user-selected provisional IDM. Coarse INS need not wait for resolution of 114's IDD5/IDM duplicate. Do not move that duplicate to 115.

Fresh readback of `group_analysis/visual_review_20261002/tables/correction_table.csv` found all 832 rows' `human_region`, `human_subregion`, `human_layer`, `reviewer`, `review_date` and `human_notes` blank. Its populated `review_category` is a machine selection field. Verified source pixels and available figures do not establish completed human anatomical review of those additional candidates.

## Accessible evidence and limitations

The [112/114 case record](../../../../notes/region_analysis_review_20261004/case_112_114_20261004/README.md) binds Excel rows, corresponding native/atlas graphs, reviewed original/regenerated images, controls and unchanged workbooks. `visual_and_original_source_readback.json` records actual historical inspection of the March **8-mm** wide-field and soma plots; `native_visual_provenance.json` binds regenerated **4-mm** context, soma views and source paths. Original soma NIfTI pixels matched regenerated data. These are recorded historical checks, not a new visual inspection in this protocol.

Exact historical originals include `X:/fMOST/251637/cube_data_251637_other_regions_20260320/Region_Unknown_0/LowRes/Plots/251637_112.swc_Unknown_0_WideField_Plot.png` and the corresponding 114 file. Related `HighRes/Plots/*SomaBlock_Plot.png` and `HighRes/Data/*SomaBlock.nii.gz` are recorded. Current lab-path accessibility is not asserted. The workbook's Folder is a retrieval cue; verify the actual linked file rather than manufacture paths from labels.

Workspace native pairs are under `notes/region_analysis_review_20261004/case_112_114_20261004/neuron-{112,114}/`, including `native_full_section_soma_locator.png`. Controls 111 (Henry R) and 422 (Henry L) have recorded same-sample native views. Original filename L prefixes remain historical labels; the explicit human R decision governs reviewed laterality.

Additional bulk assets are under `group_analysis/visual_review_20261002/{SampleID}/{NeuronID_without_suffix}/`: `soma.nii.gz`, `context.nii.gz`, sidecars, provenance and review figures where available. Own native SWCs are under its `swc_raw/{SampleID}/{NeuronID}` tree. [Native-style examples](../../../visual_review_native_style_20261008/index.html) retain the preferred grayscale high-resolution soma and green wide-field MIP with traces. `DerivedContext` is assembled from local high-resolution tiles; it is not a copied whole-section overview.

CH1 fluorescence is not established as a PI/cytoarchitecture channel. Correctly linked images can support expert coarse localization, but do not automatically establish layers, microscopic parcel borders or a precise GM/WM boundary. Keep missing/blank tiles, partial coverage and seams explicit. **252384/003's central section 12464 is unsuitable for placement review**: source strip discontinuities were independently verified. A reassuring MIP cannot repair them; use intact adjacent sections or another trustworthy context for a new decision.

## Paired native-versus-warped check

1. **Establish identity.** Bind sample/channel/NeuronID and hashes of the own native SWC, native image/crop and actual atlas SWC. Compare node IDs, parents and types; do not match only nearest coordinates. Locate the same unique root and, where available, the observed soma node in both frames. Exclude mirrored FNT comparison copies from anatomical lookup. The 112/114 correspondence evidence is case-specific.
2. **Inspect native anatomy first.** Confirm that the high-resolution root marker corresponds to the labeled soma rather than an axon, neighboring cell or gap. On wide-field and individual central/adjacent sections, judge its relation to the insular cortex, surrounding opercula/sulci and white matter using adequate field of view and same-sample reviewed controls. Record `INS`, `adjacent_non_INS`, `INS_boundary_uncertain`, or `unassessable`, with reviewer/date, image anchor and short reason. Fine parcel/layer may remain unresolved. Record screen side separately when absolute anatomical laterality is unavailable.
3. **Check crop correspondence.** Use saved native origin, XYZ spacing and axis order: `local_index=(native_xyz-origin_xyz)/spacing_xyz`. Confirm the soma and proximal edges align with signal across sections. `main_scripts/render_cached_toolkit_pair.py:27–75` checks bound raw hash, root equality and crop containment; it does not validate tissue stitching or calibration. A marker rendered with the same wrong origin can look internally consistent. Inspect unmarked sections and surrounding structures, not only overlays. Misaligned/discontinuous overview evidence is unassessable, not evidence of an ectopic cell.
4. **Inspect the warped position separately.** Show the original atlas root on the exact hashed NMT/atlas/tissue images with explicit voxel convention; retain current lookup and origin sensitivity. Native XYZ and NMT XYZ cannot be subtracted directly. Matching graph identities establishes correspondence, not transform accuracy. Without the original deformation/export lineage, record anatomical disagreement such as `native_INS_atlas_WM_discordance`; do not report a numerical registration residual or claim a reproduced warp.
5. **Discriminate explanations with controls.** A uniform 125 µm-per-axis shift under the alternative corner-origin policy indicates index sensitivity, not a demonstrated repair. Nonuniform mismatch of neighboring cells and anatomical landmarks supports registration displacement once image correspondence is verified. Native trace/image mismatch instead implicates crop/export/identity handling. A soma genuinely outside the cortical ribbon on trustworthy serial native sections may be a white-matter neuron. A soma clearly in adjacent cortex should remain adjacent cortex. If evidence cannot distinguish these explanations, retain them as unresolved rather than force INS.

The [origin sensitivity addendum](../../atlas_locations/soma_audit_20261009/coordinate_origin_uncertainty_20261009.md) compares the current client's `rint(XYZ/250)`, the author's `ceil(XYZ/250)-1` and half-open centre cells. These are diagnostic policies, not adopted corrections. Published code transforms images and SWC points separately and binds NMTv2.0 CHARM/SARM; current portal-to-NMTv2.1 ARM lineage remains unbound. See the [full Methods review](../../references/fmost_full_methods_20261009.md). Do not rewarp or shift data simply to improve agreement with atlas labels.

White-matter somata are biologically possible: [Mortazavi et al. 2016](https://doi.org/10.3389/fnana.2016.00015) studied NeuN-positive WM neurons in adult/aged rhesus monkeys; [Swiegers et al. 2021](https://doi.org/10.1002/cne.25216) documented interstitial neurons in crested macaque WM. These primary studies rule out automatic rejection solely from a warped WM label; they do not identify any present INS neuron as a WM neuron. Native tissue evidence is needed. A reviewed juxtainsular WM neuron can remain an explicit separate category; it is not automatically cortical INS.

## Separate status fields

| Field family | Preserve or record |
| --- | --- |
| Identity/retrieval | Sample/channel/NeuronID, hashes, workbook/sheet/Excel rows, original Folder, numeric-neighbor cue, nearest-INS distance with coordinate policy |
| Original labels | Every Henry Area/Layer duplicate, portal label, source table label, atlas and tissue results; no destructive precedence merge |
| Coarse human decision | `coarse_region`, `coarse_review_status`, existing Henry/user-reported versus newly inspected basis, reviewer/date, image/decision links and reason |
| Fine anatomy | Original fine parcel/layer and separate `fine_parcel_status`, `layer_status`; unresolved fine anatomy does not undo credible coarse INS evidence |
| Image QC | Cell-body/graph correspondence, crop alignment, section/context coverage, seams, channel and source provenance |
| Registration | Native/reference versions, export-origin status, current/sensitivity lookup, transform lineage if available and discrepancy hypothesis |
| Tissue/laterality | Native tissue assessment and warped tissue separately; human anatomical side, mask side and unresolved side separately |
| Eligibility | Coarse biological INS selection and reason; atlas-map eligibility separately; terminal/arbor assessment separately; retain every unassessed/excluded identity |

Numeric neighbors and NNT/nearest-neighbor or atlas-distance results retrieve candidates for review; they establish neither spatial continuity nor membership. Prefer same-sample confirmed INS and adjacent-cortex controls. A human-confirmed coarse INS neuron with unresolved registration can remain in the biological inventory while precise atlas-map eligibility stays flagged. An in-bounds atlas root or nearby ID cannot substitute for source evidence for a new inclusion.

No fine parcel is required for every coarse INS acceptance. The parent will record actual case decisions in separate derivatives; this protocol preserves all original evidence and does not itself promote new candidates.

## Actual inspection of three additional cases

On 2026-10-09, Codex independently opened each case's saved `soma_marked.png`, `soma_ortho.png` and `context_marked.png`; additionally opened 252384/047's `section_locator.png` and `context_stack.png`. These observations concern the inspected saved displays and sidecars. They are not human INS acceptance, an independent source-pixel audit or a full native-image tracing review.

| Case | Actual observed signal and context | Anatomical implication |
| --- | --- | --- |
| 252790/032 | Bright root-associated fluorescence with neurites in native XY/XZ/YZ displays; other labeled structures nearby. Soma crop loads 27 cubes without missing neighbors. Context is assembled/downsampled from own high-resolution fluorescence tiles, nominal 5.2 × 5.2 × 3 µm, 4 × 4 mm × 30 µm. Display reports 75.4% coverage, 239/324 cubes acquired and 85 missing. No full-section locator. | Supports source soma-associated signal. Partial local context does not independently resolve INS versus adjacent cortex or tissue type. It is a priority for expert native localization, not an automatic INS inclusion. |
| 252383/121 | Root coincides with a bright elongated soma-like structure and attached neurites in orthogonal native displays. Single soma cube is complete. Derived fluorescence context has conspicuous rectangular missing patches; display reports 56.3% coverage, 175/324 acquired and 149 missing cubes. No full-section locator. | A real image signal is present, but its local field and missing anatomy cannot establish a coarse INS boundary or genuine white-matter soma. Additional trustworthy context/reviewer assessment is needed. |
| 252384/047 | Root-associated bright fluorescence and neurites visible in native orthogonal displays; single soma cube complete. Copied low-resolution source context uses 5 × 5 × 3 µm, 4-mm field, ten loaded sections and no recorded missing section. Full-section locator places its crop relative to the broader tissue. The five displayed neighboring sections show horizontal intensity bands/seams. | This case has actual wider source-tissue context for expert localization. Bands/seams and unverified native-to-overview alignment limit precise boundary claims. No definite INS, WM or adjacent-cortex decision is made here; do not assume the separate 252384/003 defect proves this case unusable. |

These **derived contexts contain acquired fluorescence pixels**, assembled from high-resolution tiles; they are not tissue synthesized from SWC geometry. SWCs guide centres/overlays and some acquisition selection, which does not turn the resulting field into an unbiased full-section sample. Missing tiles are not missing tissue or absent neurons. Trace overlays remain reconstruction evidence and must be distinguished from the unmarked source-image signal.

The inspected marked-image byte hashes are:

| Case | `soma_marked.png` SHA256 | `context_marked.png` SHA256 |
| --- | --- | --- |
| 252790/032 | `6c51102ef7d1159bc2396a6666974c5cf8ff6046f87d6d497ae1301a272e2bf8` | `d7f2cfaaa3053f3d6687dc0ac5fd9c8bd4680c8c5ea8ccf5e27716e99c302631` |
| 252383/121 | `7a4ceeb8ccc2e6e97d349bccff58d29908a40149cf6b5abbf01aa424a3ad4edc` | `d03a5da9199f2926a0f9b420c877f87391b19ff7331e820ee2c687a5d49e6ce9` |
| 252384/047 | `fdb9fa2d64cffdb6f511a5d955bfea7d7d829a9bea45b6fa12c85ae0f535802a` | `999e6d4a61b03d5ee91c214b36b9a88591f95c04133d0f47b38649e64930dcaf` |

All paths are under `group_analysis/visual_review_20261002/{sample}/{id}/`; sidecars bind native spacing, crop geometry and source acquisition status. No images were regenerated or edited for this inspection.

## Across-monkey map presentation

Use the same coarse INS biological inclusion definition across animals, preserving exact Henry/user-reviewed/new-review evidence per neuron. Fine parcels can remain separate optional annotations; do not force Henry's fine scheme onto unreviewed animals. Keep a subject-level panel available even if fine labels are incomplete. Resolve `AnimalID` explicitly and pool repeated injection samples within the same animal before cross-animal summaries; retain injection/sample coverage and counts, not additional pseudo-replicates.

Show separate quantities with explicit titles:

* **Reviewed terminal arbor/field evidence:** only when image-reviewed arbor labels exist. Display actual arbor skeleton/field support or target-presence evidence; do not present smoothed graph-tip spots as reviewed terminal fields.
* **Candidate axonal graph endpoints:** exploratory endpoint count/density and neuron occupancy under the declared export/index policy. Label them candidate endpoints in every subject panel. Report selected and computable neurons separately; lack of eligible type-2 tips is unassessed proxy contribution, not zero biological innervation. Retain outside-FOV, unmapped and unresolved-compartment counts.
* **Axon trajectory length:** a separate complementary map of template-space skeleton passage/length, with its own units and denominator. Do not use it as a terminal-field or synaptic-strength label.

For honest visual comparison, use the same hashed NMTv2.1 symmetric skull-stripped structural background, matched slice coordinates, common support and shared scale per measurement. State when backgrounds or spaces differ. Include native-review and registration/coordinate-origin status per subject; candidate maps can be descriptive while their anatomical alignment remains exploratory. A coarse human INS decision need not be undone by an uncertain warped WM label, but precise cross-subject target overlap cannot be scientifically accepted from that coarse decision alone.

Average neuron contributions within animal first, then average available animal maps equally. Show n-selected/n-computable and missing animal groups; never impute absent groups with zero images. Subject-to-subject differences may reflect injection coverage, reconstruction completeness and registration as well as biology. Keep the three measurement families distinguishable; no t-map, spatial significance or reviewed synapse claim follows from this presentation plan.
