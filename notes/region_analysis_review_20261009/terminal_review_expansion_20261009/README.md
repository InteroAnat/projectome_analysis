# Expanded native-image axon-ending and passage review

Completed 2026-10-09, reviewer Codex. Seven saved panels were actually viewed: **six additional original full-graph axon leaves in six neurons/animals, plus one internal passage location**. Three new neurons have ARM-background source roots; two have precentral opercular source labels. This adds four animals to the earlier pilot, bringing the combined reviewed set to **11 individual leaves in 11 neurons across seven animals**, plus two internal passage locations. These are purposive examples, not a random sample or cohort-wide terminal assessment.

The unchanged selection remains 462 neurons. This review does not change any source label, source membership, graph, map, cohort or human decision. Existing atlas coordinates and ARM names are **declared metadata under the current lookup**, not independently accepted anatomical localization. A source ARM0 root is not proof of white matter or of insular membership. No authoritative transform was recovered or required for this native-image assessment.

## Cases actually assessed

| Exact neuron UID | Animal | Source metadata | Declared ARM6 target | Selected original leaf | Section branch nodes / original leaves | Observed result |
|---|---:|---|---|---:|---:|---|
| 252383::124.swc | 605 | Atlas background | CR_granular_insula (729) | 5857 | 2 / 3 | Compact fluorescence at the leaf location; nearby shafts and depth overlap leave completion unresolved. |
| 252790::037.swc | 797 | Atlas background | CL_granular_insula (229) | 2494 | 1 / 2 | Fluorescent shaft approaches the location; an upward shaft in later planes creates possible continuation/crossing uncertainty. |
| 252718::162.swc | 900 | Atlas background | CL_granular_insula (229) | 84 | 2 / 2 | Short bright oblique shaft at the location; noisy neighboring signal and depth continuity remain unresolved. |
| 252385::056.swc | 331 | CR_precentral_operular_area (549), portal PrCO_49 | CR_precentral_operular_area (549) | 2727 | 196 / 214 | Extensive branching in the graph; the one inspected leaf has fluorescent support but possible signal beyond its position. |
| 252527::056.swc | 945 | CR_precentral_operular_area (549), portal PrCO_49 | CR_precentral_operular_area (549) | 1224 | 208 / 203 | Extensive branching; local signal and a nearby curved shaft cannot establish endpoint completion. |
| 252714::099.swc | 631 | CL_agranular_and_dysgranular_insula (228) | CL_agranular_and_dysgranular_insula (228) | 5203 | 94 / 77 | Extensive branching; thin fluorescent approach to the ring supports local correspondence, with depth completion uncertain. |

Names above preserve the exact full official ARM key strings, including its `operular` spelling. No other atlas is used. Branch counts describe original graph topology within the target-connected section; they are not image-validated branch counts. The three extensively branched examples are **branched-field morphology candidates**. The smaller sections are local branch candidates; their few branch nodes do not establish a dense terminal field. All six selected graph leaves remain reconstructed endings with local image support and **unresolved ending completeness**. No leaf was deleted or accepted as a biological terminal.

For the extra passage example, `252790::037.swc` original internal node **7419** belongs to a 38-node CL_granular_insula section with zero original leaves and zero internal branch nodes. Its original entry is `7399→7400`, exit `7437→7438`; node7419 has an original child. The depth sweep shows local bright signal and a curved trajectory around the node. A bright spot is therefore not sufficient to call a terminal or bouton. The graph supports passage; the whole section was not image-proofread, and passage does not exclude en passant boutons.

## What the panels show

[Review panels](review_panels/) contain six `*_section_and_optical_planes.png` figures and one `252790_037_internal_passage_optical_planes.png`. Each has three full-section morphology projections followed by nine **actual, unprojected XY optical planes**, not maximum-intensity projections. The ring marks only the selected node's XY position; its Z coordinate is stated in the title. A ring on another Z plane does not imply signal at the node's full XYZ coordinate. Original graph entry/exit links are preserved, and regional cuts never manufacture a graph leaf.

Inputs use the existing nominal native sampling, 0.65 × 0.65 × 3 µm, uint16 TIFFs with shape ZYX=(90,360,360). Axes labelled native XYZ do not imply anatomical directions. Pixel centres follow the existing nominal client convention; optical resolution and calibrated image/SWC origin are unverified. Local visual correspondence is supporting evidence, not physical registration ground truth. These displays use a recorded 1st–99.7th percentile contrast window without segmentation or a fluorescence acceptance threshold.

The targets were predeclared for breadth; within each, the cached cube with most covered original leaves was selected, followed by the leaf with the largest minimum distance to a cube face. Ties use cube XYZ and node ID. This is a reproducible display-selection rule, **not a biological criterion**. The six selected leaves lie 31.7–72.5 nominal µm from their nearest cube face, so none is at a reviewed cube edge. All six face-neighbors are cached for the first four cases. `252527::056.swc` lacks the X-minus neighbor, while `252714::099.swc` lacks X-minus and Y-minus; the selected leaves lie respectively158.625, and183.098/106.203 nominal µm from those missing faces. Missing adjacent caches limit broader tracing; they are not tissue boundaries or demonstrated damage. Only the recorded local planes were inspected, not every neighboring cube or every branch.

## Coverage and interpretation

The prior hash-bound [all-462 coverage ledger](../terminal_field_assessment_20261009/native_cube_pilot_current_rint/selected462_native_image_coverage.csv) remains unchanged: 201 designated native/atlas graph pairs, 184 neurons with at least one leaf in a cached source cube, and 9,209 covered reconstructed leaves out of48,574 original axon leaves in those native graphs. Availability is not validation. The 261 absent from that native cache are unassessed for this paired review, not biological negatives. Animal936 is absent from the designated native cache; this expansion adds no native-image validation to the Henry-reviewed-only animal.

Six new leaf locations and one passage location were actually reviewed here. Together with the earlier five leaves and one passage, this is **11 individually reviewed leaf locations and two passage locations**; 451 selected neurons have not received either pilot's individual leaf review. Other leaves within the displayed sections remain unassessed. Across the184 neurons with available leaf imagery, only11 have an individually reviewed leaf in these two pilots. The 52 ARM-background source cases and all other evidence strata remain in the unchanged ledger; three additional background-source cases receive limited local axon review here, without resolving their soma/source classification.

The current target lookup is `np.rint(atlas_XYZ/250)` into pinned ARM level6. Coordinate-origin sensitivity, source-root acceptance and transformation provenance remain as previously recorded; no anatomical claim is inferred from lookup agreement. These panels do not estimate terminal density, bouton density, synapses, connectivity prevalence or completeness of long-range arbors. SWC radius was not used.

Methods follow the earlier [terminal-field assessment](../terminal_field_assessment_20261009/README.md) and the actual [local Liu Methods review](../literature_method_update_20261009/liu_local_methods.md): image continuity and graph branching are separate observations; Yufeng Liu's image-intensity/radius varicosity method is not reproduced from SWC leaves. Sang Liu's exact full Methods remain unlocated; no unavailable classifier parameter is invented.

## Provenance and readback

- [Assessment inputs](assessment_inputs.json): exact UID/animal, source/target evidence, original node IDs, component entry/exits, native/atlas/TIFF hashes, cube geometry, adjacent-cache availability and selected-node face distances.
- [Actual image assessments](codex_image_assessments.json): per-leaf observations, uncertainty and separate passage review; no anatomical/biological acceptance.
- [Passage inputs](passage_inputs.json): exact original graph entry/exit and source/display bindings.
- [Generation provenance](generation_provenance.json): input/code and six-case output hashes. Its pending-review status is historical; the separate review/readback receipt records completion without rewriting it.
- [Independent saved readback](independent_saved_readback.json): source hashes reverified, leaf status checked against every original node's children, target-connected membership independently traversed without importing the producer, section branch/leaf counts reconciled, passage identity checked, and PNG bytes/dimensions verified. This is software/spatial evidence, not independent biological acceptance.

Reproducible local generators refuse existing output destinations. All new scripts and records reside in this folder. No network request, raw-image copy, source modification, cohort promotion, staging or commit was performed.
