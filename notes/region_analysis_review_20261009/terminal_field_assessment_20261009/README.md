# Native-image terminal-field candidate assessment

**Completed 2026-10-09; reviewer: Codex.** We can assess reconstructed terminal-field candidates with the available native SWCs and cached microscopy. This pilot actually inspected five neuron/target fields, five individual candidate endings in optical planes, and one reconstructed passing segment. It does not turn the cohort's endpoint maps into reviewed terminal-arbor maps.

The authoritative fMOST-to-NMT transform is unavailable. Recovering it is **not a requirement for this native morphology/image assessment**. ARM labels below are the existing declared target metadata, conditional on the current export and lookup policy. No source coordinates, source labels, cohort membership, existing maps or published outputs were changed.

**Terminology:** SWC type 2 means an axon-labelled node, not a neuron class. A *connected axon section within an ARM region* is the review unit called a `target_component` in the code: original axon-labelled nodes joined by original edges, with both edge ends in that region. Such a section may contain branching fields and passing shafts; it is not automatically a terminal arbor. A regional cut never creates a full-graph ending. Node-type interpretation follows the source SWC convention and is separate from image-backed compartment validation.

## What was assessed

| Neuron | Declared ARM level-6 target | Original leaf | Codex assessment |
|---|---|---:|---|
| `252790::014.swc` | `CL_intermediate_agranular_insula_area` (41) | 1727 | Branched-field morphology candidate; selected ending has local fluorescent support. The whole component has 88 branch nodes and 71 original leaves, which were not all image-reviewed. |
| `252790::032.swc` | `CL_granular_insula` (229) | 2344 | Image-backed simple ending; selected component has no branch nodes and one leaf. It is not itself a terminal-field arbor. |
| `252383::018.swc` | `CR_granular_insula` (729) | 2294 | Branched-field morphology candidate; selected ending has local fluorescent support. The whole component has 56 branch nodes and 45 leaves, which were not all image-reviewed. |
| `252383::121.swc` | `SR_putamen` (1556) | 867 | Simple ending with weaker, noisy depth support. No branched putamen arbor is established by this component. |
| `252384::047.swc` | `SL_claustrum` (1004) | 6514 | Simple ending with ambiguous depth correspondence and missing adjacent image coverage. Possible continuation/tracing truncation remains unresolved. |

These are **purposive examples**, chosen to include branching, simple endings, uncertain sources and limited image coverage, across three samples/animals. They do not estimate the prevalence of arbors or terminal completeness. The two branched-field candidates are local insula-labelled fields, not evidence of complete long-range terminal-field coverage. A target-connected component is a transparent review unit; it can contain several morphological fields and passing shafts. It is not an author-model arbor segmentation or an accepted biological arbor boundary.

The additional passing example is `252790::032.swc`, target 229, nodes 7082–7128, original entry `7081→7082`, exit `7128→7129`, internal display node 7105. It has no original graph leaves or branch nodes within that target component. It demonstrates that cutting a graph at a regional boundary must not manufacture a terminal. Passage does not imply absence of en passant boutons.

The source-plane sweep particularly matters for `252384::047.swc`: leaf 6514 is 3.879 nominal µm below the cube's upper-Y block boundary, and the immediately adjacent upper-Y cube is not cached. A cache edge is not a tissue edge, proven damage, or a reason to delete the endpoint. The case remains ambiguous rather than automatically rejected or biologically accepted.

## Evidence and denominators

The unchanged selected ledger has 462 neurons. The designated local native-SWC cache contains 201 of them. All 201 numerical native/atlas SWC comparisons preserve the same node IDs, types and parent links; the five reviewed graphs additionally pass the complete structural parser. Identity correspondence does not establish registration accuracy or correct compartment labels.

Across those 201 native graphs there are 48,574 strict candidate axonal leaves. Of these, 9,209 fall in an existing local source TIFF cube; 184 neurons have at least one such covered leaf and 17 available-native neurons have none. This is **file availability**, not 9,209 image-validated endings. The 261 not found in this designated native cache are unavailable for this paired review; this is not a claim that no native source exists elsewhere.

Exactly five original leaves were individually image-reviewed here, one per selected field. The other leaves within the five fields, the rest of those neurons and the remaining 457 selected neurons have not received this pilot's image review. Biological terminals, boutons and synapses remain unassessed. No biological zero-innervation result follows from missing native data, absent cubes, an unresolved ending or an unbranched target component.

The image cache contains 68,213 TIFFs across nine sample directories, including samples outside this selected ledger. Each reviewed cube is bound by its own hash. The images are uint16 arrays of shape `(90, 360, 360)` in ZYX order, displayed after explicit conversion to XYZ. Nominal sampling is 0.65 × 0.65 × 3 µm, giving nominal block extents 234 × 234 × 270 µm. Sampling is not a measured optical resolution, and native axis names do not imply anatomical left/right or anterior/posterior.

Block origins use the existing client's nominal block convention. The current figures place displayed pixel centres consistently with that nominal index convention; no physical-origin calibration or voxel-level ground truth is inferred. Native branch lengths are labelled **nominal native µm**, not corrected tissue length. The optical-plane panels contain actual individual XY planes around each selected ending, rather than MIPs labelled as slices. MIPs remain useful overview figures; local ending assessments use the plane sweeps as well.

## Literature-based assessment criteria

The exact source Methods and deposited-code review are preserved in [fMOST full Methods](../../../group_analysis/evolution_20261008/references/fmost_full_methods_20261009.md) and its [evidence JSON](../../../group_analysis/evolution_20261008/references/fmost_full_methods_evidence_20261009.json).

* **Gao et al. 2022**, [Nature Neuroscience, DOI 10.1038/s41593-022-01041-5](https://doi.org/10.1038/s41593-022-01041-5), Zotero `L5TALCAJ` / indexed full text `IQ5H7F89`: Methods “Quality control and evaluation” and “Reconstruction,” indexed lines 1480–1482, use tracing/proofreading and user confirmation of finished branches. This motivates inspecting image continuity and distinguishing a completed branch from a graph termination. It is not a terminal detector for the present data.
* **Gao et al. 2023**, [Nature Neuroscience, DOI 10.1038/s41593-023-01339-y](https://doi.org/10.1038/s41593-023-01339-y), Zotero `SAKZIAKG` / `CJUA8PUK`: “Analysis of long-range axons,” lines 1849–1850, measures regional axon length; “Potential connectivity,” lines 1853, models possible synaptic proximity. Length, a graph leaf and a putative synapse are different observables. Its incomplete/abrupt-ending reconstruction QC at 1823–1824 motivates preserving unresolved completeness.
* **Gou et al. 2025**, [Cell, DOI 10.1016/j.cell.2025.06.005](https://doi.org/10.1016/j.cell.2025.06.005), Zotero `JWB4MYBC` / `8BNNG4KV`: STAR “Segmentation of axon arbors,” e5–e6, indexed lines 437–438 and 444–446, describes increased downstream branching density and shorter branches and/or gray-matter/nucleus entry. This motivates morphology-reviewed arbor candidates, rather than declaring every endpoint an arbor. **Gou is a different paper/author from Gao.** No arbitrary numerical density, tip-count or branch-length cutoff was imported here. This pilot does not reproduce the terminal-dendrogram GNN; inspected code lacks the needed final model, curated labels and image-derived feature data, and its linkage differs from the Methods prose.
* **Yufeng Liu et al. 2024**, [Nature Communications 15:10269, DOI 10.1038/s41467-024-54745-6](https://doi.org/10.1038/s41467-024-54745-6): “Arbor detection and analysis,” p19, treats densely packed subtrees separately from individual leaves. “Detect axonal varicosities,” pp19–20, requires image-derived intensity/radius evidence and expert annotations; p15 limits biological validation. Bright puncta in these cubes are not automatically boutons. SWC radius was not used: its provenance is unresolved and an inspected Gou warp code path stores inversion residual in that field.
* **Qiu et al. 2024**, [Science, DOI 10.1126/science.adj9198](https://doi.org/10.1126/science.adj9198), supplement p5 “Axon projections within primary targets”: distinguishes branching/ending arbors from passing trajectories. Its mouse hippocampal length threshold was not transferred to macaque insula.

This assessment therefore separates: **branched-field morphology candidate**, **image-backed simple ending**, **reconstructed passage**, **possible unresolved truncation/continuation**, and **unassessed biological terminal**. A local source image spot-check supports only the inspected location, not every inferred field branch or its ARM assignment. No fluorescence threshold, bouton-size threshold, radius filter or new biological exclusion was selected.

## Reproducibility and current outputs

Start with [Codex assessment records](codex_terminal_field_assessment.json), which contain reviewer, exact node identities, native/atlas/source-image hashes, observations, target policy, test results and review denominators. [Independent review](independent_terminal_review.md) and [its JSON receipt](independent_terminal_review.json) verify the graphs, source binding, geometry, current targets and display correction without accepting biological terminals.

The primary supporting files are:

* [All-462 native/image availability ledger](native_cube_pilot_current_rint/selected462_native_image_coverage.csv).
* [Current-policy generation provenance](native_cube_pilot_current_rint/generation_provenance.json); each neuron subdirectory contains full target-component node identities/coordinates, inputs and morphology/image overviews.
* [Plain-language morphology overviews](review_panels/morphology_display_provenance.json); five primary morphology PNGs titled “Connected axon section within an ARM region,” with original axon ends and branch points. These replace the older morphology displays for presentation only; geometry, node identities and assessment decisions are unchanged.
* [Readable optical-plane and passage figures](review_panels/display_provenance.json); six primary review panels, all actually inspected by Codex. Each panel's title records its composite neuron identity and selected original node.
* [Optical-plane source receipt](optical_plane_checks_current_rint/optical_plane_provenance.json), preserving slice/cube sources and original passage entry/exit.
* Six focused tests in [test_native_terminal_assessment.py](test_native_terminal_assessment.py), covering full-graph leaves, compartment/atlas-cut false endings, unsorted identity matching, exact cube edges, ties-to-even target sampling and displayed pixel centres.

Current ARM target sampling is `np.rint(atlas_xyz/250)`, retaining the declared current policy and ties-to-even behavior. The initial half-open diagnostic is preserved separately. Across all nodes in these five graphs, half-open lookup differs at 139 voxel indices and 11 target labels; **none of the original axon-leaf labels or the five selected connected components changes in this pilot**. This is an actual-data finding, not generic rounding equivalence or anatomical acceptance. Existing source-group root labels and coordinates were not changed.

Scripts ending `_current_rint.py` produce the primary current-policy records. Initial `native_cube_pilot` / `optical_plane_checks` outputs preserve the half-open diagnostic and earlier display; `optical_plane_checks_current_rint` has superseded cramped display panels. Use `review_panels` for readable current morphology, plane and passage displays. Generators refuse to overwrite existing destinations; no large source image is copied into this assessment folder or downloaded.

## What this completes and what remains unassessed

The feasible native-image terminal-field assessment has been performed, with inspectable morphology candidates, individual ending evidence, a passage counterexample, an ambiguous edge case, exact sources and independent software review. It does not depend on recovering an unavailable authoritative transform. It also does not require an unavailable author GNN merely to perform a transparent Codex review.

A whole-cohort terminal-field outcome would require extending the explicit review to the remaining native branches/fields, resolving missing images and local image/SWC ambiguity, and recording independent tracing/anatomical review where biological acceptance is intended. Author-GNN reproduction would additionally require its actual model, labels/features and final algorithm clarification. Those are distinct optional methods, not hidden requirements for this completed pilot. No inferential animal comparison, biological terminal count, NMT terminal-arbor map or synapse claim was generated from five purposive fields.
