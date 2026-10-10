# ARM target-parcel maps using the legacy projection-strength definition

Codex | 2026-10-10 | Descriptive analysis; anatomical acceptance remains open.

The user correctly identified that projection strength already exists in the legacy pipeline. The primary new view therefore preserves that calculation rather than substituting graph-end counts. The maps colour whole receiving ARM parcels, instead of drawing reconstructed axon trajectories. [Producer](../../group_analysis/scripts/map_legacy_strength_to_arm_parcels.py), [numerical output and map index](../../group_analysis/evolution_20261008/arm_target_strength_20261010/map_index.csv), [independent checker](verify_legacy_parcel_strength.py).

## Verified primary literature

| Reference and local Methods anchor | Specific method supported | Application and limitation |
|---|---|---|
| Gao et al. (2022), *Single-neuron projectome of mouse prefrontal cortex*, Nature Neuroscience 25:515–529, DOI [10.1038/s41593-022-01041-5](https://doi.org/10.1038/s41593-022-01041-5); Zotero L5TALCAJ / IQ5H7F89. PDF p17, “Subdivision of PFC target regions”; Figure 3f–g on PDF p7 and Figure 5d–f. | Subdivide target striatum into 14,004 cubes of 100 × 100 × 100 µm; calculate neuron-by-cube axonal lengths; cluster target cubes with similar PFC input using Spearman rank correlation and Ward linkage. Apply the procedure to mediodorsal thalamus. | This is the method recalled by the user. Our source-by-target matrices are the first descriptive step. Existing ARM parcels replace cubes; macaque ARM labels replace the mouse atlas. We have not discovered new connectivity-defined subdivisions. The paper's distance/linkage combination requires a separate implementation audit before any replication; it is not copied blindly. |
| Gao et al. (2023), *Single-neuron analysis of dendrites and axons reveals the network organization in mouse prefrontal cortex*, Nature Neuroscience 26:1111–1126, DOI [10.1038/s41593-023-01339-y](https://doi.org/10.1038/s41593-023-01339-y); Zotero SAKZIAKG / CJUA8PUK. PDF p18, “Long-range axons”. | Projection strength measured by axonal length inside each receiving region; relative target bias `(LA-LB)/(LA+LB)`. | Supports regional length as a descriptive measure, separate from drawing traces. It does not validate our all-compartment legacy leaf gate or native physical units. We preserve that gate for compatibility and label its limitations. |
| Gou et al. (2025), *Single-neuron projectomes of macaque prefrontal cortex reveal refined axon targeting and arborization*, Cell 188:3806–3822.e1–e9, DOI [10.1016/j.cell.2025.06.005](https://doi.org/10.1016/j.cell.2025.06.005); Zotero JWB4MYBC / 8BNNG4KV. PDF pp23–25, arbor segmentation and patches. | Arbor definitions use morphology and anatomical context, followed by curated training data and graph neural network segmentation. Fine arbor patches are evaluated separately within specimens because cross-specimen overlap is confounded by variability and registration. | Region-wide leaf-gated reconstruction length is not segmented terminal-arbor length. No endpoint or legacy gate is promoted to a verified terminal field, synapse or author-model output. |
| Yan et al. (2022), *Mapping brain-wide excitatory projectome of primate prefrontal cortex at submicron resolution and comparison with diffusion tractography*, eLife 11:e72534, DOI [10.7554/eLife.72534](https://doi.org/10.7554/eLife.72534); Zotero RXUGG8AT / 7974HUDI. PDF p20, registration/quantification; Figure 3 on PDF pp7–8. | Registered segmented GFP volume and parcel GFP density quantify bulk-tracing connectivity; total-signal share and maximum-density normalization answer different questions. | This is serial two-photon bulk tracing, not single-neuron fMOST. It supports interpretable parcel displays and explicit denominators, not transfer of fluorescent density to our skeleton length. Its D99/INIA19 labels are not used here. |

Exact local PDF hashes, read pages and extraction hashes are recorded in [literature receipt](legacy_target_strength_literature.json). PDFs and extracted full text remain private. Earlier [Liu Methods review](../region_analysis_review_20261009/literature_method_update_20261009/liu_local_methods.md) supports separating image-derived varicosity/arbor evidence from graph topology; it is not treated as a new reading in this increment.

## Numerical definition and hierarchy policy

For neuron `i`, target region `r`, and actual ARM level `h`, `L(i,r,h)` is the legacy retained reconstruction length:

1. Divide declared atlas-index-micrometre coordinates by the verified 250 µm/index scale. This is an encoding conversion, not a recovered registration.
2. Include edges from every SWC compartment. Calculate Euclidean edge lengths in atlas-index coordinates and round each edge to three decimals, as the legacy code does.
3. Assign the whole edge to its proximal node's rounded atlas voxel. Do not distribute it along its trajectory or move it to the graph ending.
4. Retain a positive labelled region only if it contains at least one original graph leaf of any compartment. Outside-reference leaves and atlas label zero do not become named receiving targets.
5. Compute the existing per-neuron strength `S(i,r,h) = round(log10(L(i,r,h)+1),4)`. An absent retained target in a successfully measured neuron is zero.

The same rule is freshly evaluated against each of the six actual ARM volumes. This is an explicitly documented extension: it is not historical parent pooling of an L6 table. The leaf gate and proximal assignment mean coarse-level retained lengths need not sum to fine-level retained lengths. Never sum values across hierarchy levels.

The new group summary first averages **per-neuron logged strength within each selected animal/source group**, then averages those animal means equally. All selected neurons in the source parcel count in the denominator, including measured zeros. An animal with no selected neuron in a source parcel is absent, not a zero animal. `mean(log10(L+1))` differs from `log10(mean(L)+1)`; the latter is not substituted. The raw retained lengths, animal means, group means and counts are also saved.

All 462 exact source identities have sparse six-level measurements and graph QC. Nine named level-6 ARM insula origin views contain 301 atlas-assigned neurons. The other 161 source-evidence records remain in the complete measurement ledger; uncertain sources are not assigned to fine insula parcels. Existing candidate-inclusive endpoint and axon maps remain separate evidence. No new cohort or animal is invented.

## Map scale and interpretation

Each parcel's voxels receive its group mean `S`, stored as float32 in NIfTI. Background label zero is NaN, not measured absence. Positive labels with measured zero retain zero. Official full ARM names and key-level conflicts remain in the target tables. A common linear colour scale from zero to the maximum stored group strength across all nine sources and six levels makes display values comparable; no percentile clipping, smoothing, MIP or extra log transform is used. Zero parcels are transparent in figures but remain quantitative zeros in NIfTI. The background is the pinned skull-stripped NMT v2.1 symmetric MRI, intensity window 75–920 (arbitrary MRI units), on its 0.25-mm grid.

This is a **regional summary painted onto parcels**, not a measured voxel-density map. Its values depend on the legacy atlas-index unit, reconstruction coverage and graph gate. It is not calibrated native tissue length, accepted terminal-arbor strength, neuronal population density or a statistical t-map. Sparse animal coverage, source assignment uncertainty and unavailable authoritative transforms limit anatomical interpretation. Tests establish agreement with code and source records only.

## Reproduction and validation

From `D:\projectome_analysis`, using the project Python environment:

```powershell
$python = 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe'
& $python -X utf8 -B notes/projection_maps_by_origin/test_legacy_parcel_strength.py
& $python -X utf8 -B group_analysis/scripts/map_legacy_strength_to_arm_parcels.py --output group_analysis/evolution_20261008/arm_target_strength_reproduction_20261010
& $python -X utf8 -B notes/projection_maps_by_origin/verify_legacy_parcel_strength.py --output group_analysis/evolution_20261008/arm_target_strength_reproduction_20261010 --receipt notes/projection_maps_by_origin/legacy_parcel_reproduction_readback.json
```

Use fresh output and receipt paths. The producer verifies input/reference/atlas/source hashes, validates and reads each graph once, calculates edges once and reuses them for all six target volumes. It saves per-neuron sparse data, complete named-origin neuron-by-target length and strength matrices at each level, per-animal/group source-by-target tables, 54 NIfTIs, nine figures and software/version provenance. The checker imports the independent legacy regional API (not the producer), tests representative sources from all origin parcels/monkeys, the large graph, an endpoint-ineligible case and 251637/115, then independently expands zeros/animal denominators and reads every parcel voxel in all 54 maps. Focused tests cover compartment/leaf gating, proximal assignment, level-specific labels, rounding, outside-reference leaves and unequal animal sampling.

Target-profile clustering, alternative axon/arbor measures, target-volume density and within-neuron fractional normalization can follow after explicit feature/coverage checks. They are separate analyses and are not implied by the legacy map scale.

## Saved validation outcome

The [independent readback](legacy_parcel_independent_readback.json) passed: 462 source hashes, 18 representative neurons evaluated through the separate legacy regional API at all six actual levels, and every voxel in all 54 parcel maps. Independent matrix-based animal aggregation agrees with every saved source-by-target value and denominator. Four focused legacy regressions, six source-evidence regressions and nine publication-tool tests passed. [Visual review](legacy_parcel_visual_review.json) covers all nine figures in a contact sheet, with full-size checks for ARM6 origins 41, 43 and 728. The common upper display limit is 2.5667130303030303. These checks establish software/source agreement, not anatomical or terminal-field acceptance.
