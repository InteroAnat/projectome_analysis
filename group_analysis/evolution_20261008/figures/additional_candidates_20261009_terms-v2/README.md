# Additional Unknown/near-INS candidates: readable previews

[Plain-language map terminology](../../../../docs/region_analysis_terminology.md). A candidate axon end is a non-root axon-labelled graph leaf, not a verified biological terminal. Endpoint-eligible neurons have at least one such leaf anywhere, including outside the reference field of view. Within each animal/source group, density is per eligible neuron (endpoints) or per selected neuron (axon length), and per reference mm³. Across animals, each contributing animal map has equal weight. Occupancy is the fraction of eligible neurons with at least one candidate end in a voxel, counting each neuron once.

26 unique neurons across five established animals, with no neuron or source-path overlap with the [main 436](../multimonkey_coarse_20261009_terms-v3/README.md). **OriginCandidate** contains 15 neurons whose INS lookup changes between coordinate-origin policies. **NearINS** contains 11 neurons with two non-INS lookups but a corrected distance within 2 mm of another established INS anchor. Nearby numeric IDs alone do not select a neuron. These are retrieval classifications; original labels remain unchanged and no new INS anatomy is accepted.

| View | Sheets | Open |
|---|---:|---|
| Equal-animal mean candidate endpoint density | 1 | [Candidate strata](endpoint_density_groups/space-NMTv2p1_desc-candidateEndpointDensity_matchedSlices_page-01.png) |
| Equal-animal mean axon-length density | 1 | [Candidate strata](axon_density_groups/space-NMTv2p1_desc-axonLengthDensity_matchedSlices_page-01.png) |
| Individual-animal endpoint density | 3 | [First sheet](endpoint_density_animals/space-NMTv2p1_desc-candidateEndpointDensity_matchedSlices_page-01.png); subsequent numbered sheets in the same directory |
| Individual-animal axon-length density | 3 | [First sheet](axon_density_animals/space-NMTv2p1_desc-axonLengthDensity_matchedSlices_page-01.png); subsequent numbered sheets in the same directory |

All 26 neurons have computable type-2 graph leaves: 4,063 candidate endpoints, not verified terminals. Main and preview maps remain separate. Descriptive group means use equal contributing-animal weights; OriginCandidate_R has only one contributing animal.

These figures use the same population NMT MRI, common cuts **56, 200, 87**, and display method as the main figures. MRI/map planes match; there is no MIP or smoothing. Each family's color limit is independently recorded, so color alone cannot quantify differences between main and preview sheets. Original NMT coordinates are retained under the unresolved export-origin convention, not shifted to make candidates appear inside INS. [Selection provenance](../../projection_inputs/additional_candidates_20261009/preparation_provenance.json) · [endpoint readback](../../endpoint_maps/additional_candidates_20261009_review/endpoint_run_readback.json) · [axon readback](../../projection_maps/additional_candidates_20261009_review/projection_run_readback.json) · [status and remaining dependencies](../../reports/evolution_status_20261009.md).

This wording variant preserves all underlying map values, selected cuts, MRI windows and color limits. The [display receipt](../../../../notes/region_analysis_review_20261009/readable_figures_receipt.json) records the new PNG and renderer hashes. [Original preview figures](../additional_candidates_20261009/README.md) remain unchanged.

