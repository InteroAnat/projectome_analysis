# Insula pipeline evolution — 2026-10-08

**Current evidence update:** [local Gao/Liu Methods, native terminal assessment and CM032/updated CM033 review](../../notes/region_analysis_review_20261009/scope_update_20261009.md). Original fMOST transforms are unavailable, without a recovery gate. The five-neuron native pilot adds image-backed morphology evidence; it does not convert all candidate ends into reviewed terminal fields. Existing ARM NIfTIs, six-level tables and the single selected-neuron ledger remain unchanged.

Active branch: `codex/insula-pipeline-evolution-20261008`, created from `78008cbaab71bf737dc7129c72e5fe284c3ba40f`. Existing unrelated edits and source data are preserved. [Publication status](../../notes/region_analysis_review_20261009/publication_20261009.md) records scoped commits and remote verification; cohort promotion and scientific acceptance are separate decisions.

The subsequent [region/table audit](../../notes/region_analysis_review_20261009/README.md) records assessed tables, focused repairs and final checks. Start with the [primary ARM map index](arm_mapping_20261009/README.md), [six-level projection matrices](../../notes/region_analysis_review_20261009/hierarchy_tables_20261009/README.md), [projection-profile clustering](../../notes/region_analysis_review_20261009/clustering_20261009/README.md) and [MSTIM integration](../../notes/region_analysis_review_20261009/mstim_integration_20261009/README.md). Earlier test counts and derivatives retain their dated scope.

The user's current priority is **terminal sites/fields first**, with axon trajectories as a complementary measure. The [refined goal and measurement definitions](terminal_projection_goal_20261008.md) preserve all seven requested workstreams and document the exact brain background and initial map method.

The [2026-10-09 status report](reports/evolution_status_20261009.md) records the inventory, focused repairs and classification/region QC. One **462-neuron ledger across eight monkeys** preserves the disjoint historical 436/26 source partitions. Of these, 410 have named ARM source parcels (301 direct insular labels and 109 named neighbours), while 52 remain atlas background. Human review and candidate evidence remain separate from these direct atlas assignments. The current map workflow defaults to one primary figure set; NIfTI volumes and all six table hierarchy levels remain available.

The [original Methods review and scientific basis](references/scientific_basis_20261009.md) guides interpretation. Published fMOST registration supports raw-sample-to-NMT images and separately transformed SWC points. Export-origin sensitivity, fine-parcel uncertainty and the paper/deposited-code arbor-clustering discrepancy remain explicit. The new endpoint maps contain computational candidates; biological terminal-field acceptance remains pending.

| Location | Contents and scope |
|---|---|
| [inventory/live_inventory_20261008](inventory/live_inventory_20261008/README.md) | Current all-macaque metadata inventory, source snapshots, dataset observations and independently checked identities. |
| [atlas soma locations](atlas_locations/soma_audit_20261009/delivery_readback_20261009.md) | All 8,746 identities; 1,912 available own-source roots, manual/portal/direct atlas channels, source-byte collisions and independent coordinate-origin sensitivity. |
| [coarse review and graph sources](classification/coarse_insula_review_20261009/README.md) | Human evidence, atlas/tissue policies, complete graph equality for all cached copies, native-case limitations and the [corrected 559-entry queue](classification/coarse_insula_review_20261009/distance_priority_v2_20261009/README.md). |
| [main map inputs](projection_inputs/multimonkey_coarse_20261009_v2/preparation_provenance.json) / [additional candidates](projection_inputs/additional_candidates_20261009/preparation_provenance.json) | 436 / 26 exact source-verified identities; evidence, animal identity and source hashes preserved. |
| [main endpoint readback](endpoint_maps/multimonkey_coarse_20261009_review_v2/endpoint_run_readback.json) / [preview readback](endpoint_maps/additional_candidates_20261009_review/endpoint_run_readback.json) | 108 / 39 saved maps, complete voxel/region/denominator checks; candidate endpoints only. |
| [main axon readback](projection_maps/multimonkey_coarse_20261009_review/projection_run_readback.json) / [preview readback](projection_maps/additional_candidates_20261009_review/projection_run_readback.json) | 72 / 26 saved length/density maps, independent voxel splitting and equal-animal means. |
| [primary ARM maps](arm_mapping_20261009/README.md) | Twelve group sheets with full official names and matched NMT MRI cuts; reviewed endpoint and axon NIfTIs. Earlier display variants are archived. |
| [explicit discovery rule](validation/discovery_geometry_rule_20261009.md) / [refinement readback](reference/refinement_folded_20261009/independent_readback_20261009.json) | Folded-mm mode, source/reference hashes and independent agreement on 1,223 historical identities. |
| [projection_inputs/curated_reference](projection_inputs/curated_reference/projection_manifest.csv) | Explicit source/animal/subregion/reference-hash manifest for the 260-neuron pilot. |
| [projection_maps/curated_reference](projection_maps/curated_reference/run_provenance.json) | Descriptive all-axon length/density maps, coverage accounting and code/source hashes. |
| [projection_maps/curated_reference_review](projection_maps/curated_reference_review/projection_run_readback.json) | Independent complete saved-map readback and the original labelled MIP overview figures. |
| [matched-slice figures](projection_maps/curated_reference_matched_slices_20261008/matched_slices_provenance.json) | Corrected same-plane MRI/map views, shared contrast windows, exact cuts and figure hashes; map values unchanged. |
| [crossmodal](crossmodal/input_readiness.json) | Read-only CM032/CM033 input, reference and registration-readiness audit. |
| [references](references/selected_library_evidence.json) | Verified Zotero citation metadata, method support, local export corrections and limitations. |
| [full Methods basis](references/scientific_basis_20261009.md) | Original Zotero Methods for fMOST, insula cytoarchitecture/tracing, NMT hierarchy and multimodal comparisons; exact source anchors and unresolved adaptations. |
| [current validation](validation/unittest_20261009_delivery.log) | 283 passing tests, independent real-data map checks and [source-leaf census](validation/terminal_leaf_source_audit_20261008.json); anatomical acceptance is assessed separately. |

The primary biological goal remains reviewed terminal arbors/fields. Current cross-monkey maps show candidate graph endpoints and **all type-2 axon edge length**, including arbors. Endpoint maps condition on 403/436 main neurons and 26/26 preview neurons with eligible leaves; the others remain unassessed. Axon means use all selected neurons. Group summaries weight contributing animals equally. No t-map or new anatomical acceptance is claimed.

New sources are not required to reproduce the saved runs; public preparation, mapping and review scripts are in `../scripts/`. Run each into a new output directory. Source images, canonical tables and existing figures are not overwritten. The earlier `figures/multimonkey_coarse_20261009` sheets are preserved but superseded for display by `..._layout-v2`; underlying maps are unchanged.

The historical [terminal methods note](references/terminal_site_measurement_plan_20261008.md) defines candidate endpoints, reviewed arbors, putative boutons and verified synapses separately. The new full Methods review supplies the original source detail, supersedes earlier access limitations and documents corrections. The [hardened 32-map readback](projection_maps/curated_reference_review_20261009_v2/projection_run_readback.json) also passes; software agreement does not resolve the export-origin assumption or establish anatomical registration.

The following command reproduces the earlier one-animal pilot display (choose a fresh output directory). Use the [primary ARM workflow](arm_mapping_20261009/README.md) for current multi-monkey maps; the [earlier workflow](reports/mapping_workflow_20261009.md) is retained for historical reproducibility:

```powershell
python -B group_analysis/scripts/render_projection_slices.py --run group_analysis/evolution_20261008/projection_maps/curated_reference --readback group_analysis/evolution_20261008/projection_maps/curated_reference_review/projection_run_readback.json --output group_analysis/evolution_20261008/projection_maps/curated_reference_matched_slices_repeat01
```

That earlier pilot command selects cuts separately for each map. Current multi-monkey figures instead use common XYZ cuts 56, 200, 87, preserving matching MRI/map planes without MIP. Neither selected slices nor computational zeros establish biological absence.
