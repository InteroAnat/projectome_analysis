# All-selected connected axon-section census

Completed 2026-10-09 by Codex. This census covers **all 462 selected neurons across eight established animals**, extending the small image-review examples to a complete quantitative graph ledger. It adds no cohort, map family, smoothing, statistical test or automatic anatomical acceptance.

The saved graphs contain **7,828,761 axon-labelled nodes, 75,625 original axon endings and 47,044 connected axon sections** under the declared current ARM level-6 lookup. Here an axon label means SWC compartment type 2, not a neuronal cell type. A section is a connected part of the original axon graph whose sampled nodes share one ARM label. It is not a segmented or accepted biological arbor.

| Saved graph description | Sections |
|---|---:|
| Branching with original endings | 8,942 |
| Original ending without a local bifurcation | 12,881 |
| Branching without original endings | 2,440 |
| Unbranched without original endings | 22,781 |

These counts describe graph topology and regional cuts, **not biological field or passage prevalence**. A branching section with endings is a morphology-review candidate; a simple ending does not establish an arbor. An unbranched section without endings can support a passage hypothesis, but may reflect a short regional cut and does not exclude en passant boutons. All sections retain `biological_terminal_state=unassessed` and `image_review_state=not_assessed_by_census`.

## Exact measurement contract

- **Original ending:** a non-root type-2 node with no children in the complete original graph, including non-axon/custom types. Cutting by atlas region, compartment or image block never creates an ending.
- **Connected section:** original parent–child adjacency between type-2 nodes with identical sampled ARM6 labels. Every axon-labelled node occurs in exactly one section, including atlas background and outside-reference sections. Node identities and original links are retained.
- **Branching:** local branch points have at least two children within that section. Separate fields retain full-graph branch points, axon-child branch points, target-label transitions, compartment boundaries and nodes protected from false endpoint classification by non-axon children.
- **Target lookup:** `np.rint(XYZ / declared IndexScaleUm)` with ties to even, using the pinned ARM key and NMT v2.1 image. This is the explicitly retained current node-lookup policy, not registration acceptance or a claim of equivalence to half-open voxel allocation. Official full ARM names are preserved without CHARM/SARM substitution or inferred hierarchy parents.
- **Length:** a child-type-2 edge is measured using the NMT reference affine in millimetres. Internal section length includes an edge only when both original endpoint nodes belong to the section. A sparse edge's interior can cross another label or background; this is therefore **endpoint-selected edge length, not exact rasterized regional axon length**. Node-label transition counts are node-edge approximations, not measured anatomical crossing sites. Incoming target-transition and non-axon-to-axon edges are retained separately; the three edge categories conserve total child-type-2 length.
- **Native correspondence:** all 201 designated native/NMT pairs have exactly matching node IDs, compartment types and parents. Optional native lengths use their nominal acquisition XYZ coordinates in micrometres, not calibrated tissue length. Topology agreement does not verify the deformation or source anatomy. The authoritative original transform is unavailable; recovery is not required for this bounded native review.

Of the 47,044 sections, 31,570 have positive official ARM labels, 15,473 have atlas-background labels, and one lies outside the reference. Their original endings number respectively 63,955, 11,669 and one. Background is not equated with white matter, nor are target labels used to promote an uncertain source soma to accepted insula.

## Image availability and actual review are separate

The independent native census reconciles exactly: **201 pairs, 48,574 original axon endings, 9,209 endings in cached native cubes, and 184 neurons with at least one covered ending**. The native pairs contain 25,652 sections; 1,947 sections have at least one ending in a cached cube. Coverage means a matching cube filename exists in the designated local cache; the census does not open or review image pixels. Nominal source sampling is 0.65 × 0.65 × 3 µm, not a measurement of optical resolution.

The remaining 261 selected neurons, all animal 936, lack native graphs in this designated cache. Their native lengths and coverage values remain blank/NA, never biological zero. This is a cache-scoped statement: separately recorded older case assets are not declared globally absent.

The [earlier pilot](../terminal_field_assessment_20261009/README.md) and [six-case expansion](../terminal_review_expansion_20261009/README.md) actually reviewed **11 individual original-ending locations in 11 neurons across seven animals, plus two internal passage locations**. They are purposive image examples, not complete section reviews. The other 75,614 original ending locations and all complete biological terminal fields remain unassessed. Availability of the other cached endings must not be presented as completed review or validated terminal connectivity.

[Additional image-review cues](additional_image_review_cues.csv) identify up to 20 new neurons with ARM-background sources or portal PrCO annotations and cached ending evidence. Ranking uses covered-ending count, then local branch count and identity; one section per neuron is retained, excluding the 11 prior pilot neurons only to improve future review breadth. This is a transparent availability ranking, not a biological threshold or an instruction to promote source labels. For example, `252383::097.swc` has a declared CL_granular_insula section with 129 original endings, 111 local branch points and 127 covered endings; its source ARM0 classification remains unresolved.

## Literature basis and limits

The actual full-Methods reviews distinguish **Gao** from **Gou**. [Gao et al. 2022](https://doi.org/10.1038/s41593-022-01041-5), PDF p16, motivates explicit tracing completion and image-backed independent reconstruction QC. [Gao et al. 2023](https://doi.org/10.1038/s41593-023-01339-y), PDF pp17–18, supports regional axon-length allocation while distinguishing modeled potential connectivity from observed synapses. Our census supplies neither the authors' independent human tracing nor synapse evidence; see the exact local [Gao Methods readback](../literature_method_update_20261009/gao_local_methods.md).

[Gou et al. 2025](https://doi.org/10.1016/j.cell.2025.06.005), PDF pp23–24 / STAR e5–e6, segments arbors using a terminal path-distance dendrogram and trained GNN, with branching and anatomical context. Our connected sections do not reproduce that classifier; its required final model/curated labels were not located in the inspected deposited code. [Yufeng Liu et al. 2024](https://doi.org/10.1038/s41467-024-54745-6), PDF pp19–20, defines densely packed subtrees and uses image-derived intensity/radius features for varicosity candidates. No mouse distance, size, density or intensity threshold is imported here, and SWC radius is not used as bouton evidence. Exact anchors and limitations are in the [Liu Methods readback](../literature_method_update_20261009/liu_local_methods.md). Sang Liu's exact full Methods remain unlocated and are not claimed as implemented.

An optional **reconstructed end-branch length** descriptor can complement endpoint counts: trace an original full-graph ending upstream along its original degree-one chain to a bifurcation/root or explicitly declared compartment boundary. It measures distal graph geometry, not an arbor. The existing terminal QC API already records a full-graph branch-to-root/bifurcation version and its child-type-2 portion; stopping at the first non-axon node would be a separate definition. Any future spatial map should allocate the selected original chain edges along their trajectory, not deposit whole-chain length at the ending voxel. Long unfinished unbranched trunks remain a limitation. No end-branch maps were generated here.

## Files, validation and reproduction

- [Per-neuron ledger](selected462_ARM6_sections/neuron_morphology_and_coverage.csv): all 462 identities, original source metadata, graph metrics and missing-aware coverage.
- [Per-section ledger](selected462_ARM6_sections/connected_axon_sections.csv): all 47,044 sections, official target names, original endings/branches, endpoint-selected lengths and coverage.
- [Original transition edges](selected462_ARM6_sections/original_target_and_type_crossing_edges.csv), [native ending coverage](selected462_ARM6_sections/native_original_end_cube_coverage.csv) and [source bindings](selected462_ARM6_sections/source_swc_bindings.csv): exact identities, paths and hashes.
- [Compact summary](descriptive_census_summary.json), [producer provenance](selected462_ARM6_sections/census_provenance.json), [per-neuron reconciliation](selected462_ARM6_sections/per_neuron_reconciliation.json) and [independent readback](independent_census_readback.json).

All 462 endpoint counts and total child-type-2 lengths agree with the existing projection tables. All 201 native graph identities and prior coverage counts agree exactly. An independent reader checks all saved bindings and all-neuron node/ending/three-way length conservation, then checks ten actual source graphs using a sparse undirected connected-component oracle and parent-ID-set ending detection without importing the producer or its parser. Seven focused tests cover 80 seeded synthetic trees, compartment transitions, regional false-end protection, half ties, physical affine lengths, root-only graphs, missing coverage and no-clobber behavior. These checks validate software and representation, not biological anatomy.

Run the following from the repository root with the project environment. The producer refuses existing output destinations; use a fresh destination for reproduction. Local source graphs, atlas/reference files and the existing cube-name cache are required. No source images are downloaded or copied.

```powershell
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -X utf8 -B -m unittest discover -s notes/region_analysis_review_20261009/terminal_field_census_20261009 -p test_connected_axon_census.py -v
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -X utf8 -B -c "import sys; sys.path.insert(0,'notes/region_analysis_review_20261009/terminal_field_census_20261009'); import census_connected_axon_sections as c; c.build(c.HERE/'reproduced_selected462_ARM6_sections')"
```

The exact original node memberships are additionally retained locally in `selected462_ARM6_sections/section_original_node_membership.jsonl` (87,831,295 bytes), hash-bound by producer provenance. This reproducible node ledger is deliberately excluded from the compact public payload; publication preparation records its local-only path, hash and size. It duplicates no source coordinates and can be regenerated from the bound SWCs. All other outputs are compact evidence tables; no hundreds of figures, extra biological cohorts or duplicate map families were created.
