# Reconstructed axon end-branches in the unchanged ARM ledger

Agent: Codex | Date: 2026-10-09 (Asia/Shanghai)

This is an additional descriptive measure on the **same 462 neurons**, with preserved source classifications and all eight animals. It complements original axon-end counts and whole-axon length. Its name describes graph geometry; it is not a reviewed terminal-arbor, bouton or synapse map.

## Definition and literature basis

For each **original non-root axon ending in the complete SWC**, follow the original axon chain upstream to the first original full-graph bifurcation, root or non-axon parent. Include and flag the final child-axon-labelled transition edge, then stop. The whole graph determines children and bifurcations, including non-axon children. A cropped regional boundary cannot create a new ending. Chains do not share selected edges. This differs from the older QC descriptor that can traverse intervening non-axon compartments.

Selected edges are split at reference voxel faces **along their trajectories**, measured in template-space millimetres. Entire branch length is never deposited at its ending voxel. Sparse per-neuron accumulation is reused for all six actual ARM volumes and animal maps, avoiding six graph reruns. Native tissue length, physical orientation and accepted fine anatomy are not inferred from declared NMT exports.

The [local full-Methods review](../literature_method_update_20261009/gao_local_methods.md) binds verified Zotero metadata and actual PDF pages. Gao et al. (2022), *Nature Neuroscience* 25:515–529, [doi:10.1038/s41593-022-01041-5](https://doi.org/10.1038/s41593-022-01041-5), supports tracing-completion/image-QC distinctions. Gao et al. (2023), 26:1111–1126, [doi:10.1038/s41593-023-01339-y](https://doi.org/10.1038/s41593-023-01339-y), supports regional axon length as a separate projection-allocation measure. Gou et al. (2025), *Cell* 188:3806–3822.e1–e9, [doi:10.1016/j.cell.2025.06.005](https://doi.org/10.1016/j.cell.2025.06.005), motivates original terminal-path geometry but uses additional arbor segmentation and tissue context. Our explicit chain rule is an **adaptation**, not reproduction of their GNN arbor classifier. The [local Liu review](../literature_method_update_20261009/liu_local_methods.md) explains why image-derived varicosities and dense-subtree arbors cannot be obtained from leaves alone. No mouse distance/radius threshold or smoothing parameter is borrowed.

An unfinished long unbranched shaft can qualify under this rule. A field with many terminal branches can also have important upstream arbor edges excluded. The measure narrows the selected graph geometry; it does not isolate biological terminal fields or establish tracing completion.

## Numerical outputs and denominators

The [source-bound run](selected462_ARM/run_provenance.json) records **429 eligible neurons and 33 NA**, 75,625 original axon endings, 3,992,993 selected original edges and **36,651.459961 template mm**. Of that total, 36,646.493253 mm lies inside the reference and 4.966708 mm outside. Thirty-five included final transition edges are explicitly flagged. These are descriptive sums, not tissue volumes, terminal prevalence or statistical strength estimates.

There are **67 NIfTIs**: 50 eligible animal/source means and 17 source-group means. Each map stores **template mm per end-eligible neuron per voxel**. Density is that value divided by reference voxel volume; duplicate density volumes are unnecessary. Animal/source maps average eligible neurons only. Group maps average contributing animals equally. Groups with no eligible neuron contribute no zero map. Valid zero-length or all-outside chains remain computable; no eligible ending gives a missing matrix row, not absent biological innervation.

The [six-level workbook](selected462_ARM/arm_axon_end_branch_hierarchy_tables.xlsx) contains neuron QC, the official target dictionary and separate ARM L1–L6 length matrices. Matching CSVs are named `ARM_L{level}_axon_end_branch_length_mm.csv`. Each level independently conserves total selected length, including explicit atlas background and outside-reference columns. Hierarchy levels must not be summed. All 52 unassigned source cases remain in the same ledger and tables; soma identity is not determined by an axon's target.

Source images, SWCs and numerical NIfTIs remain local, with published hashes and reproduction commands. The [427-test live-suite receipt](live_suite_20261009.json) validates software contracts. Independent real-data readback, profile sensitivity and matched-slice displays are delivered separately; their exact validation scope is recorded in their own receipts.

The [independent readback](independent_end_branch_readback.json) checks every selected original edge, every numeric/NA cell at all six levels and every voxel in all animal/group maps. The [source-relative sensitivity](profile_sensitivity/README.md) has 428 mapped-positive profiles but no balanced k2–8 partition; the 2-versus-426 diagnostic split is animal-dependent. The [matched MRI sheets](matched_slices/display_provenance.json) show 15 official named parcels, with the same X56/Y200/Z87 cuts as existing maps and a shared contrast window. No MIP, smoothing or fine anatomical acceptance is implied.

To repeat validation, profiles and displays after a fresh map build:

```powershell
python -X utf8 -B notes/region_analysis_review_20261009/axon_end_branches_20261009/independent_end_branch_readback.py --run <fresh-end-branch-directory> --receipt <fresh-readback.json>
python -X utf8 -B notes/region_analysis_review_20261009/axon_end_branches_20261009/analyse_end_branch_profiles.py --run <fresh-end-branch-directory> --review <fresh-readback.json> --prior-relative notes/region_analysis_review_20261009/clustering_20261009/ARM6_source_relative_sensitivity --output <fresh-profile-directory>
python -X utf8 -B notes/region_analysis_review_20261009/axon_end_branches_20261009/render_end_branch_slices.py --run <fresh-end-branch-directory> --readback <fresh-readback.json> --brain-mask atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_brainmask.nii.gz --output <fresh-figure-directory>
```

## Verified boundary repair

The additive [checkpoint readback](checkpoint_readback.json) freshly verifies 126 frozen file bindings, including all 67 map files, producer/independent-reader code, hierarchy/profile artifacts and displayed figures. It does not rerun the scientific analysis or confer anatomical acceptance. Reproduce this byte check with `verify_checkpoint.py --receipt <fresh-receipt.json>`; existing receipts are never overwritten.

An independent regression exposed a floating-point half-open-bin defect: adding 0.5 before flooring can move the representable float immediately below a voxel face into the upper bin. The common geometry helper now compares the fractional part with 0.5. Both whole-axon fast and crossing-edge paths are covered.

The [complete real-source impact audit](boundary_impact/README.md) checks all **7,828,761 original axon edges**, including every one of **878,039 clipped positive midpoint intervals**. Old and corrected allocations are identical on these 462 sources: zero affected edges or neurons. Existing whole-axon maps therefore require no numerical regeneration for this fix. Historical code bytes and results remain bound to their original snapshots. Synthetic failure detection does not establish an observed anatomical error.

## Reproduction

Use the existing projectome Python environment and a fresh output directory:

```powershell
python -X utf8 -B group_analysis/scripts/build_axon_end_branch_maps.py `
  --manifest notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv `
  --reference atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz `
  --atlas atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz `
  --atlas-key atlas/ARM_key_all.txt `
  --output <fresh-end-branch-directory>
```

The pure kernel is `main_scripts/terminal_sites.py::reconstructed_axon_end_branch_summary`. Focused tests include an independent forward graph-selection oracle on 128 seeded trees, every-cell interval intersections, compartment transitions, false cropped ends, densification, affine units, outside coverage, known zero versus NA and unequal animal denominators. Original endpoint calculations and established whole-axon definitions are preserved.
