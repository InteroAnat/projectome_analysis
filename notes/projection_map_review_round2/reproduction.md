# Reproduce and inspect the projection maps

Run from the repository root on the workstation holding the original source reconstructions and atlas. A public Git clone alone lacks the privately retained SWCs, NIfTIs and local PDFs. Do not invent missing transforms. These steps start with the already declared NMT-coordinate exports; they do not reproduce or validate the unavailable original registration.

## 1. Pin the environment and sources

Record `git rev-parse HEAD` and `git status --short`. Preserve unrelated modifications. Use the projectome environment: Python 3.10.20, NumPy 1.26.4, nibabel 5.4.2 and pandas 2.3.3; the display also needs Matplotlib. Actual versions and script/source hashes are saved in the numerical receipt. The installed interpreter here is `C:/Users/laika_yan/miniconda3/envs/projectome/python.exe`. Do not install over an existing environment merely to adopt that name.

Required inputs:

| Input | Established path and meaning |
|---|---|
| MRI | `atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz` |
| Coverage mask | Same directory, `NMT_v2.1_sym_brainmask.nii.gz`; affects coverage/window, not endpoint trimming |
| ARM labels | Same directory, `ARM_in_NMT_v2.1_sym.nii.gz`; contains six actual hierarchy volumes |
| ARM names | `atlas/ARM_key_all.txt`; retain background/outside and key-range conflicts |
| Hemisphere mask | `supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz` in the same atlas directory |
| Main selection | `group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints/input_manifest.csv` (436 identities) |
| Additional selection | `group_analysis/evolution_20261008/endpoint_maps/additional_candidates_20261009/input_manifest.csv` (26 disjoint identities) |
| Full final ledger | `notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv` (462) |
| Reconstructions | Exact `SWCPath` and `SWCSHA256` from those manifests, not a filename-only lookup |

The grid is 256×312×200, 0.25 mm sampling, voxel volume 0.015625 mm³. Export XYZ divided by declared `IndexScaleUm=250;250;250` gives the current template voxel-centre convention. Do not silently add an origin shift, mirror coordinates, or reinterpret these as physical micrometres. Absolute historical source paths must resolve; a different workstation requires a separately logged, hash-verified path remapping, with immutable originals preserved.

## 2. Reinspect saved maps and reproduce the new display

The following procedure was exercised end to end for this delivery. Choose a destination that does not exist; the scripts refuse to overwrite outputs.

```powershell
./notes/projection_map_review_round2/reproduce_maps.ps1 -Output work/projection_maps_recheck
```

This writes `runs.json`, runs `inspect_projection_maps.py` across the five existing run directories, then runs `render_endpoint_markers.py`. The inspector verifies original source endpoint counts, all saved voxel relations and exact identity/animal membership. The renderer requires the matching successful receipt and hashes before drawing. Its `--slice-voxels X Y Z` argument accepts voxel indices, not millimetres; defaults are 56 200 87. Keep the default cuts when comparing these sheets with existing figures.

Direct commands, equivalent to the final two driver steps:

```powershell
python -X utf8 -B notes/projection_map_review_round2/inspect_projection_maps.py --output work/projection_maps_inspection
python -X utf8 -B notes/projection_map_review_round2/render_endpoint_markers.py --run group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints --inspection work/projection_maps_inspection/inspection_receipt.json --output work/projection_endpoint_markers
```

Read `inspection_receipt.json` and `all_map_value_checks.csv`. Inspect all four PNGs and `display_provenance.json`. Require matching hashes, 452 checked NIfTIs, 462 exact identities, all numerical checks passed and no unexpected source changes. The count/density/occupancy relation uses rtol 2e-7 and atol 1e-7; equal-animal voxel readback uses rtol 3e-7 and atol 1e-7, accommodating stored float32 values. Length integration uses rtol 2e-7 and atol 1e-6 mm. These tolerances address storage rounding, not anatomical accuracy.

## 3. Optional full numerical rebuild from the same sources

```powershell
./notes/projection_map_review_round2/reproduce_maps.ps1 -Output work/projection_maps_source_rebuild -Rebuild
```

This optional path is computationally expensive. Its CLI/driver structure was checked; a second complete numerical rebuild was not performed in this display review. The established independent full-edge/region readback and new saved-map source/voxel checks are the real-data evidence delivered here. Expect processing time and disk use to vary with the local environment; do not run over frozen directories.

The driver executes the following order, with each nonzero exit code stopping the run:

1. `build_endpoint_maps.py` on each preserved manifest: complete graph leaves; candidate counts and distinct-neuron occupancy; eligible denominator includes ends outside the field of view. Outputs include count/density/occupancy NIfTIs and direct ARM L1–L6 summaries. Kernel: `main_scripts/terminal_sites.py::terminal_site_summary`; atlas lookup: `main_scripts/endpoint_atlas.py`.
2. `build_projection_maps.py` on the same two manifests: child-type-2 axon edges, including labelled transitions; split at voxel faces; template length and length-density NIfTIs. Kernel: `main_scripts/projection_maps.py`. Average all selected neurons within each animal/source group, then contributing animals equally.
3. `review_endpoint_run.py` and `review_projection_run.py` on each newly built run. These read sources/saved maps independently of the corresponding producers. Read their receipts before proceeding.
4. `build_axon_end_branch_maps.py` on the full 462 ledger: original leaf to first full-graph bifurcation/root/nonaxon parent; flag final included transition edge; raster along trajectories. Kernel: `reconstructed_axon_end_branch_summary`; no leaf-centred length deposit. Outputs are 67 length NIfTIs plus six hierarchy matrices/workbook; density is derived without duplicate volumes.
5. `independent_end_branch_readback.py --run ... --receipt ...`: forward graph-selection and independent interval-raster oracle, all original selected edges, all six hierarchy cells and every map voxel. This independent check is required before end-branch interpretation.
6. `export_arm_projection_tables.py`: supply both endpoint runs, both axon runs and the same final ledger. Produce the six-level combined workbook, sparse regional measures, targets, neuron summary and map-to-region reconciliation. Never sum across hierarchy levels. Preserve known zeros versus NA and ARM-background/outside rows.
7. `inspect_projection_maps.py --runs-json ...` against the five fresh directories, then the additive endpoint-marker renderer. Compare sums/source identities/denominators, rather than demanding identical compressed-file hashes between reruns with different timestamps.

## 4. Focused checks, display review and acceptance

```powershell
Push-Location notes/projection_map_review_round2
python -X utf8 -B -m unittest test_map_inspection
Pop-Location
```

The five tests cover cropping-created false endpoints, nonaxon children, multiple ends sharing a voxel, corrupt normalization/occupancy, and exact single-plane marker placement. The driver itself also exposed and corrected relative-path handling during the end-to-end run. Inspect every final page for readable full ARM names, MRI contrast, matched cuts, legends, overlap and clipping. Marker size is display-only and dots may overlap; NIfTIs remain the quantitative source.

Passing these checks establishes the specified numerical and display contracts. It does not accept registration, terminal fields, boutons, synapses or fine soma anatomy. Add image-reviewed evidence separately without overwriting atlas/manual/coordinate provenance. Statistical inference requires animal-level replication and justified sampling/multiple-comparison assumptions; no such inferential map is generated here.

## 5. Sorting, archiving and publication

`sort_review_artifacts.py --output <fresh-folder>` rehashes the established local artifact catalog and writes file roles/dependencies before any move. `--archive-previews` is narrowly limited to this review's two private PDF-preview PNGs. Original sources and hash-bound historical derivatives retain their established paths. The archived developer/failed runs have explicit old/new hashes and are excluded from validation and publication. Use the current README links to locate active outputs.

`notes/region_analysis_review_20261009/prepare_publication.py` inventories the reviewed source/report/figure scope and scans credentials without staging. `verify_publication_index.py` checks exact staged bytes against that scope. Stage explicit path lists only; preserve unrelated work, private scientific binaries and local paper previews. After commit/push, verify the feature branch SHA with `git ls-remote`, and read back the default branch. Record the result in the publication note.
