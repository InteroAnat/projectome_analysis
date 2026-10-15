# Data progress outputs

This folder holds the files written by `project_overview_plot`. The parser lives in `group_analysis/project_overview/`. A run replaces the output files below and leaves this description in place.

```powershell
python tools/project_overview_plot.py
```

| File | What it is |
|------|------------|
| `data_progress_table.csv` | One row per kept animal, plus TOTAL. Refreshed by parsing. |
| `data_progress_table.xlsx` | Same rows, plus `injection_plan`, `selected_neurons`, `totals`, legacy L3/L6 `Projection_Strength_*` sheets (462-row order; 306 neurons with harmonized combined data), and `ARM_regional_measures` (L1–L6 long table for all 462). The PNG is drawn from the `progress` sheet only. |
| `data_progress_table.png` | Bars, DEB lab injection grid, and the all-groups table. |
| `selected_neuron_potential_insula.csv` | One row per Selected neuron (462) with the position-rule check. Same rows as the `selected_neurons` sheet. |
| `provenance.json` | Loader paths, reality-bar numbers, live-table fetch, and the potential-insula rule with counts. |

## Live table

The live table is the Tencent Docs workbook **macaque_fMOST**, tab **Progress Details** (`BB08J2`):

https://docs.qq.com/sheet/DQXFTS21ieUVGTEpJ?tab=BB08J2

`project_overview_plot` parses that tab for animal name, fMOST id, injection sites, and tracing text. The request is the sheet opendoc URL. If the response is not the cell grid, the run uses the saved extract `group_analysis/evolution_20261008/overview_sheet_20261012/progress_details_BB08J2.tsv` and records `tsv_fallback` in `provenance.json`. A missing snapshot raises. The function does not invent animals.

DEB lab `data_status` (`Completed` or `Pending`) is the word in the Monkey Data workbook `group_analysis/docs/monkey_experiment_data_summary.xlsx`. The live tab keeps its own tracing text in `sheet_tracing_status`. Completed cells are `#C6EFCE`. Pending cells are `#B8CCE4`.

`--offline` skips the live table and does not add Other lab. The all-groups figure is the command without `--offline`.

## Full neuron region table

The newest full neuron region table is [all_neuron_review_manifest_distance_corrected.csv](../evolution_20261008/classification/coarse_insula_review_20261009/distance_priority_v2_20261009/all_neuron_review_manifest_distance_corrected.csv): 8,746 neurons, every sample, with portal region, soma atlas label, folded-box screen, and the distance to the nearest same-sample insula neuron. The 2026-10-12 recount (`per_animal_insula_summary.csv`) and the extra-dataset screen read this file. Selected is the 462-neuron ledger [neuron_summary.csv](../../notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv). Every Selected neuron is in the full table.

## Count columns

| Column | Meaning |
|--------|---------|
| Screened | Every neuron of that monkey in the full region table, all regions. The figure header is `Screened (monkey all region)`. |
| Selected | Neurons of that animal in the 462 ledger. |
| ARM insula | Soma in an ARM insula label. On the eight, this is the atlas part of Selected. |
| Added | Selected minus ARM insula. |
| Potential insula (auto rule based) | The part of Added whose soma is within 2 mm of an ARM insula voxel. Not a human label. |
| Selected by (`selected_by_reason`) | Count of each reason the neurons were selected, from the ledger `EvidenceStratum`. |

ARM insula rule: the soma is the neuron's own SWC root (`own_root_xyz_um`, NMT index = µm / 250), looked up on `atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz` level 6. Insula labels are Ia/Id, Iai, Ial, Iam/Iapm, Iapl, Ig, and Pi on both sides. The distance is a Euclidean distance transform at 0.25 mm. The check is applied to every added neuron, whatever its label (unknown, G, SII, and so on). The folded 251637 box and the same-sample anchor distance stay in the per-neuron file as reference columns.

Selection reasons:

| Reason | Short | Meaning |
|--------|-------|---------|
| HumanINS | Henry | Henry visual insula annotation |
| AtlasINS | Atlas | ARM atlas insula label |
| Candidate | Box | Spatial candidate (folded 251637 box) |
| G_exploratory | G | Gustatory G label, exploratory |
| OriginCandidate | Origin | Insula at one soma-origin lookup only |
| NearINS | Near | Non-insula label, same-sample insula neuron within 2 mm |

The per-neuron file gives `selection_reason`, `selection_reason_text`, `soma_ARM_label`, and `ARM_insula_distance_mm` for each row. 747/250432 is not in the 462, so Added, Potential, and Selected by stay blank on that row; its Spatial only column is the box screen.

Run with the `projectome` environment. The ARM check needs nibabel and scipy:

```powershell
conda run -n projectome python tools/project_overview_plot.py
```

Counts come from `load_analysis_neuron_table` (the 462-neuron ledger plus `per_animal_insula_summary.csv`). This folder is not anatomical acceptance.

## Multi-level projection strength

`project_overview_plot` copies into the workbook:

| Sheet | Source | Coverage |
|-------|--------|----------|
| `Summary`, `Projection_Strength_*` (L6 ipsi/contra, L3 ipsi/contra) | [multi_monkey_INS_combined.xlsx](../combined/multi_monkey_INS_combined.xlsx) | 306 Selected neurons present in the harmonized combined table; remaining Selected rows are blank in these sheets |
| `ARM_regional_measures` | [per_neuron_regional_measures.csv](../../notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/per_neuron_regional_measures.csv) | All 462 Selected neurons, levels 1–6 (`CandidateEndpointCount`, `AxonTemplateLengthMm`) |

Full six-level matrices and group means remain in [arm_projection_hierarchy_tables.xlsx](../../notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/arm_projection_hierarchy_tables.xlsx). Legacy strength uses log10(retained length + 1); ARM rows use the 2026-10-09 exporter contract (see hierarchy README).
