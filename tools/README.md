# Tools

Small commands that sit next to the analysis packages. The drawing and table code stays in those packages. This folder is the place to run them.

| Tool | Command | What it writes |
|------|---------|----------------|
| Project overview, all cohort groups | `python tools/project_overview_plot.py` | `group_analysis/data_progress/`: CSV, xlsx, PNG, `provenance.json` |
| CH00DS NRRD to LPS NIfTI | `python tools/nrrd_to_orient_lps.py` | One cardinal NIfTI. See the script docstring. |
| NMT flatmap | `julia --project=tools/nmt_flatmap_pipeline` | Depth model, flatmap, UVW. See [nmt_flatmap_pipeline/README.md](nmt_flatmap_pipeline/README.md). |

## Project overview

`tools/project_overview_plot.py` is the command. It calls `project_overview_plot` in `group_analysis/project_overview/track.py`. That function parses the live table and the local sources, saves a new table, then calls `plot_progress_table` in `group_analysis/project_overview/plot.py`. One run covers every group. There is no separate plot per lab.

From the repo root, with the `projectome` environment:

```powershell
python tools/project_overview_plot.py
```

The default output folder is `group_analysis/data_progress`. That folder is only the outputs of this command. Pass `--output <folder>` to write somewhere else. The command deletes `data_progress_table.csv`, `data_progress_table.xlsx`, `data_progress_table.png`, and `provenance.json` in the chosen folder before writing. It refuses `group_analysis/docs`, `insula_summary_20261012`, and `inventory/legacy_overview_20261009`.

### Live table

The live table is the Tencent Docs workbook **macaque_fMOST**, tab **Progress Details** (`BB08J2`):

https://docs.qq.com/sheet/DQXFTS21ieUVGTEpJ?tab=BB08J2

`project_overview_plot` parses that tab for animal name, fMOST id, injection sites, and tracing text. The request is the sheet opendoc URL. If the response is not the cell grid, the run uses the saved extract `group_analysis/evolution_20261008/overview_sheet_20261012/progress_details_BB08J2.tsv` and records `tsv_fallback` in `provenance.json`. A missing snapshot raises. The function does not invent animals.

DEB lab `data_status` (`Completed` or `Pending`) is the word in Monkey Data, `group_analysis/docs/monkey_experiment_data_summary.xlsx`. The live tab's tracing text stays in `sheet_tracing_status`. Completed is `#C6EFCE`. Pending is `#B8CCE4`.

`--offline` skips the live table and does not add Other lab. The all-groups figure is the command without `--offline`.

### Who is on the figure

An animal is kept when `ARM_insula_n` > 0 or `Henry_coarse_INS_n` > 0. Spatial-only is not enough, so those animals are listed under `dropped_no_insula_neuron` and are not drawn.

| Group | Who | Where they appear |
|-------|-----|-------------------|
| DEB lab | 936/251637, 605/252383, 900/252718, 945/252527, 797/252790, 948/252384, 331/252385, 631/252714 | Table, plan-vs-reality bars, cohort-size bars, injection grid |
| Other lab | Animals outside those eight that still have an insula neuron. 747/250432 is the current row (screened 234, ARM insula 8). | Table only. Not on the bars or the injection grid. Selected stays blank because 747 is not in the 462 ledger. |
| Table total | Sum of the visible rows, including Other lab | Bottom row. `ARM and spatial` is an intersection already inside ARM insula, so that total cell is an em dash. |

The last all-groups run (2026-10-12) drew 9 animals: table total screened 1732, ARM insula 309, Selected 462. The DEB lab bars stay on the eight only: screened 1498, ARM insula 301, ION expected 2429, insula cell target 2400. Those bars are not a second group plot.

Counts come from `load_analysis_neuron_table` (the 462-neuron ledger plus `per_animal_insula_summary.csv`). The 5 µm column is Yes only when `X:\fMOST\5micron_datasets` has that sample folder. This figure is not anatomical acceptance.

### Outputs

Written under `group_analysis/data_progress/` unless `--output` names another folder. See [data_progress/README.md](../group_analysis/data_progress/README.md).

| File | Content |
|------|---------|
| `data_progress_table.csv` | One row per kept animal, plus TOTAL |
| `data_progress_table.xlsx` | Same rows, an `injection_plan` sheet, and a `totals` sheet. The PNG is redrawn from this workbook. |
| `data_progress_table.png` | Bars, DEB lab injection grid, and the all-groups table |
| `provenance.json` | Loader paths, reality-bar numbers, dropped animals, live-table fetch |

`python group_analysis/scripts/build_data_progress.py` is the same function. Use `tools/project_overview_plot.py` for the all-groups figure.
