# Data progress outputs

This folder holds the files written by `project_overview_plot`. The parser lives in `group_analysis/project_overview/`. A run replaces the four output files below and leaves this description in place.

```powershell
python tools/project_overview_plot.py
```

| File | What it is |
|------|------------|
| `data_progress_table.csv` | One row per kept animal, plus TOTAL. Refreshed by parsing. |
| `data_progress_table.xlsx` | Same rows, plus `injection_plan` and `totals`. The PNG is drawn from this workbook. |
| `data_progress_table.png` | Bars, DEB lab injection grid, and the all-groups table. |
| `provenance.json` | Loader paths, reality-bar numbers, and the live-table fetch. |

## Live table

The live table is the Tencent Docs workbook **macaque_fMOST**, tab **Progress Details** (`BB08J2`):

https://docs.qq.com/sheet/DQXFTS21ieUVGTEpJ?tab=BB08J2

`project_overview_plot` parses that tab for animal name, fMOST id, injection sites, and tracing text. The request is the sheet opendoc URL. If the response is not the cell grid, the run uses the saved extract `group_analysis/evolution_20261008/overview_sheet_20261012/progress_details_BB08J2.tsv` and records `tsv_fallback` in `provenance.json`. A missing snapshot raises. The function does not invent animals.

DEB lab `data_status` (`Completed` or `Pending`) is the word in the Monkey Data workbook `group_analysis/docs/monkey_experiment_data_summary.xlsx`. The live tab keeps its own tracing text in `sheet_tracing_status`. Completed cells are `#C6EFCE`. Pending cells are `#B8CCE4`.

`--offline` skips the live table and does not add Other lab. The all-groups figure is the command without `--offline`.

Counts come from `load_analysis_neuron_table` (the 462-neuron ledger plus `per_animal_insula_summary.csv`). This folder is not anatomical acceptance.
