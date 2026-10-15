# Data progress track (insula multi-monkey)

The all-groups figure is `python tools/project_overview_plot.py`. That command calls `project_overview_plot`, which parses the live table and the local sources, then refreshes the table and figure in [`../data_progress/`](../data_progress/README.md). Usage is [tools/README.md](../../tools/README.md).

## Live table

The live table is the Tencent Docs workbook **macaque_fMOST**, tab **Progress Details** (`BB08J2`):

https://docs.qq.com/sheet/DQXFTS21ieUVGTEpJ?tab=BB08J2

`project_overview_plot` parses that tab for animal name, fMOST id, injection sites, and tracing text. If the opendoc response is not the cell grid, the run uses `group_analysis/evolution_20261008/overview_sheet_20261012/progress_details_BB08J2.tsv` and records `tsv_fallback`. DEB lab `data_status` (`Completed` or `Pending`) stays the Monkey Data word in [`monkey_experiment_data_summary.xlsx`](monkey_experiment_data_summary.xlsx). The live tracing text is a separate column.

**Manifest (edit by hand):** [`dataset_status_manifest.csv`](dataset_status_manifest.csv)

After editing the manifest, refresh the overview:

```bash
python tools/project_overview_plot.py
```

`python group_analysis/scripts/build_data_progress.py` calls the same function. The July timestamped CSV and PNG under `docs/` are older snapshots. This command does not delete them.

---

## Module

| Piece | Path |
|-------|------|
| Function | `group_analysis/project_overview/track.py` → `project_overview_plot` |
| Drawing | `group_analysis/project_overview/plot.py` → `plot_progress_table` |
| CLI | `tools/project_overview_plot.py` |
| Output folder | `group_analysis/data_progress/` |
| Live table | macaque_fMOST, tab Progress Details (`BB08J2`) |
| Manifest source | `docs/dataset_status_manifest.csv` |
| Combined insula counts | `combined/multi_monkey_INS_combined.xlsx` |
| Step1 auto-insula | latest `step1_results/{fmost}_*/tables/*_results_*.xlsx` |

---

## Outputs

| File | Description |
|------|-------------|
| `data_progress/data_progress_table.csv` | One row per kept animal **plus a TOTAL row** |
| `data_progress/data_progress_table.xlsx` | Same rows, injection plan, and totals. The PNG is drawn from this workbook. |
| `data_progress/data_progress_table.png` | Bars, DEB lab injection grid, and the all-groups table |
| `data_progress/provenance.json` | Loader paths and the live-table fetch |

Older example, not rewritten by the current command: `data_progress_table_20260701_1416.csv`.

---

## Table columns

| Column | Meaning |
|--------|---------|
| `generated_at` | Build timestamp (same on every row; repeated on TOTAL row) |
| `ion_n_traced` | Neurons on ION (`selectNeurons`) |
| `step1_auto_insula_n` | Atlas auto-insula labels in latest step1 |
| `insula_corrected_n` | Manifest hand-QC / recovery count |
| `insula_in_combined_n` | Rows in `multi_monkey_INS_combined.xlsx` |
| `insula_distribution_combined` | Subregion breakdown (e.g. `IAL:20 IAI:2`) |
| `insula_L_R_combined` | Left/right counts from combined table |
| `five_um_local` | 5 µm CH1 resample copied for Visual_toolkit low-res |
| `pipeline_stage_code` | Stages completed (0–6): ION → step1 → insula-QC → combined → 5um-local → complete |
| `pipeline_stage` | Next action (e.g. `next: combined`, `complete`, `pending (ION)`) |

---

## 797 note

ION API returns **0** SWCs for fMOST **252790** (`selectNeurons?id=252790` → `[]`). Manifest `analysis_n=92` is the experimental sheet target, not traced count. If INS neurons exist locally, upload to ION under **252790**, then refresh manifest and re-run build.

---

## Policy

Use **`multi_monkey_INS_combined.xlsx`** (unharmonized) + human soma QC for inferential work.
