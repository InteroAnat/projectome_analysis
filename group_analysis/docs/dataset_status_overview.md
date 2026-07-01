# Data progress track (insula multi-monkey)

**Manifest (edit by hand):** [`dataset_status_manifest.csv`](dataset_status_manifest.csv)

After editing the manifest, rebuild the integrated progress table and figure:

```bash
python group_analysis/scripts/build_data_progress.py
```

Legacy wrapper (same output): `python group_analysis/scripts/plot_dataset_status_overview.py`

Each run writes **timestamped** outputs and removes legacy unstamped files / old `dataset_status_*.png` figures.

---

## Module

| Piece | Path |
|-------|------|
| Package | `group_analysis/data_progress/` |
| CLI | `group_analysis/scripts/build_data_progress.py` |
| Manifest source | `docs/dataset_status_manifest.csv` |
| Combined insula counts | `combined/multi_monkey_INS_combined.xlsx` |
| Step1 auto-insula | latest `step1_results/{fmost}_*/tables/*_results_*.xlsx` |

---

## Outputs

| File | Description |
|------|-------------|
| `docs/data_progress_table_YYYYMMDD_HHMM.csv` | One row per plan monkey **plus a TOTAL row** — counts, distributions, 5 µm, pipeline stage, `generated_at` |
| `docs/figures/data_progress_table_YYYYMMDD_HHMM.png` | Summary figure for the same run |

Example: `data_progress_table_20260701_1405.csv` / `.png`

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
