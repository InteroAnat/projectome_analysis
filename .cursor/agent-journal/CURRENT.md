# Current handoff — projectome_analysis

**Updated:** 2026-07-01 (cleanup branch + data progress)

## [handoff] 当前焦点

- **分支:** `cleaner_projectome_scripts` — repo 清理 + data progress 模块；待 push GitHub
- **Data progress:** 编辑 `group_analysis/docs/dataset_status_manifest.csv` → `python group_analysis/scripts/build_data_progress.py`
- **输出:** `docs/data_progress_table_YYYYMMDD_HHMM.csv` + `docs/figures/data_progress_table_YYYYMMDD_HHMM.png`

## [scope] 相关路径

- `group_analysis/data_progress/` — table, plot, track, manifest, sources
- `group_analysis/docs/dataset_status_manifest.csv`
- `main_scripts/step3.2.run_brain_viz_meshRender.py`（brain viz，已保留）
- `.gitignore` — clutter 模式

## [verify]

- Build: `data_progress_table_20260701_1416.*` — insula 353/2400, ION 1205/1225, 5µm 1/8
- Git: `safe.directory=D:/projectome_analysis`（ownership 未在 OS 层修复）

## [next] 下一步

1. Merge / PR `cleaner_projectome_scripts` when ready
2. 797 fMOST 252790 — ION 上传或 manifest 更新
3. 可选 OS-level `takeown` + 移除 safe.directory

## Today's daily

- **`2026-07-01.md`** — cleanup branch, data progress timestamp outputs, legacy fig removal
