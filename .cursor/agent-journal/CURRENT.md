# Current handoff — projectome_analysis

**Updated:** 2026-07-07 (252385 full visual rerun in progress)

## [handoff] 当前焦点

- **252385 bulk visual rerun:** INS+PrCO, grid_radius=2, hi+lo res, PNG+NIfTI, step1 tables → `cube_data_252385_{INS|PrCO}_20260707`
- **Visual_toolkit hires:** MIP + ae05802-style `_read_tiff`; requires **`imagecodecs`** in `projectome` env
- **5µm widefield:** SMB `5micron_datasets` for 251637, 252385
- **Output root:** `\\10.102.8.200\microscopy_data\fMOST\visual\`
- **分支:** `cleaner_projectome_scripts` — visual path edits pending commit

## [scope] Visual_toolkit hires — check these keys first

Before changing plot code when hires looks wrong or user asks GUI vs bulk / contrast / grid:

1. **`imagecodecs` in `projectome` env** — without it, HTTP cubes read as 360×360 not (90,360,360)
2. **Smoke one cube** — `tifffile.imread` shape must be `(90, 360, 360)`
3. **Plot mode** — canonical **MIP** (`plot_soma_block`); GUI and bulk share `Visual_toolkit.py`
4. **`grid_radius`** — 1 (GUI default, 1 cube) vs 2 (`step3.1` bulk, 3×3×3)
5. **Contrast** — 0.5–99.5% + gamma 0.5 (same everywhere)
6. **Paths** — low-res SMB `5micron_datasets`; output UNC `visual\`; not `W:\fMOST`

Full table: **`2026-07-07.md`** → session 11:51 → `[decision] Visual_toolkit hires`.

## [scope] 相关路径

- `main_scripts/Visual_toolkit.py`, `Visual_toolkit_gui.py`
- `group_analysis/scripts/run_bulk_visual_sample.py`, `bulk_visual_multi_monkey.py`
- `main_scripts/step3.1.bulk_visual_data.py` (GRID_RADIUS=2 reference)
- `group_analysis/docs/dataset_status_manifest.csv`

## [verify]

- Env: `C:\Users\laika_yan\miniconda3\envs\projectome\python.exe`
- `pip show imagecodecs` → 2025.3.30

## [next] 下一步

1. Confirm 252385 bulk run finished (50 neurons × hi+lo, grid=2)
2. Push `cleaner_projectome_scripts` after visual commit
3. Rebuild data progress (`build_data_progress.py`)

## Today's daily

- **`2026-07-07.md`** — 5µm SMB move + bulk visual + hires revert playbook
