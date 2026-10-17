# VisualToolkit change log

One consolidated history for the package (`toolkit.py`, `gui.py`, `sagittal.py`,
`dataset_config.py`, `visual_toolkit_config.yaml`). `VISUAL_TOOLKIT_GUIDE.md` is the operator
guide; per-module headers carry `Version:` / `Last updated:` / `Change history:` and
point here.

Entries are newest first. "Local/unreleased" means the change is in the working
tree but has not been committed or published; dates are Asia/Shanghai.

## Current module versions — 2026-10-17

| Module | Version | Role |
|--------|---------|------|
| `gui.py` | 1.9.0 | Tkinter operator window |
| `toolkit.py` | 1.9.0 | rendering, `REVIEW_SETTINGS`, acquisition |
| `sagittal.py` | 1.9.0 | processed-DS panel + raw-SWC resolver |
| `montage.py` | 1.9.0 | nine-sample DS00 five-view cohort montage |
| `dataset_config.py` | 1.9.0 | dataset links / operator check |
| `visual_toolkit_config.yaml` | schema 1 | per-sample DS00, 5 µm, SWC cache, GUI output root |
| `run_bulk_visual_sample.py`, `bulk_visual_multi_monkey.py` | 1.9.0 | bulk wrappers (outside the package) |

All modules move with the package version until a released tag exists; an
unreleased change never claims a release.

## 1.9.0 — 2026-10-17 (local/unreleased)

Package integration
- Moved the DS00 five-view cohort montage implementation into
  `Visual_toolkit/montage.py`; it is now available as
  `python -m Visual_toolkit.montage`. The previous
  `tools/ch00ds_5view_montage.py` path remains a compatibility wrapper.
- Added `--source-root` and `--workbook` path overrides while preserving the existing
  source-hash checks, output-overwrite refusal, five-view cameras and JSON receipt.
- Confirmed the processed-DS sagittal renderer remains package-owned in
  `Visual_toolkit/sagittal.py` and continues to back the GUI and bulk wrappers.
- Added an offline montage regression module covering camera basis, label CLI,
  streamed hashing and the legacy wrapper import.

Validation
- Package and wrapper syntax checks pass.
- The combined seven-module suite passes **74 tests**, including five offline montage
  checks for camera geometry, label CLI, streamed hashing and legacy-wrapper import.

## 1.8.0 — 2026-10-17 (local/unreleased)

Operator release
- Publishes the consolidated VisualToolkit package: config-backed datasets, shared SWC
  resolution, Soma/Context/Sagittal panels, receipt-based cache reuse, timestamped
  figures, process logging and isolated panel failures.
- Finalises the Tkinter reviewer UI with Plotting/Saving lanes, Render all, output-folder
  access, editable in-app YAML configuration with backups, a scrollable five-stage load
  guide, aligned preview-ribbon identity/filename and the outlined **Update figures**
  action.
- Clarifies that Soma acquisition reuses local cubes and fetches only missing cubes
  online; changed saves create new timestamped PNGs while the optional fixed-name Soma
  NIfTI is overwritten.
- Adds the required version header to the package demo entry point and aligns both bulk
  wrapper version headers with the package.

Validation
- The established six-module suite passes **69 tests**.
- Live GUI checks cover the updated stage guide and scrolling, ribbon alignment,
  **Update figures**, cached-set prompt, four-panel reload and the refreshed Soma note.

## 1.7.2 — 2026-10-17 (local/unreleased)

Save lane and figure tracking
- Added a visible **Trio PNG** save tick (default on) beside **Soma NIfTI**; the
  TrioPlot is written only when its save tick is on, otherwise it is preview-only and
  the temp file is removed.
- The Soma card now explains the block arithmetic again: 1 downloaded block =
  360×360×90 voxels (234×234×270 µm); the window **diameter** of 1.5 blocks is
  ≈351 µm wide × 270 µm deep and downloads only the blocks it touches; the legacy
  **radius** r = 1/2/3 has diameter 2r−1 = 1/3/5 blocks and downloads 1/27/125.
  The visible Soma-card note now contains only the block dimensions; the
  1.5-block diameter and legacy-radius sentences are removed from the GUI.
- **Load & Auto-Fill Soma** now uses the filled teal primary style (it was an outline
  button), so the identity action reads as an action rather than a field.
- Card titles use 11 pt bold in the deep navy header colour (previously muted 9 pt); the
  header subtitle reads "DEB lab · fMOST neuron reviewer"; the settings column is 430 px
  and the action buttons are wider, keeping the column at 801 px.

Layout and readability
- Input boxes stretch to fill their card (Sample/Neuron, X/Y/Z, Context width/height/
  depth, Save-to); the Load & Auto-Fill Soma button spans the card width.
- The Context card carries a compact dataset line (5 µm sections 5×5×3 µm, default
  8×8 mm × 30 µm crop, DS00 ~100×90×100 µm).
- The figure preview has a purple top ribbon and its Soma/Context/Sagittal/TrioPlot
  tabs are a segmented grid with 1-px divider lines.
- The sample · neuron identity moved from the top-right header chip into the preview
  ribbon, immediately left of the file name; the preview caption shows only the file
  name (the "w×h shown" text is gone). The white identity and filename share the same
  lowered baseline rather than sitting at different vertical positions.
- **Update figures** is an outlined button at the right of the preview tab row, visually
  separate from the segmented panel selector; clicking it updates the preview from
  cached panels through the receipt check.
- Cards are titled **Plotting** and **Saving**; the header has a **Config YAML** button
  that opens `main_scripts/Visual_toolkit/visual_toolkit_config.yaml`.
- The Sample card has a compact scrollable stage guide for the load/acquire/review
  chain: portal raw SWC → saved `resource/swc_raw/<sample>/` file → target soma
  X/Y/Z → cached 0.65 µm Soma cubes with online fallback for missing cubes → render,
  save and **Update figures**. It states that Load & Auto-Fill does not prefetch image
  cubes. A changed save creates new timestamped PNGs, keeps older PNGs, and updates the
  receipt; the fixed-name Soma NIfTI is overwritten when selected. The Context card says
  the source dataset names (`0.65 µm CH1 blocks · 5 µm CH1 resample · CH00DS N4`); the
  Saving card explains the receipt file and the PNG-versus-NIfTI save behavior.
- Each plot card now carries its own one-line source under the fields: Soma →
  `cached or fetched online` plus `resource/cubes/<sample>/high_res_http`; Context →
  the configured 5 µm resample folder tail (or "not configured").
- The header **Config YAML** button opens an in-app **editable** viewer of
  `main_scripts/Visual_toolkit/visual_toolkit_config.yaml` with Save/Reload. Save
  validates the YAML and schema, copies the previous file to `config_backups/`, keeps a
  pristine `visual_toolkit_config.default.yaml`, and writes atomically. The loader
  falls back to the legacy `datasets.yaml` name when the new file is absent.
- Render selected/render all check the receipt and cached files first: a matching set
  loads without re-rendering (status "Cached figures reused"), and a **Force re-render**
  tick redraws even when the cache matches. Update figures remains the explicit browse
  path with the re-render prompt.
- Entry borders use a deeper slate (#8fa3b2) so the fields read as inputs.

Bulk runners
- `native_ds_swc_path` now uses the same `resolve_swc_path` as the GUI, so bulk
  exports also resolve the legacy raw-SWC cache; 251637/002 renders the Sagittal panel
  from `resource/swc_raw` instead of failing.

Validation
- 69 tests pass (GUI request/actions, panel render, bulk decoupling, contrast,
  write safety, image loader).
- Live one-neuron checks: 252527/048 (all panels), 252385/025 (corrected 5 µm path),
  251637/002 (legacy raw SWC) complete; 252383/018 and 252718/142 report
  `partial=1` with the missing 5 µm source and still produce Soma + Sagittal.
- Hidden layout check: settings column 799 px inside 1280×980; a preview-only temp PNG
  is loaded, removed, and kept in memory; Reload prompt text is exercised by unit
  tests. Live visual check of the current-code window confirmed the white identity is
  immediately left of the filename on the same lowered baseline in the purple ribbon,
  **Update figures** has a visible outline, and clicking it updated the preview from four
  cached panels for 251637/003 without error. The stage-guide live check confirmed mouse
  wheel scrolling reaches its final save/update step.

## 1.7.1 — 2026-10-17 (local/unreleased)

GUI and source resolution
- Preview shows one panel at a time (Soma / Context / Sagittal tabs); the combined
  figure is the `TrioPlot` tab and is built only on request.
- **Render all** (F6) renders the three panels and always adds the TrioPlot; it is
  styled purple to separate it from the teal **Render selected** (F5).
- **Save selected** (default on) sits beside the render actions; when off, panels are
  preview-only, their temp PNGs are deleted after the preview loads, and no receipt or
  NIfTI is written. **Open output folder** sits to its right.
- **Reload figures** button plus an automatic scan of `<output_root>/<sample>_*` load
  existing panels without re-rendering. A per-folder `visual_gui_receipt.json` records
  coordinates, settings, role files and their hashes; reload checks coordinates, the
  settings key and the file hashes, then prompts "re-render?" when anything differs.
- Output root and auto-reload come from `visual_toolkit_config.yaml` (`visual_toolkit:` block);
  the GUI still exposes the folder per run.
- A missing raw SWC for the Sagittal panel triggers one portal fetch into
  `resource/swc_raw/`, then re-resolves; the Plot-lane hint names the neuron and
  the cached count instead of a sample-wide warning.
- The toolkit's fetch/download/render lines are mirrored in a **Process log** panel and
  in the status strip; the progress bar runs for the whole render.

Sources and package layout
- `sagittal.py`, `dataset_config.py` and the YAML config moved into this package;
  the toolkit, GUI and bulk wrappers import `VisualToolkit.sagittal` /
  `.dataset_config` (no copies under `tools/` or `config/`).
- `resolve_swc_path()` order: registered `swc_cache`, then
  `resource/swc_raw/<sample>/<neuron>.swc`; provenance records `swc.source`
  (`registered_cache` / `legacy_raw_cache`).
- Panel failures are isolated: one unavailable panel is reported and the others
  still render, save and preview.
- Figures are named `<sample>_<neuron>_<region>_<YYYYMMDD-HHMMSS>_<Role>_Plot.png`
  (role suffix last, so suffix readers keep working); the NIfTI name is unchanged.

Validation
- 68 tests at the time; a live F5 at 17:18 wrote timestamped Soma/Context/Sagittal
  PNGs, the Soma NIfTI and `visual_gui_receipt.json`, did not regenerate the TrioPlot
  with its tick off, and showed the fetch/download lines in the process log.

## 1.7.0 — 2026-10-17 (local/unreleased)

- Binavi-style two-column window: settings cards left, figure preview right, navy
  header, teal accent, pale borders.
- `Plot` lane (Soma / Context / Sagittal ticks) and `Save` lane (per-PNG ticks,
  Soma NIfTI, output folder; Context NIfTI kept out of the GUI).
- Default output moved from `resource/segmented_cubes/<sample>` to
  `work/visual_gui/<sample>_<YYYYMMDD>`, resolved from the module (not cwd).
- `plot_soma_block` / `plot_widefield_context` return `output_png` for the preview.

Validation
- 22 tests at the time; a live F5 run produced all three panels plus the TrioPlot, and
  the settings column measured 796 px inside the 1280×980 window.

## 1.6.0 — 2026-10-17 (local/unreleased)

- One target marker for all panels: cyan/turquoise (`#00e5ee`) box + cross on the
  exact mapped point, drawn by a single helper.
- `REVIEW_SETTINGS` as the shared settings source (marker, soma window, context,
  widefield gain/threshold, scale bars, figure sizes).
- Soma panel uses a soma-centred 1.5-cube window (≈351 µm) with descending centred
  steps and a recorded centre offset; resizable layout with a layout-derived
  minimum size.

Validation
- 63 tests; 252527 25/25 neurons at 1.5 cubes with ≤0.5 px centre offsets; a complete
  sample re-rendered byte-identically to the pre-fix output.

## 1.5.0

- Auto-load the default neuron (`003.swc`) on startup; silent coordinate fill;
  thread-safe status updates.

Validation not recorded in the original header; behaviour retained as the legacy
baseline and covered indirectly by the GUI request tests.

## 1.4.0

- Soma region and formatted coordinates in plot titles/filenames; optional cache
  directory for bulk processing.

Validation not recorded in the original header; behaviour retained as the legacy
baseline (titles/filenames only, no voxel changes).

## Earlier

- Pre-package GUI (`main_scripts/Visual_toolkit_gui.py`): Soma Plot / Wide-field
  Plot / Plot Both, fixed 700×700 window, output under
  `resource/segmented_cubes/<sample>`.
- Reference baseline for the v1.7.0 comparison; the fixed frame clipped the status bar
  once review actions were added, which v1.6.0/v1.7.0 fixed.
