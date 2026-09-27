# Code review — insula projectome analysis

**Date:** 2026-09-26  
**Branch:** `test/liu-ven-fmost-bridge`  
**Scope:** `group_analysis/scripts`, `group_analysis/R_analysis`, `main_scripts` step pipeline + `region_analysis/`, `neuron-vis/neuronVis/IONData.py`, light pass on `gou_julia_scripts/`  
**Mode:** READ-ONLY (this file is the only new write)  
**Static checks:** `python -m py_compile` OK on priority `.py` (env `deb_fmri_pipeline`); `pyflakes` not installed; `Rscript` not on PATH (R syntax not parsed)

**Canonical data (per plan):** `group_analysis/combined/` (harmonized preferred). Do not treat `251637_results.xlsx` / legacy `neuron_tables/` as primary for multi-monkey L/R.

**NMT voxel note (verified):** `neuro_tracer.py` uses `reso = 250` µm → `x_nii = x_phys / 250` (0.25 mm voxels). Midline `x = 32000` µm ↔ voxel ~128. Lengths in `region_analysis_per_neuron._distance` are in **voxel units**, not µm (ratios cancel for proportion strength).

---

## High

| File:line | Issue | Why it matters | Concrete fix |
|-----------|--------|----------------|--------------|
| `group_analysis/R_analysis/combined_lr_primary_analysis.R:247-261` | Receiver ranking uses `best_q = pmin(p_presence_BH, p_pres_BH, p_magnitude_BH, p_mag_BH)` then `min(best_q)` across strata/families when summarising by `target`. | Violates locked BH rule (`notes/whole_insula_lr_continuation_plan_v2.md`: BH within stratum × analysis family, **never** min across families). Inflates apparent significance of asymmetric receivers; can pick hubs vs intra-insula family opportunistically. | Rank within one family at a time; report separate presence-BH and magnitude-BH columns; never `pmin` across families. Prefer `v2_combined_primary_pipeline.R` F6 pattern (after fixing its own `best_q` display — see next). |
| `group_analysis/R_analysis/v2_combined_primary_pipeline.R:1089-1110` | BH itself is correctly `group_by(stratum, hier, family)`, but ranking uses `best_q = pmin(p_pres_BH, p_mag_BH)` and `top_targets` filters `best_q < 0.05`. | Same BH-family violation for the **receiver-rank** endpoint the plan called out (§6). Manuscript “top asymmetric receivers” can be driven by whichever of presence vs magnitude is smaller. | Drop `best_q` for discovery; require a predeclared primary family (e.g. magnitude) or dual-report with family-specific FDR. Do not select targets by `pmin`. |
| `group_analysis/R_analysis/multi_monkey_lr_analysis.R:175-181`, `functional_hubs_L6.R:151-152`, `functional_hubs_analysis.R:144`, `intra_insula_connectivity.R:291-292` | Stratum `IDD5_plus_IDM` (and cousins) = `SampleID == "251637" & Region %in% c("IDD5","IDM")`. Inputs are **non-harmonized** `multi_monkey_INS_combined.xlsx`. | Plan/v2 primary stratum is `IDD5_plus_IDM_balanced` across monkeys with `n_L=81, n_R=68` on harmonized table. Older scripts understate n and exclude new-monkey keepers → wrong L/R inference for headline claims. | Point all primary R scripts at `multi_monkey_INS_combined_harmonized.xlsx`; redefine stratum as `Region %in% c("IDD5","IDM")` (all samples); assert `n_L/n_R` match context table; deprecate or clearly label 251637-only strata as sensitivity. |
| `group_analysis/scripts/04_refine_soma_region_by_coords.py:51-52,155-159` + `03b_relaxed_scan.py:96-122` | `PAD_TOL = 2.0` / pad loop labeled **mm**, but applied to `Soma_NII_*` (voxel indices; bbox X ~48–212). At 0.25 mm/voxel, pad=2 ≈ **0.5 mm**, pad=10 ≈ **2.5 mm**. | Recovery yield curves and Phase-4 rationale (“pad=2 mm”) mis-state physical tolerance; may under-recover or mis-document registration slack. | Rename to `PAD_TOL_VOX`; convert intended mm → voxels (`mm / 0.25`); rewrite comments and `relaxed_yield_curve.csv` column to `pad_vox` / `pad_mm`; re-run recovery if physical 2 mm was intended (`PAD=8`). |
| `neuron-vis/neuronVis/IONData.py:25-28,38-41,67-70,118-121,138-141,215-218,251-256` | Infinite `while status_code != 200` / recursive retry / `while leng==1` with **no timeout**, hard-coded `10.10.48.110` and `10.10.31.31`. | Any down/missing SWC hangs pipelines (step1, FNT, bulk visual). Lab-only IPs break off-network use. | Bounded retries + `requests.get(..., timeout=…)`; max attempts; env/config for hosts; keep `bulk_visual_multi_monkey.safe_fetch_raw_swc` as the pattern and route all fetches through it. |
| `main_scripts/step3.1.bulk_visual_data.py:40-47` | Defaults: hard-coded sample `251637`, `INPUT_FILE` under `neuron_tables/`, `PARENT_OUTPUT_DIR = …/Downloads/bulk_visual`. | Easy to regenerate QC into Downloads from stale tables; diverges from `group_analysis/scripts/run_bulk_visual_sample.py` (share path + combined/harmonized). | Make CLI-required args (sample, input table, out dir); default input to `group_analysis/combined/…_harmonized.xlsx`; remove Downloads default. |

---

## Medium

| File:line | Issue | Why it matters | Concrete fix |
|-----------|--------|----------------|--------------|
| `group_analysis/R_analysis/v2_combined_primary_pipeline.R:614-634` (F3A) | Plan: grey cells with `n_source < 10`; edges only if `n≥3`. Code sets `inferential_row` / `edge_above_thr` but F3A still fills all rows (incl. IDV n=3) without grey-out. | Descriptive-only strata can be read as inferential heatmaps. | `aes(alpha = inferential_row)` or `fill = ifelse(n_total<10, NA, …)`; blank text when `!edge_above_thr`. |
| `group_analysis/R_analysis/improved_panel_figures.R:12`, `combined_lr_primary_analysis.R:27`, hubs/intra scripts | Still load **non-harmonized** combined workbook while v2 uses harmonized. | Label provenance / IDD5:IDM counts differ; figures disagree with v2 manuscript path. | Single `COMBINED_XLSX` constant → harmonized; fail if file missing. |
| `group_analysis/scripts/05a_build_combined_table.py:78-80` | `NeuronID.isin(keep_neuron_ids)` without normalising `"007.swc"` vs `"7"` / int. | Silent row drop on dtype mismatch (VEN IDs 007,026,… are zero-padded). | Canonicalise: `f"{int(stem):03d}.swc"` everywhere (same as `06a_merge_henry_layers._neuron_id_from_swc`). |
| `group_analysis/scripts/06a_merge_henry_layers.py:151` | `SampleID == 251637` (int) vs possible string `"251637"` from openpyxl. | If Excel stores text, **zero** Henry merges with no hard fail (only print counts). | `str(cell).strip() == "251637"`; assert `n_merged > 0` or `n_merged == expected`. |
| `group_analysis/scripts/05c_run_global_fnt.py:144-147` vs `05a:80` | Table `NeuronUID` uses `::`; FNT UID uses `_` + strip `.swc`. Mantel rebuilds `NeuronFnt` — OK if careful; easy to join wrong key. | Silent empty Mantel intersect. | Document dual keys; add assert `length(common_ids)`; optional store both columns in Summary. |
| `main_scripts/step2_fnt-dist_pipeline.py:160-163` (and twin in `fnt-dist_pipeline*.py`) | Docstring says reflection “into the **right**”; math `x' = 64000 - x` for `x > 32000` folds **right→left** (same as `05c`). | Operator may reverse mirror “to match docs”. | Fix docstring to “right→left fold for FNT”; keep math. |
| `main_scripts/fnt_dist_clustering.py:29` | `TYPE_FILE = …/main_scripts/neuron_tables/251637_results.xlsx` (stale). | Clustering/type labels from obsolete workbook. | Point to `group_analysis/combined/` or timestamped step1 under `neuron_tables_new/`. |
| `gou_julia_scripts/gou_flatmap_minimal.jl:12-14` | Flatmap uses **NMT v2.0** + `neuron_tables/251637_INS_HE_inferred.xlsx`; step1/preproc use **NMT v2.1** + `neuron_tables_new` / combined. | Soma UV overlays can disagree with BIDS/projectome analysis space. | Prefer v2.1 surfaces if available; soma from harmonized Summary; document version pin if v2.0 is intentional Gou parity. |
| `group_analysis/scripts/05c_run_global_fnt.py:154-207` | Serial per-neuron fetch + FNT; resume only via existing files; no parallel pool. | ~300 neurons wall-clock dominated by network + FNT. | ProcessPool for local fnt-from-swc/decimate; keep ION fetch rate-limited; always skip-existing (already partial). |
| `group_analysis/R_analysis/v2_combined_primary_pipeline.R:75-84` | Cliff’s δ nested Python-style loops O(\|L\|·\|R\|) per cell. | Slow on large strata / many targets (F2/F6). | Vectorise (`outer` / `rowSums`) or package `effsize`/`rcompanion`. |
| `main_scripts/region_analysis.py:167-170` + laterality | Per-row `df.apply` for terminal laterality lists. | Slow on large samples; not wrong. | Vectorise side from prefixes; build ipsi/contra dicts in C/NumPy path. |
| `group_analysis/scripts/02_run_step1_multi.py:107-110` | Broad `except Exception` continues other samples. | One atlas/ION failure can leave incomplete cohort without failing CI. | Fail-fast flag; write error to manifest and non-zero exit if any sample fails. |
| Reproducibility (R suite) | `set.seed(42)` present in several scripts; **no** `sessionInfo()` / package versions written beside outputs; figures overwrite fixed paths. | Hard to audit what produced a PNG months later. | Write `sessionInfo()` + git SHA + input SHA256 into `outputs/.../run_meta.txt` each run; stamp dirs with date. |

---

## Low

| File:line | Issue | Why it matters | Concrete fix |
|-----------|--------|----------------|--------------|
| `insula_label_set.py` vs `06_harmonize` vs R `src_levels` | Vocab mix: atlas `Ial`/`Ig`/`Ia/Id`, manual `IAL`/`IDD5`, R sometimes `IA/ID`/`IG`. | Confusion in filters; usually uppercased in Python. | Export one YAML/CSV of allowed labels; R reads it. |
| `bulk_visual_multi_monkey.py:97-98,221-222` | Bare `except: pass` on cache read / cleanup. | Masks corrupt SWC cache. | Log and delete bad cache file. |
| `volume_tools.py:42,170` | Empty `pass` in FNTCube init paths. | Dead stubs. | Remove or implement. |
| `IONData.py:10-13` | References `inspect` without import when `__file__` missing. | Rare crash in embedded contexts. | `import inspect` or drop branch. |
| `04_refine…:128` | `summary.iterrows()` for recovery. | Fine at n~few hundred; pattern for larger cohorts. | Vectorised bbox masks. |
| Julia `insula_lr_mirror_flatmap.jl` | Hard-coded `D:\…` includes. | Portability. | `PROJECT_ROOT` env / `@__DIR__`. |

---

## Top 10 fixes (order)

1. **Stop `pmin` across BH families** in `combined_lr_primary_analysis.R` and v2 F6 ranking (`v2_combined_primary_pipeline.R`).  
2. **Retarget primary R analyses** to harmonized workbook + `IDD5_plus_IDM_balanced` (n_L=81, n_R=68); treat 251637-only as sensitivity.  
3. **Fix PAD units** in Phase 3b/4 (vox vs mm); re-document and re-run recovery if 2 mm was intended.  
4. **Hard timeouts + retry caps** in `IONData.py`; centralise hosts in config.  
5. **Canonical NeuronID** (`{int:03d}.swc`) on all joins (05a, bulk, Henry, FNT).  
6. **Assert SampleID coercion** in `06a_merge_henry_layers.py`; fail if merge count is 0.  
7. **Grey `n_source<10` / blank `n<3` edges** on F3 (and any Gou-style heatmap still missing it).  
8. **Retire Downloads / stale `251637_results.xlsx` defaults** in step3.1 and `fnt_dist_clustering.py`.  
9. **Align Julia flatmap** atlas (v2.0 vs v2.1) and soma table with combined/harmonized.  
10. **Parallelise + skip-existing** FNT stage (05c); write run metadata (seed, SHA, `sessionInfo`).

---

## Refactor suggestions (L0→L3)

Keep **step scripts as editable entry points**; do not fork second pipelines under `group_analysis` that reimplement step1/FNT/bulk.

| Layer | Role | Concrete moves |
|-------|------|----------------|
| **L0** | Pure actuators | `LateralityParser`, SWC mirror (`x'=64000-x`), NeuronID canonicaliser, atlas label strip, ION fetch-with-timeout, `fnt-from-swc` / `fnt-decimate` wrappers. |
| **L1** | Domain actuators | “Fetch NMT SWC”, “mirror for FNT”, “bbox rescue one soma”, “Henry join one row”, “BH within stratum×family”. |
| **L2** | Orchestrators | `02_run_step1_multi`, `05c_run_global_fnt`, `run_bulk_visual_sample`, `v2_combined_primary_pipeline.R` — call L1 only; own paths/manifests. |
| **L3** | Pipeline | Thin CLI/make: reference → step1 → refine → combined → harmonize → Henry → FNT → R v2; single `paths.yaml` (`COMBINED_HARMONIZED`, NMT version, ION hosts). |

**Dedup priorities:** (1) one ION client with timeouts (group wrapper already better than raw `IONData`); (2) one combined-table path constant shared by R+Python; (3) one midline/mirror helper shared by `05c` and `step2_*` (docs currently contradict each other); (4) delete or archive tutorial R that still reads `251637_results.xlsx` from the critical path.

---

## Static check notes

- Priority Python files: `py_compile` / `ast.parse` clean.  
- `pyflakes`: not in `deb_fmri_pipeline`.  
- `Rscript`: not found on this Windows PATH — R files reviewed by read only.

## Parent verification (2026-09-26)

| Finding | Verdict | Evidence |
|---|---|---|
| BH `pmin` across presence/magnitude families | **Confirmed** | `v2_combined_primary_pipeline.R:1094`, `combined_lr_primary_analysis.R:247`. Report presence and magnitude hits separately, or combine p-values before one BH pass. |
| Stale strata on non-harmonized table | **Confirmed** | `multi_monkey_lr_analysis.R:31,181` reads `multi_monkey_INS_combined.xlsx`, `IDD5_plus_IDM_251637` only. |
| PAD unit error | **Confirmed, direction corrected** | `Soma_Phys = 250 × Soma_NII` (fit on `recovery/all_refined_neurons.csv`), so NII voxel = 0.25 mm. `PAD_TOL = 2.0` = **0.5 mm**, not 2 mm: rescue was *stricter* than documented, not 4× looser. Docs (`LR_insula_analysis_review.md` §11A) overstate the tolerance. |
| **New: bboxes pool both hemispheres** | **High** | `reference/251637_subregion_bboxes.csv`: IAL X_q005–q995 = 75–195 vox, IDD5 51–210 (midline ≈ 128). X constraint is non-discriminative; rescue effectively matches Y/Z only, and overlapping boxes inflate `excluded_ambiguous_bbox`. Fix: build bboxes on mirrored X (fold R→L at 128 vox) or per hemisphere, then re-run `04_refine_soma_region_by_coords.py` and diff rescued sets. |
