# Projectome analysis — project note

**Purpose:** Living human-readable status for fMOST / insula projectome work. Agents **merge** updates; do not replace sections.

## Status snapshot (2026-08-12)

| | |
|---|---|
| **Now** | Pipeline docs include step 3.4 atlas mesh extraction (`volume_to_mesh_mz3_MONKEY.py` on Bryant `subcortex_visualization`) |
| **Next** | Continue data / visual pipeline as Binbin directs; cross-modal link to opto fMRI via `multimodal_fmri/registry/subjects.tsv` |
| **Blockers** | none noted |

**Related:** multimodal opto living note → `D:/multimodal_fmri/docs/project_note.md`  
**Env (cross-modal):** prefer `multimodal_ins_win` for new shared scripts; `projectome` env still OK for projectome-only work.

## How we use this file

| Who | What |
|-----|------|
| Human | Science decisions, animal IDs, analysis priorities |
| Agent | Refresh Status + append Changelog after substantive work |

Global rule: `~/.cursor/rules/project-living-note.mdc`.

## Open decisions

| ID | Question | Status |
|----|----------|--------|
| — | (add as needed) | open |

## Changelog

### 2026-08-12 — pipeline docs: atlas mesh extraction

- Documented Step 5.3 / `step3.4.mesh_extraction.py` in `README.md` and `main_scripts/PIPELINE_MINDMAP.md`.
- Engine: `subcortex_visualization/monkey_atlas_guide/volume_to_mesh_mz3_MONKEY.py` (local macaque fork of [anniegbryant/subcortex_visualization](https://github.com/anniegbryant/subcortex_visualization)).
- Replaced stale mindmap node `step3.4.region_flatmap.viz.py` (file does not exist). Clustered heatmap is now Step 5.4.

### 2026-07-22 — living note created

- Added `docs/project_note.md` so this repo matches the multimodal_fmri living-note convention.
