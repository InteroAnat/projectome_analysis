# SQ4/SQ5 Distance Logic Digest

## Context

This digest summarizes why SQ4/SQ5 distance handling was changed, and how to interpret the outputs after the update.

---

## Core decisions

1. **SQ4 projection distance switched to Bray on proportion-normalized neuron profiles** (`bray_prop` endpoint).
2. **SQ5 FNT panel uses `FNT01` (0-1 min-max normalized) for visualization comparability.**
3. **SQ5 mismatch panel changed from z-score mismatch to rank mismatch.**

---

## Why SQ4 moved to `bray_prop`

- `vegan::vegdist(method = "bray")` does not row-normalize input.
- Raw Bray includes both pattern and magnitude effects.
- For SQ4, the question is morphology–projection **pattern** concordance, so projection matrix should be normalized first.

Interpretation now:
- SQ4 compares FNT distance to projection-pattern distance, not projection magnitude differences.

---

## Why SQ5 mismatch changed from `|z_FNT - z_Proj|` to rank mismatch

Old panel:
- `|z_FNT - z_Proj|` is sensitive to distribution shape, tails, and variance differences.
- In sparse/non-Gaussian matrices, z-based mismatch can look unintuitive:
  - both cells can look bright in panel 1/2 but still appear dark in mismatch panel.

New panel:
- `|rank(FNT01) - rank(Proj)|` (scaled to 0-1 via rank percentiles).
- This measures disagreement in **relative ordering** of neuron-pair distances.
- More robust for sparse, skewed, non-parametric distance structure.
- Better aligned with Spearman-style concordance logic.

---

## Practical reading guide (current SQ5)

- **Panel 1 (`FNT01`)**: normalized FNT distance map (0-1 visual scale).
- **Panel 2 (`Proj`)**: Bray-prop projection distance map (pattern-focused).
- **Panel 3 (`|Δrank|`)**:
  - low/light: pair has similar relative distance rank in both modalities
  - high/dark: pair is ranked very differently between FNT and projection spaces

---

## Statistical interpretation boundaries

- This does not claim parametric assumptions.
- It is intentionally robust to sparse matrix behavior and heavy-tailed distance distributions.
- Mantel/Spearman concordance and rank mismatch are conceptually consistent.

---

## Upstream data provenance reminder

- True upstream source: `main_scripts/step2_fnt-dist_pipeline.py`
- Intermediate exporter: `R_analysis/scripts/fnt_dist_clustering.r`
- SQ4/SQ5 consumer: `R_analysis/scripts/laterality_v1_v2_INS_HE_merged_v3.Rmd`

