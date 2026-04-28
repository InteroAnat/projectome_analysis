# -*- coding: utf-8 -*-
"""Rebuild v2_combined_primary_pipeline.Rmd with sectioned chunks (run once)."""
from pathlib import Path

p = Path("v2_combined_primary_pipeline.Rmd")
lines = p.read_text(encoding="utf-8").splitlines()

if not any(l.strip() == "```{r pipeline}" for l in lines):
    raise SystemExit(
        "This script expects the original Rmd with one ```{r pipeline} chunk. "
        "If the file is already sectioned, restore the monolithic version from git, "
        "then run again."
    )

# Re-extract from FILE using fixed 1-based line ranges (must match monolithic layout)
ranges = [
    (51, 133, "01-packages-paths-helpers", "Packages, output directories, and small utilities (`build_mat`, `normalize_rows`, …)."),
    (135, 183, "02-load-hybrid", "Read harmonized workbook; build `m_l3` / `m_l6` / **`p_combo`** (L3 extrinsic + L6 intra-insula)."),
    (185, 248, "03-strata-context-orders", "Strata spec, context CSVs, `stratum_idx()`, row/column orders for figures."),
    (250, 434, "04-fig1-gou-domains", "**F1A/F1B** — Gou six functional-domain panels from `p_combo` + ontology map."),
    (436, 563, "05-fig2-key-targets", "**F2 / F2B** — Curated key targets (hybrid L3/L6 policy)."),
    (565, 714, "06-fig3-intra-insula", "**F3A–C** — Intra-insula matrices (L6), Ward ordering."),
    (716, 779, "07-fig4-gradient", "**F4** — Interoceptive gradient vs soma AP (OLS per hemisphere)."),
    (781, 832, "08-fig5-ibias", "**F5** — Length-based hemispheric bias vs AP."),
    (834, 1009, "09-fig6-composition-li", "**F6** — Projection laterality index on `p_combo`; bilateral vs one-sided strata."),
    (1011, 1075, "10-fig10-ap-octiles", "**F10** — Interoceptive targets × soma AP octiles."),
    (1077, 1182, "11-fig6-rank-supplement", "**F6 supplement** — Rank-based asymmetric receivers (BH)."),
    (1184, 1257, "12-fig7-headlines", "**F7** — Headline targets under resolved L3/L6 policy."),
    (1259, 1311, "13-fig8-loso", "**F8** — Leave-one-monkey-out sensitivity."),
    (1313, 1468, "14-fig9-mantel-fnt", "**F9 / F9b** — Mantel test; FNT vs Bray–Curtis hex + SQ5 heatmaps."),
    (1470, 1804, "15-registry-qc-captions", "Merged spec, `stat_registry`, README, P15 QC, manuscript xlsx, **FIGURE_CAPTIONS.md**."),
]

HEADING = {
    "01-packages-paths-helpers": "Packages, paths, and helpers",
    "02-load-hybrid": "Load data and build hybrid `p_combo`",
    "03-strata-context-orders": "Strata spec, context tables, orders",
    "04-fig1-gou-domains": "Figure 1 — Gou functional domains",
    "05-fig2-key-targets": "Figure 2 — Key projection targets",
    "06-fig3-intra-insula": "Figure 3 — Intra-insula connectivity",
    "07-fig4-gradient": "Figure 4 — Interoceptive AP gradient",
    "08-fig5-ibias": "Figure 5 — Ibias (axon length)",
    "09-fig6-composition-li": "Figure 6 — Composition laterality index",
    "10-fig10-ap-octiles": "Figure 10 — AP octiles × interoceptive targets",
    "11-fig6-rank-supplement": "Figure 6 supplement — Rank asymmetry",
    "12-fig7-headlines": "Figure 7 — Headline targets",
    "13-fig8-loso": "Figure 8 — Leave-one-monkey-out",
    "14-fig9-mantel-fnt": "Figures 9 / 9b — FNT vs projection",
    "15-registry-qc-captions": "Registry, QC, manuscript table, captions",
}

yaml = '''---
title: "Whole-insula L+R combined primary pipeline"
subtitle: "Outputs → `group_analysis/R_analysis/outputs/combined_primary_v2/`"
author: "projectome_analysis"
date: "`r Sys.Date()`"
output:
  html_document:
    toc: true
    toc_depth: 3
    toc_float:
      collapsed: false
      smooth_scroll: true
    number_sections: false
    anchor_sections: true
    code_folding: show
    code_download: true
    theme: cosmo
    highlight: tango
    df_print: paged
---

# Overview

This notebook runs the **insula L/R continuation analysis (v2)** end-to-end.

| | |
|--|--|
| **Hybrid profile** | `p_combo`: extra-insula targets from **L3**, intra-insula from **L6**; row-normalized per neuron. |
| **Primary input** | `group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx` |
| **Figures / stats** | `outputs/combined_primary_v2/figures/` and `stats/` |
| **Captions** | `figures/FIGURE_CAPTIONS.md` (rewritten each run) |
| **Run from CLI** | `Rscript v2_combined_primary_pipeline.R` → renders this Rmd |

**Requirements:** Pandoc on `PATH`; R packages `readxl`, `readr`, `dplyr`, `tidyr`, `ggplot2`, `stringr`, `vegan`, `scales`, `ComplexHeatmap`, `circlize`, `writexl`, `rmarkdown`.

The pipeline is split into **numbered chunks** below so the HTML table of contents matches analysis stages (F1–F10, QC, exports).

---

# Knitr setup

```{r setup, include=FALSE}
inp <- knitr::current_input()
pipeline_dir <- dirname(normalizePath(
  if (is.null(inp) || !nzchar(inp)) getwd() else inp,
  winslash = "/", mustWork = FALSE
))
knitr::opts_knit$set(root.dir = pipeline_dir)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE,
  fig.show = "hide",
  results = "asis",
  cache = FALSE
)
```

'''

out = [yaml]
file_lines = lines  # full original
for lo, hi, chunk_id, blurb in ranges:
    chunk_body = file_lines[lo - 1 : hi]
    h = HEADING.get(chunk_id, chunk_id)
    out.append("")
    out.append(f"## {h} {{#{chunk_id}}}")
    out.append("")
    out.append(blurb)
    out.append("")
    out.append(f"```{{r {chunk_id}}}")
    out.extend(chunk_body)
    out.append("```")
    out.append("")

Path("v2_combined_primary_pipeline.Rmd").write_text("\n".join(out) + "\n", encoding="utf-8")
print("Wrote v2_combined_primary_pipeline.Rmd with", len(ranges), "chunks")
