from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
GROUP = PROJECT / "group_analysis"
DOCS = GROUP / "docs"
FIGURES = DOCS / "figures"
# Refreshed table, workbook, figure, and provenance. This folder is outputs only.
OVERVIEW_OUTPUT = GROUP / "data_progress"
# Newest full neuron region table: 8746 neurons, every sample, atlas labels,
# folded-box screen, and same-sample insula anchor distances.
FULL_NEURON_REGION_TABLE = (
    GROUP
    / "evolution_20261008/classification/coarse_insula_review_20261009"
    / "distance_priority_v2_20261009/all_neuron_review_manifest_distance_corrected.csv"
)
SELECTED_LEDGER = (
    PROJECT
    / "notes/region_analysis_review_20261009/hierarchy_tables_20261009"
    / "combined_arm_projection_tables_462/neuron_summary.csv"
)
ARM_PROJECTION_HIERARCHY_XLSX = (
    PROJECT
    / "notes/region_analysis_review_20261009/hierarchy_tables_20261009"
    / "combined_arm_projection_tables_462/arm_projection_hierarchy_tables.xlsx"
)
PER_NEURON_REGIONAL_MEASURES = (
    PROJECT
    / "notes/region_analysis_review_20261009/hierarchy_tables_20261009"
    / "combined_arm_projection_tables_462/per_neuron_regional_measures.csv"
)
SCRIPTS = GROUP / "scripts"
ARM_ATLAS = PROJECT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz"
ARM_KEY = PROJECT / "atlas/ARM_key_all.txt"
MANIFEST = DOCS / "dataset_status_manifest.csv"
STEP1_DIR = GROUP / "step1_results"
COMBINED_XLSX = GROUP / "combined" / "multi_monkey_INS_combined.xlsx"
REF_INS_XLSX = PROJECT / "neuron_tables_new" / "251637_INS_HE_inferred.xlsx"

PLAN_ANIMALS = ["936", "605", "900", "945", "797", "948", "331", "631"]
PLAN_INSULA_EXPECTED = 2400
REFERENCE_FMOST = "251637"

OUT_TABLE_STEM = "data_progress_table"
OUT_FIGURE_STEM = "data_progress_table"


def stamped_outputs(file_stamp: str) -> tuple[Path, Path]:
    """Return timestamped CSV + PNG paths, e.g. data_progress_table_20260701_1405."""
    return (
        DOCS / f"{OUT_TABLE_STEM}_{file_stamp}.csv",
        FIGURES / f"{OUT_FIGURE_STEM}_{file_stamp}.png",
    )


LEGACY_FIGURE_GLOBS = (
    "dataset_status_*.png",
    f"{OUT_FIGURE_STEM}.png",
)
LEGACY_TABLE_NAMES = (
    f"{OUT_TABLE_STEM}.csv",
    f"{OUT_TABLE_STEM}.new.csv",
)
