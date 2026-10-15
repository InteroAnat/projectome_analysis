"""Attach multi-level regional projection strength to the progress workbook.

Legacy harmonized strength (log-scaled retained length) lives in
``group_analysis/combined/multi_monkey_INS_combined.xlsx`` at L3 and L6.
Six-level ARM endpoint/axon measures for all 462 Selected neurons live in
``combined_arm_projection_tables_462/per_neuron_regional_measures.csv``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .paths import (
    ARM_PROJECTION_HIERARCHY_XLSX,
    COMBINED_XLSX,
    PER_NEURON_REGIONAL_MEASURES,
    SELECTED_LEDGER,
)

# Same sheet names as group_analysis/scripts/05a_build_combined_table.py (strength only).
LEGACY_STRENGTH_SHEETS = (
    "Projection_Strength_ipsi",
    "Projection_Strength_contra",
    "Projection_Strength_L3_ipsi",
    "Projection_Strength_L3_contra",
)
LEGACY_SUMMARY_SHEET = "Summary"
UID_COLUMN = "NeuronUID"


def _read_ledger_uids() -> list[str]:
    with SELECTED_LEDGER.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    uids = [str(row[UID_COLUMN]).strip() for row in rows if str(row.get(UID_COLUMN, "")).strip()]
    if len(uids) != 462:
        raise ValueError(f"Selected ledger has {len(uids)} NeuronUID values, expected 462")
    if len(set(uids)) != 462:
        raise ValueError("Selected ledger NeuronUID values are not unique")
    return uids


def _filter_by_uid(frame: pd.DataFrame, uids: list[str], label: str) -> pd.DataFrame:
    if UID_COLUMN not in frame.columns:
        raise ValueError(f"{label} is missing column {UID_COLUMN}")
    subset = frame[frame[UID_COLUMN].astype(str).isin(uids)].copy()
    order = {uid: index for index, uid in enumerate(uids)}
    subset["_order"] = subset[UID_COLUMN].astype(str).map(order)
    subset = subset.sort_values("_order", kind="mergesort").drop(columns="_order")
    return subset.reset_index(drop=True)


def load_projection_strength_tables() -> dict:
    """Load legacy L3/L6 strength matrices and ARM L1–L6 long measures for Selected neurons."""
    uids = _read_ledger_uids()
    uid_set = set(uids)

    if not COMBINED_XLSX.is_file():
        raise FileNotFoundError(COMBINED_XLSX)
    legacy_sheets: dict[str, pd.DataFrame] = {}
    summary = pd.read_excel(COMBINED_XLSX, sheet_name=LEGACY_SUMMARY_SHEET, dtype=str)
    legacy_sheets[LEGACY_SUMMARY_SHEET] = _filter_by_uid(summary, uids, LEGACY_SUMMARY_SHEET)
    combined_present = set(legacy_sheets[LEGACY_SUMMARY_SHEET][UID_COLUMN].astype(str))
    for sheet in LEGACY_STRENGTH_SHEETS:
        frame = pd.read_excel(COMBINED_XLSX, sheet_name=sheet)
        legacy_sheets[sheet] = _filter_by_uid(frame, uids, sheet)

    if not PER_NEURON_REGIONAL_MEASURES.is_file():
        raise FileNotFoundError(PER_NEURON_REGIONAL_MEASURES)
    arm_long = pd.read_csv(PER_NEURON_REGIONAL_MEASURES, dtype=str)
    arm_long = arm_long[arm_long[UID_COLUMN].astype(str).isin(uid_set)].copy()
    arm_present = set(arm_long[UID_COLUMN].astype(str))
    arm_missing = sorted(uid_set - arm_present)
    if arm_missing and len(arm_missing) > 33:
        raise ValueError(
            "Unexpected ARM regional gap for Selected neurons: "
            f"{len(arm_missing)} missing (expected at most 33 endpoint-ineligible omissions)"
        )

    return {
        "ledger_neuron_n": len(uids),
        "legacy_combined_path": str(COMBINED_XLSX),
        "legacy_strength_sheets": list(LEGACY_STRENGTH_SHEETS),
        "legacy_strength_neuron_n": len(combined_present),
        "legacy_strength_missing_n": len(uid_set - combined_present),
        "arm_hierarchy_workbook": str(ARM_PROJECTION_HIERARCHY_XLSX),
        "arm_regional_measures_path": str(PER_NEURON_REGIONAL_MEASURES),
        "legacy_sheets": legacy_sheets,
        "arm_regional_long": arm_long,
        "arm_regional_neuron_n": len(arm_present),
        "arm_regional_missing_n": len(arm_missing),
        "metric_notes": (
            "Legacy Projection_Strength_* = log10(retained length + 1) from the harmonized "
            "multi_monkey_INS_combined workbook (L6 finest columns; L3 grouped columns). "
            "ARM per_neuron_regional_measures = six actual ARM levels with "
            "CandidateEndpointCount and AxonTemplateLengthMm for all 462 Selected neurons."
        ),
    }


def append_projection_sheets(book, bundle: dict) -> None:
    """Add legacy strength and ARM long measures as extra worksheets on an openpyxl Workbook."""
    from openpyxl.utils.dataframe import dataframe_to_rows

    for sheet_name, frame in bundle["legacy_sheets"].items():
        short = sheet_name if len(sheet_name) <= 31 else sheet_name[:31]
        ws = book.create_sheet(short)
        for row in dataframe_to_rows(frame, index=False, header=True):
            ws.append(row)
        ws.freeze_panes = "A2"

    ws = book.create_sheet("ARM_regional_measures")
    for row in dataframe_to_rows(bundle["arm_regional_long"], index=False, header=True):
        ws.append(row)
    ws.freeze_panes = "A2"
