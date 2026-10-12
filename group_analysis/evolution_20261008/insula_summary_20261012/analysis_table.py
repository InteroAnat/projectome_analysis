"""Load the current analysis neuron table. No screen and no file writes.

The evolution recount in this folder writes ``per_animal_insula_summary.csv``
and reads the 462-neuron ledger. This is the reader for that pair.
``recount_insula_summary.main`` should stay the only writer. After either
file changes, refresh the progress overview with
``group_analysis.data_progress.track.write_progress_overview``, which draws
``group_analysis.data_progress.plot.plot_progress_table``.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEDGER = (
    ROOT
    / "notes/region_analysis_review_20261009/hierarchy_tables_20261009"
    / "combined_arm_projection_tables_462/neuron_summary.csv"
)
SUMMARY = HERE / "per_animal_insula_summary.csv"

# Same eight pairs as the plan image and the recount. Not the July combined workbook.
EIGHT = {
    "936": "251637",
    "605": "252383",
    "900": "252718",
    "945": "252527",
    "797": "252790",
    "948": "252384",
    "331": "252385",
    "631": "252714",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_analysis_neuron_table() -> dict:
    """Return the 462-ledger check plus the saved per-animal recount.

    Does not import the recount screen, does not read the July combined
    workbook, and does not write the ledger or the summary. Cohort membership
    is the eight pairs above. Other animals stay in ``rows`` and are not
    added into the eight-only reality totals.
    """
    if not SUMMARY.is_file():
        raise FileNotFoundError(SUMMARY)
    if not LEDGER.is_file():
        raise FileNotFoundError(LEDGER)
    ledger = read_csv(LEDGER)
    if len(ledger) != 462:
        raise ValueError(f"Analysis ledger has {len(ledger)} rows, expected 462")
    eight_pairs = {(animal, sample) for animal, sample in EIGHT.items()}
    screened = 0
    arm = 0
    rows = []
    for record in read_csv(SUMMARY):
        animal = str(record.get("animal_id", "")).strip()
        sample = " ".join(str(record.get("fMOST_samples", "")).split())
        in_eight = (animal, sample) in eight_pairs and str(record.get("channel_variant", "")) != "yes"
        copied = dict(record)
        copied["insula_8_cohort"] = "DEB lab" if in_eight else "Other lab"
        if in_eight:
            screened += int(copied["neurons_screened"])
            arm += int(copied["ARM_insula_n"])
        rows.append(copied)
    if screened == 0 or arm == 0:
        raise ValueError("Insula-eight recount totals were empty")
    return {
        "loader": "group_analysis.evolution_20261008.insula_summary_20261012.analysis_table.load_analysis_neuron_table",
        "ledger_path": str(LEDGER),
        "summary_path": str(SUMMARY),
        "ledger_rows": len(ledger),
        "rows": rows,
        "eight_screened": screened,
        "eight_arm_insula": arm,
    }
