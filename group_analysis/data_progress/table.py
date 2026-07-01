from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .manifest import load_manifest, plan_cohort
from .paths import PLAN_INSULA_EXPECTED, REFERENCE_FMOST
from .sources import load_combined_stats, load_reference_distribution, load_step1_stats

STAGES: list[tuple[str, str]] = [
    ("ION", "ion_n"),
    ("step1", "step1"),
    ("insula-QC", "insula_corrected"),
    ("combined", "combined_insula_n"),
    ("5um-local", "five_um_ready"),
]


def _step1_complete(row: pd.Series) -> bool:
    if int(row.get("step1_ok", 0) or 0) != 1:
        return False
    if str(row.get("fmost_key", "")) == REFERENCE_FMOST:
        return True
    xlsx = str(row.get("step1_xlsx") or "").strip()
    return bool(xlsx)


def _stage_done(row: pd.Series, key: str) -> bool:
    if key == "ion_n":
        return int(row.get("ion_n", 0) or 0) > 0
    if key == "step1":
        return _step1_complete(row)
    if key == "insula_corrected":
        return int(row.get("insula_corrected", 0) or 0) > 0
    if key == "combined_insula_n":
        return int(row.get("combined_insula_n", 0) or 0) > 0
    if key == "five_um_ready":
        return bool(row.get("five_um_ready"))
    return False


def _pipeline_info(row: pd.Series) -> tuple[int, str]:
    """Return (stages_completed 0–6, next action label)."""
    completed = 0
    for label, key in STAGES:
        if _stage_done(row, key):
            completed += 1
            continue
        if completed == 0:
            return 0, "pending (ION)"
        return completed, f"next: {label}"
    return 6, "complete"


def build_progress_table(
    manifest: pd.DataFrame | None = None,
    generated_at: datetime | None = None,
) -> tuple[pd.DataFrame, dict]:
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).astimezone()

    if manifest is None:
        manifest = load_manifest()

    base = plan_cohort(manifest).copy()
    fmost_ids = [str(int(x)) for x in base["fmost_id"] if pd.notna(x)]

    step1 = load_step1_stats(fmost_ids)
    combined_summary, _distribution_long = load_combined_stats()

    table = base.copy()
    table["fmost_key"] = table["fmost_id"].apply(lambda x: str(int(x)) if pd.notna(x) else "")

    table = table.merge(step1, left_on="fmost_key", right_on="fmost_id", how="left", suffixes=("", "_s1"))
    if "fmost_id_s1" in table.columns:
        table = table.drop(columns=["fmost_id_s1"])
    if "fmost_id" in table.columns and "fmost_id_x" not in table.columns:
        table = table.rename(columns={"fmost_id": "fmost_id_manifest"})
    elif "fmost_id_x" in table.columns:
        table = table.rename(columns={"fmost_id_x": "fmost_id_manifest"}).drop(columns=["fmost_id_y"], errors="ignore")

    combined_summary = combined_summary.rename(columns={"fmost_id": "fmost_key"})
    table = table.merge(combined_summary, on="fmost_key", how="left")

    for col in ("combined_insula_n", "step1_all_n", "step1_auto_insula_n"):
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype(int)
        else:
            table[col] = 0

    missing_step1 = (table["step1_all_n"] == 0) & (table["ion_n"] > 0) & (table["step1_ok"] == 1)
    table.loc[missing_step1, "step1_all_n"] = table.loc[missing_step1, "ion_n"]

    ref_dist = load_reference_distribution()
    ref_mask = table["fmost_key"] == REFERENCE_FMOST
    if ref_mask.any():
        table.loc[ref_mask, "combined_distribution"] = table.loc[ref_mask, "combined_distribution"].fillna(ref_dist)
        table.loc[ref_mask & (table["combined_insula_n"] == 0), "combined_insula_n"] = table.loc[
            ref_mask, "insula_corrected"
        ]

    pipeline = table.apply(_pipeline_info, axis=1, result_type="expand")
    table["pipeline_stage_code"] = pipeline[0].astype(int)
    table["pipeline_stage"] = pipeline[1]

    table["sort_key"] = pd.to_numeric(table["analysis_n"], errors="coerce").fillna(-1)
    table = table.sort_values("sort_key", ascending=False)

    ts = generated_at.strftime("%Y-%m-%d %H:%M %Z")

    progress = pd.DataFrame(
        {
            "generated_at": ts,
            "monkey_id": table["animal"],
            "fmost_id": table["fmost_key"],
            "data_status": table["data_status"],
            "plan_injections": table["plan_injections"],
            "injection_sites": table["injection_sites"],
            "analysis_n_expected": table["analysis_n"],
            "ion_n_traced": table["ion_n"],
            "step1_done": table["step1_ok"],
            "step1_all_neurons": table["step1_all_n"],
            "step1_auto_insula_n": table["step1_auto_insula_n"],
            "insula_corrected_n": table["insula_corrected"],
            "insula_in_combined_n": table["combined_insula_n"],
            "insula_distribution_combined": table.get("combined_distribution", "—").fillna("—"),
            "insula_distribution_step1_auto": table.get("step1_auto_distribution", "—").fillna("—"),
            "insula_L_R_combined": table.get("combined_lr", "—").fillna("—"),
            "five_um_local": table["five_um_label"],
            "pipeline_stage_code": table["pipeline_stage_code"],
            "pipeline_stage": table["pipeline_stage"],
            "notes": table["notes"],
        }
    )

    totals = {
        "generated_at": ts,
        "file_stamp": generated_at.strftime("%Y%m%d_%H%M"),
        "monkeys": len(progress),
        "analysis_expected": int(pd.to_numeric(progress["analysis_n_expected"], errors="coerce").fillna(0).sum()),
        "ion_traced": int(progress["ion_n_traced"].sum()),
        "insula_expected_plan": PLAN_INSULA_EXPECTED,
        "insula_corrected": int(progress["insula_corrected_n"].sum()),
        "insula_in_combined": int(progress["insula_in_combined_n"].sum()),
        "five_um_ready": int(progress["five_um_local"].astype(str).str.lower().str.startswith("yes").sum()),
    }

    return progress, totals


def append_total_row(progress: pd.DataFrame, totals: dict) -> pd.DataFrame:
    row = {c: "—" for c in progress.columns}
    row.update(
        {
            "generated_at": totals["generated_at"],
            "monkey_id": "TOTAL",
            "fmost_id": "—",
            "data_status": f"{totals['monkeys']} monkeys",
            "injection_sites": "plan cohort",
            "analysis_n_expected": totals["analysis_expected"],
            "ion_n_traced": totals["ion_traced"],
            "insula_corrected_n": totals["insula_corrected"],
            "insula_in_combined_n": totals["insula_in_combined"],
            "insula_distribution_combined": f"plan target {totals['insula_expected_plan']}",
            "five_um_local": f"{totals['five_um_ready']}/{totals['monkeys']}",
            "pipeline_stage_code": "—",
            "pipeline_stage": "—",
            "notes": (
                f"ION {totals['ion_traced']}/{totals['analysis_expected']}; "
                f"insula {totals['insula_corrected']}/{totals['insula_expected_plan']}; "
                f"combined {totals['insula_in_combined']}; "
                f"5µm {totals['five_um_ready']}/{totals['monkeys']}"
            ),
        }
    )
    return pd.concat([progress, pd.DataFrame([row])], ignore_index=True)
