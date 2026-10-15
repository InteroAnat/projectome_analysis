"""Write a fresh overview that adds Progress Details identities.

Reads the saved BB08J2 extract, checks that build_progress_table still
resolves the eight animal/sample pairs, and writes a new CSV. It does
not overwrite the 20261009 overview.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from group_analysis.data_progress import table
from group_analysis.data_progress.sheet_overview import (
    LEGACY_EIGHT,
    SHEET_HEADERS,
    assert_overview,
    extend_overview_with_sheet,
    parse_progress_details_tsv,
)

LEGACY_OVERVIEW = (
    Path(__file__).resolve().parents[1]
    / "evolution_20261008/inventory/legacy_overview_20261009/per_monkey_insula_inventory.csv"
)
DEFAULT_JSON = Path(
    r"C:\Users\laika_yan\.cursor\browser-logs"
    r"\cdp-response-Runtime.evaluate-2026-10-12T04-53-22-137Z.json"
)
SHEET_URL = "https://docs.qq.com/sheet/DQXFTS21ieUVGTEpJ?tab=BB08J2&login_t=1791780626031"
SKIPPED = [
    "NO.",
    "性别",
    "出生日期",
    "注射实验日期",
    "核磁日期",
    "使用病毒",
    "cre终稀释倍数",
    "注射体积(ul)",
    "等待时间(min)",
    "灌流日期",
    "伦理文件批号",
    "成像单位/方式",
    "备注(notes)",
    "成像进度(fMOST imaging)",
    "成像数据拷贝(Imaging data Copy)",
    "追踪数量(Number of traced neurons)",
    "配准进度(Image registration)",
    "入库进度(Database Import)",
    "入库数量(Number for analysis)",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sheet_text(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    packed = json.loads(payload["result"]["value"])
    if packed.get("id") != "BB08J2" or packed.get("name") != "Progress Details":
        raise ValueError(f"Extract is not Progress Details BB08J2: {packed.get('name')} {packed.get('id')}")
    return packed["tsv"], {
        "tab_name": packed["name"],
        "tab_id": packed["id"],
        "grid_rows": packed["rows"],
        "grid_cols": packed["cols"],
        "nonempty_rows": packed["nonempty"],
    }


def confirm_progress_pairs() -> int:
    """Run the existing progress function and require the eight pairs."""
    progress, _totals = table.build_progress_table()
    got = set(zip(progress["monkey_id"].astype(str), progress["fmost_id"].astype(str)))
    missing = [pair for pair in LEGACY_EIGHT.items() if pair not in got]
    if missing:
        raise ValueError(f"build_progress_table is missing pairs {missing}")
    return len(progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evolution_20261008/overview_sheet_20261012",
    )
    parser.add_argument("--sheet-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--legacy-overview", type=Path, default=LEGACY_OVERVIEW)
    args = parser.parse_args()
    if not args.legacy_overview.is_file():
        raise FileNotFoundError(args.legacy_overview)
    before = sha256(args.legacy_overview)
    sheet_text, grid = load_sheet_text(args.sheet_json)
    animals = parse_progress_details_tsv(sheet_text)
    progress_n = confirm_progress_pairs()
    legacy = pd_read(args.legacy_overview)
    original_columns = list(legacy.columns)
    overview = extend_overview_with_sheet(legacy, animals)
    assert_overview(overview)
    eight = overview[overview["in_legacy_eight"] == "yes"]
    for column in original_columns:
        if not eight[column].astype(str).equals(legacy[column].astype(str)):
            raise ValueError(f"Legacy column changed: {column}")
    if sha256(args.legacy_overview) != before:
        raise ValueError("Legacy overview file changed while building the new table")

    args.output.mkdir(parents=True, exist_ok=True)
    animals.to_csv(args.output / "sheet_animals.csv", index=False)
    (args.output / "progress_details_BB08J2.tsv").write_text(sheet_text, encoding="utf-8")
    overview.to_csv(args.output / "per_monkey_overview_with_sheet.csv", index=False)
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    outside = animals.loc[
        ~animals["sheet_fmost_id"].map(lambda value: value.strip() in set(LEGACY_EIGHT.values())),
        ["sheet_animal_name", "sheet_fmost_id", "sheet_fmost_cell_kind", "sheet_tracing_status"],
    ]
    receipt = {
        "url": SHEET_URL,
        "document_title": "macaque_fMOST",
        "tab_name": grid["tab_name"],
        "tab_id": grid["tab_id"],
        "retrieved_at": "2026-10-12T12:53:22+08:00",
        "written_at": now,
        "retrieval": "Spreadsheet cell values from the open view-only grid. A login button was visible and was not used.",
        "grid_rows": grid["grid_rows"],
        "grid_cols": grid["grid_cols"],
        "nonempty_rows": grid["nonempty_rows"],
        "headers": SHEET_HEADERS,
        "columns_kept": ["注射动物_ID", "fMOST_ID", "注射脑区", "追踪进度(Neurite tracing)"],
        "columns_skipped": SKIPPED,
        "unread_tabs": ["Progress Overview", "统计表", "苏州测试用fMOST注射信息", "工作表1"],
        "unread_reason": "Those tabs were listed but not loaded. This table uses BB08J2 only.",
        "cells_unreadable_on_BB08J2": [],
        "animal_250432_name_cell": "747",
        "absent_strings": ["ION2508004", "ION2508005", "2508004", "250432ch2"],
        "build_progress_table_rows": progress_n,
        "build_progress_table_eight_pairs_resolved": True,
        "legacy_overview_sha256": before,
        "legacy_overview_rewritten": False,
        "neuron_screen_run": False,
        "anatomical_acceptance": False,
        "animals_outside_eight": outside.to_dict("records"),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "animals": len(animals),
        "overview_rows": len(overview),
        "outside_eight": len(outside),
        "progress_table_rows": progress_n,
        "output": str(args.output),
        "name_250432": "747",
    }, ensure_ascii=False))


def pd_read(path: Path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False)


if __name__ == "__main__":
    main()
