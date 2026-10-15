"""Add Tencent-sheet animal identity onto the legacy eight-monkey overview.

`build_progress_table` stays unchanged so older receipts keep the same
columns. This wrapper only attaches fields that identify an animal or
say whether tracing has started, then appends animals the sheet names
that are outside the eight.
"""
from __future__ import annotations

import pandas as pd

# Short animal id -> fMOST sample. These are the legacy overview pairs.
LEGACY_EIGHT = {
    "331": "252385",
    "605": "252383",
    "631": "252714",
    "797": "252790",
    "900": "252718",
    "936": "251637",
    "945": "252527",
    "948": "252384",
}

# Headers read from Progress Details (tab BB08J2), in sheet order.
SHEET_HEADERS = [
    "NO.",
    "注射动物_ID",
    "fMOST_ID",
    "性别",
    "出生日期",
    "注射实验日期",
    "核磁日期",
    "注射脑区",
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
    "追踪进度(Neurite tracing)",
    "追踪数量(Number of traced neurons)",
    "配准进度(Image registration)",
    "入库进度(Database Import)",
    "入库数量(Number for analysis)",
]

# Kept because they identify the animal or report tracing progress.
KEPT_SHEET_COLUMNS = {
    "注射动物_ID": "sheet_animal_name",
    "fMOST_ID": "sheet_fmost_id",
    "注射脑区": "sheet_injection_sites",
    "追踪进度(Neurite tracing)": "sheet_tracing_status",
}

OVERVIEW_EXTRA_COLUMNS = [
    "sheet_animal_name",
    "sheet_fmost_id",
    "sheet_fmost_cell_kind",
    "sheet_injection_sites",
    "sheet_tracing_status",
    "in_legacy_eight",
    "neuron_screen_status",
]


def _single_fmost_token(cell: str) -> str:
    """Return a sample id only when the sheet cell is one integer.

    A cell with two ids is left unsplit. The sheet does not say those
    ids are channel variants, so they stay one quoted cell.
    """
    text = str(cell).strip()
    tokens = text.split()
    if len(tokens) == 1 and tokens[0].isdigit():
        return tokens[0]
    return ""


def fmost_cell_kind(cell: str) -> str:
    text = str(cell).strip()
    if not text:
        return "empty"
    if _single_fmost_token(text):
        return "single"
    return "multiple_ids_not_split"


def _clean_cell(text: str) -> str:
    """Strip edges. Collapse a line-broken cell without touching internal spaces."""
    raw = str(text)
    if "\\n" not in raw:
        return raw.strip()
    parts = []
    for part in raw.split("\\n"):
        part = part.strip()
        if part and part not in parts:
            parts.append(part)
    return " ".join(parts)


def _is_region(site: str) -> bool:
    text = site.strip()
    if not text or text == "/":
        return False
    # The 935 row uses this cell for a death note, not a brain region.
    if "死亡" in text:
        return False
    return True


def _name_matches_legacy(sheet_name: str, monkey_id: str) -> bool:
    if sheet_name == monkey_id:
        return True
    return sheet_name.startswith(monkey_id + " ") or sheet_name.startswith(monkey_id + "(")


def parse_progress_details_tsv(text: str) -> pd.DataFrame:
    """Collapse merged animal blocks into one row of useful fields.

    Child rows repeat injection sites under a blank animal cell. The
    animal name and fMOST id are taken only from the first row of each
    block. Blank name cells stay blank.
    """
    lines = [line for line in text.split("\n") if line != ""]
    if not lines:
        raise ValueError("Progress Details extract is empty")
    header = lines[0].split("\t")
    if header[: len(SHEET_HEADERS)] != SHEET_HEADERS:
        raise ValueError(f"Unexpected sheet header: {header[: len(SHEET_HEADERS)]}")
    width = len(SHEET_HEADERS)
    animals: list[dict] = []
    current: dict | None = None
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        extra = [c.strip() for c in cells[width:] if c.strip()]
        if extra:
            raise ValueError(f"Unread values past the known headers: {extra[:5]}")
        row = {name: _clean_cell(cells[i]) for i, name in enumerate(SHEET_HEADERS)}
        if row["NO."] or row["注射动物_ID"] or row["fMOST_ID"]:
            if current is not None:
                animals.append(current)
            current = {
                "sheet_no": row["NO."],
                "sheet_animal_name": row["注射动物_ID"],
                "sheet_fmost_id": row["fMOST_ID"],
                "sites": [],
                "sheet_tracing_status": row["追踪进度(Neurite tracing)"],
            }
        elif current is None:
            raise ValueError("Injection row appears before any animal")
        if current["sheet_animal_name"] != row["注射动物_ID"] and row["注射动物_ID"]:
            raise ValueError("Child row carries a second animal name")
        if current["sheet_fmost_id"] != row["fMOST_ID"] and row["fMOST_ID"]:
            raise ValueError("Child row carries a second fMOST id")
        site = row["注射脑区"]
        if _is_region(site) and site not in current["sites"]:
            current["sites"].append(site)
        if not current["sheet_tracing_status"] and row["追踪进度(Neurite tracing)"]:
            current["sheet_tracing_status"] = row["追踪进度(Neurite tracing)"]
    if current is not None:
        animals.append(current)
    frame = pd.DataFrame(animals)
    if frame.empty:
        raise ValueError("No animal rows parsed from Progress Details")
    frame["sheet_injection_sites"] = frame["sites"].map(lambda sites: " | ".join(sites))
    frame["sheet_fmost_cell_kind"] = frame["sheet_fmost_id"].map(fmost_cell_kind)
    frame = frame.drop(columns=["sites"])
    if frame["sheet_no"].duplicated().any():
        raise ValueError("Duplicate sheet animal numbers")
    return frame.reset_index(drop=True)


def extend_overview_with_sheet(legacy: pd.DataFrame, animals: pd.DataFrame) -> pd.DataFrame:
    """Keep every legacy column for the eight, then append sheet-only animals.

    Neuron-count cells for added animals stay empty. This does not screen
    neurons and does not treat a sheet row as anatomical acceptance.
    """
    if set(zip(legacy["monkey_id"], legacy["fmost_id"])) != set(LEGACY_EIGHT.items()):
        raise ValueError("Legacy overview is not the eight known animal/sample pairs")
    needed = {"sheet_animal_name", "sheet_fmost_id", "sheet_injection_sites", "sheet_tracing_status"}
    missing = needed - set(animals.columns)
    if missing:
        raise ValueError(f"Sheet animal table missing {sorted(missing)}")

    by_sample: dict[str, dict] = {}
    for record in animals.to_dict("records"):
        token = _single_fmost_token(record["sheet_fmost_id"])
        if token not in set(LEGACY_EIGHT.values()):
            continue
        if token in by_sample:
            raise ValueError(f"Sheet lists sample {token} twice")
        by_sample[token] = record
    absent = set(LEGACY_EIGHT.values()) - set(by_sample)
    if absent:
        raise ValueError(f"Sheet has no row for legacy samples {sorted(absent)}")

    out = legacy.copy()
    attached = []
    for monkey_id, fmost_id in zip(out["monkey_id"], out["fmost_id"]):
        record = by_sample[fmost_id]
        if not _name_matches_legacy(record["sheet_animal_name"], monkey_id):
            raise ValueError(
                f"Sheet animal cell {record['sheet_animal_name']!r} "
                f"does not match legacy monkey {monkey_id}"
            )
        attached.append(record)
    out["sheet_animal_name"] = [row["sheet_animal_name"] for row in attached]
    out["sheet_fmost_id"] = [row["sheet_fmost_id"] for row in attached]
    out["sheet_fmost_cell_kind"] = [fmost_cell_kind(row["sheet_fmost_id"]) for row in attached]
    out["sheet_injection_sites"] = [row["sheet_injection_sites"] for row in attached]
    out["sheet_tracing_status"] = [row["sheet_tracing_status"] for row in attached]
    out["in_legacy_eight"] = "yes"
    out["neuron_screen_status"] = "legacy_inventory_counts"

    legacy_samples = set(LEGACY_EIGHT.values())
    extras = []
    for record in animals.to_dict("records"):
        token = _single_fmost_token(record["sheet_fmost_id"])
        if token in legacy_samples:
            continue
        row = {column: "" for column in out.columns}
        # Exact sheet text. An empty name stays empty; no id is invented.
        row["monkey_id"] = record["sheet_animal_name"]
        row["fmost_id"] = record["sheet_fmost_id"]
        row["sheet_animal_name"] = record["sheet_animal_name"]
        row["sheet_fmost_id"] = record["sheet_fmost_id"]
        row["sheet_fmost_cell_kind"] = fmost_cell_kind(record["sheet_fmost_id"])
        row["sheet_injection_sites"] = record["sheet_injection_sites"]
        row["sheet_tracing_status"] = record["sheet_tracing_status"]
        row["in_legacy_eight"] = "no"
        row["neuron_screen_status"] = "not_screened"
        if "anatomical_acceptance" in row:
            row["anatomical_acceptance"] = "False"
        extras.append(row)
    if extras:
        out = pd.concat([out, pd.DataFrame(extras)], ignore_index=True)
    return out


def assert_overview(frame: pd.DataFrame) -> None:
    """Smoke checks for the eight pairs and the 250432 name cell."""
    eight = frame[frame["in_legacy_eight"] == "yes"]
    got = set(zip(eight["monkey_id"], eight["fmost_id"]))
    if got != set(LEGACY_EIGHT.items()):
        raise AssertionError(f"Eight pairs changed: {got}")
    hit = frame[frame["sheet_fmost_id"] == "250432"]
    if len(hit) != 1:
        raise AssertionError(f"Expected one 250432 row, found {len(hit)}")
    name = hit.iloc[0]["sheet_animal_name"]
    if name != "747":
        raise AssertionError(f"250432 animal cell is {name!r}, not '747'")
    if (frame["monkey_id"] == "").any() and (frame.loc[frame["monkey_id"] == "", "sheet_animal_name"] != "").any():
        raise AssertionError("A named sheet animal was written with an empty monkey id")
    banned = {"ION2508004", "ION2508005", "2508004", "250432ch2"}
    present = set(frame["monkey_id"]) | set(frame["fmost_id"]) | set(frame["sheet_animal_name"]) | set(frame["sheet_fmost_id"])
    found = banned & present
    if found:
        raise AssertionError(f"Portal alias or unlisted channel entered the overview: {found}")
    added = frame[frame["in_legacy_eight"] == "no"]
    count_cols = [
        "snapshot_atlas_INS_n",
        "selected_projectome_n",
        "henry_visual_INS_in_selected_n",
    ]
    for column in count_cols:
        if column in added.columns and added[column].astype(str).str.strip().ne("").any():
            raise AssertionError(f"Added animals must not carry {column}")
