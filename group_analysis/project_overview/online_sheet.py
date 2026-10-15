"""Read the macaque fMOST Progress Details tab and attach useful identity fields.

The default URL is the share link for workbook macaque_fMOST, tab BB08J2.
That tab's title is Progress Details. A request without the share query
returns HTTP 401. A request with the share query returns workbook metadata,
not the cell grid, so a verified local snapshot of the same tab is the
fallback. A failed read raises OnlineSheetError and does not invent rows.
"""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .paths import GROUP
from .sheet_overview import LEGACY_EIGHT, parse_progress_details_tsv

# Saved extract of tab BB08J2. Used only after a live read fails.
SAVED_PROGRESS_DETAILS_TSV = (
    GROUP / "evolution_20261008" / "overview_sheet_20261012" / "progress_details_BB08J2.tsv"
)

# Share link the user supplied. login_t is part of that link, not a separate secret.
DEFAULT_SHEET_URL = (
    "https://docs.qq.com/sheet/DQXFTS21ieUVGTEpJ?tab=BB08J2&login_t=1791780626031"
)
DOC_ID = "DQXFTS21ieUVGTEpJ"
TAB_ID = "BB08J2"
TAB_NAME = "Progress Details"

SHEET_COLUMNS = [
    "sheet_animal_name",
    "sheet_fmost_id",
    "sheet_fmost_cell_kind",
    "sheet_injection_sites",
    "sheet_tracing_status",
    "in_legacy_eight",
    "neuron_screen_status",
    "anatomical_acceptance",
]


class OnlineSheetError(RuntimeError):
    """The Progress Details grid could not be read. No animal row was invented."""


def opendoc_url(sheet_url: str) -> str:
    """Build the sheet's own opendoc URL from a docs.qq.com sheet link."""
    parsed = urllib.parse.urlparse(sheet_url)
    query = urllib.parse.parse_qs(parsed.query)
    tab = (query.get("tab") or [TAB_ID])[0]
    if tab != TAB_ID:
        raise OnlineSheetError(f"Refusing tab {tab!r}; Progress Details is {TAB_ID}")
    params = {
        "tab": tab,
        "id": DOC_ID,
        "u": "",
        "noEscape": "1",
        "normal": "1",
        "outformat": "1",
        "wb": "1",
        "startrow": "0",
        "endrow": "400",
    }
    if "login_t" in query and query["login_t"]:
        params["login_t"] = query["login_t"][0]
    return "https://docs.qq.com/dop-api/opendoc?" + urllib.parse.urlencode(params)


def fetch_progress_details_tsv(sheet_url: str = DEFAULT_SHEET_URL, timeout: float = 30) -> str:
    """Fetch tab BB08J2. Raise if the response is not the cell grid."""
    url = opendoc_url(sheet_url)
    request = urllib.request.Request(url, headers={"User-Agent": "projectome-overview"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise OnlineSheetError(
            f"Progress Details opendoc returned HTTP {exc.code}. "
            "A link without the share query is rejected with 401."
        ) from exc
    except urllib.error.URLError as exc:
        raise OnlineSheetError(f"Progress Details opendoc could not be reached: {exc.reason}") from exc
    if b"250432" not in raw or "\t747\t".encode() not in raw:
        raise OnlineSheetError(
            "Progress Details opendoc returned workbook metadata, not the cell grid. "
            "The follow-up get/sheet call needs a browser session and was not used to invent rows."
        )
    raise OnlineSheetError(
        "Progress Details payload contained cell markers but no TSV decoder is available for that body."
    )


def load_progress_details(
    sheet_url: str | None = None,
    snapshot: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load Progress Details animals. Live grid first, then a saved TSV.

    An omitted snapshot falls back to ``progress_details_BB08J2.tsv`` when
    that file exists. A missing file raises OnlineSheetError. No animal row
    is invented, and an empty animal cell stays empty.
    """
    url = sheet_url or DEFAULT_SHEET_URL
    meta = {
        "url": url,
        "tab_id": TAB_ID,
        "tab_name": TAB_NAME,
        "fetch": "",
        "fetch_error": "",
        "snapshot": "",
    }
    try:
        text = fetch_progress_details_tsv(url)
        meta["fetch"] = "live_opendoc"
    except OnlineSheetError as exc:
        meta["fetch_error"] = str(exc)
        path = SAVED_PROGRESS_DETAILS_TSV if snapshot is None else Path(snapshot)
        if not path.is_file():
            raise OnlineSheetError(f"Live read failed and snapshot is missing: {path}") from exc
        text = path.read_text(encoding="utf-8")
        meta["fetch"] = "tsv_fallback"
        meta["snapshot"] = str(path.resolve())
    animals = parse_progress_details_tsv(text)
    meta["animal_rows"] = int(len(animals))
    return animals, meta


def _blank_row(columns: list[str]) -> dict:
    return {column: "" for column in columns}


def merge_sheet_into_progress(progress: pd.DataFrame, animals: pd.DataFrame) -> pd.DataFrame:
    """Attach useful sheet fields. Existing eight pairs stay. New animals are appended.

    Count cells on appended animals stay empty. Channel ids that the sheet
    does not list are not created.
    """
    eight = set(zip(progress["monkey_id"].astype(str), progress["fmost_id"].astype(str)))
    missing = [pair for pair in LEGACY_EIGHT.items() if pair not in eight]
    if missing:
        raise ValueError(f"Progress table is missing legacy pairs {missing}")

    by_sample: dict[str, dict] = {}
    for record in animals.to_dict("records"):
        token = str(record["sheet_fmost_id"]).strip()
        if token in set(LEGACY_EIGHT.values()):
            if token in by_sample:
                raise ValueError(f"Progress Details lists sample {token} twice")
            by_sample[token] = record
    absent = set(LEGACY_EIGHT.values()) - set(by_sample)
    if absent:
        raise ValueError(f"Progress Details has no row for legacy samples {sorted(absent)}")

    out = progress.copy()
    names, sites, tracing, kinds = [], [], [], []
    for monkey_id, fmost_id in zip(out["monkey_id"].astype(str), out["fmost_id"].astype(str)):
        record = by_sample[fmost_id]
        sheet_name = str(record["sheet_animal_name"])
        if sheet_name != monkey_id and not sheet_name.startswith(monkey_id + " ") and not sheet_name.startswith(monkey_id + "("):
            raise ValueError(f"Progress Details animal {sheet_name!r} does not match monkey {monkey_id}")
        names.append(sheet_name)
        sites.append(record["sheet_injection_sites"])
        tracing.append(record["sheet_tracing_status"])
        kinds.append(record["sheet_fmost_cell_kind"])
    out["sheet_animal_name"] = names
    out["sheet_fmost_id"] = out["fmost_id"].astype(str)
    out["sheet_fmost_cell_kind"] = kinds
    out["sheet_injection_sites"] = sites
    out["sheet_tracing_status"] = tracing
    out["in_legacy_eight"] = "yes"
    out["neuron_screen_status"] = "cohort_progress_counts"
    out["anatomical_acceptance"] = "False"

    known_samples = set(out["fmost_id"].astype(str))
    extras = []
    for record in animals.to_dict("records"):
        sample = str(record["sheet_fmost_id"]).strip()
        if sample in known_samples:
            continue
        row = _blank_row(list(out.columns))
        row["monkey_id"] = record["sheet_animal_name"]
        row["fmost_id"] = record["sheet_fmost_id"]
        row["sheet_animal_name"] = record["sheet_animal_name"]
        row["sheet_fmost_id"] = record["sheet_fmost_id"]
        row["sheet_fmost_cell_kind"] = record["sheet_fmost_cell_kind"]
        row["sheet_injection_sites"] = record["sheet_injection_sites"]
        row["sheet_tracing_status"] = record["sheet_tracing_status"]
        row["in_legacy_eight"] = "no"
        row["neuron_screen_status"] = "not_screened"
        row["anatomical_acceptance"] = "False"
        extras.append(row)
    if extras:
        out = pd.concat([out, pd.DataFrame(extras)], ignore_index=True)
    return out
