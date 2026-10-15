"""Recount every manifest neuron with the existing insula rules and write a summary.

Does not edit the 462 ledger, canonical workbooks, SWCs, or published maps.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "group_analysis/evolution_20261008/extra_insula_screen_20261012"))
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
import screen_extra_insula as screen  # noqa: E402
from group_analysis.project_overview.sheet_overview import parse_progress_details_tsv  # noqa: E402

OUT = Path(__file__).resolve().parent
SHEET_TSV = ROOT / "group_analysis/evolution_20261008/overview_sheet_20261012/progress_details_BB08J2.tsv"
LEGACY = ROOT / "group_analysis/evolution_20261008/inventory/legacy_overview_20261009/per_monkey_insula_inventory.csv"
LEDGER = ROOT / "notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv"
ENDPOINT = ROOT / "group_analysis/evolution_20261008/soma_origin_maps_20261010/count_reconciliation/per_monkey_counts_history_and_evidence.csv"
PRIOR_SUMMARY = ROOT / "group_analysis/evolution_20261008/extra_insula_screen_20261012/extra_dataset_summary_20261012.csv"

EIGHT_ORDER = ["936", "605", "900", "945", "797", "948", "331", "631"]
EIGHT = {
    "936": "251637", "605": "252383", "631": "252714", "797": "252790",
    "900": "252718", "936": "251637", "945": "252527", "948": "252384", "331": "252385",
}
SAMPLE_TO_EIGHT = {sample: animal for animal, sample in EIGHT.items()}
COMPLETED = PatternFill("solid", fgColor="C6EFCE")
PENDING = PatternFill("solid", fgColor="B8CCE4")
HEADER = PatternFill("solid", fgColor="D9E2F3")


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def blank_counts():
    return Counter()


def add_neuron(counts: Counter, row: dict) -> None:
    counts["neurons_screened"] += 1
    if row["coordinate_status"] == "present":
        counts["neurons_with_coordinates"] += 1
    else:
        counts["neurons_missing_coordinates"] += 1
    arm = row["ARM_insula"] == "true"
    spatial = row["passes_spatial_screen"] == "true"
    if arm:
        counts["ARM_insula_n"] += 1
    if arm and spatial:
        counts["both_n"] += 1
    elif arm:
        counts["ARM_only_n"] += 1
    elif spatial:
        counts["spatial_only_n"] += 1
    if row["passes_spatial_screen"] == "unassessed":
        counts["spatial_unassessed_n"] += 1


def sheet_bindings(animals):
    """Map a single fMOST token, or every digit token in one animal cell, to that animal."""
    by_sample = {}
    named = []
    for record in animals.to_dict("records"):
        name = str(record["sheet_animal_name"]).strip()
        cell = str(record["sheet_fmost_id"]).strip()
        named.append(record)
        tokens = [token for token in cell.split() if token.isdigit()]
        if not tokens:
            continue
        for token in tokens:
            if token in by_sample and by_sample[token]["sheet_animal_name"] != name:
                raise ValueError(f"Sample {token} is named twice on Progress Details")
            by_sample[token] = record
    return by_sample, named


def animal_key(sample: str, by_sample: dict) -> tuple[str, str, str]:
    """Return animal id, assignment source, and whether this sample is a channel."""
    if sample in SAMPLE_TO_EIGHT:
        return SAMPLE_TO_EIGHT[sample], "legacy_eight", "no"
    if sample.lower().endswith("ch2"):
        return "", "channel_not_named_on_progress_details", "yes"
    record = by_sample.get(sample)
    if record:
        return str(record["sheet_animal_name"]).strip(), "progress_details_tab", "no"
    return "", "no_animal_name_in_inventory_or_progress_details", "no"


def main() -> None:
    if not SHEET_TSV.is_file():
        raise FileNotFoundError(SHEET_TSV)
    animals = parse_progress_details_tsv(SHEET_TSV.read_text(encoding="utf-8"))
    hit = animals[animals["sheet_fmost_id"].astype(str).str.strip() == "250432"]
    if len(hit) != 1 or str(hit.iloc[0]["sheet_animal_name"]).strip() != "747":
        raise SystemExit("Saved Progress Details extract does not put 747 on the 250432 row")
    by_sample, named_rows = sheet_bindings(animals)
    if "250432ch2" in by_sample or any("250432ch2" in str(row["sheet_fmost_id"]) for row in named_rows):
        raise SystemExit("Progress Details names 250432ch2; the channel rule must be revisited")

    names, full = screen.key_tables()
    insula, _ = screen.build_insula_label_set(str(screen.KEY))
    boxes = screen.load_boxes(screen.BOXES)
    atlas = np.asarray(nib.load(str(screen.ATLAS)).dataobj)[..., 0, 5]
    manifest = screen.read_rows(screen.MANIFEST)
    prior_ids = {row["NeuronID"] for row in screen.read_rows(screen.PRIOR29)}
    prior_summary = {row["SampleID"]: row for row in read_csv(PRIOR_SUMMARY)}

    per_neuron_250432 = []
    by_sample_counts: dict[str, Counter] = defaultdict(blank_counts)
    spatial_mismatches = []
    for row in manifest:
        portal = screen.lookup_portal(row, atlas, names, full, insula)
        swc = screen.swc_record(row, names, full, insula)
        if row.get("coordinate_status") == "present":
            scanned = screen.spatial_screen(
                [float(row[f"nmt_{axis}_mm"]) * 1000.0 for axis in "xyz"], boxes, 2.0
            )
            recomputed = set(filter(None, scanned["bbox_hits"].split(";")))
            stored = set(filter(None, (row.get("bbox_hits") or "").split(";")))
            if recomputed != stored:
                spatial_mismatches.append(f"{row['sample']}|{row['neuron_id']}")
            spatial = "true" if recomputed else "false"
        else:
            spatial = "unassessed"
        arm = screen.primary_arm(swc, portal)
        counted = {
            "sample": row["sample"],
            "neuron_id": row["neuron_id"],
            "ARM_insula": arm["insula"],
            "passes_spatial_screen": spatial,
            "coordinate_status": row.get("coordinate_status") or "",
            "swc_root_ARM_insula": swc["insula"],
        }
        add_neuron(by_sample_counts[row["sample"]], counted)
        if row["sample"] == "250432":
            per_neuron_250432.append(counted)

    union = {
        row["neuron_id"] for row in per_neuron_250432
        if row["swc_root_ARM_insula"] == "true" or row["passes_spatial_screen"] == "true"
    }
    recount_250432 = by_sample_counts["250432"]
    cross = {
        "neurons_screened": recount_250432["neurons_screened"],
        "ARM_insula_n": recount_250432["ARM_insula_n"],
        "spatial_only_n": recount_250432["spatial_only_n"],
        "both_n": recount_250432["both_n"],
        "union_n": len(union),
        "prior_expected": {"screened": 234, "ARM_insula": 8, "spatial_only": 21, "both": 2, "union": 29},
        "union_ids_match_prior29": union == prior_ids,
        "only_in_recount": sorted(union - prior_ids),
        "only_in_prior29": sorted(prior_ids - union),
        "stored_bbox_mismatches_n": len(spatial_mismatches),
        "stored_bbox_mismatch_examples": spatial_mismatches[:8],
    }
    prior_row = prior_summary.get("250432", {})
    cross["prior_extra_screen_row"] = {
        "neurons_screened": prior_row.get("neurons_screened", ""),
        "ARM_insula_n": prior_row.get("ARM_insula_n", ""),
        "spatial_only_n": prior_row.get("spatial_only_n", ""),
        "both_n": prior_row.get("both_n", ""),
    }

    henry = Counter()
    ledger_endpoint = Counter()
    if LEDGER.is_file():
        for row in read_csv(LEDGER):
            sample = row.get("sample") or row.get("SampleID") or ""
            if sample == "251637" and truthy(row.get("Henry_coarse_INS_visual_evidence", "")):
                henry[sample] += 1
            if truthy(row.get("EndpointEligible", "")):
                ledger_endpoint[sample] += 1
    endpoint_already = {}
    for row in read_csv(ENDPOINT):
        endpoint_already[row["SampleID"]] = row["endpoint_eligible"]

    legacy_status = {}
    for row in read_csv(LEGACY):
        legacy_status[row["fmost_id"]] = row["overview_data_status"]

    groups: dict[tuple, dict] = {}

    def ensure(kind: str, animal: str, source: str) -> dict:
        key = (kind, animal, source)
        if key not in groups:
            groups[key] = {
                "animal_id": animal,
                "identity_source": source,
                "samples": [],
                "counts": Counter(),
                "tracing": "",
                "monkey_data_status": "",
                "channel": "no",
            }
        return groups[key]

    for record in named_rows:
        name = str(record["sheet_animal_name"]).strip()
        cell = str(record["sheet_fmost_id"]).strip()
        tokens = [token for token in cell.split() if token.isdigit()]
        if any(token in SAMPLE_TO_EIGHT for token in tokens):
            continue
        group = ensure("named", name, "progress_details_tab")
        group["tracing"] = record["sheet_tracing_status"]
        group["sheet_no"] = str(record["sheet_no"])
        group["sheet_fmost_cell"] = cell
        if not tokens:
            group["samples"].append("")
        for token in tokens:
            if token not in group["samples"]:
                group["samples"].append(token)
                for field, value in by_sample_counts.get(token, Counter()).items():
                    group["counts"][field] += value

    for animal in EIGHT_ORDER:
        sample = EIGHT[animal]
        group = ensure("eight", animal, "legacy_eight")
        group["samples"] = [sample]
        group["counts"] = by_sample_counts.get(sample, Counter())
        sheet_row = by_sample.get(sample)
        if sheet_row:
            group["tracing"] = sheet_row["sheet_tracing_status"]
        group["monkey_data_status"] = legacy_status.get(sample, "")

    seen_channel_bases = set()
    for sample, counts in sorted(by_sample_counts.items()):
        animal, source, is_channel = animal_key(sample, by_sample)
        if animal:
            continue
        base = screen.channel_base(sample)
        if base in SAMPLE_TO_EIGHT or base in by_sample:
            group = ensure("channel", base, source)
            group["animal_id"] = ""
            group["channel"] = "yes"
            group["channel_of"] = base
        else:
            group = ensure("unresolved", base, source)
            if sample != base:
                group["channel"] = "yes"
        if sample not in group["samples"]:
            group["samples"].append(sample)
            for field, value in counts.items():
                group["counts"][field] += value
        seen_channel_bases.add(base)

    rows = []
    ordered = (
        [groups[("eight", animal, "legacy_eight")] for animal in EIGHT_ORDER]
        + [item for key, item in groups.items() if key[0] == "named"]
        + [item for key, item in groups.items() if key[0] == "channel"]
        + [item for key, item in groups.items() if key[0] == "unresolved"]
    )
    named_order = {str(record["sheet_no"]): i for i, record in enumerate(named_rows)}
    ordered[8:8 + sum(key[0] == "named" for key in groups)] = sorted(
        [item for key, item in groups.items() if key[0] == "named"],
        key=lambda item: named_order.get(next((str(r["sheet_no"]) for r in named_rows if str(r["sheet_animal_name"]).strip() == item["animal_id"]), "999"), 999),
    )

    for item in ordered:
        counts = item["counts"]
        samples = [sample for sample in item["samples"] if sample]
        sample_text = "; ".join(samples)
        if item["animal_id"] == "138":
            sample_text = "221720  221747"
        henry_n = ""
        if item["animal_id"] == "936":
            henry_n = henry.get("251637", 0)
        endpoint = ""
        if len(samples) == 1 and samples[0] in endpoint_already:
            endpoint = endpoint_already[samples[0]]
        screened = counts["neurons_screened"]
        rows.append({
            "animal_id": item["animal_id"],
            "fMOST_samples": sample_text,
            "identity_source": item["identity_source"],
            "channel_variant": item["channel"],
            "channel_of": item.get("channel_of", ""),
            "neurons_screened": screened if screened or samples else "",
            "neurons_with_coordinates": counts["neurons_with_coordinates"] if screened else "",
            "neurons_missing_coordinates": counts["neurons_missing_coordinates"] if screened else "",
            "ARM_insula_n": counts["ARM_insula_n"] if screened else "",
            "ARM_only_n": counts["ARM_only_n"] if screened else "",
            "spatial_only_n": counts["spatial_only_n"] if screened else "",
            "both_n": counts["both_n"] if screened else "",
            "Henry_coarse_INS_n": henry_n,
            "endpoint_eligible_already_computed": endpoint,
            "monkey_data_status": item["monkey_data_status"],
            "progress_details_tracing_status": item["tracing"],
            "anatomical_acceptance": "False",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    # This write is what analysis_table.load_analysis_neuron_table reads.
    # After it changes, refresh project_overview_plot / plot_progress_table.
    csv_path = OUT / "per_animal_insula_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "insula_summary"
    sheet.append(fieldnames)
    for cell in sheet[1]:
        cell.fill = HEADER
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    status_column = fieldnames.index("monkey_data_status") + 1
    for record in rows:
        sheet.append([record[name] for name in fieldnames])
        status = record["monkey_data_status"]
        fill = COMPLETED if status == "Completed" else PENDING if status == "Pending" else None
        if fill is not None:
            sheet.cell(sheet.max_row, status_column).fill = fill
    sheet.auto_filter.ref = sheet.dimensions
    sheet.freeze_panes = "A2"
    sheet.row_dimensions[1].height = 32
    for column in sheet.columns:
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = 18 if letter != "B" else 28
    xlsx_path = OUT / "per_animal_insula_summary.xlsx"
    workbook.save(xlsx_path)

    (OUT / "recount_250432_crosscheck.json").write_text(
        json.dumps(cross, indent=2) + "\n", encoding="utf-8"
    )
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    hits = [
        record for record in rows
        if record["animal_id"] in set(EIGHT_ORDER) | {"747"}
        or (str(record["ARM_insula_n"]).isdigit() and int(record["ARM_insula_n"]) > 0)
        or (str(record["spatial_only_n"]).isdigit() and int(record["spatial_only_n"]) > 0)
        or (str(record["both_n"]).isdigit() and int(record["both_n"]) > 0)
    ]
    (OUT / "recount_provenance.json").write_text(json.dumps({
        "written_at": now,
        "progress_details_tab": "Progress Details",
        "progress_details_tab_id": "BB08J2",
        "progress_details_250432_row": {
            "spreadsheet_row": 141,
            "NO.": "20",
            "注射动物_ID": "747",
            "fMOST_ID": "250432",
            "注射脑区": "L-CLA-1",
            "追踪进度": "completed",
            "read_from": "open view-only grid, tab BB08J2, reconfirmed in this session",
        },
        "250432ch2_on_progress_details": False,
        "manifest_neurons": len(manifest),
        "samples_recounted": len(by_sample_counts),
        "anatomical_acceptance": False,
        "added_to_462_ledger": False,
        "rules": {
            "ARM": "rint(soma_um/250) on ARM_in_NMT_v2.1_sym slice [..., 0, 5]; label 0 is unassigned, not white matter; SWC-root label preferred when present",
            "spatial": "folded X = abs(Soma_NII_X-128); q0.005-q0.995 bounds times 0.25 mm; inclusive 2 mm pad",
            "Henry": "existing selected-ledger flag, counted only for 251637 / 936",
            "endpoint": "copied from the already computed reconciliation table when that sample has a value",
        },
        "crosscheck_250432": cross,
        "ledger_endpoint_eligible_by_sample": dict(ledger_endpoint),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "animals": len(rows),
        "manifest": len(manifest),
        "cross": {key: cross[key] for key in ("neurons_screened", "ARM_insula_n", "spatial_only_n", "both_n", "union_n", "union_ids_match_prior29", "stored_bbox_mismatches_n")},
        "rows_with_hits_or_eight_or_747": len(hits),
        "csv": str(csv_path),
        "xlsx": str(xlsx_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
