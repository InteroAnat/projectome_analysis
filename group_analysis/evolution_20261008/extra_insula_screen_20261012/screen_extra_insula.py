"""Screen inventory datasets outside the eight-monkey table.

Reads saved coordinates and existing SWC-root ARM fields. Looks up portal
soma coordinates in ARM with the current rint rule. Does not relabel neurons,
edit the 462 ledger, or write maps.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "group_analysis" / "scripts"))
from audit_insula_inventory import load_boxes, spatial_screen  # noqa: E402
from insula_label_set import build_insula_label_set, normalize_label  # noqa: E402

EIGHT = {
    "251637", "252383", "252714", "252790",
    "252718", "252527", "252385", "252384",
}
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / (
    "group_analysis/evolution_20261008/classification/coarse_insula_review_20261009/"
    "distance_priority_v2_20261009/all_neuron_review_manifest_distance_corrected.csv"
)
PRIOR29 = ROOT / (
    "group_analysis/evolution_20261008/soma_origin_maps_20261010/source_investigation/"
    "dataset250432_selected29_candidates.csv"
)
BOXES = ROOT / "group_analysis/staging_20260926/reference/251637_subregion_bboxes_folded.csv"
ATLAS = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz"
KEY = ROOT / "atlas/ARM_key_all.txt"
SAMPLE_INVENTORY = ROOT / "group_analysis/evolution_20261008/inventory/sample_inventory.csv"
LIVE_SAMPLE_INVENTORY = ROOT / (
    "group_analysis/evolution_20261008/inventory/live_inventory_20261008/sample_inventory.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def channel_base(sample: str) -> str:
    lower = sample.lower()
    if lower.endswith("ch2"):
        return sample[: -len("ch2")]
    return sample


def key_tables():
    names, full = {}, {}
    with KEY.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            index = int(row["Index"])
            names[index] = row["Abbreviation"]
            full[index] = row["Full_Name"]
    return names, full


def lookup_portal(row, arm, names, full, insula):
    if row.get("coordinate_status") != "present":
        return {"status": "unassessed", "index": "", "abbreviation": "", "full_name": "", "insula": "unassessed"}
    try:
        xyz_um = [float(row[f"nmt_{axis}_mm"]) * 1000.0 for axis in "xyz"]
    except (TypeError, ValueError):
        return {"status": "unassessed", "index": "", "abbreviation": "", "full_name": "", "insula": "unassessed"}
    encoded = np.asarray(xyz_um, dtype=float) / 250.0
    if encoded.shape != (3,) or not np.all(np.isfinite(encoded)):
        return {"status": "unassessed", "index": "", "abbreviation": "", "full_name": "", "insula": "unassessed"}
    voxel = np.rint(encoded).astype(int)
    if not np.all((voxel >= 0) & (voxel < np.asarray(arm.shape))):
        return {"status": "out_of_bounds", "index": "", "abbreviation": "", "full_name": "", "insula": "unassessed"}
    index = int(arm[tuple(voxel)])
    if index == 0:
        return {
            "status": "atlas_label_0_unassigned", "index": 0, "abbreviation": "0",
            "full_name": "Atlas label 0 (unassigned)", "insula": "false",
        }
    abbreviation = names.get(index, f"Unknown_{index}")
    return {
        "status": "atlas_label", "index": index, "abbreviation": abbreviation,
        "full_name": full.get(index, ""),
        "insula": "true" if normalize_label(abbreviation) in insula else "false",
    }


def swc_record(row, names, full, insula):
    status = row.get("own_lookup_status") or ""
    if status not in {"atlas_label", "atlas_background"}:
        return {"status": "unassessed", "index": "", "abbreviation": "", "full_name": "", "insula": "unassessed"}
    index = int(row["own_atlas_id"])
    if index == 0:
        return {
            "status": "atlas_label_0_unassigned", "index": 0, "abbreviation": "0",
            "full_name": "Atlas label 0 (unassigned)", "insula": "false",
        }
    abbreviation = names.get(index, row.get("own_atlas_label") or f"Unknown_{index}")
    return {
        "status": status, "index": index, "abbreviation": abbreviation,
        "full_name": full.get(index, ""),
        "insula": "true" if normalize_label(abbreviation) in insula else "false",
    }


def existing_source(row):
    raw = row.get("own_source_paths") or ""
    if raw:
        try:
            paths = json.loads(raw)
        except json.JSONDecodeError:
            paths = []
        for path in paths:
            if (ROOT / path).exists():
                return path
    raw_soma = row.get("portal_soma_source") or ""
    if raw_soma:
        try:
            meta = json.loads(raw_soma)
        except json.JSONDecodeError:
            meta = {}
        if isinstance(meta, dict) and meta.get("file"):
            return meta["file"]
    return ""


def evidence_class(arm_insula, spatial):
    if arm_insula == "conflict":
        return "conflict"
    arm_yes = arm_insula == "true"
    spatial_yes = spatial == "true"
    if arm_yes and spatial_yes:
        return "both"
    if arm_yes:
        return "ARM insula"
    if spatial_yes:
        return "spatial candidate"
    if arm_insula == "unassessed" or spatial == "unassessed":
        return "unassessed"
    return "neither"


def primary_arm(swc, portal):
    if swc["insula"] != "unassessed" and portal["insula"] != "unassessed" and swc["insula"] != portal["insula"]:
        return {
            "index": "", "abbreviation": "", "full_name": "", "insula": "conflict",
            "source": "conflict_swc_root_vs_portal_coordinate",
        }
    if swc["insula"] != "unassessed":
        return {**swc, "source": "swc_root_manifest"}
    if portal["insula"] != "unassessed":
        return {**portal, "source": "portal_soma_coordinate"}
    return {
        "index": "", "abbreviation": "", "full_name": "", "insula": "unassessed",
        "source": "unassessed",
    }


def main():
    names, full = key_tables()
    insula, _ = build_insula_label_set(str(KEY))
    boxes = load_boxes(BOXES)
    atlas = np.asarray(nib.load(str(ATLAS)).dataobj)[..., 0, 5]
    rows = read_rows(MANIFEST)
    prior_ids = {row["NeuronID"] for row in read_rows(PRIOR29)}
    extra = [row for row in rows if row["sample"] not in EIGHT]
    spatial_mismatches = []
    output_rows = []
    for row in extra:
        portal = lookup_portal(row, atlas, names, full, insula)
        swc = swc_record(row, names, full, insula)
        if row.get("coordinate_status") == "present":
            screen = spatial_screen([float(row[f"nmt_{axis}_mm"]) * 1000.0 for axis in "xyz"], boxes, 2.0)
            recomputed = set(filter(None, screen["bbox_hits"].split(";")))
            stored = set(filter(None, (row.get("bbox_hits") or "").split(";")))
            if recomputed != stored:
                spatial_mismatches.append(f"{row['sample']}|{row['neuron_id']}")
            spatial = "true" if recomputed else "false"
            bbox = screen["bbox_hits"]
        else:
            spatial = "unassessed"
            bbox = ""
        arm = primary_arm(swc, portal)
        conflict = swc["insula"] != "unassessed" and portal["insula"] != "unassessed" and swc["insula"] != portal["insula"]
        output_rows.append({
            "SampleID": row["sample"],
            "NeuronID": row["neuron_id"],
            "monkey_id": "unknown",
            "ARM_index": arm["index"],
            "ARM_abbreviation": arm["abbreviation"],
            "ARM_full_name": arm["full_name"],
            "ARM_lookup_source": arm["source"],
            "ARM_insula": arm["insula"],
            "passes_spatial_screen": spatial,
            "bbox_hits": bbox,
            "evidence_class": evidence_class(arm["insula"], spatial),
            "swc_root_ARM_index": swc["index"],
            "swc_root_ARM_full_name": swc["full_name"],
            "swc_root_ARM_insula": swc["insula"],
            "portal_coordinate_ARM_index": portal["index"],
            "portal_coordinate_ARM_full_name": portal["full_name"],
            "portal_coordinate_ARM_insula": portal["insula"],
            "arm_source_conflict": "true" if conflict else "false",
            "coordinate_status": row.get("coordinate_status") or "",
            "source_path": existing_source(row),
            "in_prior_250432_union29": "yes" if row["sample"] == "250432" and row["neuron_id"] in prior_ids else (
                "no" if row["sample"] == "250432" else "not_applicable"
            ),
        })

    union = {
        row["NeuronID"] for row in output_rows
        if row["SampleID"] == "250432" and (
            row["swc_root_ARM_insula"] == "true" or row["passes_spatial_screen"] == "true"
        )
    }
    if union != prior_ids:
        raise SystemExit(f"250432 union {len(union)} != prior 29 {len(prior_ids)}; only in union {sorted(union-prior_ids)[:10]}; only in prior {sorted(prior_ids-union)[:10]}")
    if spatial_mismatches:
        raise SystemExit(f"{len(spatial_mismatches)} spatial-screen mismatches, first {spatial_mismatches[:5]}")

    fieldnames = list(output_rows[0])
    table_path = OUT / "extra_insula_candidates_20261012.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    by_sample = {}
    for row in output_rows:
        item = by_sample.setdefault(row["SampleID"], Counter())
        item["neurons_screened"] += 1
        if row["coordinate_status"] == "present":
            item["neurons_with_coordinates"] += 1
        else:
            item["neurons_missing_coordinates"] += 1
        item[row["evidence_class"]] += 1
        if row["ARM_insula"] == "true":
            item["ARM_insula_n"] += 1
        if row["passes_spatial_screen"] == "true":
            item["spatial_pass_n"] += 1
        if row["swc_root_ARM_insula"] != "unassessed":
            item["swc_root_lookup_n"] += 1

    sample_rows = read_rows(SAMPLE_INVENTORY)
    live_rows = read_rows(LIVE_SAMPLE_INVENTORY)
    inventory_ids = {row["sample"] for row in sample_rows}
    live_ids = {row["sample"] for row in live_rows}
    summary = []
    for sample in sorted(set(by_sample) | (inventory_ids - EIGHT) | (live_ids - EIGHT)):
        counts = by_sample.get(sample, Counter())
        base = channel_base(sample)
        if sample != base:
            role = "channel_variant"
        elif sample in {"252334", "252985"}:
            role = "tracker_alias_conflict"
        else:
            role = "dataset_without_verified_monkey_id"
        summary.append({
            "SampleID": sample,
            "channel_base": base,
            "dataset_role": role,
            "monkey_id": "unknown",
            "in_parent_sample_inventory": sample in inventory_ids,
            "in_live_sample_inventory": sample in live_ids,
            "neurons_screened": counts["neurons_screened"],
            "neurons_with_coordinates": counts["neurons_with_coordinates"],
            "neurons_missing_coordinates": counts["neurons_missing_coordinates"],
            "ARM_insula_n": counts["ARM_insula_n"],
            "spatial_pass_n": counts["spatial_pass_n"],
            "both_n": counts["both"],
            "ARM_insula_only_n": counts["ARM insula"],
            "spatial_only_n": counts["spatial candidate"],
            "neither_n": counts["neither"],
            "unassessed_n": counts["unassessed"],
            "conflict_n": counts["conflict"],
            "swc_root_lookup_n": counts["swc_root_lookup_n"],
        })
    summary_path = OUT / "extra_dataset_summary_20261012.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    subset = [row for row in output_rows if row["SampleID"] == "250432"]
    cross = {
        "prior_union_n": len(prior_ids),
        "recomputed_union_n": len(union),
        "union_ids_match_prior29": True,
        "swc_ARM_insula": sum(row["swc_root_ARM_insula"] == "true" for row in subset),
        "spatial_pass": sum(row["passes_spatial_screen"] == "true" for row in subset),
        "both": sum(row["evidence_class"] == "both" for row in subset),
        "ARM_insula_only": sum(row["evidence_class"] == "ARM insula" for row in subset),
        "spatial_only": sum(row["evidence_class"] == "spatial candidate" for row in subset),
        "ARM_and_not_in_bbox": sorted(
            row["NeuronID"] for row in subset
            if row["swc_root_ARM_insula"] == "true" and row["passes_spatial_screen"] != "true"
        ),
        "both_ids": sorted(row["NeuronID"] for row in subset if row["evidence_class"] == "both"),
        "spatial_only_ids": sorted(row["NeuronID"] for row in subset if row["evidence_class"] == "spatial candidate"),
    }
    totals = Counter(row["evidence_class"] for row in output_rows)
    hash_paths = [
        MANIFEST, PRIOR29, BOXES, ATLAS, KEY, SAMPLE_INVENTORY, LIVE_SAMPLE_INVENTORY,
        ROOT / "group_analysis/docs/dataset_status_manifest.csv",
        ROOT / "group_analysis/docs/monkey_experiment_data_summary.xlsx",
        ROOT / "group_analysis/docs/data_progress_table_20260701_1416.csv",
        ROOT / "group_analysis/docs/data_progress_table_20260713_1649.csv",
        ROOT / "group_analysis/docs/data_progress_table_20260727_1620.csv",
        ROOT / "group_analysis/portal_audit_20260926/portal_macaque_samples.csv",
        ROOT / "notes/bulk_visual_review_20261002/portal_snapshot/250432_sample_info.json",
        ROOT / "group_analysis/evolution_20261008/inventory/live_inventory_20261008/live_snapshots/20261008T152413Z/catalog_sample_info.json",
        ROOT / "projectome.sqlite",
        ROOT / "group_analysis/evolution_20261008/soma_origin_maps_20261010/source_investigation/dataset250432_all234_source_inventory.csv",
        ROOT / "group_analysis/step1_results/250432_20260927_151627_region_analysis/reports/analysis_summary.txt",
    ]
    provenance = {
        "created_local": datetime.now().astimezone().isoformat(),
        "anatomical_acceptance": False,
        "added_to_462_ledger": False,
        "between_animal_map": False,
        "t_map": False,
        "eight_monkey_samples_excluded": sorted(EIGHT),
        "neurons_in_manifest": len(rows),
        "extra_neurons_screened": len(output_rows),
        "evidence_class_totals": dict(totals),
        "coordinate_present": sum(row["coordinate_status"] == "present" for row in output_rows),
        "coordinate_missing": sum(row["coordinate_status"] != "present" for row in output_rows),
        "spatial_mismatches": spatial_mismatches,
        "dataset250432_crosscheck": cross,
        "parent_sample_inventory_n": len(sample_rows),
        "live_sample_inventory_n": len(live_rows),
        "samples_only_in_live_inventory": sorted(live_ids - inventory_ids),
        "samples_only_in_parent_inventory": sorted(inventory_ids - live_ids),
        "samples_in_neuron_manifest_not_live_inventory": sorted({row["sample"] for row in rows} - live_ids),
        "atlas_slice": "[..., 0, 5]",
        "arm_policy": "current_rint_zero_center = rint(soma_um / 250); label 0 is unassigned, not white matter",
        "spatial_policy": "folded X = abs(mm - 32); Henry/reference q0.005-q0.995 bounds times 0.25 mm; inclusive 2 mm pad",
        "sources": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in hash_paths},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (OUT / "run_stats.json").write_text(json.dumps({
        "totals": dict(totals),
        "cross": cross,
        "summary": summary,
        "coordinate_present": provenance["coordinate_present"],
        "coordinate_missing": provenance["coordinate_missing"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"extra": len(output_rows), "totals": dict(totals), "cross": {k: cross[k] for k in cross if not k.endswith("ids") and k != "ARM_and_not_in_bbox"}}, indent=2))


if __name__ == "__main__":
    main()
