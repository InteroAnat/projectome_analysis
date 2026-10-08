"""Read-only, snapshot-based all-macaque injection/source/candidate inventory.

Default inputs are persisted metadata, never raw images or SWCs. Outputs are a
new audit, not a cohort or human tracker. Optional capture is bounded metadata
GETs only; failed endpoints remain missing and never become zero counts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from cohort import MIDLINE_NII_X, NII_VOXEL_MM
from insula_label_set import build_insula_label_set
from region_labels import normalize_portal_region_label, EXPLICIT_UNKNOWN_LABELS

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("group_analysis/evolution_20261008/inventory")
BASE_URL = "http://10.10.48.110/neuronbrowser/api/user"
ADJACENT = {"G", "RI", "RETROINSULA", "CL", "PROM", "SII", "TPT",
            "AREA_7OP", "AREA_44", "F4", "F5"}
UNKNOWN = set(EXPLICIT_UNKNOWN_LABELS)
FALSE_VALUES = {"", "0", "false", "no", "none", "null"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(path, root, checked_at="", url=""):
    path = Path(path)
    try:
        name = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        name = str(path.resolve())
    return {"file": name, "sha256": sha256(path), "checked_at": checked_at, "url": url}


def truth(value):
    return str(value or "").strip().lower() not in FALSE_VALUES


def base_label(value):
    """Strip only the terminal numeric atlas ID; preserve area and side names."""
    return normalize_portal_region_label(value)


def is_macaque(row):
    text = str(row.get("spicies") or row.get("species") or "").lower()
    return str(row.get("project_id") or "").lower() == "monkey" or any(
        token in text for token in ("monkey", "macaque", "macaca", "rhesus", "猕猴", "恒河"))


def finite_xyz(values):
    try:
        xyz = tuple(float(v) for v in values)
    except (ValueError, TypeError):
        return None
    return xyz if len(xyz) == 3 and all(math.isfinite(v) for v in xyz) else None


def timestamp_order(value):
    if not value:
        return float("-inf")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()


def load_boxes(path):
    boxes = []
    for row in read_csv(path):
        if row.get("x_frame") != "Soma_NII_X_folded":
            raise ValueError("Inventory requires the folded-X reference, not bilateral X bounds")
        if float(row["midline_nii_x"]) != MIDLINE_NII_X or float(row["nii_voxel_mm"]) != NII_VOXEL_MM:
            raise ValueError("Folded reference geometry disagrees with cohort constants")
        boxes.append((row["sub_region"], *(float(row[f"{axis}_{bound}"]) * NII_VOXEL_MM
                      for axis in "XYZ" for bound in ("lo_q005", "hi_q995"))))
    return boxes


def spatial_screen(xyz_um, boxes, pad_mm=2.0):
    """NMT portal um -> mm; folded X=abs(X_mm - 32 mm). No laterality acceptance."""
    xyz = finite_xyz(xyz_um) if xyz_um is not None else None
    if xyz is None:
        return {"coordinate_status": "missing", "bbox_status": "missing_coordinates",
                "bbox_hits": "", "coordinate_side_candidate": ""}
    x, y, z = (v / 1000.0 for v in xyz)
    midline = MIDLINE_NII_X * NII_VOXEL_MM
    folded = abs(x - midline)
    hits = [label for label, xl, xh, yl, yh, zl, zh in boxes
            if xl - pad_mm <= folded <= xh + pad_mm and yl - pad_mm <= y <= yh + pad_mm
            and zl - pad_mm <= z <= zh + pad_mm]
    return {"coordinate_status": "present", "bbox_status": "inside_candidate" if hits else "outside_screen",
            "bbox_hits": ";".join(hits), "nmt_x_mm": x, "nmt_y_mm": y, "nmt_z_mm": z,
            "folded_x_mm": folded,
            "coordinate_side_candidate": "L" if x < midline else "R" if x > midline else "midline"}


def classify_neuron(region, xyz_um, boxes, insula, excluded=False, pad_mm=2.0):
    label = base_label(region)
    result = spatial_screen(xyz_um, boxes, pad_mm)
    atlas_ins = label in insula
    reasons = []
    if atlas_ins:
        reasons.append("atlas_INS")
    elif label == "PRCO":
        reasons.append("atlas_PrCO")
    elif label in UNKNOWN:
        reasons.append("atlas_explicit_unknown")
    elif label in ADJACENT:
        reasons.append("atlas_adjacent_review")
    elif not label:
        reasons.append("missing_region_metadata")
    if result["bbox_hits"]:
        reasons.append("folded_251637_bbox_candidate")
    potential = bool(atlas_ins or result["bbox_hits"]) and not excluded
    if not excluded and not atlas_ins and result["coordinate_status"] == "missing":
        potential = None
    union_g = True if label == "G" and not excluded else potential
    return {**result, "portal_region_base": label, "atlas_INS": atlas_ins, "atlas_G": label == "G",
            "screen_reasons": ";".join(reasons), "excluded": bool(excluded),
            "potential_INS_candidate": potential, "atlasINS_or_G_or_foldedBox_candidate": union_g,
            "anatomical_membership": "not_assessed", "layer_identity": "not_assessed",
            "VEN_identity": "not_assessed"}


def exact_records(records, sample, issues, kind):
    """Never normalize a sample/channel or neuron name; reject ambiguous duplicates."""
    grouped = {}
    for row in records:
        if str(row.get("sampleid") or "") != sample:
            issues.append({"sample": sample, "kind": kind, "issue": "sample_identity_mismatch", "record": row})
            continue
        name = str(row.get("name") or "")
        if not name:
            issues.append({"sample": sample, "kind": kind, "issue": "missing_neuron_id", "record": row})
            continue
        grouped.setdefault(name, []).append(row)
    valid = {}
    for name, rows in grouped.items():
        if len(rows) != 1:
            issues.append({"sample": sample, "neuron_id": name, "kind": kind,
                           "issue": "ambiguous_duplicate_identity", "count": len(rows)})
        else:
            valid[name] = rows[0]
    return valid


def load_snapshots(root, directories, issues):
    """Latest dated endpoint wins, including a latest failure. Verify saved hashes."""
    endpoints = {}
    for directory in directories:
        manifest_path = Path(directory) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["records"]:
            sample, kind = str(record.get("sample", "")), record["kind"]
            path = root / record["file"] if record.get("file") else None
            if path is not None and not path.exists():
                path = Path(directory) / Path(record["file"].replace("\\", "/")).name
            payload, status = None, "missing"
            if record.get("http_status") == 200 and path is not None and path.exists():
                if record.get("sha256") and sha256(path) != record["sha256"]:
                    raise ValueError(f"Snapshot hash mismatch: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                status = "ok" if isinstance(payload, list) else "invalid_payload"
            elif record.get("http_status") != 200:
                status = "request_failed"
            key = (sample, kind)
            if key not in endpoints or timestamp_order(record.get("checked_at", "")) >= timestamp_order(endpoints[key]["checked_at"]):
                endpoints[key] = {"data": payload if status == "ok" else None, "status": status,
                                  "checked_at": record.get("checked_at", ""), "record": record,
                                  "source": provenance(path, root, record.get("checked_at", ""), record.get("url", ""))
                                  if path is not None and path.exists() else dict(record)}
                if status != "ok":
                    issues.append({"sample": sample, "kind": kind, "issue": status,
                                   "checked_at": record.get("checked_at", ""), "error": record.get("error", "")})
    return endpoints


def parse_series_evidence(path):
    """Read the saved scoped inventory; this is dated evidence, not a live check."""
    text = Path(path).read_text(encoding="utf-8")
    date = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", text).group(1)
    series = {}
    for line in text.splitlines():
        match = re.match(r"\| (\d{6})(?: reference)? \| `([^`]+)` \| ([\d,]+) \|", line)
        if match:
            series[match[1]] = {"status": "verified_copied_series", "relative_path": match[2],
                                "slice_count": int(match[3].replace(",", "")), "checked_at": date,
                                "channel": "CH1", "source": str(path)}
    missing = re.search(r"(\d{6}[^\n]+) have no copied 5micron series", text)
    if missing:
        for sample in re.findall(r"\b\d{6}\b", missing[1]):
            series[sample] = {"status": "not_copied_at_scoped_root", "checked_at": date, "source": str(path)}
    return series


def reconcile_series(sample, tracker, series, directory_observation=None):
    evidence = series.get(sample, {})
    status = evidence.get("status", "not_in_saved_source_inventory")
    directory_date, directory_status = "", "not_checked"
    if directory_observation:
        directory_date = directory_observation["checked_at"]
        directories = directory_observation["directories"]
        present = any(name.startswith(sample + "-") or name.startswith(sample + "_") for name in directories)
        directory_status = "present_at_scoped_root" if present else "absent_at_scoped_root"
        if not present:
            status = "not_copied_at_scoped_root"
    claim = tracker.get("5_micron_data_copied", "")
    if status == "verified_copied_series":
        discrepancy = "tracker_no_vs_verified_series" if claim and not claim.lower().startswith("yes") else ""
    elif claim.lower().startswith("yes"):
        discrepancy = "tracker_yes_without_matching_verified_series"
    else:
        discrepancy = ""
    return {"copied5um_status": status,
            "copied5um_evidence_date": directory_date if directory_status == "absent_at_scoped_root" else evidence.get("checked_at", ""),
            "copied5um_slice_evidence_date": evidence.get("checked_at", "") if evidence.get("status") == "verified_copied_series" else "",
            "copied5um_relative_path": evidence.get("relative_path", ""),
            "copied5um_slice_count": evidence.get("slice_count", ""),
            "copied5um_channel": evidence.get("channel", ""),
            "copied5um_directory_observation_date": directory_date,
            "copied5um_directory_observation_status": directory_status,
            "copied5um_directory_observation_root": directory_observation.get("root", "") if directory_observation else "",
            "copied5um_tracker_claim": claim, "copied5um_discrepancy": discrepancy,
            "copied5um_evidence_scope": "saved slice inventory plus separately dated directory observation; CH1 not verified PI/cytoarchitecture"}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_inventory(root, output, snapshot_dirs=None, pad_mm=2.0, series_observation_path=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output != root / DEFAULT_OUTPUT and root / DEFAULT_OUTPUT not in output.parents:
        raise ValueError(f"Output must stay under {root / DEFAULT_OUTPUT}")
    if pad_mm < 0 or not math.isfinite(pad_mm):
        raise ValueError("Padding must be finite and nonnegative")
    sources = {
        "catalog": root / "group_analysis/portal_audit_20260926/portal_macaque_samples.csv",
        "aggregate": root / "group_analysis/portal_audit_20260928/portal_refresh_vs_20260926.csv",
        "tracker": root / "group_analysis/docs/dataset_status_manifest.csv",
        "visual_manifest": root / "group_analysis/visual_review_20261002/manifest/identity_manifest.csv",
        "boxes": root / "group_analysis/staging_20260926/reference/251637_subregion_bboxes_folded.csv",
        "series": root / "notes/bulk_visual_review_20261002/server_inventory.md",
        "labels": root / "group_analysis/scripts/insula_label_set.py",
        "atlas_key": root / "atlas/ARM_key_all.txt",
        "cohort_constants": root / "group_analysis/scripts/cohort.py",
        "portal_normalizer": root / "main_scripts/region_labels.py",
    }
    source_prov = {key: provenance(path, root) for key, path in sources.items()}
    if series_observation_path is None:
        series_observation_path = root / DEFAULT_OUTPUT / "source_root_observation_20261008.json"
    directory_observation = None
    if Path(series_observation_path).exists():
        directory_observation = json.loads(Path(series_observation_path).read_text(encoding="utf-8"))
        source_prov["directory_observation"] = provenance(series_observation_path, root, directory_observation["checked_at"])
    issues = []
    snapshot_dirs = snapshot_dirs or [root / "notes/bulk_visual_review_20261002/portal_snapshot"]
    endpoints = load_snapshots(root, snapshot_dirs, issues)
    source_prov["snapshot_manifests"] = [provenance(Path(directory) / "manifest.json", root) for directory in snapshot_dirs]
    catalog = {r["fMOST_id"]: r for r in read_csv(sources["catalog"])}
    catalog_prov = {sid: source_prov["catalog"] for sid in catalog}
    # Prefix-filtered sampleInfo may return channel siblings. Preserve each exact
    # fMOST_id separately, and never assign it its parent's soma/series evidence.
    for (requested, kind), endpoint in endpoints.items():
        if kind != "sample_info" or endpoint["status"] != "ok":
            continue
        for info in endpoint["data"]:
            sid = str(info.get("fMOST_id") or "")
            if sid and (is_macaque(info) or sid in catalog):
                catalog[sid] = info
                catalog_prov[sid] = endpoint["source"]
    aggregate = {r["sample"]: r for r in read_csv(sources["aggregate"])}
    full_catalog = endpoints.get(("", "sample_info"))
    current_ids = {str(r.get("fMOST_id")) for r in full_catalog["data"]} if full_catalog and full_catalog["status"] == "ok" else None
    trackers = {r["fmost_id"]: r for r in read_csv(sources["tracker"])}
    visual = {}
    for row in read_csv(sources["visual_manifest"]):
        key = (row["SampleID"], row["NeuronID"])
        if key in visual:
            raise ValueError(f"Duplicate visual manifest identity: {key}")
        visual[key] = row
    series, boxes = parse_series_evidence(sources["series"]), load_boxes(sources["boxes"])
    insula, _ = build_insula_label_set(str(sources["atlas_key"]))
    all_samples = sorted(set(catalog) | set(trackers) | set(aggregate)
                         | {sid for sid, _ in visual} | {sid for sid, _ in endpoints if sid})
    sample_rows, neuron_rows = [], []
    for sample in all_samples:
        info, tracker = catalog.get(sample, {}), trackers.get(sample, {})
        neuron_ep, soma_ep = endpoints.get((sample, "neurons")), endpoints.get((sample, "soma"))
        neurons = exact_records(neuron_ep["data"], sample, issues, "neurons") if neuron_ep and neuron_ep["status"] == "ok" else {}
        somata = exact_records(soma_ep["data"], sample, issues, "soma") if soma_ep and soma_ep["status"] == "ok" else {}
        selected = {nid: row for (sid, nid), row in visual.items() if sid == sample}
        local_rows = []
        for name in sorted(set(neurons) | set(somata) | set(selected)):
            neuron, soma, review = neurons.get(name, {}), somata.get(name, {}), selected.get(name, {})
            xyz = finite_xyz(soma.get(k) for k in ("somax", "somay", "somaz"))
            coord_source = "portal_getSoma_NMT_physical_um" if xyz is not None else "missing"
            if xyz is None and review.get("nmt_frame"):
                xyz_nii = finite_xyz(review.get(k) for k in ("nmt_nii_x", "nmt_nii_y", "nmt_nii_z"))
                if xyz_nii is not None:
                    xyz = tuple(v * NII_VOXEL_MM * 1000 for v in xyz_nii)
                    coord_source = f"saved_visual_manifest:{review['nmt_frame']}"
            excluded = truth(neuron.get("exclude")) or truth(soma.get("exclude"))
            region = neuron.get("region", review.get("portal_region", ""))
            screen = classify_neuron(region, xyz, boxes, insula, excluded, pad_mm)
            item = {"sample": sample, "neuron_id": name, "uid": f"{sample}|{name}",
                    "in_saved_portal_neurons": name in neurons, "in_saved_portal_soma": name in somata,
                    "in_visual_manifest": name in selected, "portal_region": region,
                    "portal_exclude": neuron.get("exclude", ""), "soma_exclude": soma.get("exclude", ""),
                    "coordinate_source": coord_source, **screen,
                    "visual_review_category": review.get("review_category", ""),
                    "visual_selection_reasons": review.get("selection_reasons", ""),
                    "staging_refined_label": review.get("staging_refined_label", ""),
                    "staging_region_source": review.get("staging_region_source", ""),
                    "step1_source": review.get("step1_source", ""),
                    "portal_neuron_source": json.dumps(neuron_ep["source"], ensure_ascii=True) if name in neurons else "",
                    "portal_soma_source": json.dumps(soma_ep["source"], ensure_ascii=True) if name in somata else "",
                    "visual_manifest_source": json.dumps(source_prov["visual_manifest"]) if name in selected else ""}
            if truth(review.get("nmt_coordinate_disagreement")):
                item["coordinate_disagreement"] = True
            local_rows.append(item)
        neuron_rows.extend(local_rows)
        full_list = bool(neuron_ep and neuron_ep["status"] == "ok")
        historical = aggregate.get(sample, {})
        aggregate_ok = historical.get("http_status") == "200" and not historical.get("status", "").startswith("fetch_fail")
        latest_failed = neuron_ep is not None and neuron_ep["status"] != "ok"
        counts = Counter(base_label(n.get("region")) for n in neurons.values())
        not_in_current_catalog = current_ids is not None and sample not in current_ids
        usable_aggregate = aggregate_ok and not latest_failed and not not_in_current_catalog
        n_listed = len(neurons) if full_list else int(historical["n_portal_now"]) if usable_aggregate else None
        n_ins = sum(n["atlas_INS"] for n in local_rows if n["in_saved_portal_neurons"]) if full_list else int(historical["n_insula_now"]) if usable_aggregate else None
        n_prco = counts["PRCO"] if full_list else int(historical["n_prco_now"]) if usable_aggregate else None
        potential_lower_bound = sum(n["potential_INS_candidate"] is True for n in local_rows) if full_list else None
        unassessed_potential = sum(n["potential_INS_candidate"] is None for n in local_rows) if full_list else None
        potential = potential_lower_bound if full_list and unassessed_potential == 0 else None
        g_union_lower_bound = sum(n["atlasINS_or_G_or_foldedBox_candidate"] is True for n in local_rows) if full_list else None
        g_union_unassessed = sum(n["atlasINS_or_G_or_foldedBox_candidate"] is None for n in local_rows) if full_list else None
        injection_region, injection_structure = str(info.get("injection_region") or ""), str(info.get("injection_structure") or "")
        tokens = {base_label(part) for part in re.split(r"[;\n\r]+", injection_region + ";" + injection_structure) if part.strip()}
        inj_terms = sorted(tokens & insula)
        current_monkey = tracker.get("animal", "")
        historical_monkey = info.get("monkey_id_tracker", "")
        row = {"sample": sample, "channel_variant": bool(re.search(r"ch\d+$", sample, re.I)),
               "project_id": info.get("project_id", ""), "macaque_metadata": is_macaque(info),
               "metadata_status": "missing_from_catalog" if not info or truth(info.get("missing_from_sampleinfo")) else "present",
               "latest_full_catalog_status": "absent_historical_tracker_alias_only" if not_in_current_catalog else "present" if current_ids is not None else "no_fresh_full_catalog",
               "tracker_monkey_id": current_monkey, "historical_audit_monkey_id": historical_monkey,
               "identity_note": "historical audit tracker identity was absent/unverified; current tracker maps this monkey to a different exact fMOST ID" if historical_monkey and not current_monkey else "",
               "tracker_injection_claim": tracker.get("injection_sites", ""),
               "portal_injection_region": injection_region, "portal_injection_structure": injection_structure,
               "portal_injection_insula_terms": ";".join(inj_terms),
               "portal_injection_evidence_status": "metadata_terms_only_not_injection_verification" if injection_region or injection_structure else "missing",
               "injection_time_source_value": info.get("injection_time", ""),
               "sample_metadata_tracing_count": info.get("tracing_cell_number", ""),
               "neuron_list_status": neuron_ep["status"] if neuron_ep else "absent_current_catalog_historical_alias" if not_in_current_catalog else "aggregate_only" if aggregate_ok else "missing",
               "soma_endpoint_status": soma_ep["status"] if soma_ep else "missing",
               "neuron_evidence_date": neuron_ep["checked_at"] if neuron_ep else "2026-09-28" if aggregate_ok else "",
               "n_listed_reconstructions": n_listed, "n_atlas_INS": n_ins, "n_atlas_PrCO": n_prco,
               "n_historical_listed_reconstructions_20260928": historical.get("n_portal_now", "") if aggregate_ok else None,
               "n_listed_delta_vs_20260928": n_listed - int(historical["n_portal_now"]) if n_listed is not None and aggregate_ok else None,
               "n_atlas_G": counts["G"] if full_list else None,
               "n_atlas_explicit_unknown": sum(counts[v] for v in UNKNOWN) if full_list else None,
               "n_missing_region_metadata": counts[""] if full_list else None,
               "n_excluded_neuron_identities": sum(n["excluded"] for n in local_rows) if full_list else None,
               "n_coordinate_present": sum(n["coordinate_status"] == "present" for n in local_rows) if local_rows else None,
               "n_coordinate_missing": sum(n["coordinate_status"] == "missing" for n in local_rows) if local_rows else None,
               "n_potential_INS_candidate": potential,
               "n_potential_INS_candidate_lower_bound": potential_lower_bound,
               "n_potential_INS_unassessed": unassessed_potential,
               "n_atlasINS_or_G_or_foldedBox_candidate": g_union_lower_bound if full_list and g_union_unassessed == 0 else None,
               "n_atlasINS_or_G_or_foldedBox_candidate_lower_bound": g_union_lower_bound,
               "n_atlasINS_or_G_or_foldedBox_unassessed": g_union_unassessed,
               "potential_count_status": "incomplete_missing_coordinates_read_lower_bound" if unassessed_potential else "complete_bounded_snapshot_screen" if full_list else "missing_per_neuron_snapshot",
               "n_visual_manifest_identities": len(selected),
               "n_visual_manifest_only_identities": len(set(selected) - set(neurons)) if full_list else None,
               **reconcile_series(sample, tracker, series, directory_observation),
               "sample_metadata_source": json.dumps(catalog_prov.get(sample, {})),
               "neuron_count_source": json.dumps(neuron_ep["source"] if neuron_ep else source_prov["aggregate"] if aggregate_ok else {}),
               "tracker_source": json.dumps(source_prov["tracker"]) if tracker else "",
               "anatomical_acceptance": "not_assessed"}
        sample_rows.append(row)
    ranked = sorted(sample_rows, key=lambda r: (r["channel_variant"], -(r["n_atlas_INS"] or 0),
                    -(r["n_potential_INS_candidate_lower_bound"] or 0), r["sample"]))
    for rank, row in enumerate(ranked, 1):
        row["priority_rank_INS_then_potential"] = rank
        row["priority_scope"] = "channel_variants_separate; dated evidence; unknown counts not zero"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "sample_inventory.csv", ranked)
    write_csv(output / "neuron_inventory.csv", neuron_rows)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": "saved_snapshot_audit",
               "n_samples": len(sample_rows), "n_macaque_catalog_samples": sum(is_macaque(r) for r in catalog.values()),
               "n_channel_variants": sum(r["channel_variant"] for r in sample_rows),
               "n_samples_with_full_neuron_snapshot": sum(r["neuron_list_status"] == "ok" for r in sample_rows),
               "n_samples_aggregate_only": sum(r["neuron_list_status"] == "aggregate_only" for r in sample_rows),
               "n_historical_tracker_aliases_absent_current_catalog": sum(r["latest_full_catalog_status"] == "absent_historical_tracker_alias_only" for r in sample_rows),
               "n_neuron_identity_rows": len(neuron_rows), "n_verified_copied_series": len([s for s in series.values() if s["status"] == "verified_copied_series"]),
               "padding_mm": pad_mm, "coordinate_rule": "portal um/1000; folded X=abs(X_mm-32); folded reference q0.005-q0.995 NII bounds*0.25 + padding_mm",
               "source_files": source_prov, "source_root_observation": directory_observation,
               "snapshot_endpoints": [e["source"] | {"status": e["status"]} for e in endpoints.values()],
               "issues": issues, "limitations": ["All outputs are candidates only; no anatomical, layer, laterality, injection-site or VEN acceptance.",
                    "Reconstruction-list entries are not proof of complete reconstruction; metadata tracing_cell_number is a separate field.",
                    "Missing or failed per-neuron/soma endpoints remain unassessed; aggregate-only INS counts use historical vocabulary.",
                    "CH1 slice evidence is dated 2026-10-02. Any later shallow folder observation has its own date and does not validate slices or PI/cytoarchitecture.",
                    "Excluded identities remain visible and cannot enter potential counts; exact sample/channel identity is preserved."]}
    (output / "provenance.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = ["# All-macaque insula inventory", "", f"Generated from saved evidence: {len(sample_rows)} exact sample IDs; {len(neuron_rows)} exact sample/neuron identities.", "",
             f"Full per-neuron metadata exists for {summary['n_samples_with_full_neuron_snapshot']} samples. {summary['n_samples_aggregate_only']} samples have historical aggregate counts only. Blank CSV counts mean missing, never zero.", "",
             "Injection claims, portal injection tags, reconstruction-list counts and candidate soma counts are separate evidence fields. Channel variants remain separate sample IDs.", "",
             "Priority ranks sort actual atlas INS counts, then observed folded-box candidate lower bounds; channel variants are listed separately at the end. Incomplete potential totals are blank and retain explicit observed lower bounds. Separate G flags and INS-or-G-or-box counts preserve G without anatomical promotion. Read evidence dates and missing-count status before using a rank.", "",
             "`neuron_inventory.csv` records every saved portal neuron plus visual-manifest-only identities, source hashes, labels, exclusion flags and candidate screen. `sample_inventory.csv` covers all catalog/tracker identities. `provenance.json` records sources and issues.", "",
             *[f"- {text}" for text in summary["limitations"]], "",
             "Rerun: `python -B group_analysis/scripts/audit_insula_inventory.py`. Use `--snapshot-dir` for an additional persisted manifest. The optional `--capture-live --max-requests N` records bounded metadata GETs only under the new output folder; it never downloads images or SWCs.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def capture_live(root, directory, max_requests=150, timeout=10.0):
    """Bounded metadata-only snapshot; no retry loops and no source-share access."""
    import requests
    if not 1 <= max_requests <= 200 or not 0 < timeout <= 30:
        raise ValueError("Require 1..200 requests and 0 < timeout <= 30 seconds")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    records, used = [], 0

    def get(sample, kind, endpoint, params):
        nonlocal used
        if used >= max_requests:
            return None
        used += 1
        url = f"{BASE_URL}/{endpoint}?{urlencode(params)}"
        record = {"sample": sample, "kind": kind, "url": url,
                  "checked_at": datetime.now(timezone.utc).isoformat(), "http_status": None}
        data = None
        try:
            response = requests.get(url, timeout=timeout)
            record["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("Expected list metadata payload")
            path = directory / f"{sample or 'catalog'}_{kind}.json"
            path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
            record.update(file=str(path.relative_to(root)), sha256=sha256(path), count=len(data))
        except (requests.RequestException, ValueError) as exc:
            record["error"] = str(exc)
            record["http_status"] = record["http_status"] if record["http_status"] != 200 else None
            data = None
        records.append(record)
        (directory / "manifest.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
        return data

    catalog = get("", "sample_info", "getSampleInfo", {"project_id": ""})
    if catalog is None:
        return
    samples = sorted({str(r["fMOST_id"]) for r in catalog if r.get("fMOST_id") and is_macaque(r)})
    for sample in samples:
        get(sample, "neurons", "selectNeurons", {"id": sample})
        get(sample, "soma", "getSoma", {"fMOST_id": sample})
        if used >= max_requests:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-dir", type=Path, action="append", default=[])
    parser.add_argument("--pad-mm", type=float, default=2.0)
    parser.add_argument("--series-observation", type=Path)
    parser.add_argument("--capture-live", action="store_true")
    parser.add_argument("--max-requests", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    if output != root / DEFAULT_OUTPUT and root / DEFAULT_OUTPUT not in output.parents:
        parser.error("Output must stay under the dedicated evolution inventory directory")
    snapshots = [root / "notes/bulk_visual_review_20261002/portal_snapshot"] + args.snapshot_dir
    if args.capture_live:
        directory = output / "live_snapshots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        capture_live(root, directory, args.max_requests, args.timeout)
        snapshots.append(directory)
    result = build_inventory(root, output, snapshots, args.pad_mm, args.series_observation)
    print(json.dumps({key: result[key] for key in result if key.startswith("n_")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
