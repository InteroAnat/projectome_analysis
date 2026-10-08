"""Prepare a separate, hash-bound preview of spatial INS review candidates.

The main cohort is excluded by exact identity and resolved source path. An
origin-sensitive INS lookup receives OriginCandidate; otherwise both lookups
must be non-INS and the corrected same-sample anchor distance must be <=2 mm
for NearINS. Numeric ID proximity alone never selects a neuron. These labels
describe retrieval evidence, not accepted insular anatomy or terminal fields.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd

from build_projection_maps import ROOT, checked_manifest, sha256


CURRENT = "portal_coordinate_current_rint_zero_center_insula_candidate"
EDGE = "portal_coordinate_published_ceil_edge_one_based_to_zero_insula_candidate"
DISTANCE = "nearest_same_exact_sample_INS_anchor_mm"


def _read_unique(path, required):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty or not set(required) <= set(frame.columns):
        raise ValueError(f"Missing explicit inventory fields: {path}")
    if frame.duplicated(["sample", "neuron_id"]).any():
        raise ValueError("Duplicate exact candidate identity")
    if not frame.uid.eq(frame["sample"] + "|" + frame.neuron_id).all():
        raise ValueError("UID differs from exact sample/neuron identity")
    return frame


def prepare(sources, inventory_delivery, triage, triage_provenance, main_manifest,
            animal_map, reference, output, *, input_root=ROOT):
    """Write a fresh candidate manifest and complete corrected-triage ledger."""
    input_root, output = Path(input_root), Path(output)
    if output.exists():
        raise FileExistsError("Use a new candidate-preview destination")
    planned = [output / name for name in ("projection_manifest.csv", "selection_ledger.csv",
                                          "preparation_provenance.json")]
    keys = [str(path.resolve()).replace("\\", "/").casefold() for path in planned]
    if len(set(keys)) != len(keys) or any(len(str(path.resolve())) > 240 for path in planned):
        raise ValueError("Candidate output paths collide or exceed the portable 240-character limit")
    paths = {key: Path(value) for key, value in {
        "audited_sources": sources, "inventory_delivery": inventory_delivery,
        "corrected_triage": triage, "triage_provenance": triage_provenance,
        "main_manifest": main_manifest, "animal_registry": animal_map,
        "reference": reference}.items()}
    hashes = {key: sha256(path) for key, path in paths.items()}
    delivery = json.loads(paths["inventory_delivery"].read_text(encoding="utf-8"))
    if delivery["artifacts"].get(paths["audited_sources"].name) != hashes["audited_sources"]:
        raise ValueError("Source inventory differs from audited delivery")
    # This filename is the explicit provenance artifact in the audited delivery.
    inventory_provenance = paths["inventory_delivery"].parent / "provenance.json"
    if delivery["artifacts"].get(inventory_provenance.name) != sha256(inventory_provenance):
        raise ValueError("Audited inventory provenance changed")
    dependencies = {"inventory_provenance": (inventory_provenance, sha256(inventory_provenance))}
    inventory = json.loads(inventory_provenance.read_text(encoding="utf-8"))
    for name, item in inventory["inputs"].items():
        path = Path(item["path"])
        path = path if path.is_absolute() else input_root / path
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Audited input changed: {name}")
        dependencies[name] = (path, item["sha256"])
    for name in ("animal_registry", "reference"):
        if inventory["inputs"][name]["sha256"] != hashes[name]:
            raise ValueError(f"{name} differs from audited inventory")
    correction = json.loads(paths["triage_provenance"].read_text(encoding="utf-8"))
    if correction["input_delivery"]["sha256"] != hashes["inventory_delivery"]:
        raise ValueError("Corrected triage belongs to a different audited delivery")
    if correction["outputs"].get(paths["corrected_triage"].name) != hashes["corrected_triage"]:
        raise ValueError("Corrected triage artifact changed")
    for name, expected in correction["protected_prior_file_sha256"].items():
        path = paths["inventory_delivery"].parent / name
        if sha256(path) != expected:
            raise ValueError(f"Protected prior inventory changed: {name}")
        dependencies[f"protected_{name}"] = (path, expected)

    common = {"sample", "neuron_id", "uid", "excluded", "registry_animal",
              "animal_aggregation_eligible", "map_source_graph_status", "map_selected_source",
              "map_selected_sha256", "map_source_selection_rule", "portal_region", "own_mask_side",
              "exclusive_map_group"}
    source_frame = _read_unique(paths["audited_sources"], common)
    triage_frame = _read_unique(paths["corrected_triage"], common | {
        CURRENT, EDGE, DISTANCE, "nearest_INS_anchor_uid", "nearest_INS_anchor_evidence_basis", "anchor_distance_status"})
    source_rows = {row["uid"]: row for row in source_frame.to_dict("records")}
    main_records = checked_manifest(paths["main_manifest"], input_root)
    if any(row["ReferenceSHA256"].lower() != hashes["reference"] for row in main_records):
        raise ValueError("Main cohort binds a different reference")
    main_uids = {row["SampleID"] + "|" + row["NeuronID"] for row in main_records}
    main_paths = {row["source_path"] for row in main_records}
    registry = pd.read_csv(paths["animal_registry"], dtype=str, keep_default_na=False)
    if not {"fmost_id", "animal"} <= set(registry.columns) or registry.fmost_id.duplicated().any():
        raise ValueError("Animal registry requires unique exact sample rows")
    animals = dict(zip(registry.fmost_id, registry.animal))
    image = nib.load(paths["reference"])
    if (image.shape != (256, 312, 200) or image.header.get_xyzt_units()[0] != "mm"
            or not np.allclose(image.header.get_zooms()[:3], [.25] * 3)):
        raise ValueError("Cached SWC encoding requires the audited 0.25-mm NMT grid")

    selected, ledger = [], []
    for triage_row in triage_frame.to_dict("records"):
        uid, sample = triage_row["uid"], triage_row["sample"]
        source_row = source_rows.get(uid)
        group = ""
        if uid in main_uids:
            reason = "already_in_main_manifest"
        elif triage_row["excluded"] == "True":
            reason = "excluded_source_identity"
        elif not animals.get(sample):
            reason = "missing_established_animal_identity"
        elif source_row is None or triage_row["map_source_graph_status"] != "numeric_graph_verified_all_copies":
            reason = "no_verified_audited_source_graph"
        else:
            if any(triage_row[key] != source_row[key] for key in common):
                raise ValueError(f"Corrected triage changed original source evidence: {uid}")
            if (source_row["excluded"] != "False" or source_row["animal_aggregation_eligible"] != "True"
                    or source_row["registry_animal"] != animals[sample]):
                raise ValueError(f"Candidate source registry/exclusion evidence disagrees: {uid}")
            if any(triage_row[key] not in ("True", "False") for key in (CURRENT, EDGE)):
                raise ValueError(f"Missing explicit INS origin-policy lookup: {uid}")
            if triage_row[CURRENT] != triage_row[EDGE]:
                kind = "OriginCandidate"
            else:
                try:
                    distance = float(triage_row[DISTANCE])
                except ValueError:
                    distance = np.nan
                if (triage_row[CURRENT] == "False" and np.isfinite(distance) and 0 <= distance <= 2):
                    anchor_uid = triage_row["nearest_INS_anchor_uid"]
                    basis = json.loads(triage_row["nearest_INS_anchor_evidence_basis"])
                    if (anchor_uid == uid or not anchor_uid.startswith(sample + "|")
                            or triage_row["anchor_distance_status"] != "other_established_anchor_found"
                            or not isinstance(basis, list) or not basis
                            or any(value not in ("Henry_coarse_INS_visual_annotation", "original_portal_INS_label")
                                   for value in basis)):
                        raise ValueError(f"Near-INS distance lacks another established same-sample anchor: {uid}")
                    kind = "NearINS"
                else:
                    kind = ""
            if not kind:
                reason = "no_spatial_selection_evidence_ID_cue_not_sufficient"
            else:
                side = source_row["own_mask_side"]
                if side not in ("L", "R"):
                    raise ValueError(f"Missing audited own-root hemisphere: {uid}")
                group = f"{kind}_{side}"
                source = Path(source_row["map_selected_source"])
                source = (source if source.is_absolute() else input_root / source).resolve(strict=True)
                if source in main_paths:
                    raise ValueError("Candidate source path overlaps the main cohort")
                selected.append({**source_row, **{"Triage_" + key: value for key, value in triage_row.items()},
                    "AnimalID": animals[sample], "SampleID": sample, "NeuronID": source_row["neuron_id"],
                    "Subregion": group, "OriginalSourceLabel": source_row["portal_region"],
                    "SWCPath": str(source), "SWCSHA256": source_row["map_selected_sha256"],
                    "ReferenceSHA256": hashes["reference"], "CoordinateFrame": "atlas_index_um",
                    "IndexScaleUm": "250;250;250", "EvidenceStratum": kind,
                    "AnatomyStatus": "candidate_only_not_anatomically_accepted",
                    "CoordinateOriginStatus": "unresolved_current_center_encoding_used_no_shift",
                    "RegistrationStatus": "cached_NMT_export_not_independently_anatomically_accepted",
                    "ImageTerminalStatus": "not_available", "PreviewSelectionRule":
                    "different_current_vs_edge_INS_lookup" if kind == "OriginCandidate" else
                    "both_non_INS_and_other_established_same_sample_anchor_within_2mm"})
                reason = "selected_candidate_preview_only"
        ledger.append({**triage_row, "preview_group": group, "preview_selection_status": reason})
    if not selected:
        raise ValueError("No additional spatial candidates remain")
    payload = pd.DataFrame(selected).to_csv(index=False).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="projectome-candidate-preflight-") as temporary:
        temporary_manifest = Path(temporary) / "projection_manifest.csv"
        temporary_manifest.write_bytes(payload)
        checked_manifest(temporary_manifest, input_root)
    if (any(sha256(path) != hashes[key] for key, path in paths.items())
            or any(sha256(path) != expected for path, expected in dependencies.values())
            or any(sha256(row["SWCPath"]) != row["SWCSHA256"] for row in selected)):
        raise RuntimeError("A candidate preparation input changed")
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "projection_manifest.csv"
    with manifest.open("xb") as stream:
        stream.write(payload)
    ledger_path = output / "selection_ledger.csv"
    with ledger_path.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(ledger).to_csv(stream, index=False)
    report = {
        "status": "software_verified_candidate_preview_manifest", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root.resolve()), "preparer_sha256": sha256(__file__),
        "inputs": {key: {"path": str(path.resolve()), "sha256": hashes[key]} for key, path in paths.items()},
        "audited_dependency_inputs": {key: {"path": str(path.resolve()), "sha256": expected}
                                      for key, (path, expected) in dependencies.items()},
        "selected_neurons": len(selected), "selected_animals": sorted({row["AnimalID"] for row in selected}),
        "main_selected_neurons": len(main_records), "main_overlap_neurons": 0,
        "selection_ledger_rows": len(ledger),
        "selection_ledger_status_counts": pd.Series([row["preview_selection_status"] for row in ledger]).value_counts().to_dict(),
        "groups": pd.DataFrame(selected).groupby(["AnimalID", "Subregion"]).size().reset_index(name="n_selected").to_dict("records"),
        "manifest_sha256": sha256(manifest), "selection_ledger_sha256": sha256(ledger_path),
        "selection_policy": "origin-policy INS disagreement; otherwise both non-INS plus corrected other established same-sample anchor distance <=2 mm; ID-only proximity excluded",
        "coordinate_policy": "cached current XYZ/250 center encoding retained; alternative origin is retrieval sensitivity only",
        "terminal_policy": "candidate graph endpoints and all-axon length only; reviewed terminal fields unavailable",
        "anatomical_acceptance": "none; candidate preview only", "canonical_promotion": False,
        "main_manifest_modified": False, "source_files_mutated": False}
    with (output / "preparation_provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "inventory-delivery", "triage", "triage-provenance", "main-manifest",
                 "animal-map", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = prepare(args.sources, args.inventory_delivery, args.triage, args.triage_provenance,
                     args.main_manifest, args.animal_map, args.reference, args.output, input_root=args.input_root)
    print(f"{result['status']}: {result['selected_neurons']} neurons, {len(result['selected_animals'])} animals")
