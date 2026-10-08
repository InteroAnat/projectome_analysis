"""Prepare a unique, evidence-stratified map manifest from the audited inventory.

Human coarse INS, atlas-only INS, folded-box candidates and additional G
candidates remain disjoint groups. Missing animal identities remain in the
selection ledger; they are never invented or pooled into animal means.
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


GROUPS = {f"{group}_{side}" for group in ("HumanINS", "AtlasINS", "Candidate", "G_exploratory")
          for side in ("L", "R")}


def prepare(sources, inventory_provenance, animal_map, reference, output, *, inventory_delivery, input_root=ROOT):
    sources, inventory_provenance, animal_map, reference, output, input_root = map(
        Path, (sources, inventory_provenance, animal_map, reference, output, input_root))
    if output.exists():
        raise FileExistsError("Use a new manifest destination")
    paths = {"audited_sources": sources, "inventory_provenance": inventory_provenance,
             "inventory_delivery": Path(inventory_delivery),
             "animal_registry": animal_map, "reference": reference}
    hashes = {key: sha256(path) for key, path in paths.items()}
    delivery = json.loads(paths["inventory_delivery"].read_text(encoding="utf-8"))
    for name, path in (("audited_sources", sources), ("inventory_provenance", inventory_provenance)):
        if delivery["artifacts"].get(path.name) != hashes[name]:
            raise ValueError("Selection source or provenance differs from the audited delivery artifact")
    evidence = json.loads(inventory_provenance.read_text(encoding="utf-8"))
    bound_inputs = {}
    for name, item in evidence["inputs"].items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = input_root / path
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Audited input changed: {name}")
        bound_inputs[name] = (path, item["sha256"])
    if evidence["inputs"]["animal_registry"]["sha256"] != hashes["animal_registry"]:
        raise ValueError("Animal registry differs from the audited inventory")
    if evidence["inputs"]["reference"]["sha256"] != hashes["reference"]:
        raise ValueError("Reference differs from the audited inventory")
    frame = pd.read_csv(sources, dtype=str, keep_default_na=False)
    required = {"sample", "neuron_id", "uid", "exclusive_map_group", "registry_animal", "excluded",
                "animal_aggregation_eligible", "map_selected_source", "map_selected_sha256",
                "map_source_graph_status", "map_source_selection_rule", "portal_region",
                "Henry_coarse_INS_visual_evidence", "atlas_INS", "potential_INS_candidate", "atlas_G"}
    if frame.empty or not required <= set(frame.columns) or frame.duplicated(["sample", "neuron_id"]).any():
        raise ValueError("Audited source inventory requires unique explicit identities and evidence fields")
    if not frame.uid.eq(frame["sample"] + "|" + frame.neuron_id).all():
        raise ValueError("Audited UID differs from exact sample/neuron identity")
    registry = pd.read_csv(animal_map, dtype=str, keep_default_na=False)
    if not {"fmost_id", "animal"} <= set(registry.columns) or registry.fmost_id.duplicated().any():
        raise ValueError("Animal registry requires unique exact sample identities")
    animals = dict(zip(registry.fmost_id, registry.animal))
    image = nib.load(reference)
    if (image.shape != (256, 312, 200) or image.header.get_xyzt_units()[0] != "mm"
            or not np.allclose(image.header.get_zooms()[:3], [.25] * 3)):
        raise ValueError("Cached SWC convention requires the declared 0.25-mm NMT grid")
    rows, ledger = [], []
    for row in frame.to_dict("records"):
        group = row["exclusive_map_group"]
        if group not in GROUPS:
            continue
        if row["excluded"] not in ("True", "False"):
            raise ValueError("Missing explicit exclusion status")
        checks = {"HumanINS": "Henry_coarse_INS_visual_evidence", "AtlasINS": "atlas_INS",
                  "Candidate": "potential_INS_candidate", "G_exploratory": "atlas_G"}
        kind = group.rsplit("_", 1)[0]
        if row[checks[kind]] != "True":
            raise ValueError("Exclusive group lacks its declared selection evidence")
        if kind != "HumanINS" and row["Henry_coarse_INS_visual_evidence"] == "True":
            raise ValueError("Human INS evidence was overwritten by a weaker selection stratum")
        if kind in ("Candidate", "G_exploratory") and row["atlas_INS"] == "True":
            raise ValueError("Atlas INS evidence was overwritten by a candidate stratum")
        if kind == "G_exploratory" and row["potential_INS_candidate"] == "True":
            raise ValueError("Strict INS candidate was placed in the additional G group")
        animal = animals.get(row["sample"], "")
        if animal != row["registry_animal"]:
            raise ValueError("Audited animal identity disagrees with exact registry row")
        eligible = row["animal_aggregation_eligible"] == "True"
        if eligible != bool(animal):
            raise ValueError("Animal aggregation flag disagrees with registry availability")
        reason = ("excluded_source_identity" if row["excluded"] == "True" else
                  "missing_established_animal_identity" if not eligible else "selected")
        ledger.append({**row, "mapping_selection_status": reason})
        if reason != "selected":
            continue
        if row["map_source_graph_status"] != "numeric_graph_verified_all_copies":
            raise ValueError("Selected source graph was not verified across cached variants")
        source = Path(row["map_selected_source"])
        if not source.is_absolute():
            source = input_root / source
        source = source.resolve(strict=True)
        rows.append({**row, "AnimalID": animal, "SampleID": row["sample"], "NeuronID": row["neuron_id"],
            "Subregion": group, "OriginalSourceLabel": row["portal_region"],
            "SWCPath": str(source), "SWCSHA256": row["map_selected_sha256"],
            "ReferenceSHA256": hashes["reference"], "CoordinateFrame": "atlas_index_um",
            "IndexScaleUm": "250;250;250", "EvidenceStratum": kind,
            "AnatomyStatus": "Henry_visual_coarse_INS" if kind == "HumanINS" else
                "atlas_label_not_human_confirmed" if kind == "AtlasINS" else "candidate_only_not_anatomically_accepted",
            "FineParcelStatus": "original_annotations_retained_not_required_for_coarse_grouping",
            "CoordinateOriginStatus": "unresolved_current_center_encoding_used_no_shift",
            "RegistrationStatus": "cached_NMT_export_not_independently_anatomically_accepted",
            "ImageTerminalStatus": "not_available"})
    if not rows:
        raise ValueError("No eligible source neurons remain")
    payload = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    # Reuse the builder's full source/identity/hash validation before creating
    # a deliverable directory. Failed sources leave no partial preparation.
    with tempfile.TemporaryDirectory(prefix="projectome-manifest-check-") as temporary:
        preflight = Path(temporary) / "projection_manifest.csv"
        preflight.write_bytes(payload)
        checked_manifest(preflight, input_root)
    if (any(sha256(path) != hashes[key] for key, path in paths.items())
            or any(sha256(path) != expected for path, expected in bound_inputs.values())):
        raise RuntimeError("A preparation input changed")
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "projection_manifest.csv"
    with manifest.open("xb") as stream:
        stream.write(payload)
    ledger_path = output / "selection_ledger.csv"
    with ledger_path.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(ledger).to_csv(stream, index=False)
    report = {"status": "software_verified_inventory_manifest", "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: {"path": str(path.resolve()), "sha256": hashes[key]} for key, path in paths.items()},
        "preparer_sha256": sha256(__file__), "source_inventory_rows": len(frame),
        "selection_ledger_rows": len(ledger), "selected_neurons": len(rows),
        "selected_animals": sorted({row["AnimalID"] for row in rows}),
        "groups": pd.DataFrame(rows).groupby(["AnimalID", "Subregion"]).size().reset_index(name="n_selected").to_dict("records"),
        "manifest_sha256": sha256(manifest), "selection_ledger_sha256": sha256(ledger_path),
        "selection_policy": "disjoint evidence strata and hemisphere; each exact identity once; no coordinate relabeling",
        "coordinate_policy": "current cached XYZ/250 center encoding; export origin and registration unresolved",
        "terminal_policy": "candidate graph leaves and all-axon length only; reviewed terminal fields unavailable",
        "animal_policy": "exact human registry mapping; missing identities excluded from animal aggregation",
        "source_policy": "audited deterministic source after equal complete seven-column numeric graphs across variants",
        "canonical_promotion": False, "anatomical_acceptance": "no new acceptance", "source_files_mutated": False}
    with (output / "preparation_provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "inventory-provenance", "inventory-delivery", "animal-map", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = prepare(args.sources, args.inventory_provenance, args.animal_map, args.reference, args.output,
                     inventory_delivery=args.inventory_delivery, input_root=args.input_root)
    print(f"{result['status']}: {result['selected_neurons']} neurons, {len(result['selected_animals'])} animals")
