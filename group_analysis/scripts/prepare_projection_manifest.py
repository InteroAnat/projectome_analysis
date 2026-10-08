"""Preserve explicit workbook membership in a hashed projection-map manifest.

This adapter uses the existing cached-atlas-SWC convention XYZ/250 -> NMT
voxel indices. It preserves source labels unless a separately hashed reviewed
assignment is supplied. The coordinate convention is a code declaration, not
proof of native registration. Source workbooks/SWCs are never changed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_projection_maps import checked_manifest, sha256


def prepare(table, animal_map, swc_root, reference, output, *, label_column="Soma_Region",
            assignments=None):
    table, animal_map, swc_root, reference, output = map(Path, (table, animal_map, swc_root, reference, output))
    if output.exists():
        raise FileExistsError("Manifest preparation requires a new output directory")
    table_hash, animals_hash, reference_hash = sha256(table), sha256(animal_map), sha256(reference)
    source = pd.read_excel(table, sheet_name="Summary", dtype=str, keep_default_na=False)
    if (source.empty or not {"SampleID", "NeuronID", label_column} <= set(source.columns)
            or source.duplicated(["SampleID", "NeuronID"]).any()):
        raise ValueError("Source Summary must have unique explicit sample/neuron identities and labels")
    registry = pd.read_csv(animal_map, dtype=str, keep_default_na=False)
    if not {"fmost_id", "animal"} <= set(registry.columns) or registry.fmost_id.duplicated().any():
        raise ValueError("Animal registry requires unique fmost_id and explicit animal columns")
    animal_lookup = dict(zip(registry.fmost_id, registry.animal))
    image = nib.load(reference)
    if image.shape != (256, 312, 200) or not np.allclose(image.header.get_zooms()[:3], [.25]*3):
        raise ValueError("Existing cached-SWC index convention requires the declared 0.25-mm NMT grid")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Existing NMT reference must declare millimetres")
    overrides = {}
    assignment_hash = sha256(assignments) if assignments else None
    if assignments:
        record = json.loads(Path(assignments).read_text(encoding="utf-8"))
        for item in record["annotations"]:
            key = (str(record["sample_id"]), item["NeuronID"])
            if key in overrides:
                raise ValueError("Reviewed assignments contain duplicate identities")
            overrides[key] = (item, record["swc_sha256"][item["NeuronID"]])
        if not set(overrides) <= set(zip(source.SampleID, source.NeuronID)):
            raise ValueError("Reviewed assignment identity is absent from source Summary")
    rows = []
    for original in source.to_dict("records"):
        sample, neuron = original["SampleID"], original["NeuronID"]
        if sample not in animal_lookup:
            raise ValueError(f"No established animal registry identity for {sample}")
        # Basenames and sample identities are validated again by the builder.
        if Path(neuron).name != neuron or Path(sample).name != sample:
            raise ValueError("Source sample/neuron identities must be basenames")
        swc = swc_root / sample / neuron
        swc_hash = sha256(swc)
        label, status = original[label_column], "source_curated_labels_no_new_acceptance"
        note = "original source label retained"
        if (sample, neuron) in overrides:
            assignment, expected_swc = overrides[(sample, neuron)]
            if swc_hash != expected_swc:
                raise ValueError("Reviewed assignment SWC hash differs from selected source")
            label, note = assignment["Soma_Region"], assignment["status"]
            status = "reviewed_assignment_provisional_parcel" if "provisional" in note.lower() else "reviewed_assignment"
        rows.append({
            "AnimalID": animal_lookup[sample], "SampleID": sample, "NeuronID": neuron,
            "Subregion": label, "OriginalSourceLabel": original[label_column],
            "SWCPath": str(swc.resolve()), "SWCSHA256": swc_hash,
            "ReferenceSHA256": reference_hash, "CoordinateFrame": "atlas_index_um",
            "IndexScaleUm": "250;250;250", "AnatomyStatus": status, "AnatomyNote": note,
            "RegistrationStatus": "cached_atlas_registration_not_independently_accepted",
        })
    if (sha256(table) != table_hash or sha256(animal_map) != animals_hash
            or sha256(reference) != reference_hash
            or (assignments and sha256(assignments) != assignment_hash)):
        raise RuntimeError("Manifest source changed during preparation")
    output.mkdir(parents=True)
    manifest = output / "projection_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    checked = checked_manifest(manifest, ROOT)
    if [(row["SampleID"], row["NeuronID"]) for row in checked] != list(zip(source.SampleID, source.NeuronID)):
        raise RuntimeError("Prepared manifest membership/order differs from source Summary")
    declaration = ROOT / "main_scripts/neuro_tracer.py"
    provenance = {
        "status": "software_verified_manifest", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": str(table.resolve()), "source_table_sha256": table_hash,
        "source_sheet": "Summary", "source_label_column": label_column,
        "animal_registry": str(animal_map.resolve()), "animal_registry_sha256": animals_hash,
        "animal_identity_status": "existing human tracker mapping; acquisition/reconstruction counts not used",
        "reviewed_assignments": str(Path(assignments).resolve()) if assignments else None,
        "reviewed_assignments_sha256": assignment_hash, "n_overrides": len(overrides),
        "reference": str(reference.resolve()), "reference_sha256": reference_hash,
        "source_coordinate_declaration": str(declaration), "source_coordinate_declaration_sha256": sha256(declaration),
        "coordinate_mapping": "cached atlas SWC XYZ micrometres / 250 -> NMT v2.1 voxel index; no warp or mirroring",
        "grid_shape": list(image.shape), "affine_mm": image.affine.tolist(),
        "registration_acceptance": "not established by code declaration, source hashes or this manifest",
        "canonical_promotion": False, "neurons": len(rows),
        "manifest_sha256": sha256(manifest), "source_files_mutated": False,
    }
    (output / "preparation_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--animal-map", type=Path, required=True)
    parser.add_argument("--swc-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-column", default="Soma_Region")
    parser.add_argument("--reviewed-assignments", type=Path)
    args = parser.parse_args()
    result = prepare(args.table, args.animal_map, args.swc_root, args.reference, args.output,
                     label_column=args.label_column, assignments=args.reviewed_assignments)
    print(f"{result['status']}: {result['neurons']} neurons; {result['n_overrides']} explicit overrides")
