"""Read-only baseline audit against the current declared ARM voxel grid.

This checks table consistency, not the anatomical validity of SWC registration.
No physical laterality is inferred from an affine or from coordinate medians.
"""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def declared_side(label):
    for prefix, side in (("CL_", "L"), ("SL_", "L"), ("CR_", "R"),
                         ("SR_", "R"), ("L-", "L"), ("R-", "R"),
                         ("L_", "L"), ("R_", "R")):
        if str(label).startswith(prefix):
            return side
    return "Unknown"


def main():
    atlas_path = ROOT / "atlas/ARM_in_NMT_v2.1_sym.nii.gz"
    key_path = ROOT / "atlas/ARM_key_all.txt"
    img = nib.load(atlas_path)
    atlas = np.asanyarray(img.dataobj)[..., 0, 5]
    key = pd.read_csv(key_path, sep="\t")
    names = dict(zip(key.Index, key.Abbreviation))
    paths = sorted((ROOT / "group_analysis/step1_results").glob("*/tables/*.xlsx"))
    for folder in ("neuron_tables", "neuron_tables_new"):
        path = ROOT / folder / "251637_INS_HE_inferred.xlsx"
        if path.exists():
            paths.append(path)
    reports, records = [], []
    for path in paths:
        df = pd.read_excel(path, sheet_name="Summary")
        mismatches = side_conflicts = coordinate_ambiguous = 0
        for _, row in df.iterrows():
            xyz = np.array([row.get("Soma_NII_" + axis) for axis in "XYZ"], dtype=float)
            valid = np.isfinite(xyz).all()
            vox = np.rint(xyz).astype(int) if valid else None
            inside = valid and all(0 <= vox[d] < atlas.shape[d] for d in range(3))
            label = names.get(int(atlas[tuple(vox)]), "Unknown_" + str(int(atlas[tuple(vox)]))) if inside else "Out_of_Bounds"
            stored = str(row.Soma_Region)
            # Persisted soma coordinates are rounded to four decimals. Exact
            # half-voxel cases require the original SWC before alleging a bug.
            boundary = bool(valid and np.any(np.abs((xyz % 1) - 0.5) <= 0.00005))
            direct = stored.startswith(("CL_", "CR_", "SL_", "SR_", "Unknown_")) or stored == "Out_of_Bounds"
            mismatch = direct and stored != label
            side = declared_side(stored)
            conflict = side != "Unknown" and str(row.get("Soma_Side")) != side
            mismatches += int(mismatch)
            side_conflicts += int(conflict)
            coordinate_ambiguous += int(boundary)
            records.append({"source": str(path.relative_to(ROOT)), "SampleID": row.get("SampleID", "251637"),
                            "NeuronID": row.NeuronID, "stored_region": stored, "atlas_lookup": label,
                            "direct_atlas_label": direct, "region_mismatch": mismatch,
                            "rounding_boundary": boundary, "label_side": side,
                            "stored_side": row.get("Soma_Side"), "label_side_conflict": conflict,
                            "atlas_label_side": declared_side(label), "voxel": str(vox.tolist()) if valid else "invalid"})
        reports.append({"source": str(path.relative_to(ROOT)), "sha256": digest(path),
                        "rows": len(df), "duplicate_ids": int(df.NeuronID.duplicated().sum()),
                        "direct_region_mismatches": mismatches, "label_side_conflicts": side_conflicts,
                        "rounding_boundary_rows": coordinate_ambiguous})
    rows = pd.DataFrame(records)
    rows.to_csv(OUT / "existing_table_region_baseline_rows.csv", index=False)
    report = {"status": "baseline_observed", "interpretation": "Mechanical ARM lookup using persisted voxel coordinates; registration and anatomical laterality remain unvalidated.",
              "atlas": {"path": str(atlas_path.relative_to(ROOT)), "sha256": digest(atlas_path),
                        "shape": img.shape, "level": 6, "key_sha256": digest(key_path)},
              "sources": reports, "checked_rows": len(rows),
              "direct_region_mismatches": int(rows.region_mismatch.sum()),
              "label_side_conflicts": int(rows.label_side_conflict.sum())}
    (OUT / "existing_table_region_baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
