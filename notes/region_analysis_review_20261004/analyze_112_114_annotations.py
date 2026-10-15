"""Compare original Henry sides against the actual atlas roots, preserving rows."""
import hashlib
import json
from pathlib import Path
from datetime import datetime
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "case_112_114_20261004"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = ROOT / "R_analysis/tables/somainfo_Henry_2026.04.03.xlsx"
    df = pd.read_excel(source, "251637 mostly insula")
    mask_path = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz"
    mask = np.asanyarray(nib.load(mask_path).dataobj)
    rows, missing = [], []
    for i, row in df.iterrows():
        if pd.isna(row.SWC) or pd.isna(row.Area): continue
        nid = f"{int(row.SWC):03d}.swc"
        path = ROOT / "main_scripts/processed_neurons/251637" / nid
        if not path.exists():
            missing.append(nid); continue
        nodes = np.loadtxt(path, comments="#", ndmin=2)
        xyz = nodes[nodes[:,6] == -1][0,2:5] / 250
        voxel = np.rint(xyz).astype(int)
        inside = np.all((voxel >= 0) & (voxel < np.array(mask.shape)))
        side = {1: "R", 2: "L"}.get(int(mask[tuple(voxel)]), "Unknown") if inside else "Unknown"
        declared = "L" if str(row.Area).startswith("L-") else "R" if str(row.Area).startswith("R-") else "Unknown"
        rows.append({"excel_row": int(i)+2, "NeuronID": nid, "Area": row.Area,
                     "Layer": int(row.Layer) if pd.notna(row.Layer) else None, "manual_side": declared,
                     "atlas_mask_side": side, "root_voxel_xyz": xyz.tolist(), "source_swc_sha256": digest(path),
                     "conflict": declared in ("L","R") and side in ("L","R") and declared != side})
    graphs = []
    for stem in ("112", "114"):
        raw = np.loadtxt(OUT / f"raw_{stem}.swc", comments="#", ndmin=2)
        atlas = np.loadtxt(OUT / f"atlas_{stem}.swc", comments="#", ndmin=2)
        graph = {"NeuronID": stem + ".swc", "same_node_identity_type_radius_parent":
                 raw.shape == atlas.shape and np.array_equal(raw[:,[0,1,5,6]], atlas[:,[0,1,5,6]])}
        variants = []
        for path in (ROOT / "main_scripts/processed_neurons/251637" / (stem + ".swc"),
                     ROOT / "group_analysis/fnt/nmt_swcs/251637" / (stem + ".swc"),
                     ROOT / "group_analysis/fnt/mirrored_swcs" / ("251637_" + stem + ".swc")):
            if not path.exists(): continue
            nodes = np.loadtxt(path, comments="#", ndmin=2)
            variants.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                             "numeric_graph_equal_live_atlas": np.array_equal(nodes, atlas),
                             "root_xyz": nodes[nodes[:,6] == -1][0,2:5].tolist()})
        graph["variants"] = variants
        graphs.append(graph)
    duplicate = df[df.SWC.duplicated(keep=False) & df.SWC.notna()]
    result = {"status": "annotation_and_identity_readback_complete", "observed_at": datetime.now().astimezone().isoformat(),
        "henry_source": source.relative_to(ROOT).as_posix(), "henry_sha256": digest(source), "mask_sha256": digest(mask_path),
        "checked_annotation_rows": len(rows), "unique_neurons": len({r["NeuronID"] for r in rows}),
        "conflict_rows": [r for r in rows if r["conflict"]], "unavailable_source_ids": missing,
        "case_rows": [r for r in rows if r["NeuronID"] in ("112.swc","114.swc")],
        "neighboring_annotation_rows": [r for r in rows if 110 <= int(r["NeuronID"][:-4]) <= 116],
        "duplicate_rows": [{"excel_row": int(i)+2, "SWC": int(r.SWC), "Area": r.Area,
                            "Layer": int(r.Layer) if pd.notna(r.Layer) else None} for i,r in duplicate.iterrows()],
        "graphs": graphs,
        "interpretation": "Identity/cached atlas geometry and manual annotations checked; source-provided registration transforms and absolute native orientation still required for anatomical acceptance"}
    (OUT / "annotation_and_identity_readback.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("checked_annotation_rows","unique_neurons","conflict_rows","duplicate_rows","neighboring_annotation_rows")}))

if __name__ == "__main__":
    main()
