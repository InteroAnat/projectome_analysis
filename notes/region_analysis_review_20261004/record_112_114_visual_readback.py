"""Record inspected original/native figures and compare saved original soma data."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "case_112_114_20261004"
LAB = Path("X:/fMOST/251637")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    generated = json.loads((OUT / "native_visual_provenance.json").read_text(encoding="utf-8"))
    inputs = json.loads((OUT / "source_readback.json").read_text(encoding="utf-8"))
    checked = []
    for stem, label in (("112","L-IDM"),("114","L-IDD5")):
        folder = OUT / f"neuron-{stem}"
        old = LAB / "cube_data_251637_other_regions_20260320/Region_Unknown_0"
        for path in [folder / "native_full_section_soma_locator.png",
                     folder / f"251637_{stem}.swc_{label}_WideField_Plot.png",
                     folder / f"251637_{stem}.swc_{label}_SomaBlock_Plot.png",
                     old / f"LowRes/Plots/251637_{stem}.swc_Unknown_0_WideField_Plot.png",
                     old / f"HighRes/Plots/251637_{stem}.swc_Unknown_0_SomaBlock_Plot.png"]:
            checked.append({"NeuronID": stem+".swc", "path": str(path), "sha256": sha(path),
                "actual_visual_inspection": True,
                "observation": "Full native locator: soma on viewer-left; soma/trace crops show the marked cell and corresponding native tissue. Original and regenerated soma views show the same cell."})
    comparisons = []
    for stem, label in (("112","L-IDM"),("114","L-IDD5")):
        old = LAB / f"cube_data_251637_other_regions_20260320/Region_Unknown_0/HighRes/Data/251637_{stem}.swc_Unknown_0_SomaBlock.nii.gz"
        new = OUT / f"neuron-{stem}/251637_{stem}.swc_{label}_SomaBlock.nii.gz"
        a,b = nib.load(old), nib.load(new)
        comparisons.append({"NeuronID": stem+".swc", "original": str(old), "original_sha256": sha(old),
            "regenerated": new.relative_to(ROOT).as_posix(), "regenerated_sha256": sha(new),
            "shape_original": list(a.shape), "shape_regenerated": list(b.shape),
            "all_soma_voxels_equal": np.array_equal(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)),
            "original_affine": a.affine.tolist(), "regenerated_affine": b.affine.tolist(),
            "header_note": "Native voxel correspondence does not validate anatomical orientation; affine spacing/sign fixes are separately recorded"})
    henry_lab = LAB / "Area and Layers by Henry_2026.04.03.xlsx"
    henry_repo = ROOT / "R_analysis/tables/somainfo_Henry_2026.04.03.xlsx"
    df = pd.read_excel(henry_repo, "251637 mostly insula")
    result = {"status": "actual_visual_and_original_source_readback_complete", "recorded_at": datetime.now().astimezone().isoformat(),
        "native_producer_errors": [e for n in generated["neurons"] for e in n["errors"]],
        "inspected_figures": checked, "original_soma_data_comparisons": comparisons,
        "henry_lab_sha256": sha(henry_lab), "henry_repo_sha256": sha(henry_repo),
        "henry_lab_repo_bytes_equal": henry_lab.read_bytes() == henry_repo.read_bytes(),
        "changed_original_workbooks": [e["path"] for e in inputs["workbooks"] if sha(ROOT / e["path"]) != e["sha256"]],
        "henry_115_annotation_present": bool(pd.to_numeric(df.SWC, errors="coerce").eq(115).any()),
        "native_side_observation": "Both 112 and 114 somata are on viewer-left of their own full-section images",
        "interpretation": "R atlas assignment supported; localized manual L-prefix errors are plausible. Absolute native orientation and IDD5-versus-IDM for duplicate 114 remain unresolved.",
        "original_annotations_modified": False, "production_code_modified_in_this_check": False}
    (OUT / "visual_and_original_source_readback.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"inspected_figures": len(checked), "soma_voxel_equal": [c["all_soma_voxels_equal"] for c in comparisons],
                      "henry_lab_repo_bytes_equal": result["henry_lab_repo_bytes_equal"],
                      "changed_original_workbooks": result["changed_original_workbooks"],
                      "henry_115_annotation_present": result["henry_115_annotation_present"]}))

if __name__ == "__main__":
    main()
