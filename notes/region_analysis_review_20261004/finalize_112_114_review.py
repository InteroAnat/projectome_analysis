"""Save the targeted case conclusion and merge it into the living project note."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "case_112_114_20261004"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    controls_path = OUT / "same_sample_native_controls.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    henry_path = ROOT / "R_analysis/tables/somainfo_Henry_2026.04.03.xlsx"
    henry = pd.read_excel(henry_path, "251637 mostly insula")
    mask_path = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz"
    mask = np.asanyarray(nib.load(mask_path).dataobj)
    inspected = []
    for control in controls["controls"]:
        nid = control["NeuronID"]
        rows = henry[pd.to_numeric(henry.SWC, errors="coerce").eq(int(nid[:-4]))]
        assert len(rows) == 1 and rows.Area.iloc[0] == control["manual_area"]
        source = ROOT / "main_scripts/processed_neurons/251637" / nid
        nodes = np.loadtxt(source, comments="#", ndmin=2)
        xyz = nodes[nodes[:,6] == -1][0,2:5] / 250
        side = {1:"R",2:"L"}[int(mask[tuple(np.rint(xyz).astype(int))])]
        assert side == control["side"]
        assert digest(ROOT / control["figure"]) == control["figure_sha256"]
        inspected.append({**control,"verified_henry_row":int(rows.index[0])+2,"atlas_root_xyz_voxels":xyz.tolist(),
            "atlas_source_sha256":digest(source),"actual_visual_inspection":True,
            "observed_native_screen_side":"viewer-left" if side=="R" else "viewer-right"})
    record = {"status":"same_sample_control_visual_readback_passed","recorded_at":datetime.now().astimezone().isoformat(),
        "producer_sha256":digest(controls_path),"henry_source_sha256":digest(henry_path),
        "mask_sha256":digest(mask_path),"controls":inspected,
        "conclusion":"112/114 occupy the same displayed hemisphere as agreeing Henry-R control111; Henry-L control422 is on the opposite side.",
        "acceptance_limit":"This is a same-sample check anchored to existing human annotations, not an independent acquisition-orientation certificate."}
    (OUT / "same_sample_control_visual_readback.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    text = """# 251637 neurons 112 and 114: targeted laterality check

Agent: Codex. Date: 2026-10-04 (Asia/Shanghai). Branch: `codex/fmost-methods-review-20261002`.

**The combined evidence favors R hemisphere for both neurons and localized errors in their manual L prefixes.** This is a working conclusion anchored to the supplied atlas and Henry's other annotations. The original annotations remain intact; absolute acquisition orientation and the actual registration transforms were not independently recovered.

| Neuron | Original Henry annotation | Fresh atlas root X, voxels | Official mask | Own native full-section soma |
| --- | --- | --- | --- | --- |
| 112.swc | L-IDM, layer 3; Excel row 113 | 211.5384 | R | Viewer-left, section 9118 |
| 114.swc | L-IDD5 and L-IDM, both layer 3; Excel rows 115/116 | 209.5244 | R | Viewer-left, section 8580 |

## What was checked

- Fresh server atlas SWCs are numerically identical to the cached atlas-space graphs. The separate FNT NMT copies also agree. Mirrored FNT comparison copies have different X coordinates and were not substituted into region lookup.
- Own native raw SWCs and atlas SWCs have identical node IDs, node types, radii and parent links: 7,474 nodes for 112 and 11,744 for 114. They are the corresponding graphs in different coordinate frames.
- All 296 annotated rows in the original Henry sheet were compared against actual cached atlas roots: 295 distinct neurons; 293 agree in hemisphere. Only 112 and 114 conflict (three conflicting rows because 114 occurs twice). No source SWC was missing.
- Nearby annotated 110, 111, 113 and 116 all have manual R labels and nearby R atlas roots. Same-sample native controls were actually inspected: 111 (Henry R-IDD5 / atlas R) is on viewer-left; 422 (Henry L-IDD5 / atlas L) is on viewer-right. Both 112/114 occupy the same displayed hemisphere as the R control.
- The lab Henry workbook on `X:/fMOST/251637/` is byte-identical to the repository copy. The contradictory prefixes and duplicate 114 therefore exist in the original supplied sheet, not just in a later table export.
- The original March bulk high-resolution and 8-mm wide-field plots for both neurons were found on the lab drive and actually inspected. Newly generated high-resolution, 4-mm wide-field and full-section views were also inspected, using the public toolkit plotting methods. The two regenerated soma NIfTIs match every voxel of the original March soma data. Header spacing/sign corrections remain a separate implementation difference.
- The current portal metadata does not supply injection coordinates or an independent orientation declaration. No source-provided raw-to-atlas transformation record was recovered in the checked paths.

## What remains unresolved

114 has two region entries, IDD5 and IDM. The duplicate handling in the existing Henry-layer merge is already documented: it prefers the leaf matching the harmonized label. That is a downstream selection policy, not a second anatomical assessment. There is no Henry row for 115, and the duplicate occurs between 113 and 116; a row-identity typo is possible but unproven. Do not silently move the second 114 annotation to 115.

For review, the candidate hemisphere correction is **R for 112 and 114**. Keep 112's IDM/layer 3 as the supplied parcel/layer annotation; 114's IDD5-versus-IDM identity remains ambiguous. Changing those parcel/layer labels requires architecture review. The isolated reruns retain the original L labels with an explicit conflict and resolved side Unknown; accepted workbooks have not been replaced.

## Evidence and reversibility

`source_readback.json` binds thirty source workbooks and bounded portal/raw/atlas requests. `annotation_and_identity_readback.json` preserves exact Excel row numbers, neighboring labels, duplicate rows and graph variants. `native_visual_provenance.json` binds current public-toolkit acquisitions and outputs. `visual_and_original_source_readback.json` records ten inspected original/generated images, complete original soma-voxel comparisons and unchanged workbooks. `same_sample_control_visual_readback.json` records two inspected controls with independently re-read Henry/atlas labels.

The first isolated helper invocation used the wrong export keyword and wrote no images; its failure ledger was retained. The helper was corrected to the existing `acquisition_metadata` interface and the successful rerun has zero producer errors. No production code, original images or annotations were changed during this targeted check. New evidence remains in this dated case folder.
"""
    (OUT / "README.md").write_text(text,encoding="utf-8")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry=f"""### Targeted native check of 251637/112 and /114 ({stamp})

**Agent: Codex | Date: {stamp} (Asia/Shanghai)**

Fresh server atlas graphs match the cached sources for both historical conflicts. Direct native full-section, high-resolution soma and public-style wide-field checks place both somata on viewer-left, in the same displayed hemisphere as same-sample agreeing Henry-R control 111; Henry-L control 422 lies on viewer-right. The original March bulk views were also inspected, and both regenerated soma NIfTIs match every original voxel. Original lab/repository Henry workbooks are byte-identical; all thirty source workbooks checked remain unchanged.

Across Henry's full sheet, **293/295 distinct neurons agree with the official atlas hemisphere**; only 112/114 conflict. Adjacent 110/111/113/116 are all annotated R with nearby R roots. **Evidence favors localized manual L-prefix errors and R hemisphere for 112/114**, conditional on the supplied atlas and the other human annotations. Original labels remain preserved; isolated outputs still flag the conflicts rather than silently correcting accepted data. No independent acquisition-orientation certificate or source registration transform was recovered.

114 is entered twice (L-IDD5 and L-IDM, both layer 3), and 115 has no entry. The existing Henry-layer merge already documents its duplicate-selection policy; region identity remains unresolved and the second entry must not automatically be reassigned to 115. Detailed case sources, exact Excel rows, original/current figures and hash-bound readbacks: [112/114 targeted review](../notes/region_analysis_review_20261004/case_112_114_20261004/README.md). This check changed no production code or original annotations.

"""
    note=ROOT / "docs/project_note.md"
    current=note.read_text(encoding="utf-8")
    marker="### Completed region software and visual review"
    index=current.index(marker)
    note.write_text(current[:index]+entry+current[index:],encoding="utf-8")
    review=OUT.parent / "README.md"
    with review.open("a",encoding="utf-8") as stream:
        stream.write("\n## Targeted follow-up for 112/114\n\nThe [native and original-source case review](case_112_114_20261004/README.md) favors localized manual L-prefix errors: both match the R side of same-sample controls, while 293/295 distinct Henry annotations agree with atlas sides. Original labels remain unchanged. The duplicate 114 parcel entry and absolute acquisition/transform provenance remain unresolved.\n")
    print(json.dumps({"status":"targeted_review_recorded","native_controls_verified":len(inspected),"production_edits":0}))

if __name__ == "__main__":
    main()
