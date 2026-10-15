"""Render actual soma coordinates and endpoints against bundled NMT references."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "main_scripts"))
from swc_validation import parse_swc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def side(name):
    for prefix, value in [("CL_", "L"), ("SL_", "L"), ("CR_", "R"), ("SR_", "R"),
                          ("L-", "L"), ("R-", "R"), ("L_", "L"), ("R_", "R")]:
        if str(name).startswith(prefix):
            return value
    return "Unknown"


def main():
    bundle = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym"
    paths = {"template": bundle / "NMT_v2.1_sym_SS.nii.gz",
             "official_arm": bundle / "ARM_in_NMT_v2.1_sym.nii.gz",
             "analysis_arm": ROOT / "atlas/ARM_in_NMT_v2.1_sym.nii.gz",
             "lr_brainmask": bundle / "supplemental_masks/NMT_v2.1_sym_LR_brainmask.nii.gz",
             "lr_plane": bundle / "supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz",
             "key": ROOT / "atlas/ARM_key_all.txt", "bundle_readme": ROOT / "atlas/NMT_v2.1_sym/readme.md"}
    images = {name: nib.load(path) for name, path in paths.items() if name not in ("key", "bundle_readme")}
    template = np.asanyarray(images["template"].dataobj)
    arm = np.asanyarray(images["official_arm"].dataobj)[..., 0, 5]
    analysis = np.asanyarray(images["analysis_arm"].dataobj)[..., 0, 5]
    plane = np.asanyarray(images["lr_plane"].dataobj)
    brainmask = np.asanyarray(images["lr_brainmask"].dataobj)
    for name, image in images.items():
        if image.shape[:3] != template.shape or not np.allclose(image.affine, images["template"].affine):
            raise ValueError(f"Reference grid mismatch: {name}")
    if not np.array_equal(arm, analysis):
        raise ValueError("Analysis ARM level 6 differs from bundled official ARM")
    key = pd.read_csv(paths["key"], sep="\t")
    names = dict(zip(key.Index, key.Abbreviation))
    # Establish mask-value semantics from the documented CL/CR/SL/SR labels
    # in the official bundle, recording the entire contingency, not medians.
    label_sides = np.zeros(int(arm.max()) + 1, dtype=np.uint8)
    for rid, name in names.items():
        label_sides[int(rid)] = {"L": 1, "R": 2, "Unknown": 0}[side(name)]
    arm_sides = label_sides[arm.astype(int)]
    contingency, mask_to_side = {}, {}
    for value in (1, 2):
        counts = {s: int(np.count_nonzero((plane == value) & (arm_sides == code))) for s, code in [("L", 1), ("R", 2)]}
        contingency[str(value)] = counts
        if min(counts.values()) != 0 or max(counts.values()) == 0:
            raise ValueError(f"Mask and documented atlas hemispheres disagree: {counts}")
        mask_to_side[value] = max(counts, key=counts.get)
    sources, tables = [], []
    for path in sorted((ROOT / "group_analysis/step1_results").glob("*/tables/*.xlsx")):
        df = pd.read_excel(path, "Summary")
        df["source"] = str(path.relative_to(ROOT))
        tables.append(df)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
    refined_path = ROOT / "neuron_tables_new/251637_INS_HE_inferred.xlsx"
    refined = pd.read_excel(refined_path, "Summary")
    refined["SampleID"] = "251637"
    refined["source"] = str(refined_path.relative_to(ROOT))
    sources.append({"path": str(refined_path.relative_to(ROOT)), "sha256": sha(refined_path)})
    all_rows = pd.concat(tables + [refined], ignore_index=True)
    findings = []
    for _, row in all_rows.iterrows():
        vox = np.rint([row["Soma_NII_" + a] for a in "XYZ"]).astype(int)
        inside = all(0 <= vox[d] < plane.shape[d] for d in range(3))
        reference = mask_to_side[int(plane[tuple(vox)])] if inside else "Unknown"
        brain_side = mask_to_side.get(int(brainmask[tuple(vox)]), "Outside_brainmask") if inside else "Out_of_Bounds"
        declared = side(row.Soma_Region)
        findings.append({"SampleID": str(row.SampleID), "NeuronID": row.NeuronID,
                         "Soma_Region": row.Soma_Region, "stored_side": row.Soma_Side,
                         "label_side": declared, "mask_plane_side": reference,
                         "mask_brain_side": brain_side,
                         "label_mask_conflict": declared in ("L", "R") and reference != declared,
                         "stored_side_mask_conflict": str(row.Soma_Side) in ("L", "R") and reference != str(row.Soma_Side),
                         "source": row.source})
    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(OUT / "atlas_mask_laterality_rows.csv", index=False)
    hemisphere_colors = {"L": "#3a9ae040", "R": "#ff9f4540"}
    cmap = ListedColormap(["#00000000"] + [hemisphere_colors[mask_to_side[v]] for v in (1, 2)])
    template_vmax = np.percentile(template[template > 0], 99)
    slice_axes = [(1, (0, 2), "Coronal"), (2, (0, 1), "Axial"), (0, (1, 2), "Sagittal")]

    def draw(ax, xyz, axis, displayed, title):
        position = int(np.rint(xyz[axis]))
        ax.imshow(np.take(template, position, axis=axis).T, cmap="gray", origin="lower", vmin=0, vmax=template_vmax)
        ax.imshow(np.take(brainmask, position, axis=axis).T, cmap=cmap, origin="lower", vmin=0, vmax=2)
        ax.scatter(xyz[displayed[0]], xyz[displayed[1]], marker="+", color="magenta", s=100, linewidths=1.6)
        if 0 in displayed:
            ax.axvline(127.5, color="white", linestyle=":", lw=0.7)
            ax.text(5, 5, "L", color="#3a9ae0", weight="bold")
            ax.text(plane.shape[0]-15, 5, "R", color="#ff9f45", weight="bold")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("voxel " + "XYZ"[displayed[0]], fontsize=8)
        ax.set_ylabel("voxel " + "XYZ"[displayed[1]], fontsize=8)

    figures = []
    # Each monkey: deterministic left, right, and unlabeled examples; original
    # coordinates are overlaid at their actual slice, with no reflection.
    for df in tables:
        sample = str(df.SampleID.iloc[0])
        selections = []
        sides = df.Soma_Region.map(side)
        for s in ("L", "R", "Unknown"):
            candidates = df[sides.eq(s)].sort_values("NeuronID")
            if not candidates.empty:
                selections.append(candidates.iloc[0])
        fig, axs = plt.subplots(len(selections), 3, figsize=(12, 4 * len(selections)), squeeze=False)
        for row_index, row in enumerate(selections):
            xyz = np.array([row["Soma_NII_" + a] for a in "XYZ"])
            for col, (axis, displayed, title) in enumerate(slice_axes):
                draw(axs[row_index, col], xyz, axis, displayed, f"{row.NeuronID} | {row.Soma_Region} | stored {row.Soma_Side}\n{title}, slice {int(np.rint(xyz[axis]))}")
        fig.suptitle(f"Sample {sample}: soma vs official NMT hemisphere masks\nBlue=L, orange=R, magenta=soma; atlas voxel frame; no mirroring", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = FIG / f"sample-{sample}_space-NMTv2p1_desc-somaLaterality_examples.png"
        fig.savefig(path, dpi=130); plt.close(fig)
        figures.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "neurons": selections and [r.NeuronID for r in selections]})
    # The two known conflicts: roots and all endpoints from their actual cached
    # atlas-space SWC. Endpoint projections are explicitly marked as projections.
    for neuron in ("112.swc", "114.swc"):
        row = refined[refined.NeuronID.eq(neuron)].iloc[0]
        swc = ROOT / "neuron-vis/resource/swc/251637" / neuron
        nodes = parse_swc(swc.read_text(), source=str(swc))
        root = next(r for r in nodes if r[6] == -1)
        xyz = np.array(root[2:5]) / 250
        parents = {r[6] for r in nodes}
        ends = np.array([r[2:5] for r in nodes if r[0] not in parents]) / 250
        if not np.allclose(xyz, [row["Soma_NII_" + a] for a in "XYZ"], atol=0.00005):
            raise ValueError("Source root differs from stored soma")
        fig, axs = plt.subplots(2, 3, figsize=(13, 9))
        for col, (axis, displayed, title) in enumerate(slice_axes):
            draw(axs[0, col], xyz, axis, displayed, f"Root: {title}")
            ax = axs[1, col]
            ax.imshow(np.max(template, axis=axis).T, cmap="gray", origin="lower")
            for endpoint in ends:
                vox = np.rint(endpoint).astype(int)
                inside = all(0 <= vox[d] < plane.shape[d] for d in range(3))
                color = {"L": "#3a9ae0", "R": "#ff9f45"}.get(mask_to_side.get(int(plane[tuple(vox)])) if inside else None, "red")
                ax.scatter(endpoint[displayed[0]], endpoint[displayed[1]], color=color, s=10)
            ax.scatter(xyz[displayed[0]], xyz[displayed[1]], marker="+", color="magenta", s=100)
            ax.set_title(f"All {len(ends)} endpoints: {title} projection", fontsize=10)
        finding = findings_df[(findings_df.SampleID.eq("251637")) & findings_df.NeuronID.eq(neuron)].iloc[0]
        fig.suptitle(f"251637 / {neuron}: refined {row.Soma_Region}; stored side {row.Soma_Side}; mask side {finding.mask_plane_side}\nLabel-mask conflict={finding.label_mask_conflict}; atlas-space source only; no native registration acceptance", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = FIG / f"sample-251637_neuron-{neuron[:-4]}_space-NMTv2p1_desc-labelSideConflict_somaEndpoints.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        figures.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "swc_source": str(swc.relative_to(ROOT)), "swc_sha256": sha(swc)})
    report = {"status": "rendered_pending_visual_inspection", "coordinate_mapping": "cached atlas SWC micrometres / 250 -> NMT voxels; not affine world-coordinate conversion",
              "sources": sources + [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths.values()],
              "official_analysis_arm_level6_equal": True, "all_reference_grids_equal": True,
              "mask_value_to_side": mask_to_side, "mask_semantic_contingency": contingency,
              "overlay_palette": {str(v): hemisphere_colors[mask_to_side[v]] for v in (1, 2)},
              "checked_rows": len(findings_df), "label_mask_conflicts": int(findings_df.label_mask_conflict.sum()),
              "stored_side_mask_conflicts": int(findings_df.stored_side_mask_conflict.sum()), "figures": figures,
              "scope_limit": "Atlas-frame consistency; original fMOST-to-NMT registration and human refined-label validity require source review."}
    (OUT / "atlas_laterality_visual_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "checked_rows", "label_mask_conflicts", "stored_side_mask_conflicts", "mask_value_to_side")}, indent=2))


if __name__ == "__main__":
    main()
