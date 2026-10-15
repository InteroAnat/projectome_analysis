"""Actual full-section locator contact sheets; source bulk PNGs remain unchanged."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    manifest_path = ROOT / "group_analysis/visual_review_20261002/manifest/identity_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    mask_path = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz"
    mask = np.asanyarray(nib.load(mask_path).dataobj)
    folder = OUT / "native_contact_sheets"
    folder.mkdir(exist_ok=True)
    records, missing = [], []
    for sample in ("252384", "252385", "252527", "252714"):
        selected = manifest[manifest.SampleID.eq(sample)].sort_values("NeuronID")
        available = []
        for _, row in selected.iterrows():
            path = ROOT / "group_analysis/visual_review_20261002" / sample / Path(row.NeuronID).stem / "section_locator.png"
            if path.is_file():
                available.append((row, path))
            else:
                missing.append({"UID": row.UID, "reason": "no existing full-section locator"})
        for start in range(0, len(available), 20):
            page = available[start:start+20]
            fig, axes = plt.subplots(4, 5, figsize=(20, 16), facecolor="white")
            sources = []
            for ax in axes.flat:
                ax.axis("off")
            for ax, (row, path) in zip(axes.flat, page):
                image = np.array(Image.open(path).convert("RGB"))
                prov_path = path.with_name("provenance.json")
                prov = json.loads(prov_path.read_text(encoding="utf-8"))
                xyz = np.array([float(row["nmt_nii_"+a]) if row["nmt_nii_"+a] else np.nan for a in "xyz"])
                side = "Unknown"
                if np.isfinite(xyz).all():
                    v = np.rint(xyz).astype(int)
                    if np.all((v >= 0) & (v < np.array(mask.shape))):
                        side = {1: "R", 2: "L"}.get(int(mask[tuple(v)]), "Unknown")
                ax.imshow(image)
                ax.set_title(f"{row.UID} | atlas-mask side {side}\n{row.step1_soma_region or 'no step1 label'} | native X={prov['native_soma_x_um']:.0f} um",
                             fontsize=9, color={"L":"#1856ac", "R":"#b65b00", "Unknown":"#444444"}[side])
                sources.append({"UID": row.UID, "locator_path": path.relative_to(ROOT).as_posix(),
                                "locator_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "provenance_sha256": hashlib.sha256(prov_path.read_bytes()).hexdigest(),
                                "atlas_mask_side": side, "native_x_um": prov["native_soma_x_um"]})
            page_index = start // 20 + 1
            fig.suptitle(f"Native full-section locator review | sample {sample} | page {page_index}\n"
                         "Original white boxes mark wide-field crops. Atlas side is a comparison label; native anatomical orientation is unverified.", fontsize=13)
            fig.tight_layout(rect=(0,0,1,0.95))
            path = folder / f"sample-{sample}_desc-nativeLocator_page-{page_index:02d}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "sample": sample, "page": page_index, "sources": sources, "status": "rendered_pending_actual_image_inspection"})
    report = {"status": "rendered_pending_actual_image_inspection", "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(), "pages": records,
              "locator_count": sum(len(page["sources"]) for page in records), "missing_locators": missing,
              "bulk_inputs_modified": False}
    (OUT / "native_contact_sheet_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(records), "locator_count": report["locator_count"], "missing": len(missing)}))


if __name__ == "__main__":
    main()
