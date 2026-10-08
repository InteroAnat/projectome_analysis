"""Repair the clipped QC colorbar text without repeating regional calculations."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import nibabel as nib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
from summarize_mstim_arm import plot_qc
from export_arm_projection_tables import sha256


def main():
    base = Path(__file__).resolve().parent
    run = base / "cm032_regional_comparison"
    output = base / "cm032_readable_qc"
    if output.exists():
        raise FileExistsError("Use a fresh readable display directory")
    receipt = run / "integration_provenance.json"
    record = json.loads(receipt.read_text(encoding="utf-8"))
    warp = json.loads((base / "cm032_warp_reproduction/warp_reproduction.json").read_text(encoding="utf-8"))
    sources = {str(receipt): sha256(receipt)}
    for name, expected in record["outputs"].items():
        path = run / name
        if sha256(path) != expected:
            raise ValueError("Original integration derivative changed")
        sources[str(path)] = expected
    inputs = [ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz",
              Path(warp["outputs"]["reproduced_contrast"]["path"]),
              Path(warp["outputs"]["analysis_coverage"]["path"]),
              Path("D:/multimodal_fmri/derivatives/norm/sub-cm032/ses-20160309_ana/cm032_20160309_ana_NMT_Warped.nii.gz")]
    for path in inputs:
        if record["source_sha256"][str(path.resolve())] != sha256(path):
            raise ValueError("Bound display image changed")
        sources[str(path)] = sha256(path)
    output.mkdir()
    png = output / "cm032_alignment_and_response_qc.png"
    images = [nib.load(path) for path in inputs]
    display = plot_qc(images[0], *[image.get_fdata() for image in images[1:]], png)
    if any(sha256(path) != expected for path, expected in sources.items()):
        raise RuntimeError("A bound display source changed")
    result = {"status": "display_correction_completed", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "reason": "Original long vertical colorbar text clipped; wrapped label and tight saved bounds",
              "sources": sources, "display": display,
              "script_sha256": sha256(__file__),
              "renderer_sha256": sha256(ROOT / "group_analysis/scripts/summarize_mstim_arm.py"),
              "output_sha256": {png.name: sha256(png)},
              "regional_values_recomputed": False, "scientific_acceptance": False}
    (output / "display_provenance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(png)}))


if __name__ == "__main__":
    main()
