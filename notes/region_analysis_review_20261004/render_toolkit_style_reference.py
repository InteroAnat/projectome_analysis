"""Exercise unchanged public plot interfaces on actual persisted native crops.

These separate examples demonstrate toolkit styling without replacing bulk
outputs or reloading server pixels. They use the current corrected geometry.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "toolkit_style_reference"
sys.path.insert(0, str(ROOT / "main_scripts"))
from Visual_toolkit import Visual_toolkit


def main():
    sample, neuron = "252385", "050"
    source = ROOT / "group_analysis/visual_review_20261002" / sample / neuron
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    swc = ROOT / "group_analysis/visual_review_20261002/swc_raw" / sample / f"{neuron}.swc"
    if not swc.is_file():
        raise FileNotFoundError(swc)
    nodes = np.loadtxt(swc, comments="#", ndmin=2)
    by_id = {int(row[0]): SimpleNamespace(x=row[2], y=row[3], z=row[4]) for row in nodes}
    edges = [SimpleNamespace(data=[by_id[int(row[6])], by_id[int(row[0])]]) for row in nodes if row[6] != -1]
    soma = [provenance[f"native_soma_{axis}_um"] for axis in "xyz"]
    OUT.mkdir(exist_ok=True)
    toolkit = Visual_toolkit(sample, cache_dir=str(OUT / "cache"))
    inputs = []
    for kind, method in (("soma", toolkit.plot_soma_block), ("context", toolkit.plot_widefield_context)):
        path = source / f"{kind}.nii.gz"
        image = nib.load(path)
        volume = np.transpose(np.asanyarray(image.dataobj), (2,1,0))
        origin = provenance[f"{kind}_origin_xyz_um"]
        spacing = provenance[f"declared_spacing_{'high' if kind == 'soma' else 'low'}_xyz_um"]
        options = {"swc_tree": SimpleNamespace(edges=edges)} if kind == "context" else {}
        method(volume, origin, spacing, soma, f"{neuron}.swc", soma_region="CR_PrCO",
               output_dir=str(OUT), **options)
        inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    outputs = [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(OUT.glob("*.png"))]
    (OUT / "provenance.json").write_text(json.dumps({"public_methods": ["plot_soma_block", "plot_widefield_context"],
        "inputs": inputs, "raw_swc_sha256": hashlib.sha256(swc.read_bytes()).hexdigest(),
        "outputs": outputs, "toolkit_sha256": hashlib.sha256((ROOT / "main_scripts/Visual_toolkit.py").read_bytes()).hexdigest(),
        "style": "Existing toolkit grayscale soma MIP/cyan marker and green wide-field signal/red SWC edges",
        "bulk_outputs_modified": False, "registration_acceptance": "not established"}, indent=2), encoding="utf-8")
    print(json.dumps({"style_reference_images": len(outputs), "output": str(OUT)}))


if __name__ == "__main__":
    main()
