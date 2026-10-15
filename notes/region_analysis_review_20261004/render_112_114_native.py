"""Render own-native soma and context using public toolkit style, plus locators."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "case_112_114_20261004"
sys.path.insert(0, str(ROOT / "main_scripts"))
from Visual_toolkit import Visual_toolkit, RESOLUTION_HIGH, RESOLUTION_LOW
from swc_validation import parse_swc

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    report = {"status": "rendering", "started_at": datetime.now().astimezone().isoformat(),
              "public_toolkit_sha256": digest(ROOT / "main_scripts/Visual_toolkit.py"), "neurons": []}
    report_path = OUT / "native_visual_provenance.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    toolkit = Visual_toolkit("251637", cache_dir=str(OUT / "native_cache"), low_res_slice_cache_limit=2)
    for stem, label in (("112", "L-IDM"), ("114", "L-IDD5")):
        folder = OUT / f"neuron-{stem}"
        folder.mkdir(exist_ok=True)
        source = OUT / f"raw_{stem}.swc"
        nodes = parse_swc(source.read_text(encoding="utf-8"), source=str(source))
        root = next(row for row in nodes if row[6] == -1)
        soma = list(root[2:5])
        by_id = {int(row[0]): SimpleNamespace(x=row[2], y=row[3], z=row[4]) for row in nodes}
        edges = [SimpleNamespace(data=[by_id[int(row[6])], by_id[int(row[0])]]) for row in nodes if row[6] != -1]
        record = {"NeuronID": stem + ".swc", "manual_label_comparison": label, "native_root_xyz_um": soma,
                  "raw_sha256": digest(source), "acquisitions": [], "errors": []}
        report["neurons"].append(record); save()
        try:
            context, origin, spacing = toolkit.get_low_res_widefield(soma, width_um=4000, height_um=4000, depth_um=30, allow_partial=True)
            metadata = toolkit.last_low_res_metadata
            toolkit.export_data(context, origin, spacing, stem + ".swc", suffix="WideField", soma_region=label,
                                output_dir=str(folder), acquisition_metadata=metadata, soma_coords=soma)
            toolkit.plot_widefield_context(context, origin, spacing, soma, stem + ".swc", soma_region=label,
                                          output_dir=str(folder), swc_tree=SimpleNamespace(edges=edges))
            z = int(soma[2] / RESOLUTION_LOW[2])
            section_path = toolkit._get_low_res_slice_path(z)
            full = toolkit._read_cached_section(section_path)
            # Locator has no anatomical side glyphs: native XY display only.
            shown = full[::8, ::8]
            positive = shown[shown > 0]
            vmax = float(np.percentile(positive, 99.5)) if len(positive) else 1
            fig, ax = plt.subplots(figsize=(9,7))
            ax.imshow(shown, cmap="gray", origin="upper", vmin=0, vmax=vmax)
            ax.scatter(soma[0]/RESOLUTION_LOW[0]/8, soma[1]/RESOLUTION_LOW[1]/8,
                       marker="+", color="cyan", s=150, linewidths=1.8)
            ax.set_title(f"251637/{stem}: own native CH1 section {z}\nManual {label}; atlas-mask R; native orientation unverified")
            ax.set_xlabel("Native X increases to viewer-right (downsampled pixels)")
            ax.set_ylabel("Native Y increases downwards (downsampled pixels)")
            fig.tight_layout(); fig.savefig(folder / "native_full_section_soma_locator.png", dpi=140); plt.close(fig)
            record["acquisitions"].append({"kind": "context", "metadata": metadata, "central_full_section": {
                "path": str(section_path), "sha256": digest(Path(section_path)), "shape_yx": list(full.shape),
                "soma_pixel_xy": [soma[0]/RESOLUTION_LOW[0], soma[1]/RESOLUTION_LOW[1]]}})
            del full, context
        except Exception as exc:
            record["errors"].append({"kind": "context", "error": str(exc)})
        save()
        try:
            high, origin, spacing = toolkit.get_high_res_block(soma, grid_radius=1, allow_partial=True)
            metadata = toolkit.last_high_res_metadata
            toolkit.export_data(high, origin, spacing, stem + ".swc", suffix="SomaBlock", soma_region=label,
                                output_dir=str(folder), acquisition_metadata=metadata, soma_coords=soma)
            toolkit.plot_soma_block(high, origin, spacing, soma, stem + ".swc", soma_region=label, output_dir=str(folder))
            record["acquisitions"].append({"kind": "soma", "metadata": metadata})
            del high
        except Exception as exc:
            record["errors"].append({"kind": "soma", "error": str(exc)})
        record["outputs"] = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size}
                             for p in sorted(folder.glob("*")) if p.is_file()]
        save()
        print(json.dumps({"neuron": stem, "errors": record["errors"], "outputs": len(record["outputs"])}), flush=True)
    toolkit.close()
    report["status"] = "rendered_pending_actual_inspection"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()

if __name__ == "__main__":
    main()
