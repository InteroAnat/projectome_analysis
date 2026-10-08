"""Render saved native crops through the original public toolkit plot methods.

Primary views are the high-resolution soma MIP and low-resolution wide-field
MIP. High-resolution-derived context is explicitly separated from LowRes.
This command reads existing crops/SWCs only; it does not acquire, align, repair
or promote data. Existing outputs are never replaced.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np

from Visual_toolkit import Visual_toolkit
from swc_validation import parse_swc

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_pair(source_root, output_root, sample, neuron, label):
    folder = source_root / sample / neuron
    metadata_path = folder / "provenance.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = source_root / "swc_raw" / sample / (neuron + ".swc")
    nodes = parse_swc(raw.read_text(encoding="utf-8"), source=str(raw))
    soma = list(next(row for row in nodes if row[6] == -1)[2:5])
    expected = [metadata["native_soma_" + axis + "_um"] for axis in "xyz"]
    if not np.allclose(soma, expected, atol=1e-8, rtol=0):
        raise ValueError("Raw root and saved crop provenance disagree")
    if digest(raw) != metadata["swc_sha256"]:
        raise ValueError("Raw SWC differs from bound crop source")
    points = {int(row[0]): SimpleNamespace(x=row[2], y=row[3], z=row[4]) for row in nodes}
    tree = SimpleNamespace(edges=[SimpleNamespace(data=[points[int(row[6])], points[int(row[0])]])
                                  for row in nodes if row[6] != -1])
    # Plot methods need only these display fields. Avoid constructor acquisition
    # caches and SSH setup: this is an offline rendering operation.
    toolkit = Visual_toolkit.__new__(Visual_toolkit)
    toolkit.sample_id = sample
    toolkit.output_dir = str(output_root / sample / neuron)
    record = {"SampleID": sample, "NeuronID": neuron + ".swc", "display_label": label,
              "label_role": "original step1 label; not an accepted correction",
              "native_soma_xyz_um": soma, "raw_swc": str(raw),
              "raw_swc_sha256": digest(raw), "outputs": [], "inputs": [],
              "lowres_alignment_status": "not independently established",
              "anatomical_acceptance": False}
    for kind in ("soma", "context"):
        path = folder / (kind + ".nii.gz")
        sidecar = Path(str(path) + ".json")
        if not path.is_file() or not sidecar.is_file():
            record.setdefault("unavailable", []).append(kind)
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("array_axis_order") != "XYZ":
            raise ValueError("Saved NIfTI axis order must be explicit XYZ")
        origin, spacing = meta["origin_xyz_um"], meta["spacing_xyz_um"]
        image = nib.load(path)
        if list(image.shape) != meta["shape_xyz"]:
            raise ValueError("NIfTI shape and source sidecar disagree")
        volume = np.transpose(np.asanyarray(image.dataobj), (2, 1, 0))
        local = (np.asarray(soma) - origin) / spacing
        if not np.all((local >= 0) & (local < np.array(volume.shape)[::-1])):
            raise ValueError("Raw soma is outside saved crop")
        derived = kind == "context" and bool(meta.get("derived_from"))
        group = "HighRes" if kind == "soma" else "DerivedContext" if derived else "LowRes"
        suffix = "SomaBlock" if kind == "soma" else "DerivedContext" if derived else "WideField"
        target = output_root / sample / neuron / group / "Plots"
        target.mkdir(parents=True, exist_ok=False)
        before = digest(path)
        if kind == "soma":
            toolkit.plot_soma_block(volume, origin, spacing, soma, neuron + ".swc",
                                    soma_region=label, suffix=suffix, output_dir=str(target))
        else:
            toolkit.plot_widefield_context(volume, origin, spacing, soma, neuron + ".swc",
                                           soma_region=label, suffix=suffix,
                                           swc_tree=tree, output_dir=str(target))
        produced = list(target.glob("*.png"))
        if len(produced) != 1 or digest(path) != before:
            raise RuntimeError("Expected one plot and unchanged source pixels")
        record["inputs"].append({"path": str(path), "sha256": before,
                                 "sidecar_sha256": digest(sidecar), "origin_xyz_um": origin,
                                 "spacing_xyz_um": spacing, "complete": meta.get("complete")})
        caption = "High-resolution soma MIP" if kind == "soma" else (
            "Derived local context MIP; not a whole-section LowRes acquisition" if derived
            else "Copied low-resolution wide-field MIP; native-to-overview alignment unverified")
        if meta.get("complete") is False:
            caption += "; PARTIAL source coverage"
        record["outputs"].append({"path": produced[0].relative_to(output_root).as_posix(),
                                  "sha256": digest(produced[0]), "kind": kind, "caption": caption})
        del volume, image
    record["inputs"].append({"path": str(metadata_path), "sha256": digest(metadata_path)})
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "group_analysis/visual_review_20261002")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Sample/neuron, e.g. 252384/003; repeat to select. Default: all saved cases.")
    args = parser.parse_args(argv)
    source = args.source_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Style outputs must be separate from the original bulk tree")
    if output.exists():
        raise FileExistsError(output)
    with (source / "manifest/identity_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    labels = {(row["SampleID"], row["NeuronID"].removesuffix(".swc")): row["step1_soma_region"] or "Unknown_0"
              for row in manifest}
    cases = args.case or [sample + "/" + neuron for sample, neuron in sorted(labels)]
    selected = []
    for case in cases:
        sample, neuron = case.split("/")
        if not sample.isdigit() or not neuron.isdigit() or (sample, neuron) not in labels:
            raise ValueError("Case must match a manifest sample/neuron identity")
        if (sample, neuron) in selected:
            raise ValueError("Duplicate case")
        selected.append((sample, neuron))
    output.mkdir(parents=True)
    report = {"status": "rendering", "toolkit_sha256": digest(ROOT / "main_scripts/Visual_toolkit.py"),
              "renderer_sha256": digest(Path(__file__)), "source_manifest_sha256": digest(source / "manifest/identity_manifest.csv"),
              "public_plot_methods": ["plot_soma_block", "plot_widefield_context"],
              "style": "Original grayscale soma/cyan marker and green wide-field/red traces/white marker, both MIP",
              "original_bulk_modified": False, "coordinates_changed": False, "records": [], "errors": []}
    report_path = output / "render_provenance.json"
    for sample, neuron in selected:
        try:
            report["records"].append(render_pair(source, output, sample, neuron, labels[sample, neuron]))
        except Exception as exc:
            report["errors"].append({"SampleID": sample, "NeuronID": neuron, "error": str(exc)})
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["status"] = "rendered_pending_actual_inspection" if not report["errors"] else "failed_cases_recorded"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cards = []
    for record in report["records"]:
        views = "".join('<figure><img src="{}"><figcaption>{}</figcaption></figure>'.format(
            html.escape(item["path"]), html.escape(item["caption"])) for item in record["outputs"])
        cards.append('<article><h2>{} / {}</h2><p>Source label: {}. Alignment is not accepted by rendering.</p><div>{}</div></article>'.format(
            html.escape(record["SampleID"]), html.escape(record["NeuronID"]), html.escape(record["display_label"]), views))
    page = '<!doctype html><meta charset="utf-8"><title>Native toolkit views</title><style>body{background:#171717;color:#eee;font:16px sans-serif;margin:24px}article{margin:30px 0}article div{display:flex;flex-wrap:wrap}figure{margin:8px;width:min(46%,720px)}img{width:100%}figcaption{padding:8px}</style><h1>Original toolkit presentation</h1><p>Primary soma and wide-field views. Partial derived contexts are labelled separately. Original bulk files are unchanged.</p>' + "".join(cards)
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(report["records"]), "errors": report["errors"]}))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
