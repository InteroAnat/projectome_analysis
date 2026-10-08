"""Editable per-subject region-analysis entrypoint.

CHARM/SARM hierarchy tables take priority over ARM label-level metadata.
Local-source and refined-label options support isolated, reproducible review.
Importing this script does not start an analysis or contact the neuron server.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import nibabel as nib
import pandas as pd

from region_analysis import PopulationRegionAnalysis
from region_analysis.hemisphere import AtlasHemisphereReference

# Per-subject editable parameters: preserve this script as the working surface.
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = "251637"
NEURON_IDS = None  # e.g. ["001.swc", "003.swc"]; None processes the source list.
ATLAS_PATH = ROOT / "atlas/ARM_in_NMT_v2.1_sym.nii.gz"
TABLE_PATH = ROOT / "atlas/ARM_key_all.txt"
TEMPLATE_PATH = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz"
HEMISPHERE_MASK_PATH = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz"
CORTEX_HIERARCHY_CSV = ROOT / "atlas/CHARM_key_table_v2.csv"
SUBCORTICAL_HIERARCHY_CSV = ROOT / "atlas/SARM_key_table_v2.csv"
OUTPUT_BASE = ROOT / "output"

def main(neuron_ids=None, sample_id=SAMPLE_ID, show_plots=False,
         swc_source_dir=None, output_base=None, generate_plots=True,
         refined_soma_path=None):
    if isinstance(neuron_ids, str):
        neuron_ids = [neuron_ids]
    if neuron_ids is not None and (not neuron_ids or len(set(neuron_ids)) != len(neuron_ids)):
        raise ValueError("Neuron selection must be nonempty and unique")
    entries = None
    if swc_source_dir is not None:
        swc_source_dir = Path(swc_source_dir).resolve(strict=True)
        if not swc_source_dir.is_dir():
            raise ValueError("Local SWC source must be a directory")
        selected = neuron_ids if neuron_ids is not None else sorted(p.name for p in swc_source_dir.glob("*.swc"))
        if not selected:
            raise ValueError("No local SWC neurons selected")
        entries = [{"sampleid": str(sample_id), "name": name} for name in selected]
        for entry in entries:
            if Path(entry["name"]).name != entry["name"] or not entry["name"].endswith(".swc"):
                raise ValueError("Neuron IDs must be SWC basenames")
    reference = AtlasHemisphereReference.from_files(
        str(HEMISPHERE_MASK_PATH), str(ATLAS_PATH), str(TABLE_PATH), level=6)
    pop = PopulationRegionAnalysis(
        sample_id=str(sample_id), atlas=nib.load(ATLAS_PATH).get_fdata(),
        atlas_table=pd.read_csv(TABLE_PATH, sep="\t"), template_img=nib.load(TEMPLATE_PATH),
        arm_key_path=str(TABLE_PATH), cortex_hierarchy_csv=str(CORTEX_HIERARCHY_CSV),
        subcortical_hierarchy_csv=str(SUBCORTICAL_HIERARCHY_CSV),
        output_base=str(output_base or OUTPUT_BASE), create_output_folder=True,
        show_plots=show_plots, neuron_list=entries,
        swc_source_dir=str(swc_source_dir) if swc_source_dir is not None else None,
        hemisphere_reference=reference,
    )
    pop.process(neuron_id=neuron_ids, level=6)
    if pop.plot_dataframe.empty:
        raise RuntimeError(f"No neurons processed; load failures: {pop.load_failures}")
    refined_source = None
    if refined_soma_path is not None:
        path = Path(refined_soma_path).resolve(strict=True)
        labels = pd.read_excel(path, sheet_name="Summary")
        if labels.NeuronID.isna().any() or labels.NeuronID.duplicated().any():
            raise ValueError("Refined soma source IDs must be present and unique")
        if "SampleID" in labels and not labels.SampleID.astype(str).eq(str(sample_id)).all():
            raise ValueError("Refined soma source contains another sample")
        if labels.Soma_Region.isna().any():
            raise ValueError("Refined soma labels must be present")
        refined_source = f"{path}:sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
        pop.apply_soma_labels(dict(zip(labels.NeuronID.astype(str), labels.Soma_Region)), refined_source)
    output_path = pop.save_all(include_strength_levels=[3], generate_plots=generate_plots)
    tables = Path(output_path) / "tables"
    workbook = tables / f"{sample_id}_results.xlsx"
    renamed = tables / f"{sample_id}_results_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    if renamed.exists():
        raise FileExistsError(renamed)
    workbook.rename(renamed)
    pop.result_workbook = renamed
    provenance = {"sample_id": str(sample_id), "coordinate_mapping": "atlas SWC micrometres / 250 -> NMT voxels",
                  "hemisphere_reference": reference.source, "mask_value_to_side": reference.value_to_side,
                  "refined_soma_source": refined_source,
                  "swc_source_dir": str(swc_source_dir) if swc_source_dir is not None else "IONData",
                  "requested_neuron_ids": neuron_ids, "load_failures": pop.load_failures,
                  "processed": len(pop.plot_dataframe), "workbook": renamed.name,
                  "anatomical_registration_acceptance": "not established by this software run"}
    (Path(output_path) / "reports/run_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[COMPLETE] {renamed}; processed={len(pop.plot_dataframe)}; load_failures={len(pop.load_failures)}")
    return pop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default=SAMPLE_ID)
    parser.add_argument("--neuron-id", nargs="+", default=NEURON_IDS)
    parser.add_argument("--local-swc-dir", type=Path)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--refined-soma", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()
    result = main(args.neuron_id, args.sample_id, args.show_plots, args.local_swc_dir,
                  args.output_base, not args.no_plots, args.refined_soma)
    if result.load_failures:
        raise SystemExit(2)
