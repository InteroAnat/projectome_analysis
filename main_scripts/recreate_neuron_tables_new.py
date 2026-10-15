"""Rebuild refined subtables from fresh local-SWC analysis in a new output tree.

Historical template membership and manual soma annotations are preserved.
Derived classifications, hierarchy and projection splits are recomputed together.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = "251637"
SOURCE_TABLES_DIR = ROOT / "neuron_tables"
REFINED_SOMA_XLSX = SOURCE_TABLES_DIR / "251637_INS_HE_inferred.xlsx"
SWC_SOURCE_DIR = ROOT / "neuron-vis/resource/swc" / SAMPLE_ID


def source_files_for_subtables(source_dir):
    files = []
    for path in sorted(Path(source_dir).glob(f"{SAMPLE_ID}_*.xlsx")):
        if "results_clustered" in path.name.lower():
            continue
        with pd.ExcelFile(path) as workbook:
            if "Summary" not in workbook.sheet_names:
                continue
            ids = pd.read_excel(workbook, "Summary")
        if "NeuronID" not in ids:
            print(f"[SKIP] Aggregate Summary without neuron identities: {path.name}")
            continue
        if not ids.empty:
            if ids.NeuronID.isna().any() or ids.NeuronID.duplicated().any():
                raise ValueError(f"Template IDs must be present and unique: {path}")
            files.append(path)
    return files


def extract_neuron_ids(path):
    return pd.read_excel(path, "Summary", usecols=["NeuronID"]).NeuronID.astype(str).tolist()


def build_subtable(base_sheets, neuron_ids):
    selected = set(neuron_ids)
    if not selected.issubset(set(base_sheets["Summary"].NeuronID.astype(str))):
        raise ValueError("Fresh master results do not cover every requested template neuron")
    return {name: df[df.NeuronID.astype(str).isin(selected)].copy() if "NeuronID" in df else df.copy()
            for name, df in base_sheets.items()}


def write_workbook(path, sheets):
    # Refuse replacement, including accidental reuse of an old accepted tree.
    if path.exists():
        raise FileExistsError(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def main(source_dir=SOURCE_TABLES_DIR, refined_soma=REFINED_SOMA_XLSX,
         swc_source_dir=SWC_SOURCE_DIR, output_dir=None):
    sources = source_files_for_subtables(source_dir)
    if not sources:
        raise ValueError("No source templates found")
    memberships = {path: extract_neuron_ids(path) for path in sources}
    ids = sorted(set(nid for selected in memberships.values() for nid in selected))
    swc_source_dir = Path(swc_source_dir).resolve(strict=True)
    missing = [nid for nid in ids if not (swc_source_dir / nid).is_file()]
    if missing:
        raise FileNotFoundError(f"Local atlas-space SWC coverage incomplete: {missing}")
    output_dir = Path(output_dir or ROOT / "output" / f"251637_refined_subtables_{datetime.now():%Y%m%d_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("region_step1", ROOT / "main_scripts/step1.run_region_analysis.py")
    step1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step1)
    pop = step1.main(ids, SAMPLE_ID, swc_source_dir=swc_source_dir,
                     output_base=output_dir / "master_analysis", generate_plots=False,
                     refined_soma_path=refined_soma)
    if pop.load_failures:
        raise RuntimeError(f"Master analysis has load failures: {pop.load_failures}")
    sheets = pd.read_excel(pop.result_workbook, sheet_name=None)
    targets = output_dir / "subtables"
    targets.mkdir()
    records = []
    for path, selected in memberships.items():
        target = targets / path.name
        rebuilt = build_subtable(sheets, selected)
        write_workbook(target, rebuilt)
        readback = pd.read_excel(target, "Summary")
        if set(readback.NeuronID.astype(str)) != set(selected):
            raise RuntimeError(f"Subtable membership changed: {target}")
        records.append({"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "output": str(target), "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "neuron_ids": selected})
        print(f"[OK] {target.name}: {len(selected)} neurons")
    report = {"status": "fresh_derived_subtables_written", "master_workbook": str(pop.result_workbook),
              "manual_source": str(refined_soma), "outputs": records,
              "conflicting_annotations": pop.plot_dataframe.loc[pop.plot_dataframe.Soma_Side_Conflict,
                   ["NeuronID", "Soma_Region", "Soma_Side_Name", "Soma_Side_Reference"]].to_dict(orient="records"),
              "acceptance": "isolated derivatives; no source workbook replacement"}
    (output_dir / "subtable_rebuild_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_TABLES_DIR)
    parser.add_argument("--refined-soma", type=Path, default=REFINED_SOMA_XLSX)
    parser.add_argument("--local-swc-dir", type=Path, default=SWC_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    main(args.source_dir, args.refined_soma, args.local_swc_dir, args.output_dir)
