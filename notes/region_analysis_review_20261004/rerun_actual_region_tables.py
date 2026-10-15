"""Run the real step1 path on the union of existing tables and bulk candidates.

Independent numpy graph/atlas lookup and XLSX readback verify the declared
software contract. This is not acceptance of native-to-atlas registration.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "main_scripts"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_side(label):
    for prefix, side in (("CL_", "L"), ("SL_", "L"), ("CR_", "R"), ("SR_", "R"),
                         ("L-", "L"), ("L_", "L"), ("R-", "R"), ("R_", "R")):
        if str(label).startswith(prefix):
            return side
    return "Unknown"


def inventory():
    baseline = json.loads((OUT / "existing_table_region_baseline.json").read_text(encoding="utf-8"))
    scope = defaultdict(set)
    protected = []
    for record in baseline["sources"]:
        path = ROOT / record["source"]
        protected.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
        df = pd.read_excel(path, "Summary")
        if path.parent.name in ("neuron_tables", "neuron_tables_new"):
            sample = "251637"
        else:
            sample = path.name.split("_")[0]
        scope[sample].update(df.NeuronID.astype(str))
    manifest = ROOT / "group_analysis/visual_review_20261002/manifest/identity_manifest.csv"
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    for sample, selected in df.groupby("SampleID"):
        scope[sample].update(selected.NeuronID)
    protected.append({"path": manifest.relative_to(ROOT).as_posix(), "sha256": digest(manifest)})
    canonical = ROOT / "group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx"
    protected.append({"path": canonical.relative_to(ROOT).as_posix(), "sha256": digest(canonical)})
    sources, missing = [], []
    for sample, ids in sorted(scope.items()):
        for nid in sorted(ids):
            path = ROOT / "neuron-vis/resource/swc" / sample / nid
            record = {"SampleID": sample, "NeuronID": nid, "path": path.relative_to(ROOT).as_posix()}
            if path.is_file():
                sources.append({**record, "sha256": digest(path), "bytes": path.stat().st_size})
            else:
                missing.append(record)
    report = {"status": "local_source_inventory", "scope": {s: sorted(ids) for s, ids in sorted(scope.items())},
              "requested_neurons": sum(map(len, scope.values())), "available": len(sources),
              "missing": missing, "swc_sources": sources, "protected_inputs": protected,
              "coordinate_mapping": "cached atlas-space SWC XYZ / 250; native raw SWCs are not substituted"}
    (OUT / "actual_rerun_inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def independent_graph(path, atlas, mask, labels):
    nodes = np.loadtxt(path, comments="#", ndmin=2)
    ids, parents = nodes[:, 0].astype(int), nodes[:, 6].astype(int)
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1 or len(set(ids)) != len(ids):
        raise ValueError("SWC root/identity invariant failed")
    root = int(roots[0])
    xyz = nodes[:, 2:5] / 250.0
    vox = np.rint(xyz).astype(int)
    inside = np.all((vox >= 0) & (vox < np.array(atlas.shape)), axis=1)
    region = np.full(len(nodes), "Out_of_Bounds", dtype=object)
    rid = np.zeros(len(nodes), dtype=int)
    rid[inside] = atlas[tuple(vox[inside].T)]
    region[inside] = [labels.get(i, f"Unknown_{i}") for i in rid[inside]]
    leaves = ~np.isin(ids, parents[parents != -1])
    terminals = sorted(set(region[leaves]))
    # Independent graph edges; no tracer branch construction reused.
    id_to_row = {nid: i for i, nid in enumerate(ids)}
    children = np.flatnonzero(parents != -1)
    if any(p not in id_to_row for p in parents[children]):
        raise ValueError("SWC parent missing")
    starts = np.array([id_to_row[p] for p in parents[children]], dtype=int)
    # Preserve the documented scalar Python rounding contract. NumPy rounding
    # can differ by 0.001 at binary half ties even for identical graph edges.
    points = xyz.tolist()
    lengths = np.array([round(math.sqrt(sum((a - b) ** 2 for a, b in
                        zip(points[child], points[start]))), 3)
                        for child, start in zip(children, starts)])
    projection = defaultdict(float)
    for i, length in zip(starts, lengths):
        if inside[i] and rid[i] > 0 and region[i] in terminals:
            projection[region[i]] += float(length)
    side = {1: "R", 2: "L"}.get(int(mask[tuple(vox[root])]), "Unknown") if inside[root] else "Unknown"
    return {"soma": str(region[root]), "xyz": xyz[root].tolist(), "reference_side": side,
            "terminals": terminals, "total_length": float(lengths.sum()), "projection": dict(projection),
            "nodes": len(nodes), "endpoints": int(leaves.sum())}


def audit_sample(pop, sample, sources, atlas, mask, labels, hierarchy_paths, manual):
    label_ids = {name: index for index, name in labels.items()}
    sheets = pd.read_excel(pop.result_workbook, sheet_name=None)
    summary = sheets["Summary"].set_index("NeuronID", drop=False)
    terminals = sheets["Terminal_Sites"]
    all_lengths = sheets.get("Projection_Length_all", pd.DataFrame(columns=["NeuronID"])).set_index("NeuronID")
    failures, records = [], []
    def check(condition, nid, detail):
        if not condition:
            failures.append({"SampleID": sample, "NeuronID": nid, "check": detail})
    def matrix_row(name, nid):
        df = sheets.get(name)
        if df is None:
            return {}
        selected = df[df.NeuronID.eq(nid)]
        if len(selected) != 1:
            return {}
        return {c: float(selected.iloc[0][c]) for c in df.columns if c not in ("NeuronID", "Neuron_Type")
                and pd.notna(selected.iloc[0][c]) and selected.iloc[0][c] != 0}
    for entry in sources:
        nid = entry["NeuronID"]
        if nid not in summary.index:
            check(False, nid, "available local source absent from actual output")
            continue
        actual = independent_graph(ROOT / entry["path"], atlas, mask, labels)
        row = summary.loc[nid]
        check(not isinstance(row, pd.DataFrame), nid, "unique output identity")
        stored_label = manual.get(nid, actual["soma"])
        label_side = declared_side(stored_label)
        conflict = label_side != "Unknown" and actual["reference_side"] != "Unknown" and label_side != actual["reference_side"]
        side = "Unknown" if conflict else label_side if label_side != "Unknown" else actual["reference_side"]
        check(row.Soma_Region == stored_label, nid, "soma label / retained manual annotation")
        if manual:
            check(row.Soma_Region_Auto == actual["soma"], nid, "retained automatic soma lookup")
        check(row.Soma_Side == side and bool(row.Soma_Side_Conflict) == conflict, nid, "reference side / conflict")
        check(row.Soma_Side_Reference == actual["reference_side"], nid, "full precision mask lookup")
        target_ids = [label_ids.get(label, -1) for label in actual["terminals"]]
        if any(1169 <= i <= 1325 or 1669 <= i <= 1825 or 1083 <= i <= 1107 or 1583 <= i <= 1607 for i in target_ids):
            expected_type = "PT"
        elif any(1111 <= i <= 1168 or 1611 <= i <= 1668 for i in target_ids):
            expected_type = "CT"
        elif any(1051 <= i <= 1061 or 1551 <= i <= 1561 for i in target_ids):
            expected_type = "ITs"
        elif any((side == "L" and 501 <= i <= 1000) or (side == "R" and 1 <= i <= 500) for i in target_ids):
            expected_type = "ITc"
        elif any((side == "R" and 501 <= i <= 1000) or (side == "L" and 1 <= i <= 500) for i in target_ids):
            expected_type = "ITi"
        else:
            expected_type = "Unclassified"
        check(row.Neuron_Type == expected_type, nid, "independent declared ARM-range type contract")
        check(np.allclose([row["Soma_NII_" + a] for a in "XYZ"], actual["xyz"], atol=0.000051, rtol=0), nid, "full precision source root")
        selected = terminals[terminals.NeuronID.eq(nid)]
        check(sorted(selected.Terminal_Region.tolist()) == actual["terminals"], nid, "all graph-endpoint region labels, including outside atlas")
        check(row.Terminal_Count == len(actual["terminals"]), nid, "unique terminal-region count")
        check(row.Length_Unit == "voxel" and abs(row.Total_Length - actual["total_length"]) < 0.00001, nid, "all parent-child graph lengths / units")
        observed = ({c: float(all_lengths.loc[nid, c]) for c in all_lengths.columns
                     if c != "Neuron_Type" and all_lengths.loc[nid, c] != 0} if nid in all_lengths.index else {})
        check(set(observed) == set(actual["projection"]), nid, "retained atlas-labelled projection regions")
        check(all(abs(observed.get(c, 0) - v) < 0.00001 for c, v in actual["projection"].items()), nid, "parent-voxel projection length allocation")
        expected = {"Ipsilateral": defaultdict(float), "Contralateral": defaultdict(float), "Unknown": defaultdict(float)}
        terminal_counts = defaultdict(int)
        for label in actual["terminals"]:
            target_side = declared_side(label)
            lat = "Unknown" if "Unknown" in (side, target_side) else "Ipsilateral" if side == target_side else "Contralateral"
            terminal_counts[lat] += 1
            exported_lat = selected[selected.Terminal_Region.eq(label)].Laterality.iloc[0]
            check(exported_lat == lat, nid, "terminal sheet laterality")
            term_row = selected[selected.Terminal_Region.eq(label)].iloc[0]
            for level in (1, 3, 6):
                parent = hierarchy_paths.get(label, {}).get(level)
                value = term_row.get(f"Terminal_L{level}")
                check((pd.isna(value) if parent is None else value == parent), nid, f"independent CSV terminal hierarchy L{level}")
        for label, length in actual["projection"].items():
            target_side = declared_side(label)
            lat = "Unknown" if "Unknown" in (side, target_side) else "Ipsilateral" if side == target_side else "Contralateral"
            expected[lat][label[3:] if label.startswith(("CL_", "CR_", "SL_", "SR_")) else label] += length
        for lat, count_col, total_col, suffix in [("Ipsilateral", "N_Ipsilateral", "Total_Ipsilateral_Length", "ipsi"),
                ("Contralateral", "N_Contralateral", "Total_Contralateral_Length", "contra"),
                ("Unknown", "N_Laterality_Unknown", "Total_Unknown_Laterality_Length", None)]:
            check(row[count_col] == terminal_counts[lat], nid, "unique target " + count_col)
            check(abs(row[total_col] - sum(expected[lat].values())) < 0.00001, nid, "laterality scalar length " + lat)
            if suffix:
                split = matrix_row("Projection_Length_" + suffix, nid)
                check(set(split) == set(expected[lat]) and all(abs(split.get(c, 0)-v) < 0.00001 for c,v in expected[lat].items()), nid, "projection length split " + lat)
                strength = matrix_row("Projection_Strength_" + suffix, nid)
                check(set(strength) == set(expected[lat]) and all(abs(strength.get(c, 0) - round(float(np.log10(v + 1)), 4)) < 0.00001
                      for c,v in expected[lat].items()), nid, "projection log-strength split " + lat)
        check(abs(sum(sum(x.values()) for x in expected.values()) - sum(observed.values())) < 0.00001, nid, "ipsi+contra+unknown conservation")
        soma_sheet = sheets["Soma_Hierarchy"].set_index("NeuronID").loc[nid]
        for level in range(1, 7):
            expected_parent = hierarchy_paths.get(stored_label, {}).get(level)
            value = soma_sheet.get(f"L{level}")
            check((pd.isna(value) if expected_parent is None else value == expected_parent), nid, f"independent CSV soma hierarchy L{level}")
        records.append({"SampleID": sample, "NeuronID": nid, "source_sha256": entry["sha256"],
                        "atlas_soma": actual["soma"], "stored_soma": stored_label, "reference_side": actual["reference_side"],
                        "resolved_side": side, "conflict": conflict, "node_count": actual["nodes"],
                        "endpoint_count": actual["endpoints"], "unique_endpoint_regions": len(actual["terminals"]),
                        "total_length_voxels": actual["total_length"], "retained_projection_length_voxels": sum(actual["projection"].values()),
                        "Neuron_Type": row.Neuron_Type})
    # Every exported neuron type agrees across all sheets that declare it.
    for name, df in sheets.items():
        if "NeuronID" in df and "Neuron_Type" in df:
            for _, row in df.iterrows():
                check(row.Neuron_Type == summary.loc[row.NeuronID, "Neuron_Type"], row.NeuronID, "cross-sheet type " + name)
    return {"sample": sample, "workbook": str(pop.result_workbook.relative_to(ROOT)), "sha256": digest(pop.result_workbook),
            "checked": len(records), "records": records, "failures": failures, "load_failures": pop.load_failures}


def run(inv):
    spec = importlib.util.spec_from_file_location("region_step1", ROOT / "main_scripts/step1.run_region_analysis.py")
    step1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step1)
    atlas = np.asanyarray(nib.load(step1.ATLAS_PATH).dataobj)[..., 0, 5]
    mask = np.asanyarray(nib.load(step1.HEMISPHERE_MASK_PATH).dataobj)
    key = pd.read_csv(step1.TABLE_PATH, sep="\t")
    labels = dict(zip(key.Index, key.Abbreviation))
    hierarchy_paths = {}
    for path in (step1.CORTEX_HIERARCHY_CSV, step1.SUBCORTICAL_HIERARCHY_CSV):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            hierarchy_paths[row.Level_6_abbr] = {lv: row[f"Level_{lv}_abbr"] for lv in range(1, 7)
                                                if pd.notna(row[f"Level_{lv}_abbr"])}
    refined = ROOT / "neuron_tables/251637_INS_HE_inferred.xlsx"
    manual_df = pd.read_excel(refined, "Summary")
    manual = dict(zip(manual_df.NeuronID.astype(str), manual_df.Soma_Region))
    run_base = OUT / "runs" / f"all_sources_{datetime.now():%Y%m%d_%H%M%S}"
    run_base.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
              "inventory_sha256": digest(OUT / "actual_rerun_inventory.json"), "samples": [],
              "missing_local_sources": inv["missing"], "anatomical_acceptance": "not established"}
    report_path = run_base / "independent_readback.json"
    for sample in inv["scope"]:
        sources = [entry for entry in inv["swc_sources"] if entry["SampleID"] == sample]
        ids = [entry["NeuronID"] for entry in sources]
        source_dirs = {Path(entry["path"]).parent for entry in sources}
        if len(source_dirs) != 1:
            raise ValueError("Each isolated batch must have one explicit source directory")
        pop = step1.main(ids, sample, swc_source_dir=ROOT / next(iter(source_dirs)),
                         output_base=run_base, generate_plots=False,
                         refined_soma_path=refined if sample == "251637" else None)
        result = audit_sample(pop, sample, sources, atlas, mask, labels, hierarchy_paths, manual if sample == "251637" else {})
        report["samples"].append(result)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"sample_checked": sample, "rows": result["checked"], "failures": len(result["failures"])}), flush=True)
        del pop
        gc.collect()
    changed = [entry for entry in inv["protected_inputs"] if digest(ROOT / entry["path"]) != entry["sha256"]]
    report["changed_protected_inputs"] = changed
    count = sum(len(s["failures"]) + len(s["load_failures"]) for s in report["samples"])
    report["status"] = "passed_available_sources" if not count and not changed else "failed_readback"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "actual_rerun_latest.json").write_text(json.dumps({"report": report_path.relative_to(ROOT).as_posix(),
        "sha256": digest(report_path), "status": report["status"], "failure_count": count,
        "checked": sum(s["checked"] for s in report["samples"]), "missing_sources": len(inv["missing"])}), encoding="utf-8")
    if count or changed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--recovered", action="store_true")
    args = parser.parse_args()
    if args.recovered:
        inv = json.loads((OUT / "actual_rerun_inventory.json").read_text(encoding="utf-8"))
        recovery = json.loads((OUT / "missing_atlas_source_recovery.json").read_text(encoding="utf-8"))
        entries = [entry for entry in recovery["results"] if entry["status"] == "own_atlas_swc_recovered"]
        if recovery["status"] != "bounded_own_source_requests_finished":
            raise RuntimeError("Atlas source recovery has not finished")
        for entry in entries:
            if digest(ROOT / entry["path"]) != entry["sha256"]:
                raise RuntimeError("Recovered source hash changed")
        inv["scope"] = {sample: sorted(entry["NeuronID"] for entry in entries if entry["SampleID"] == sample)
                        for sample in sorted({entry["SampleID"] for entry in entries})}
        inv["missing"] = [entry for entry in recovery["results"] if entry["status"] != "own_atlas_swc_recovered"]
        inv["swc_sources"] = entries
        inv["available"] = len(entries)
        inv["requested_neurons"] = len(recovery["results"])
    else:
        inv = inventory()
    print(json.dumps({"requested": inv["requested_neurons"], "available": inv["available"], "missing": len(inv["missing"])}), flush=True)
    if args.run:
        run(inv)
