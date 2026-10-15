"""Independently recheck saved real-data exports, without rerunning production."""
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import pandas as pd
import rerun_actual_region_tables as audit

ROOT, OUT = audit.ROOT, audit.OUT

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

def main():
    inv = read(OUT / "actual_rerun_inventory.json")
    recovery = read(OUT / "missing_atlas_source_recovery.json")
    cached_path = OUT / "runs/all_sources_20261004_071659/independent_readback.json"
    recovered_path = OUT / "runs/all_sources_20261004_072050/independent_readback.json"
    cached, recovered = read(cached_path), read(recovered_path)
    if cached["status"] == "running" or recovered["status"] == "running":
        raise RuntimeError("Producers must finish before final readback")
    spec = importlib.util.spec_from_file_location("step1_readback", ROOT / "main_scripts/step1.run_region_analysis.py")
    step1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step1)
    atlas = np.asanyarray(nib.load(step1.ATLAS_PATH).dataobj)[..., 0, 5]
    mask = np.asanyarray(nib.load(step1.HEMISPHERE_MASK_PATH).dataobj)
    key = pd.read_csv(step1.TABLE_PATH, sep="\t")
    labels = dict(zip(key.Index, key.Abbreviation))
    hierarchy = {}
    for path in (step1.CORTEX_HIERARCHY_CSV, step1.SUBCORTICAL_HIERARCHY_CSV):
        for _, row in pd.read_csv(path).iterrows():
            hierarchy[row.Level_6_abbr] = {lv: row[f"Level_{lv}_abbr"] for lv in range(1, 7) if pd.notna(row[f"Level_{lv}_abbr"])}
    manual_df = pd.read_excel(ROOT / "neuron_tables/251637_INS_HE_inferred.xlsx", "Summary")
    manual = dict(zip(manual_df.NeuronID.astype(str), manual_df.Soma_Region))
    report = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
        "checker_sha256": audit.digest(Path(audit.__file__)), "readback_script_sha256": audit.digest(Path(__file__)),
        "original_checker_findings": [{"sample": s["sample"], "failures": s["failures"]} for s in cached["samples"] if s["failures"]],
        "rounding_policy": "Independent parent-child edges, scalar Python round(distance, 3); existing production policy unchanged",
        "samples": [], "subtables": [], "acceptance": "Software contracts only; native registration and fine anatomy remain unaccepted"}
    path = OUT / "final_actual_region_readback.json"
    for previous, entries, tag in [(cached, inv["swc_sources"], "cached"),
            (recovered, recovery["results"], "recovered")]:
        for old in previous["samples"]:
            sources = [e for e in entries if e["SampleID"] == old["sample"]]
            pop = SimpleNamespace(result_workbook=ROOT / old["workbook"], load_failures=old["load_failures"])
            result = audit.audit_sample(pop, old["sample"], sources, atlas, mask, labels, hierarchy,
                                        manual if old["sample"] == "251637" else {})
            result["batch"] = tag
            report["samples"].append(result)
            save(path, report)
            print(json.dumps({"batch": tag, "sample": old["sample"], "checked": result["checked"], "failures": len(result["failures"])}), flush=True)
    rebuild = read(OUT / "runs/refined_subtables_20261004_072110/subtable_rebuild_provenance.json")
    master_path = ROOT / rebuild["master_workbook"]
    master = pd.read_excel(master_path, sheet_name=None)
    provenance = read(master_path.parents[1] / "reports/run_provenance.json")
    swc_dir = Path(provenance["swc_source_dir"])
    sources = [{"SampleID": "251637", "NeuronID": nid, "path": (swc_dir / nid).relative_to(ROOT).as_posix(),
                "sha256": audit.digest(swc_dir / nid)} for nid in master["Summary"].NeuronID.astype(str)]
    result = audit.audit_sample(SimpleNamespace(result_workbook=master_path, load_failures=[]),
                               "251637", sources, atlas, mask, labels, hierarchy, manual)
    result["batch"] = "full_historical_master"
    report["samples"].append(result)
    print(json.dumps({"master_checked": result["checked"], "failures": len(result["failures"])}), flush=True)
    for entry in rebuild["outputs"]:
        output = ROOT / entry["output"]
        sheets = pd.read_excel(output, sheet_name=None)
        ids = set(pd.read_excel(entry["source"], "Summary").NeuronID.astype(str))
        failures = []
        if set(sheets) != set(master):
            failures.append("sheet names differ from fresh master")
        if set(sheets["Summary"].NeuronID.astype(str)) != ids:
            failures.append("original template membership differs")
        for name, df in master.items():
            expected = df[df.NeuronID.astype(str).isin(ids)] if "NeuronID" in df else df
            try:
                pd.testing.assert_frame_equal(sheets[name].reset_index(drop=True), expected.reset_index(drop=True),
                                              check_dtype=False, rtol=1e-12, atol=1e-10)
            except (AssertionError, KeyError) as exc:
                failures.append(f"{name}: {str(exc)[:500]}")
        if audit.digest(Path(entry["source"])) != entry["source_sha256"]:
            failures.append("source workbook changed")
        if audit.digest(output) != entry["output_sha256"]:
            failures.append("derived workbook changed since producer readback")
        report["subtables"].append({"output": output.relative_to(ROOT).as_posix(), "sha256": audit.digest(output),
                                   "neurons": len(ids), "sheets": len(sheets), "failures": failures})
    all_sources = inv["swc_sources"] + recovery["results"] + sources
    report["changed_sources"] = [e for e in all_sources if audit.digest(ROOT / e["path"]) != e["sha256"]]
    report["changed_protected_inputs"] = [e for e in inv["protected_inputs"] if audit.digest(ROOT / e["path"]) != e["sha256"]]
    report["unique_neurons"] = len({(r["SampleID"], r["NeuronID"]) for s in report["samples"] for r in s["records"]})
    report["checked_export_rows"] = sum(s["checked"] for s in report["samples"])
    failures = sum(len(s["failures"]) + len(s["load_failures"]) for s in report["samples"]) + sum(len(s["failures"]) for s in report["subtables"])
    report["failure_count"] = failures
    report["status"] = "passed" if not failures and not report["changed_sources"] and not report["changed_protected_inputs"] else "failed_readback"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save(path, report)
    print(json.dumps({k: report[k] for k in ("status", "unique_neurons", "checked_export_rows", "failure_count")}), flush=True)
    return 0 if report["status"] == "passed" else 1

if __name__ == "__main__":
    raise SystemExit(main())
