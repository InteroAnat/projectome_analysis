"""Collect bounded own-source and annotation evidence for 251637/112 and /114."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "case_112_114_20261004"
sys.path.insert(0, str(ROOT / "main_scripts"))
from swc_validation import parse_swc

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

def encode(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    return str(value)

def main():
    OUT.mkdir(exist_ok=True)
    report = {"started_at": datetime.now().astimezone().isoformat(), "sample": "251637",
              "workbooks": [], "requests": [], "graphs": [], "inputs_overwritten": False}
    def save():
        (OUT / "source_readback.json").write_text(json.dumps(report, indent=2, default=encode), encoding="utf-8")
    targets = {"112.swc", "114.swc"}
    files = sorted((ROOT / "neuron_tables").glob("251637*.xlsx")) + sorted((ROOT / "neuron_tables_new").glob("251637*.xlsx"))
    files += [ROOT / "R_analysis/tables/somainfo_Henry_2026.04.03.xlsx"]
    for path in files:
        if not path.exists(): continue
        workbook = pd.ExcelFile(path)
        record = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path.read_bytes()), "sheets": []}
        names = workbook.sheet_names if "Henry" in path.name else ["Summary"] if "Summary" in workbook.sheet_names else []
        for name in names:
            df = pd.read_excel(workbook, name)
            matches = pd.Series(False, index=df.index)
            for col in df.columns:
                if str(col).lower() in ("neuronid", "neuron_id", "id", "name", "neuron", "swc", "no", "index"):
                    text = df[col].astype(str).str.strip()
                    matches |= text.isin(targets | {"112", "114", "112.0", "114.0"})
            selected = df.loc[matches].replace({np.nan: None})
            record["sheets"].append({"name": name, "columns": [str(c) for c in df.columns], "matched_rows": selected.to_dict(orient="records")})
        workbook.close()
        report["workbooks"].append(record)
    save()
    def get(url, label, params=None):
        rec = {"label": label, "url": url, "params": params, "observed_at": datetime.now().astimezone().isoformat()}
        try:
            response = requests.get(url, params=params, timeout=(5,25))
            rec.update(http_status=response.status_code, bytes=len(response.content), sha256=digest(response.content))
            response.raise_for_status()
            path = OUT / (label + (".swc" if label.startswith(("raw_", "atlas_")) else ".json"))
            if path.exists() and path.read_bytes() != response.content:
                raise RuntimeError("Refusing to replace differing evidence")
            path.write_bytes(response.content)
            rec["path"] = path.relative_to(ROOT).as_posix()
            report["requests"].append(rec); save()
            return response
        except Exception as exc:
            rec["error"] = str(exc)
            report["requests"].append(rec); save()
            return None
    base = "http://10.10.48.110/neuronbrowser/api/user/"
    info = get(base + "getSampleInfo", "sample_info", {"project_id": "", "fMOST_id": "251637"})
    exact = [] if info is None else [r for r in info.json() if str(r.get("fMOST_id")) == "251637"]
    # Portal deployments use 'fMOSTID' or 'fMOST_id'; never assume another sample.
    if not exact and info is not None:
        exact = [r for r in info.json() if str(r.get("fMOSTID", r.get("sampleid", r.get("id")))) == "251637"]
    report["exact_sample_info"] = exact
    get(base + "getSoma", "portal_somata", {"fMOST_id": "251637"})
    project = str(exact[0]["project_id"]) if len(exact) == 1 else None
    for nid in sorted(targets):
        stem = nid[:-4]
        current = get(base + "getSWC", "atlas_" + stem, {"sampleid": "251637", "neuronid": nid})
        raw = get(f"http://10.10.31.31/swc/newswc/{project}/251637/swc_raw/{nid}", "raw_" + stem) if project else None
        for label, response in (("atlas", current), ("raw", raw)):
            if response is None: continue
            nodes = np.array(parse_swc(response.text, source=response.url), dtype=float)
            root = nodes[nodes[:,6] == -1][0]
            rec = {"NeuronID": nid, "frame": label, "nodes": len(nodes), "root_xyz": root[2:5].tolist(), "source_sha256": digest(response.content)}
            if label == "atlas":
                cached = ROOT / "neuron-vis/resource/swc/251637" / nid
                old = np.loadtxt(cached, comments="#", ndmin=2)
                rec["cached_numeric_graph_equal"] = np.array_equal(nodes, old)
                rec["cached_sha256"] = digest(cached.read_bytes())
            report["graphs"].append(rec)
        save()
    report["status"] = "bounded_source_checks_finished"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    print(json.dumps({"workbooks": len(report["workbooks"]), "exact_sample_records": len(exact), "graphs": report["graphs"],
                      "request_errors": [r for r in report["requests"] if "error" in r]}, default=encode), flush=True)

if __name__ == "__main__":
    main()
