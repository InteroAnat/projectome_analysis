"""Bounded own-identity atlas-SWC retrieval into a separate review input folder.

Uses the existing IONData atlas-SWC endpoint. Each sample first requires a
known cached anchor to match the endpoint's current complete numeric graph.
No native raw-coordinate file or different identity is substituted.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(ROOT / "neuron-vis/neuronVis"))
import IONData
from swc_validation import parse_swc


def main():
    inventory = json.loads((OUT / "actual_rerun_inventory.json").read_text(encoding="utf-8"))
    endpoint = f"http://{IONData.IONData().ip}/neuronbrowser/api/user/getSWC"
    report = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
              "endpoint": endpoint, "anchors": [], "results": [], "source_cache_modified": False}
    def save():
        (OUT / "missing_atlas_source_recovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    allowed = set()
    for sample in sorted({entry["SampleID"] for entry in inventory["missing"]}):
        anchor = next(entry for entry in inventory["swc_sources"] if entry["SampleID"] == sample)
        observation = {"SampleID": sample, "NeuronID": anchor["NeuronID"], "cached_sha256": anchor["sha256"]}
        try:
            response = requests.get(endpoint, params={"sampleid": sample, "neuronid": anchor["NeuronID"]}, timeout=(5, 25))
            response.raise_for_status()
            server = np.array(parse_swc(response.text, source=response.url), dtype=float)
            cached = np.loadtxt(ROOT / anchor["path"], comments="#", ndmin=2)
            equal = server.shape == cached.shape and np.array_equal(server, cached)
            observation.update(status="matched_complete_numeric_graph" if equal else "endpoint_graph_changed",
                               response_sha256=hashlib.sha256(response.content).hexdigest(), nodes=len(server))
            if equal:
                allowed.add(sample)
        except Exception as exc:
            observation.update(status="anchor_unavailable", error=str(exc))
        report["anchors"].append(observation)
        save()
    def fetch(entry):
        sample, nid = entry["SampleID"], entry["NeuronID"]
        record = {"SampleID": sample, "NeuronID": nid}
        if sample not in allowed:
            return {**record, "status": "not_requested_anchor_not_validated"}
        try:
            response = requests.get(endpoint, params={"sampleid": sample, "neuronid": nid}, timeout=(5, 25))
            record.update(url=response.url, http_status=response.status_code, response_bytes=len(response.content),
                          response_sha256=hashlib.sha256(response.content).hexdigest())
            response.raise_for_status()
            nodes = parse_swc(response.text, source=response.url)
            target = OUT / "recovered_atlas_swcs" / sample / nid
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = response.text.encode("utf-8")
            if target.exists() and target.read_bytes() != payload:
                raise RuntimeError("Refusing to replace a different recovered source")
            target.write_bytes(payload)
            record.update(status="own_atlas_swc_recovered", path=target.relative_to(ROOT).as_posix(),
                          sha256=hashlib.sha256(payload).hexdigest(), nodes=len(nodes))
        except Exception as exc:
            record.update(status="own_atlas_swc_unavailable", error=str(exc))
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, entry) for entry in inventory["missing"]]
        for job in as_completed(jobs):
            result = job.result()
            report["results"].append(result)
            save()
            print(json.dumps({"SampleID": result["SampleID"], "NeuronID": result["NeuronID"], "status": result["status"]}), flush=True)
    report["status"] = "bounded_own_source_requests_finished"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["recovered"] = sum(entry["status"] == "own_atlas_swc_recovered" for entry in report["results"])
    report["unavailable"] = len(report["results"]) - report["recovered"]
    save()
    print(json.dumps({"recovered": report["recovered"], "unavailable": report["unavailable"]}), flush=True)


if __name__ == "__main__":
    main()
