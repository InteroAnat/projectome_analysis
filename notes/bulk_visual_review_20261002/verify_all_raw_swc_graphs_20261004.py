"""Read back all native raw SWC graphs against full-soma source lineage.

Local-only structural validation uses the project's strict, tested parser.
It does not establish tracing completeness, anatomical identity or registration.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from verify_all_soma_pixels_20261003 import HERE, OUT, ROOT, save, sha

sys.path.insert(0, str(ROOT / "main_scripts"))
from swc_validation import parse_swc


def main():
    manifest = OUT / "manifest/identity_manifest.csv"
    manifest_hash = sha(manifest)
    soma_path = HERE / "all_soma_source_pixel_readback_20261003.json"
    soma_hash = sha(soma_path)
    soma = json.loads(soma_path.read_text())
    assert soma["status"] == "passed" and soma["manifest_sha256"] == manifest_hash
    baseline = {record["uid"]: record for record in soma["records"]}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(baseline) == len({row["UID"] for row in rows}) == 832
    assert set(baseline) == {row["UID"] for row in rows}
    parser_path = ROOT / "main_scripts/swc_validation.py"
    parser_hash = sha(parser_path)
    output = HERE / "all_raw_swc_graph_readback_20261004.json"
    result = {"status": "running", "manifest_sha256": manifest_hash,
              "full_soma_audit_sha256": soma_hash, "script_sha256": sha(Path(__file__)),
              "strict_parser_sha256": parser_hash, "selected_identities": len(rows),
              "server_reads": 0, "network_requests": 0, "anatomical_acceptance": False,
              "tracing_completeness": False, "records": [], "failures": []}
    save(output, result)
    for row in rows:
        uid, sample, neuron = row["UID"], row["SampleID"], row["NeuronID"]
        assert uid == f"{sample}|{neuron}"
        raw = OUT / "swc_raw" / sample / neuron
        prov_path = OUT / sample / Path(neuron).stem / "provenance.json"
        try:
            prov = json.loads(prov_path.read_text())
            assert prov["UID"] == uid
            own_url = f"http://10.10.31.31/swc/newswc/Monkey/{sample}/swc_raw/{neuron}"
            source = baseline[uid]
            if source["outcome"] == "swc_unavailable":
                assert not raw.exists() and prov["coverage_status"] == "swc_unavailable"
                assert prov.get("nmt_fallback") is False
                assert any(own_url in error and "after 3 requests" in error for error in prov["errors"])
                record = {"uid": uid, "outcome": "own_swc_unavailable", "node_count": 0,
                          "own_raw_url": own_url, "source_errors": prov["errors"]}
            else:
                digest = sha(raw)
                baseline_raw_hash = source["raw_sha256"] if source["outcome"] == "center_unavailable" else source["artifact_hashes"][str(raw)]
                assert digest == prov["swc_sha256"] == baseline_raw_hash
                assert prov["swc_url"] == own_url
                parsed = parse_swc(raw.read_text(encoding="utf-8"), source=str(raw))
                assert len(parsed) == prov["swc_node_count"]
                root = next(node for node in parsed if node[6] == -1)
                np.testing.assert_allclose(root[2:5], [prov[f"native_soma_{axis}_um"] for axis in "xyz"], atol=1e-9)
                assert np.isfinite(root[2:5]).all() and (np.array(root[2:5]) >= 0).all()
                assert sha(raw) == digest, "Raw SWC changed during graph validation"
                fresh = json.loads(prov_path.read_text())
                assert fresh["UID"] == uid and fresh["swc_sha256"] == digest and fresh["swc_url"] == own_url
                assert fresh["swc_node_count"] == len(parsed)
                np.testing.assert_allclose(root[2:5], [fresh[f"native_soma_{axis}_um"] for axis in "xyz"], atol=1e-9)
                record = {"uid": uid, "outcome": "graph_valid", "own_raw_url": own_url,
                          "raw_sha256": digest, "raw_bytes": raw.stat().st_size,
                          "node_count": len(parsed), "edge_count": len(parsed) - 1,
                          "root_id": root[0], "root_xyz_um": list(root[2:5]),
                          "node_type_counts": dict(Counter(str(node[1]) for node in parsed)),
                          "stable_raw_during_read": True, "soma_source_lineage_matches": True}
                del parsed
            result["records"].append(record)
            print(json.dumps({"uid": uid, "outcome": record["outcome"], "nodes": record["node_count"]}), flush=True)
        except Exception as exc:
            result["failures"].append({"uid": uid, "type": type(exc).__name__, "error": str(exc)})
            print(json.dumps(result["failures"][-1]), flush=True)
        result["observed_utc"] = datetime.now(timezone.utc).isoformat()
        save(output, result)
    assert sha(manifest) == manifest_hash and sha(soma_path) == soma_hash and sha(parser_path) == parser_hash
    result["checked_identities"] = len(result["records"])
    result["outcome_counts"] = dict(Counter(record["outcome"] for record in result["records"]))
    result["total_nodes"] = sum(record["node_count"] for record in result["records"])
    result["status"] = "passed" if not result["failures"] and len(result["records"]) == len(rows) else "failed"
    result["observed_utc"] = datetime.now(timezone.utc).isoformat()
    save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"records", "failures"}}), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
