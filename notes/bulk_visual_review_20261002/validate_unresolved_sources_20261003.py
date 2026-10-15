"""Validate full unresolved raw SWCs independently of the live image manifest."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))
from swc_validation import parse_swc

probe_path = HERE / "unresolved_source_probe_all_20261003.json"
probe = json.loads(probe_path.read_text(encoding="utf-8"))
destination = HERE / "unresolved_validated_raw"
destination.mkdir(exist_ok=True)
output = HERE / "unresolved_full_source_validation_20261003.json"
manifest = ROOT / "group_analysis/visual_review_20261002/manifest/identity_manifest.csv"
# Match the actual filename without modifying either candidate population.
if not manifest.exists():
    candidates = list(manifest.parent.glob("*manifest*.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot identify live manifest: {candidates}")
    manifest = candidates[0]
manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
max_file = 10 * 1024**2
max_total = 256 * 1024**2
transferred = 0
records = []

def save(state):
    data = {
        "observed_utc": datetime.now(timezone.utc).isoformat(), "state": state,
        "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256((ROOT / "main_scripts/swc_validation.py").read_bytes()).hexdigest(),
        "selected_identities": len(probe["records"]), "records": records,
        "downloaded_bytes_this_run": transferred, "max_file_bytes": max_file,
        "max_total_bytes": max_total, "request_timeout_seconds": 20,
        "manifest_sha256_before": manifest_sha,
        "manifest_unchanged": hashlib.sha256(manifest.read_bytes()).hexdigest() == manifest_sha,
        "image_requests": 0, "anatomical_acceptance": False,
        "scope": "Structural raw-source validation only; portal metadata remains unresolved; no cohort promotion",
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)

save("running")
for source in probe["records"]:
    sample, neuron = source["uid"].split("|", 1)
    target = destination / sample / neuron
    result = {"uid": source["uid"], "url": source["url"], "metadata_status": "unresolved"}
    try:
        expected = int(source["content_range"].rsplit("/", 1)[1])
        if expected > max_file:
            raise ValueError("Source exceeds per-file admission bound")
        if target.exists():
            payload = target.read_bytes()
            result["source"] = "existing_separate_validated_cache"
        else:
            if transferred + expected > max_total:
                raise ValueError("Source exceeds total transfer admission bound")
            with urlopen(source["url"], timeout=20) as response:
                if response.status != 200:
                    raise ValueError(f"Full download status {response.status}")
                payload = response.read(min(max_file, max_total - transferred) + 1)
            transferred += len(payload)
            result["source"] = "full_raw_download"
        if len(payload) != expected:
            raise ValueError(f"Full length {len(payload)} differs from probed {expected}")
        if hashlib.sha256(payload[:source["bytes_read"]]).hexdigest() != source["prefix_sha256"]:
            raise ValueError("Full source prefix differs from recorded probe")
        rows = parse_swc(payload.decode("utf-8-sig"), source=source["url"])
        root = next(row[2:5] for row in rows if row[6] == -1)
        if list(root) != source["candidate_root_xyz"]:
            raise ValueError("Validated root differs from prefix root")
        if min(root) < 0:
            raise ValueError("Root is outside nonnegative native index domain")
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        result.update(status="structurally_valid", bytes=len(payload), nodes=len(rows),
                      sha256=hashlib.sha256(payload).hexdigest(), raw_root_xyz=list(root),
                      local_path=str(target.relative_to(ROOT)), graph_validated=True)
    except Exception as error:
        result.update(status="validation_failed", error=str(error), graph_validated=False)
    records.append(result)
    save("running")
    print(json.dumps({"uid": result["uid"], "status": result["status"], "completed": len(records)}), flush=True)
save("terminal_complete")
print(json.dumps({"state": "terminal_complete", "validated": sum(r["graph_validated"] for r in records),
                  "failed": sum(not r["graph_validated"] for r in records), "downloaded_bytes": transferred}), flush=True)
