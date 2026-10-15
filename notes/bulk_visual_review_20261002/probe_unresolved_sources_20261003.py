"""Bounded raw-prefix probes; never place truncated files in the SWC cache."""
import csv
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
queue = ROOT / "group_analysis/visual_review_20261002/manifest/metadata_resolution_queue.csv"
with queue.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
targets = {"251730|226.swc", "251730|743.swc", "252383|136.swc", "252383|140.swc"}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--all", action="store_true", help="Probe every unresolved-metadata identity")
args = parser.parse_args()
records = []
for row in rows:
    if not args.all and row["UID"] not in targets:
        continue
    url = f"http://10.10.31.31/swc/newswc/Monkey/{row['SampleID']}/swc_raw/{row['NeuronID']}"
    result = {"uid": row["UID"], "url": url, "requested_range": "bytes=0-4095",
              "timeout_seconds": 8, "maximum_prefix_bytes": 4096}
    try:
        with urlopen(Request(url, headers={"Range": "bytes=0-4095"}), timeout=8) as response:
            data = response.read(4096)
            result.update(http_status=response.status, bytes_read=len(data),
                          content_type=response.headers.get("Content-Type"),
                          content_range=response.headers.get("Content-Range"),
                          prefix_sha256=hashlib.sha256(data).hexdigest())
        root = None
        for line in data.decode("utf-8-sig", errors="replace").splitlines():
            fields = line.split("#", 1)[0].split()
            if len(fields) < 7:
                continue
            try:
                values = [float(value) for value in fields[:7]]
                if values[6] == -1 and all(math.isfinite(value) for value in values) and min(values[2:5]) >= 0:
                    root = values[2:5]
                    break
            except ValueError:
                continue
        result.update(candidate_root_xyz=root, prefix_contains_root=root is not None,
                      full_graph_validated=False, coordinate_frame_calibrated=False)
    except HTTPError as error:
        result.update(http_status=error.code, bytes_read=0, error=str(error))
    except (URLError, OSError) as error:
        result.update(http_status=None, bytes_read=0, error=str(error))
    records.append(result)
summary = {"captured_utc": datetime.now(timezone.utc).isoformat(), "metadata_queue_size": len(rows),
           "probed_identities": len(records), "prefix_bytes_read": sum(row["bytes_read"] for row in records),
           "records": records, "manifest_modified": False, "image_requests": 0,
           "scope": "Bounded raw prefixes only; no full SWC, graph, anatomical, or cohort acceptance"}
filename = "unresolved_source_probe_all_20261003.json" if args.all else "unresolved_source_probe_20261003.json"
(HERE / filename).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary))
