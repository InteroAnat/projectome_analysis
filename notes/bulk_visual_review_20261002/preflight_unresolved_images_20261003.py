"""Bounded own-sample TIFF prefix probes and 4 mm plan admission, no image cache."""
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))
from fmost_derived_context import context_plan

source = HERE / "unresolved_full_source_validation_20261003.json"
validation = json.loads(source.read_text(encoding="utf-8"))
records = []
output = HERE / "unresolved_image_preflight_20261003.json"
def save(state):
    data = {"observed_utc": datetime.now(timezone.utc).isoformat(), "state": state,
            "raw_validation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "selected_identities": len(validation["records"]), "records": records,
            "prefix_limit_bytes": 4096, "timeout_seconds": 8, "image_cache_written": False,
            "full_tiff_decoded": False, "coordinate_frame_calibrated": False, "anatomical_acceptance": False}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
save("running")
for record in validation["records"]:
    if not record.get("graph_validated"):
        continue
    sample = record["uid"].split("|", 1)[0]
    root = record["raw_root_xyz"]
    cubes, plan = context_plan(root, max_cubes=800, max_source_bytes=26 * 1024**3)
    # Independent nominal native-to-cube conversion, not atlas coordinates.
    x, y, z = [math.floor(root[i] / spacing) // block
               for i, (spacing, block) in enumerate(zip((.65, .65, 3), (360, 360, 90)))]
    if [x, y, z] != plan["central_cube_xyz"]:
        raise ValueError("Independent raw-root source index differs from planned central cube")
    url = f"http://bap.cebsit.ac.cn/monkeydata/{sample}/cube/{z}/{x}_{y}_{z}.tif"
    item = {"uid": record["uid"], "raw_sha256": record["sha256"],
            "central_cube_xyz": [x, y, z], "url": url, "field_4mm_admitted": cubes is not None,
            "cube_count": plan["cube_count"], "working_bytes_estimate": plan["working_bytes_estimate"]}
    try:
        with urlopen(Request(url, headers={"Range": "bytes=0-4095"}), timeout=8) as response:
            payload = response.read(4096)
            item.update(http_status=response.status, prefix_bytes=len(payload),
                        content_range=response.headers.get("Content-Range"),
                        prefix_sha256=hashlib.sha256(payload).hexdigest(),
                        tiff_magic=payload[:4] in (b"II\x2a\x00", b"MM\x00\x2a", b"II\x2b\x00", b"MM\x00\x2b"))
    except HTTPError as error:
        item.update(http_status=error.code, prefix_bytes=0, error=str(error))
    except (URLError, OSError) as error:
        item.update(http_status=None, prefix_bytes=0, error=str(error))
    records.append(item)
    save("running")
    print(json.dumps({"uid": item["uid"], "http_status": item["http_status"], "completed": len(records)}), flush=True)
save("terminal_complete")
print(json.dumps({"state": "terminal_complete", "probed": len(records),
                  "admitted_4mm": sum(r["field_4mm_admitted"] for r in records),
                  "tiff_prefixes": sum(r.get("tiff_magic", False) for r in records)}), flush=True)
