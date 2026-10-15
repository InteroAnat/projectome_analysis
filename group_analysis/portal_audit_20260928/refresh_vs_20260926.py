"""Compare live ION selectNeurons counts to the 2026-09-26 portal audit.

Writes only under group_analysis/portal_audit_20260928/.
Bounded requests; does not use IONData retry loops.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(r"D:\projectome_analysis")
PREV = PROJECT / "group_analysis" / "portal_audit_20260926"
OUT = PROJECT / "group_analysis" / "portal_audit_20260928"
SCRIPTS = PROJECT / "group_analysis" / "scripts"
BASE = "http://10.10.48.110/neuronbrowser/api/user"
SLEEP_S = 0.2
TIMEOUT_S = 30

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PREV))
from audit_portal import (  # noqa: E402
    is_macaque,
    portal_region_base,
)
from insula_label_set import build_insula_label_set  # noqa: E402

INSULA_LABELS, _ = build_insula_label_set()
OUT.mkdir(parents=True, exist_ok=True)


def fetch_json(url: str):
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=TIMEOUT_S)
            last = r.status_code
            if r.status_code == 200:
                return r.status_code, r.json(), ""
            err = f"http_{r.status_code}"
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {e}"
            last = None
        if attempt < 2:
            time.sleep(SLEEP_S * (attempt + 1))
    return last, None, err


def main() -> int:
    prev = pd.read_csv(PREV / "portal_vs_current_diff.csv", dtype={"sample": str})
    prev_samples = pd.read_csv(PREV / "portal_macaque_samples.csv", dtype={"fMOST_id": str})
    prev_n = dict(zip(prev["sample"], prev["n_portal"]))
    prev_ins = dict(zip(prev["sample"], prev["n_insula_atlas"]))
    prev_prco = dict(zip(prev["sample"], prev["n_prco"]))

    status, info, err = fetch_json(f"{BASE}/getSampleInfo?project_id=")
    if status != 200 or not isinstance(info, list):
        print(f"sampleInfo failed: {status} {err}")
        return 1
    print(f"sampleInfo {status} n={len(info)}")
    live_mac = {
        str(r.get("fMOST_id")): r
        for r in info
        if r.get("fMOST_id") and is_macaque(r)
    }
    prev_ids = set(prev_samples["fMOST_id"].astype(str))
    new_samples = sorted(set(live_mac) - prev_ids)
    print(f"new macaque samples vs 2026-09-26: {new_samples or 'none'}")

    audit_ids = sorted(set(prev["sample"].astype(str)) | set(new_samples))
    rows = []
    id_rows = []
    for i, sid in enumerate(audit_ids, 1):
        st, data, e = fetch_json(f"{BASE}/selectNeurons?id={sid}")
        time.sleep(SLEEP_S)
        neurons = data if isinstance(data, list) else []
        ids = []
        n_ins = n_prco = 0
        regions = Counter()
        for n in neurons:
            name = str(n.get("name") or "")
            if not name:
                continue
            base = portal_region_base(str(n.get("region") or ""))
            regions[base or "<empty>"] += 1
            if base in INSULA_LABELS:
                n_ins += 1
            if base == "PRCO":
                n_prco += 1
            ids.append((name, str(n.get("region") or ""), base))
        n_now = len(ids)
        n_then = prev_n.get(sid)
        delta = None if n_then is None or st != 200 else int(n_now) - int(n_then)
        if st != 200:
            tag = f"fetch_fail:{st}:{e}"
        elif sid in new_samples:
            tag = "new_sample"
        elif delta == 0 and n_ins == int(prev_ins.get(sid, -1)) and n_prco == int(prev_prco.get(sid, -1)):
            tag = "unchanged"
        elif delta and delta > 0:
            tag = "grown"
        elif delta and delta < 0:
            tag = "shrunk"
        else:
            tag = "region_relabel"
        top = " | ".join(f"{k}:{v}" for k, v in regions.most_common(8))
        rows.append(dict(
            sample=sid,
            http_status=st,
            n_portal_20260926=n_then if n_then is not None else "",
            n_portal_now=n_now if st == 200 else "",
            delta_n=delta if delta is not None else "",
            n_insula_20260926=prev_ins.get(sid, ""),
            n_insula_now=n_ins if st == 200 else "",
            n_prco_20260926=prev_prco.get(sid, ""),
            n_prco_now=n_prco if st == 200 else "",
            status=tag,
            top_regions_now=top,
            is_new_sample=int(sid in new_samples),
        ))
        if st == 200 and tag != "unchanged":
            for name, region, base in ids:
                id_rows.append(dict(
                    sample=sid, neuron_id=name, soma_region=region,
                    soma_region_base=base, status=tag,
                ))
        if i % 10 == 0 or tag != "unchanged":
            print(f"[{i}/{len(audit_ids)}] {sid} {tag} n={n_now} delta={delta} ins={n_ins} prco={n_prco}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "portal_refresh_vs_20260926.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(id_rows).to_csv(
        OUT / "changed_sample_neuron_ids.csv", index=False, encoding="utf-8-sig"
    )
    changed = out[out["status"] != "unchanged"]
    print("--- changed ---")
    if changed.empty:
        print("none")
    else:
        cols = [
            "sample", "status", "n_portal_20260926", "n_portal_now", "delta_n",
            "n_insula_20260926", "n_insula_now", "n_prco_20260926", "n_prco_now",
        ]
        print(changed[cols].to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
