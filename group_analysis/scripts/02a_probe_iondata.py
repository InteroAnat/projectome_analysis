"""
Phase 2 prep: Probe IONData server for the 4 new monkey sample IDs.

Lists how many neurons each sample contains and writes a manifest so we know
what we're about to process before launching the heavy Step 1 pipeline.

Output:
  group_analysis/step1_results/iondata_probe.csv
"""
from __future__ import annotations

import os
import sys
import pandas as pd

PROJECT_ROOT = r"D:\projectome_analysis"
NEUROVIS = os.path.join(PROJECT_ROOT, "neuron-vis", "neuronVis")
if NEUROVIS not in sys.path:
    sys.path.insert(0, NEUROVIS)

import IONData  # type: ignore

NEW_SAMPLES = ["251730", "252383", "252384", "252385"]

OUT_DIR = os.path.join(PROJECT_ROOT, "group_analysis", "step1_results")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> int:
    iondata = IONData.IONData()
    rows = []
    for sid in NEW_SAMPLES:
        try:
            lst = iondata.getNeuronListBySampleID(sid)
            n = len(lst) if lst else 0
            sample_regions = []
            if lst:
                for entry in lst[:10]:
                    sample_regions.append(str(entry.get("region", "")).strip())
            rows.append(dict(
                sample_id=sid,
                n_neurons=n,
                first10_regions=" | ".join(sample_regions),
                ok=n > 0,
            ))
            print(f"[{sid}] n={n}  first10_regions: {sample_regions[:5]}")
        except Exception as e:
            rows.append(dict(sample_id=sid, n_neurons=0,
                             first10_regions=f"ERROR: {e}", ok=False))
            print(f"[{sid}] ERROR: {e}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "iondata_probe.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print(df.to_string(index=False))
    return 0 if df["ok"].any() else 1


if __name__ == "__main__":
    sys.exit(main())
