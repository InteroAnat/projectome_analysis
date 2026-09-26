"""Compare new step1 Summary vs previous run for the same sample_id."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

PROJECT = Path(r"D:\projectome_analysis")
STEP1 = PROJECT / "group_analysis" / "step1_results"

ANIMAL_TO_FMOST = {
    "605": "252383",
    "331": "252385",
    "945": "252527",
    "797": "252790",
    "900": "252718",
    "631": "252714",
}


def latest_results_xlsx(sample_id: str, exclude_dir: str | None = None) -> Path | None:
    pattern = str(STEP1 / f"{sample_id}_*_region_analysis" / "tables" / f"{sample_id}_results*.xlsx")
    paths = [Path(p) for p in glob.glob(pattern)]
    if exclude_dir:
        paths = [p for p in paths if exclude_dir not in str(p)]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def load_summary(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Summary")


def compare_pair(old_path: Path, new_path: Path, animal: str, sid: str) -> dict:
    old = load_summary(old_path)
    new = load_summary(new_path)
    old_ids = set(old["NeuronID"].astype(str))
    new_ids = set(new["NeuronID"].astype(str))
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    common = old_ids & new_ids
    region_changes = []
    if common and "Soma_Region" in old.columns and "Soma_Region" in new.columns:
        o = old.set_index(old["NeuronID"].astype(str))
        n = new.set_index(new["NeuronID"].astype(str))
        for nid in sorted(common):
            ro, rn = str(o.at[nid, "Soma_Region"]), str(n.at[nid, "Soma_Region"])
            if ro != rn:
                region_changes.append((nid, ro, rn))
    return {
        "animal": animal,
        "sample_id": sid,
        "old_xlsx": str(old_path),
        "new_xlsx": str(new_path),
        "n_old": len(old),
        "n_new": len(new),
        "n_added": len(added),
        "n_removed": len(removed),
        "n_soma_region_changed": len(region_changes),
        "added_ids_head": added[:10],
        "removed_ids_head": removed[:10],
        "region_changes_head": region_changes[:15],
    }


def main() -> int:
    rows = []
    for animal, sid in ANIMAL_TO_FMOST.items():
        all_xlsx = sorted(
            glob.glob(str(STEP1 / f"{sid}_*_region_analysis/tables/{sid}_results*.xlsx"))
        )
        if len(all_xlsx) < 2:
            newest = latest_results_xlsx(sid)
            rows.append(
                {
                    "animal": animal,
                    "sample_id": sid,
                    "status": "no_prior_run" if newest else "no_runs",
                    "n_new": len(load_summary(Path(newest))) if newest else 0,
                    "new_xlsx": newest,
                }
            )
            continue
        paths = [Path(p) for p in all_xlsx]
        paths.sort(key=lambda p: p.stat().st_mtime)
        old_path, new_path = paths[-2], paths[-1]
        rec = compare_pair(old_path, new_path, animal, sid)
        rec["status"] = "compared"
        rows.append(rec)

    out_csv = STEP1 / "step1_batch2_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}\n")
    for r in rows:
        print(f"--- {r.get('animal')} / {r.get('sample_id')} [{r.get('status')}] ---")
        for k, v in r.items():
            if k in ("animal", "sample_id", "status"):
                continue
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
