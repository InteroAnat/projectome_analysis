"""Reconcile prior labels, boundary coordinates and twelve historical corrections."""
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import rerun_actual_region_tables as audit

ROOT, OUT = audit.ROOT, audit.OUT

def main():
    report_path = OUT / "final_actual_region_readback.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "passed":
        raise RuntimeError("Final saved-export readback must pass first")
    by_id = {(r["SampleID"], r["NeuronID"]): r for s in report["samples"] for r in s["records"]}
    inventory = json.loads((OUT / "actual_rerun_inventory.json").read_text(encoding="utf-8"))
    sources = {(e["SampleID"], e["NeuronID"]): e for e in inventory["swc_sources"]}
    baseline = pd.read_csv(OUT / "existing_table_region_baseline_rows.csv", dtype={"SampleID": str})
    boundaries, changes = [], []
    for _, old in baseline.iterrows():
        key = (old.SampleID, old.NeuronID)
        new = by_id[key]
        if old.rounding_boundary:
            swc = ROOT / sources[key]["path"]
            nodes = np.loadtxt(swc, comments="#", ndmin=2)
            xyz = (nodes[nodes[:, 6] == -1][0, 2:5] / 250).tolist()
            boundaries.append({"source_workbook": old.source, "SampleID": old.SampleID, "NeuronID": old.NeuronID,
                "full_precision_source_root_voxels": xyz, "rounded_full_precision_voxel": np.rint(xyz).astype(int).tolist(),
                "persisted_baseline_voxel": json.loads(old.voxel), "baseline_atlas_lookup": old.atlas_lookup,
                "fresh_atlas_lookup": new["atlas_soma"], "fresh_stored_soma": new["stored_soma"],
                "source_sha256": sources[key]["sha256"]})
        if old.direct_atlas_label and old.stored_region != new["atlas_soma"]:
            changes.append({"source": old.source, "SampleID": old.SampleID, "NeuronID": old.NeuronID,
                            "previous": old.stored_region, "recomputed_full_precision": new["atlas_soma"]})
    ids = ["112.swc", "114.swc", "422.swc", "423.swc", "439.swc", "440.swc", "444.swc",
           "458.swc", "459.swc", "460.swc", "461.swc", "469.swc"]
    historical = [{**by_id[("251637", nid)], "historical_correction": "April Unclassified -> ITi",
                   "interpretation": "Original annotation retained; current classification withheld for label-mask conflict"
                       if by_id[("251637", nid)]["conflict"] else "Fresh declared type-rule result"} for nid in ids]
    result = {"status": "reconciled_from_passed_outputs", "recorded_at": datetime.now().astimezone().isoformat(),
        "final_readback_sha256": audit.digest(report_path), "boundary_rows": boundaries,
        "direct_atlas_soma_changes": changes, "historical_manual_type_cases": historical,
        "unique_conflicts": [r for r in by_id.values() if r["conflict"]],
        "accepted_sources_replaced": False, "anatomical_acceptance": "not established"}
    (OUT / "final_boundary_and_manual_cases.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"boundary_rows": len(boundaries), "direct_atlas_soma_changes": len(changes),
                      "manual_cases": len(historical), "unique_conflicts": len(result["unique_conflicts"])}))

if __name__ == "__main__":
    main()
