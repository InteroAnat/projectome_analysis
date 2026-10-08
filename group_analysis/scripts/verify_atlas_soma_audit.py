"""Independent persisted readback; no audit helper or production tracer reuse."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
from insula_label_set import build_insula_label_set, normalize_label


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--readback-stem", default="independent_readback")
    args = parser.parse_args()
    out = args.audit_dir.resolve()
    out.relative_to(ROOT / "group_analysis/evolution_20261008/atlas_locations")
    if Path(args.readback_stem).name != args.readback_stem:
        raise ValueError("Readback stem must be a basename")
    target = out / (args.readback_stem + ".json")
    if target.exists():
        raise FileExistsError(target)
    provenance = json.loads((out / "provenance.json").read_text())
    live = rows(ROOT / provenance["inventory"]["path"])
    audit = rows(out / "neuron_soma_locations.csv")
    sources = rows(out / "source_inventory.csv")
    identity = lambda r: (r["sample"], r["neuron_id"])
    assert len(audit) == len({identity(r) for r in audit}) == 8746
    assert {identity(r) for r in audit} == {identity(r) for r in live}
    grouped = defaultdict(list)
    for source in sources:
        grouped[identity(source)].append(source)
    assert len(sources) == len({r["source_path"] for r in sources})
    assert set(grouped) <= {identity(r) for r in live}
    snapshot_cache = {}
    portal_mismatch, portal_missing = [], 0
    for r in live:
        evidence = json.loads(r["portal_soma_source"])
        path = evidence["file"]
        if path not in snapshot_cache:
            payload = (ROOT / path).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == evidence["sha256"]
            raw = json.loads(payload)
            assert isinstance(raw, list)
            snapshot_cache[path] = {x["name"]: x for x in raw}
            assert len(raw) == len(snapshot_cache[path])
        soma = snapshot_cache[path].get(r["neuron_id"])
        xyz = [soma.get("soma" + a) for a in "xyz"] if soma else [None] * 3
        valid = all(isinstance(x, (int, float)) and np.isfinite(x) for x in xyz)
        if not valid:
            portal_missing += 1
            assert r["coordinate_status"] != "present"
        else:
            encoded_um = [float(r["nmt_" + a + "_mm"]) * 1000 for a in "xyz"]
            if not np.allclose(xyz, encoded_um, atol=1e-8, rtol=0):
                portal_mismatch.append(identity(r))
    assert not portal_mismatch
    assert portal_missing == 337
    # Independent np.loadtxt full numeric graph and direct ARM6 indexing, rather
    # than the audit streaming parser. One source per sample/cache root, both
    # reviewed cases, plus a half-boundary and background case when available.
    selected, strata = [], set()
    for r in sources:
        key = (r["sample"], str(Path(r["source_path"]).parent.parent))
        if key not in strata or (r["sample"], r["neuron_id"]) in (("251637", "112.swc"), ("251637", "114.swc")):
            selected.append(r)
            strata.add(key)
    for field, value in (("half_voxel_boundary", "True"), ("atlas_label", "Unknown_0")):
        match = next((r for r in sources if r.get(field) == value), None)
        if match and match not in selected:
            selected.append(match)
    bundle = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym"
    arm = np.asanyarray(nib.load(bundle / "ARM_in_NMT_v2.1_sym.nii.gz").dataobj)[..., 0, 5]
    plane = np.asanyarray(nib.load(bundle / "supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz").dataobj)
    checks = []
    for r in selected:
        path = ROOT / r["source_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["sha256"]
        graph = np.loadtxt(path, comments="#", ndmin=2)
        roots = graph[graph[:, 6] == -1]
        assert len(roots) == 1
        xyz = roots[0, 2:5]
        assert np.array_equal(xyz, json.loads(r["root_xyz_um"]))
        point = tuple(np.round(xyz / 250).astype(int))
        assert list(point) == json.loads(r["rounded_voxel_xyz"])
        inside = all(0 <= p < arm.shape[d] for d, p in enumerate(point))
        if inside:
            assert int(arm[point]) == int(r["atlas_id"])
            assert {1: "R", 2: "L"}.get(int(plane[point]), "Unknown") == r["mask_side"]
        checks.append({"sample": r["sample"], "neuron_id": r["neuron_id"], "source_path": r["source_path"],
                       "sha256": r["sha256"], "root_and_lookup": "fresh_independent_match"})
    original = pd.read_excel(ROOT / provenance["inputs"]["original_manual"]["path"], "251637 mostly insula")
    original = original[original.SWC.notna() & original.Area.notna()]
    assert len(original) == 296 and len(set(original.SWC)) == 295
    assert sum(len(json.loads(r.get("original_manual_rows") or "[]")) for r in audit) == 296
    reviewed = [r for r in audit if r.get("reviewed_region")]
    assert {identity(r) for r in reviewed} == {("251637", "112.swc"), ("251637", "114.swc")}
    assert all(r["reviewed_region"] == "R-IDM" and r["reviewed_source_hash_verified"] == "True" for r in reviewed)
    assert "provisional" in next(r for r in reviewed if r["neuron_id"] == "114.swc")["reviewed_status"]
    insula, _ = build_insula_label_set(str(ROOT / "atlas/ARM_key_all.txt"))
    candidate_rows = []
    for r in audit:
        own_assessed = r["own_swc_status"].startswith("own_atlas_root")
        portal_assessed = bool(r.get("portal_coordinate_atlas_label"))
        candidate_rows.append({"sample": r["sample"], "neuron_id": r["neuron_id"],
                               "portal_supplied_region": r["portal_region"], "inherited_portal_INS_candidate": r["atlas_INS"],
                               "inherited_potential_INS_candidate": r["potential_INS_candidate"],
                               "own_direct_ARM6_label": r.get("own_atlas_label", ""),
                               "own_direct_INS_label_set_candidate": normalize_label(r["own_atlas_label"]) in insula if own_assessed else "",
                               "own_direct_G_candidate": normalize_label(r["own_atlas_label"]) == "G" if own_assessed else "",
                               "own_direct_unknown_background": r.get("own_direct_unknown_background", ""),
                               "own_direct_INS_hierarchy_only": r.get("own_direct_INS_hierarchy", ""),
                               "portal_coordinate_direct_ARM6_label": r.get("portal_coordinate_atlas_label", ""),
                               "portal_coordinate_direct_INS_label_set_candidate": normalize_label(r["portal_coordinate_atlas_label"]) in insula if portal_assessed else "",
                               "portal_coordinate_direct_INS_hierarchy_only": r.get("portal_coordinate_direct_INS_hierarchy", ""),
                               "scientific_membership": "candidate_only_not_accepted"})
    with (out / (args.readback_stem + "_candidates.csv")).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    result = {"status": "passed", "exact_identities": len(audit), "fresh_source_files": len(sources),
              "verifier_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "identities_with_at_least_one_checked_root": sum(any(r["source_status"] == "root_checked" for r in v) for v in grouped.values()),
              "identities_without_any_local_atlas_source": len(audit) - len(grouped),
              "per_neuron_status_counts": dict(Counter(r["own_swc_status"] for r in audit)),
              "source_hash_provenance_counts": dict(Counter(r["prior_hash_status"] for r in sources)),
              "source_cache_counts": dict(Counter(str(Path(r["source_path"]).parent.parent).replace("\\", "/") for r in sources)),
              "source_root_consensus_conflicts": [list(k) for k, v in grouped.items() if len({r.get("root_xyz_um") for r in v}) > 1],
              "portal_raw_snapshots_hash_verified": len(snapshot_cache), "portal_missing_coordinates": portal_missing,
              "portal_saved_inventory_coordinate_mismatches": portal_mismatch, "representative_independent_root_reads": checks,
              "manual_original_rows_preserved": len(original), "manual_original_distinct_neurons": len(set(original.SWC)),
              "reviewed_cases": [{k: r[k] for k in ("sample", "neuron_id", "own_atlas_label", "own_mask_side", "reviewed_region", "reviewed_status", "reviewed_source_hash_verified")} for r in reviewed],
              "descriptive_disagreement_counts": {k: sum(r.get(k) == "True" for r in audit) for k in
                    ("own_portal_coordinate_mismatch_gt_1um", "own_portal_direct_label_difference", "own_vs_portal_supplied_region_difference", "portal_coordinate_vs_portal_supplied_region_difference", "own_label_mask_conflict", "own_vs_portal_supplied_side_conflict", "reviewed_mask_side_conflict")},
              "own_lookup_label_counts": dict(Counter(r.get("own_atlas_label", "unassessed") or "unassessed" for r in audit)),
              "portal_coordinate_lookup_label_counts": dict(Counter(r.get("portal_coordinate_atlas_label", "unassessed") or "unassessed" for r in audit)),
              "portal_supplied_explicit_unknown": sum(r["portal_region_base"].startswith("UNKNOWN") for r in audit),
              "portal_supplied_region_missing": sum(not r["portal_region"] for r in audit),
              "direct_lookup_candidate_definition": "Shared insula_label_set: ARM Full_Name contains insula plus curated labels, excludes retroinsula and subcortical SL_Pi; distinct from INS-only hierarchy branch",
              "direct_lookup_candidate_counts": {k: sum(r[k] is True for r in candidate_rows) for k in ("own_direct_INS_label_set_candidate", "own_direct_G_candidate", "portal_coordinate_direct_INS_label_set_candidate")},
              "direct_lookup_INS_hierarchy_only": {k: sum(r.get(k) == "True" for r in audit) for k in ("own_direct_INS_hierarchy", "portal_coordinate_direct_INS_hierarchy")},
              "registration_direction": "user-described raw fMOST brain and neurons to NMT; original transform/version provenance pending",
              "scientific_acceptance": "pending native transform/version and landmark evidence; descriptive differences do not adjudicate server label assignment"}
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "exact_identities", "fresh_source_files", "identities_with_at_least_one_checked_root", "identities_without_any_local_atlas_source", "per_neuron_status_counts", "descriptive_disagreement_counts")}))


if __name__ == "__main__":
    main()
