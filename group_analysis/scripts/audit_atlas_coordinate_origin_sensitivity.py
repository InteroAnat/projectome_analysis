"""Separate coordinate-origin/tie sensitivity; never changes current labels.

Apply three index policies to the same local ARM6/LR volumes. The published
NMTv2.0 CHARM/SARM code is evidence of a convention, not proof that the portal
SWC export follows it. This is not replication of the published atlas labels.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
from insula_label_set import build_insula_label_set, normalize_label

POLICIES = ("current_rint_zero_center", "published_ceil_edge_one_based_to_zero", "halfopen_floor_zero_center")


def indices(xyz_um, policy):
    encoded = np.asarray(xyz_um, dtype=float) / 250.0
    if encoded.shape != (3,) or not np.all(np.isfinite(encoded)):
        raise ValueError("Missing/nonfinite encoded coordinates")
    if policy == POLICIES[0]:
        return np.rint(encoded).astype(int)
    if policy == POLICIES[1]:
        return np.ceil(encoded).astype(int) - 1
    if policy == POLICIES[2]:
        return np.floor(encoded + .5).astype(int)
    raise ValueError(policy)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def annotation(xyz_um, policy, arm, plane, names, insula):
    voxel = indices(xyz_um, policy)
    inside = bool(np.all((voxel >= 0) & (voxel < np.asarray(arm.shape))))
    if not inside:
        return {"voxel": voxel.tolist(), "status": "out_of_bounds", "label": "Out_of_Bounds", "side": "Unknown", "insula_candidate": None}
    pos = tuple(voxel)
    rid = int(arm[pos])
    label = names.get(rid, f"Unknown_{rid}")
    return {"voxel": voxel.tolist(), "status": "atlas_background" if rid == 0 else "atlas_label", "label": label,
            "side": {1: "R", 2: "L"}.get(int(plane[pos]), "Unknown"), "insula_candidate": normalize_label(label) in insula}


def comparison(a, b):
    if a["status"] == "out_of_bounds" or b["status"] == "out_of_bounds":
        return "same_out_of_bounds_unknown" if a["status"] == b["status"] else "out_of_bounds_status_changed"
    if a["label"] == b["label"]:
        return "same_background_unknown" if a["status"] == "atlas_background" else "same_nonbackground_label"
    if a["status"] == "atlas_background":
        return "background_to_nonbackground_label"
    if b["status"] == "atlas_background":
        return "nonbackground_label_to_background"
    return "different_nonbackground_labels"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.relative_to(ROOT / "group_analysis/evolution_20261008/atlas_locations")
    if out.exists():
        raise FileExistsError(out)
    audit_dir = args.audit_dir.resolve()
    rows = read_csv(audit_dir / "neuron_soma_locations.csv")
    sources = read_csv(audit_dir / "source_inventory.csv")
    source_groups = defaultdict(list)
    for source in sources:
        assert digest(ROOT / source["source_path"]) == source["sha256"], source["source_path"]
        source_groups[(source["sample"], source["neuron_id"])].append(source)
    bundle = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym"
    paths = {"ARM6": bundle / "ARM_in_NMT_v2.1_sym.nii.gz",
             "LR_plane": bundle / "supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz",
             "key": ROOT / "atlas/ARM_key_all.txt",
             "published_code": ROOT / "references/analysis-code_gou_etal_2025/monkeyrec/zz0atlas/src/zz0atlas.jl"}
    code = paths["published_code"].read_text(encoding="utf-8")
    assert "ceil.(Int32, x ./ res)" in code and "(x .- 0.5f0) .* res" in code
    arm_img, plane_img = nib.load(paths["ARM6"]), nib.load(paths["LR_plane"])
    assert arm_img.shape[:3] == plane_img.shape and np.array_equal(arm_img.affine, plane_img.affine)
    arm = np.asanyarray(arm_img.dataobj)[..., 0, 5]
    plane = np.asanyarray(plane_img.dataobj)
    with paths["key"].open(encoding="utf-8-sig") as stream:
        names = {int(r["Index"]): r["Abbreviation"] for r in csv.DictReader(stream, delimiter="\t")}
    insula, _ = build_insula_label_set(str(paths["key"]))
    output, counts = [], defaultdict(Counter)
    for row in rows:
        result = {"sample": row["sample"], "neuron_id": row["neuron_id"],
                  "reviewed_region": row.get("reviewed_region", ""), "reviewed_status": row.get("reviewed_status", ""),
                  "own_swc_status": row["own_swc_status"], "own_source_hashes": row["own_source_hashes"],
                  "portal_supplied_region": row["portal_region"], "scientific_acceptance": "origin_export_transform_unverified_sensitivity_only"}
        for channel in ("own_root", "portal_coordinate"):
            xyz = None
            if channel == "own_root" and row["own_swc_status"].startswith("own_atlas_root"):
                xyz = json.loads(row["own_root_xyz_um"])
                assert all(json.loads(s["root_xyz_um"]) == xyz for s in source_groups[(row["sample"], row["neuron_id"])])
            elif channel == "portal_coordinate" and row["coordinate_status"] == "present":
                xyz = [float(row["nmt_" + a + "_mm"]) * 1000 for a in "xyz"]
            if xyz is None:
                result[channel + "_status"] = "missing_unassessed"
                counts[channel]["missing_unassessed"] += 1
                continue
            result[channel + "_status"] = "coordinate_assessed_sensitivity_only"
            variants = {}
            for policy in POLICIES:
                value = annotation(xyz, policy, arm, plane, names, insula)
                variants[policy] = value
                for key, val in value.items():
                    result[channel + "_" + policy + "_" + key] = json.dumps(val) if isinstance(val, list) else val
                counts[channel + ":" + policy][value["status"]] += 1
                counts[channel + ":" + policy + ":side"][value["side"]] += 1
                counts[channel + ":" + policy]["INS_label_set_candidate"] += value["insula_candidate"] is True
            current = variants[POLICIES[0]]
            saved = row["own_atlas_label"] if channel == "own_root" else row["portal_coordinate_atlas_label"]
            assert saved == current["label"]
            for alternate in POLICIES[1:]:
                value = variants[alternate]
                label_comparison = comparison(current, value)
                result[channel + "_vs_" + alternate + "_comparison"] = label_comparison
                counts[channel + ":vs:" + alternate][label_comparison] += 1
                counts[channel + ":vs:" + alternate]["voxel_changed"] += current["voxel"] != value["voxel"]
                both_sides = current["side"] in ("L", "R") and value["side"] in ("L", "R")
                side_comparison = "same_known_side" if both_sides and current["side"] == value["side"] else "different_known_side" if both_sides else "unknown_side_in_either_policy"
                result[channel + "_vs_" + alternate + "_side_comparison"] = side_comparison
                counts[channel + ":vs:" + alternate][side_comparison] += 1
        output.append(result)
    assert len(output) == len({(r["sample"], r["neuron_id"]) for r in output}) == 8746
    manifest = {"observed_at": datetime.now().astimezone().isoformat(), "status": "sensitivity_only_no_convention_change",
                "exact_neuron_identities": len(output), "source_hashes_freshly_reverified": len(sources),
                "user_described_registration_direction": "raw fMOST brain and neurons warped into NMT; original transform/version receipts pending",
                "policies": {POLICIES[0]: "np.rint(XYZ/250), zero-based centers; exact halves ties to even (existing client)",
                             POLICIES[1]: "literal deposited pos2idx ceil(XYZ/res) Julia1-based, converted to zero-based ceil(XYZ/250)-1; idx2pos=(index-0.5)*res; right boundary included",
                             POLICIES[2]: "floor(XYZ/250+0.5), zero-based centers, half-open center cells; differs from rint at exact half ties"},
                "numeric_evaluation": "Float64 current stored coordinates; original exporter numeric precision and origin unknown",
                "atlas_scope": "All policies applied to the SAME local NMTv2.1 symmetric ARM6 and LR plane to isolate coordinate-policy sensitivity. Published code declares NMTv2.0 CHARM+SARM; this is NOT reproduction of its atlas labels.",
                "published_source_lines": {"NMTv2.0_CHARM_SARM_references": "207-213", "pos2idx_idx2pos": "289-294"},
                "unknowns": ["Whether these SWC/getSoma exports use deposited zz0atlas edge-origin positions", "Original registration transforms/landmark acceptance", "Server atlas/reference version and label-generation implementation"],
                "counts": {k: dict(v) for k, v in counts.items()},
                "reviewed_cases": [r for r in output if r["reviewed_region"]],
                "input_audit": {"path": audit_dir.relative_to(ROOT).as_posix(), "neuron_rows_sha256": digest(audit_dir / "neuron_soma_locations.csv"), "source_inventory_sha256": digest(audit_dir / "source_inventory.csv")},
                "references": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for k, p in paths.items()},
                "code_sha256": digest(Path(__file__)), "no_external_writes": True, "network_requests": 0,
                "no_automatic_relabeling": True, "endpoint_projection_maps_generated": False}
    out.mkdir(parents=True)
    fields = list(dict.fromkeys(k for r in output for k in r))
    with (out / "per_neuron_origin_sensitivity.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    (out / "provenance_and_counts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"identities": len(output), "counts": manifest["counts"]}), flush=True)


if __name__ == "__main__":
    main()
