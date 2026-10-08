"""Read-only, exact-identity cached atlas-root and separate portal soma audit.

No download or relabeling. Atlas SWC XYZ are declared physical *index* um:
XYZ/250 gives NMT voxel indices, not NIfTI affine world coordinates.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main_scripts"))
from region_labels import normalize_region_label, normalize_portal_region_label

LIVE = "group_analysis/evolution_20261008/inventory/live_inventory_20261008/neuron_inventory.csv"
REVIEW = "notes/region_analysis_review_20261004"
BUNDLE = "atlas/NMT_v2.1_sym/NMT_v2.1_sym"
SOURCE_ROOTS = (
    ("neuron-vis/resource/swc", "atlas cache declared by Oct4 actual_rerun_inventory and rerun_actual_region_tables"),
    ("main_scripts/processed_neurons/251637", "ION atlas getNeuronByID cache; neuro_tracer monkey XYZ/250; 112/114 hash-bound reviewed evidence"),
    (REVIEW + "/recovered_atlas_swcs", "Oct4 bounded getSWC recovery with same-sample graph-matched anchors and per-identity source hashes"),
    ("group_analysis/fnt/nmt_swcs", "05c_run_global_fnt fetch_nmt_swc getNeuronByID cache; unmirrored NMT SWC source, not mirrored derivative"),
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def side(label):
    for prefix, value in (("CL_", "L"), ("SL_", "L"), ("CR_", "R"), ("SR_", "R"),
                          ("L-", "L"), ("L_", "L"), ("R-", "R"), ("R_", "R")):
        if str(label).startswith(prefix):
            return value
    return "Unknown"


def portal_side(label):
    # Portal CL_208 means anatomical CL with numeric region ID, not a
    # hemisphere prefix. Strip the declared index suffix before side parsing.
    text = str(label).strip()
    head, separator, tail = text.rpartition("_")
    if separator and tail.isdigit():
        text = head
    return side(text)


def index_coordinates(xyz_um):
    xyz = np.asarray(xyz_um, dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("Missing or nonfinite coordinates")
    return xyz / 250.0


def lookup(xyz_um, atlas, plane, brain, labels, hierarchy):
    """Match production np.round ties-to-even; expose half-voxel sensitivity."""
    xyz = index_coordinates(xyz_um)
    voxel = np.rint(xyz).astype(int)
    result = {"index_xyz": xyz.tolist(), "rounded_voxel_xyz": voxel.tolist(),
              "half_voxel_boundary": bool(np.any(np.isclose(xyz - np.floor(xyz), .5, atol=1e-9, rtol=0)))}
    if np.any(voxel < 0) or np.any(voxel >= np.asarray(atlas.shape)):
        return {**result, "lookup_status": "out_of_bounds", "atlas_label": "Out_of_Bounds", "mask_side": "Unknown"}
    pos = tuple(voxel)
    rid = int(atlas[pos])
    name = labels.get(rid, f"Unknown_{rid}")
    mask_side = {1: "R", 2: "L"}.get(int(plane[pos]), "Unknown")
    return {**result, "lookup_status": "atlas_background" if rid == 0 else "atlas_label",
            "atlas_id": rid, "atlas_label": name, "hierarchy": hierarchy.get(name, []),
            "mask_side": mask_side, "brainmask_value": int(brain[pos]),
            "atlas_label_mask_conflict": side(name) != "Unknown" and mask_side != "Unknown" and side(name) != mask_side}


def root_from_swc(path):
    """Fresh full-file root read, independent of tracer/cache-result records.

    Validates finite XYZ, seven columns, unique node IDs and one -1 root;
    graph branches/projections and registration are outside this audit.
    """
    roots, ids, count = [], set(), 0
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            fields = text.split()
            if len(fields) != 7:
                raise ValueError(f"line {line_number}: expected seven SWC columns")
            values = [float(x) for x in fields]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"line {line_number}: nonfinite SWC values")
            node, parent = values[0], values[6]
            if not node.is_integer() or not parent.is_integer() or node in ids:
                raise ValueError(f"line {line_number}: duplicate/noninteger node or parent ID")
            ids.add(node)
            count += 1
            if parent == -1:
                roots.append((int(node), values[2:5]))
    if len(roots) != 1:
        raise ValueError(f"expected one root; found {len(roots)}")
    return {"root_node_id": roots[0][0], "root_xyz_um": roots[0][1], "node_count": count}


def resolve_sources(records):
    if not records:
        return "missing_unassessed", None
    if any(row["source_status"] != "root_checked" for row in records):
        return "source_invalid_unassessed", None
    if len({row["sha256"] for row in records}) > 1:
        # Source bytes/whole graphs remain distinct. Coordinate consensus is
        # nevertheless assessable without choosing one graph as authoritative.
        keys = ("root_node_id", "root_xyz_um", "index_xyz", "rounded_voxel_xyz", "lookup_status",
                "atlas_label", "mask_side", "half_voxel_boundary", "hierarchy", "atlas_id", "atlas_label_mask_conflict", "brainmask_value")
        essential = ("root_xyz_um", "index_xyz", "rounded_voxel_xyz", "lookup_status", "atlas_label", "mask_side")
        if all(k in r for k in essential for r in records) and all(
                all(r[k] == records[0][k] for r in records) for k in essential):
            consensus = {k: records[0].get(k) if all(r.get(k) == records[0].get(k) for r in records) else None for k in keys}
            return "own_atlas_root_consensus_multiple_source_hashes", consensus
        return "source_hash_collision_unresolved", None
    return "own_atlas_root_checked", records[0]


def reviewed_assignment(sample, neuron, sources, review):
    if sample != review["sample_id"]:
        return {}
    annotation = next((a for a in review["annotations"] if a["NeuronID"] == neuron), None)
    if not annotation:
        return {}
    expected = review["swc_sha256"][neuron]
    matching = any(s.get("sha256") == expected and s.get("source_status") == "root_checked" for s in sources)
    return {"reviewed_region": annotation["Soma_Region"], "reviewed_layer": annotation["Soma_Layer"],
            "reviewed_status": annotation["status"], "reviewed_source_hash_verified": matching,
            "reviewed_effective_status": "reviewed_hash_bound" if matching else "reviewed_source_not_verified_unassessed",
            "reviewed_original_labels": annotation["Original_Henry_Labels"]}


def key_maps(path):
    rows = read_csv_tsv(path)
    labels, hierarchy, stack = {}, {}, {}
    for row in sorted(rows, key=lambda r: int(r["Index"])):
        name, level = row["Abbreviation"], int(row["First_Level"])
        stack = {k: v for k, v in stack.items() if k < level}
        stack[level] = name
        labels[int(row["Index"])] = name
        hierarchy[name] = [stack[k] for k in sorted(stack)]
    return labels, hierarchy


def read_csv_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def flat(prefix, obj):
    return {prefix + k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in obj.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / LIVE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verified-source-readback", type=Path,
                        help="Reuse fresh prior per-file root records ONLY after matching exact path and freshly recomputed SHA256; independently verify representative roots afterward")
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    out.relative_to(ROOT / "group_analysis/evolution_20261008/atlas_locations")
    if out.exists():
        raise FileExistsError(f"No-clobber output already exists: {out}")
    neurons = read_csv(args.inventory)
    identities = [(r["sample"], r["neuron_id"]) for r in neurons]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate exact inventory identity")
    scope = set(identities)
    # Inventory only documented atlas roots; no native, mirrored, alias or broad disk search.
    candidates, excluded = [], []
    for directory, declaration in SOURCE_ROOTS:
        base = ROOT / directory
        paths = base.glob("*.swc") if base.name == "251637" else base.glob("*/*.swc")
        for path in sorted(paths):
            identity = (path.parent.name, path.name)
            row = {"sample": identity[0], "neuron_id": identity[1], "source_path": relative(path),
                   "coordinate_declaration": declaration}
            if identity in scope:
                candidates.append(row)
            else:
                excluded.append({**row, "reason": "exact identity absent from live inventory; no alias mapping"})
    for nid in ("112.swc", "114.swc"):
        path = ROOT / REVIEW / "case_112_114_20261004" / ("atlas_" + nid)
        if path.exists() and ("251637", nid) in scope:
            candidates.append({"sample": "251637", "neuron_id": nid, "source_path": relative(path),
                               "coordinate_declaration": "explicit exact 112/114 fresh atlas getSWC source identity checked in annotation_and_identity_readback"})
    paths = {"template": ROOT / BUNDLE / "NMT_v2.1_sym_SS.nii.gz",
             "ARM6": ROOT / BUNDLE / "ARM_in_NMT_v2.1_sym.nii.gz",
             "LR_plane": ROOT / BUNDLE / "supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz",
             "LR_brainmask": ROOT / BUNDLE / "supplemental_masks/NMT_v2.1_sym_LR_brainmask.nii.gz",
             "key": ROOT / "atlas/ARM_key_all.txt",
             "reviewed": ROOT / REVIEW / "case_112_114_20261004/idm_assignment_20261008/reviewed_soma_assignments.json",
             "original_manual": ROOT / "R_analysis/tables/somainfo_Henry_2026.04.03.xlsx",
             "prior_source_inventory": ROOT / REVIEW / "actual_rerun_inventory.json",
             "recovery": ROOT / REVIEW / "missing_atlas_source_recovery.json",
             "coordinate_code": ROOT / "main_scripts/neuro_tracer.py",
             "IONData_client": ROOT / "neuron-vis/neuronVis/IONData.py",
             "unmirrored_FNT_cache_code": ROOT / "group_analysis/scripts/05c_run_global_fnt.py",
             "audit_code": Path(__file__).resolve()}
    images = {k: nib.load(paths[k]) for k in ("template", "ARM6", "LR_plane", "LR_brainmask")}
    template = images["template"]
    for name, img in images.items():
        if img.shape[:3] != template.shape or not np.array_equal(img.affine, template.affine):
            raise ValueError(f"Reference grid mismatch: {name}")
    if template.shape != (256, 312, 200):
        raise ValueError("Unexpected reference shape")
    atlas = np.asanyarray(images["ARM6"].dataobj)[..., 0, 5]
    plane = np.asanyarray(images["LR_plane"].dataobj)
    brain = np.asanyarray(images["LR_brainmask"].dataobj)
    labels, hierarchy = key_maps(paths["key"])
    contingency = {}
    for value in (1, 2):
        frequencies = np.bincount(atlas[plane == value].astype(int).ravel(), minlength=max(labels) + 1)
        counts = {s: sum(int(frequencies[rid]) for rid, label in labels.items() if side(label) == s) for s in ("L", "R")}
        if counts[{1: "R", 2: "L"}[value]] == 0 or counts[{1: "L", 2: "R"}[value]]:
            raise ValueError("Official atlas/mask side semantics disagree")
        contingency[str(value)] = counts
    review = json.loads(paths["reviewed"].read_text(encoding="utf-8"))
    manual = defaultdict(list)
    manual_table = pd.read_excel(paths["original_manual"], "251637 mostly insula")
    for i, row in manual_table.iterrows():
        if pd.isna(row.SWC) or pd.isna(row.Area):
            continue
        nid = f"{int(row.SWC):03d}.swc"
        manual[nid].append({"excel_row": int(i + 2), "area": str(row.get("Area", "")),
                            "layer": str(row.get("Layer", ""))})
    prior = json.loads(paths["prior_source_inventory"].read_text(encoding="utf-8"))
    expected_hashes = {r["path"].replace("\\", "/"): r["sha256"] for r in prior["swc_sources"]}
    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))
    expected_hashes.update({r["path"]: r["sha256"] for r in recovery["results"] if r["status"] == "own_atlas_swc_recovered"})
    previous, reused = {}, 0
    if args.verified_source_readback:
        previous_rows = read_csv(args.verified_source_readback)
        previous = {r["source_path"]: r for r in previous_rows}
        if len(previous) != len(previous_rows):
            raise ValueError("Duplicate source identity/path in reusable readback")
    sources = defaultdict(list)
    for n, candidate in enumerate(candidates, 1):
        path = ROOT / candidate["source_path"]
        record = {**candidate, "sha256": digest(path), "bytes": path.stat().st_size}
        expected = expected_hashes.get(candidate["source_path"])
        if candidate["source_path"] == review["swc_source_dir"] + "/" + candidate["neuron_id"] and candidate["neuron_id"] in review["swc_sha256"]:
            expected = expected or review["swc_sha256"][candidate["neuron_id"]]
        record["prior_expected_sha256"] = expected or ""
        record["prior_hash_status"] = "matched" if expected == record["sha256"] else "changed" if expected else "not_previously_hash_bound"
        try:
            old = previous.get(candidate["source_path"])
            if old and old["sha256"] == record["sha256"] and old["source_status"] == "root_checked":
                if (old["sample"], old["neuron_id"]) != (candidate["sample"], candidate["neuron_id"]):
                    raise ValueError("Reusable exact source identity mismatch")
                root = {"root_node_id": int(old["root_node_id"]), "root_xyz_um": json.loads(old["root_xyz_um"]), "node_count": int(old["node_count"])}
                record["root_read_method"] = "prior_fresh_full_file_root_read_exact_path_hash_reverified"
                reused += 1
            else:
                root = root_from_swc(path)
                record["root_read_method"] = "fresh_full_file_root_read"
            result = lookup(root["root_xyz_um"], atlas, plane, brain, labels, hierarchy)
            record.update(source_status="root_checked", **root, **result)
        except (OSError, ValueError) as exc:
            record.update(source_status="source_invalid_unassessed", error=str(exc))
        sources[(candidate["sample"], candidate["neuron_id"])].append(record)
        if n % 250 == 0:
            print(json.dumps({"fresh_sources_checked": n, "total": len(candidates)}), flush=True)
    results, discrepancies = [], []
    for original in neurons:
        sample, nid = original["sample"], original["neuron_id"]
        records = sources[(sample, nid)]
        status, own = resolve_sources(records)
        result = {**original, "own_swc_status": status, "own_source_count": len(records),
                  "own_source_paths": json.dumps([s["source_path"] for s in records]),
                  "own_source_hashes": json.dumps([s["sha256"] for s in records]),
                  "portal_assignment_lineage": "unknown server label-generation/transform receipts; working user chain raw fMOST brain and neurons warped to NMT",
                  "native_transform_landmark_acceptance": "pending", "automatic_relabeling": False}
        result["own_distinct_source_hash_count"] = len({s["sha256"] for s in records})
        result["source_graph_equivalence"] = "not_adjudicated_multiple_byte_sources" if result["own_distinct_source_hash_count"] > 1 else "one_byte_source" if records else "missing_unassessed"
        if own:
            result.update(flat("own_", {k: own[k] for k in ("root_node_id", "root_xyz_um", "index_xyz", "rounded_voxel_xyz", "lookup_status", "atlas_label", "mask_side", "half_voxel_boundary")}))
            result["own_hierarchy"] = json.dumps(own.get("hierarchy", []))
            result["own_atlas_id"] = own.get("atlas_id", "")
            result["own_brainmask_value"] = own.get("brainmask_value", "")
            result["own_label_mask_conflict"] = own.get("atlas_label_mask_conflict", False)
        portal = None
        if original["coordinate_status"] == "present":
            try:
                # Inventory index-mm values were xyz_um/1000, not affine world mm.
                portal_um = [float(original[f"nmt_{a}_mm"]) * 1000 for a in "xyz"]
                portal = lookup(portal_um, atlas, plane, brain, labels, hierarchy)
                result.update(flat("portal_coordinate_", portal))
                if own:
                    delta = np.asarray(own["root_xyz_um"]) - np.asarray(portal_um)
                    result["own_minus_portal_xyz_um"] = json.dumps(delta.tolist())
                    result["own_portal_distance_um"] = float(np.linalg.norm(delta))
                    result["own_portal_coordinate_mismatch_gt_1um"] = bool(np.max(np.abs(delta)) > 1)
                    result["own_portal_direct_label_difference"] = own["atlas_label"] != portal["atlas_label"]
            except ValueError as exc:
                result["portal_coordinate_lookup_status"] = "invalid_unassessed"
                result["portal_coordinate_error"] = str(exc)
        else:
            result["portal_coordinate_lookup_status"] = "missing_unassessed"
        supplied = original["portal_region"]
        for prefix, value in (("own", own), ("portal_coordinate", portal)):
            if value:
                result[prefix + "_direct_INS_hierarchy"] = any(normalize_region_label(x) == "INS" for x in value.get("hierarchy", []))
                result[prefix + "_direct_G"] = normalize_region_label(value["atlas_label"]) == "G"
                result[prefix + "_direct_unknown_background"] = value["atlas_label"] == "Unknown_0"
            if value and supplied:
                result[prefix + "_vs_portal_supplied_region_difference"] = normalize_region_label(value["atlas_label"]) != normalize_portal_region_label(supplied)
                supplied_side, root_side = portal_side(supplied), value["mask_side"]
                result[prefix + "_vs_portal_supplied_side_conflict"] = supplied_side != "Unknown" and root_side != "Unknown" and supplied_side != root_side
        if sample == "251637":
            result["original_manual_rows"] = json.dumps(manual.get(nid, []))
            if own:
                result["original_manual_side_conflicts"] = json.dumps([a for a in manual.get(nid, []) if side(a["area"]) != "Unknown" and side(a["area"]) != own["mask_side"]])
        reviewed = reviewed_assignment(sample, nid, records, review)
        result.update(flat("", reviewed))
        if own and reviewed:
            result["reviewed_mask_side_conflict"] = side(reviewed["reviewed_region"]) != own["mask_side"]
        flags = [k for k, v in result.items() if ("difference" in k or "conflict" in k or "mismatch" in k) and v is True]
        if status in ("source_hash_collision_unresolved", "source_invalid_unassessed", "own_atlas_root_consensus_multiple_source_hashes"):
            flags.append(status)
        if flags:
            discrepancies.append({"sample": sample, "neuron_id": nid, "flags": "|".join(flags),
                                  "portal_supplied_region": supplied, "own_root_atlas_label": result.get("own_atlas_label", ""),
                                  "portal_coordinate_atlas_label": result.get("portal_coordinate_atlas_label", ""),
                                  "own_portal_distance_um": result.get("own_portal_distance_um", ""),
                                  "interpretation": "descriptive provenance difference; assignment lineage/registration not adjudicated"})
        results.append(result)
    counts = dict(Counter(r["own_swc_status"] for r in results))
    summary = []
    for sample in sorted({r["sample"] for r in results}):
        subset = [r for r in results if r["sample"] == sample]
        summary.append({"sample": sample, "live_neurons": len(subset), **dict(Counter(r["own_swc_status"] for r in subset)),
                        "portal_coordinates_present": sum(r["coordinate_status"] == "present" for r in subset)})
    provenance = {"observed_at": datetime.now().astimezone().isoformat(), "scope_exact_live_identities": len(results),
                  "unique_exact_identities": len(set(identities)), "source_files_checked": len(candidates),
                  "source_status_counts": dict(Counter(s["source_status"] for records in sources.values() for s in records)),
                  "own_neuron_status_counts": counts, "portal_coordinates_present": sum(r["coordinate_status"] == "present" for r in results),
                  "portal_coordinates_missing": sum(r["coordinate_status"] != "present" for r in results),
                  "discrepancy_neurons": len(discrepancies), "coordinate_encoding": "atlas SWC and portal physical index um / 250 = NMT voxel; affine maps index to world only for physical display, never inverse-applied to encoded SWC",
                  "lookup_policy": "ARM level 6 (5D volume [...,0,5]); np.rint ties to even; rounded bounds; half-voxel boundaries recorded",
                  "mask_values": {"1": "R", "2": "L", "0": "Unknown"}, "mask_label_contingency": contingency,
                  "reference_grid": {"shape": list(template.shape), "affine": template.affine.tolist(), "units": list(template.header.get_xyzt_units())},
                  "inputs": {k: {"path": relative(p), "sha256": digest(p)} for k, p in paths.items()},
                  "inventory": {"path": relative(args.inventory), "sha256": digest(args.inventory)},
                  "source_roots": SOURCE_ROOTS, "excluded_sources": excluded,
                  "explicitly_excluded_locations": ["main_scripts/processed_neurons/251637_b (no exact sample alias)", "group_analysis/visual_review_20261002/swc_raw (native)", "raw_swcs descendants (native)", "mirrored_swcs/FNT graphs (not unmodified atlas source SWC)"],
                  "manual_original_source_hash_matches_review": digest(paths["original_manual"]) == review["original_henry_source"]["sha256"],
                  "portal_supplied_label_lineage": "unknown server label-generation; working user chain raw fMOST brain/neurons warped to NMT, transform/version receipts not recovered",
                  "acceptance": "computational atlas-space lookup only; no native transform/landmark acceptance; no source or manual labels overwritten",
                  "network_requests": 0, "cached_result_reuse": bool(reused), "source_root_records_reused_after_hash_reverification": reused,
                  "reused_source_readback": {"path": relative(args.verified_source_readback), "sha256": digest(args.verified_source_readback)} if args.verified_source_readback else None,
                  "source_selection_policy": "No source graph silently selected; different byte hashes with exact root/lookup consensus yield a separate consensus status; graph equivalence unadjudicated"}
    provenance["strata"] = {name: {"denominator": sum(r[name] == "True" for r in results),
                                  "own_roots_checked": sum(r[name] == "True" and r["own_swc_status"].startswith("own_atlas_root") for r in results)}
                             for name in ("atlas_INS", "atlas_G", "potential_INS_candidate", "atlasINS_or_G_or_foldedBox_candidate")}
    out.mkdir(parents=True)
    write_csv(out / "source_inventory.csv", [flat("", s) for records in sources.values() for s in records])
    write_csv(out / "neuron_soma_locations.csv", results)
    write_csv(out / "discrepancies.csv", discrepancies)
    write_csv(out / "sample_coverage.csv", summary)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    (out / "README.md").write_text(
        "# Cached atlas soma location audit\n\n"
        f"Checked {len(results)} exact live neuron identities; {counts}. All {len(candidates)} eligible local atlas SWC files were freshly hashed; {reused} root records reused only after exact path/hash revalidation, and the remainder freshly read. Independently verify representative roots with verify_atlas_soma_audit.py. Different byte hashes never silently select a graph; exact coordinate/lookup agreement is reported as root consensus with graph equivalence unadjudicated.\n\n"
        "SWC XYZ/250 and portal physical-index um/250 give NMT voxel indices. These values are not affine world millimetres. ARM level 6 and official LR masks share the recorded NMT v2.1 symmetric grid. Missing SWCs remain unassessed; portal-coordinate lookups are a separate preliminary channel.\n\n"
        "Source labels, candidate screens, original manual rows and hash-bound reviewed assignments are separate columns. 112 retains reviewed R-IDM; 114 remains provisional R-IDM. The working registration chain is raw fMOST brain and neurons warped to NMT. Portal-supplied label-generation and transform/version receipts remain unverified. Disagreements are descriptive findings, not proof of a portal bug or accepted relabeling.\n\n"
        "No raw native SWCs, mirrored FNT copies, downloads or canonical edits were used. Native transform and landmark acceptance remain pending.\n", encoding="utf-8")
    print(json.dumps({k: provenance[k] for k in ("scope_exact_live_identities", "source_files_checked", "own_neuron_status_counts", "portal_coordinates_missing", "discrepancy_neurons")}), flush=True)


if __name__ == "__main__":
    main()
