"""Independent NumPy/mm readback of isolated folded candidate refinement."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token(value):
    text = str(value).strip().upper()
    for prefix in ("CL_", "CR_", "L-", "R-", "L_", "R_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip()


def main():
    record_path = HERE / "refinement_provenance.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(record["geometry_mode"] == "folded-mm" and record["padding_mm"] == 2, "Unexpected declared geometry")
    bindings = record["reference_inputs"]
    for source in bindings.values():
        require(digest(source["path"]) == source["sha256"], "Changed reference")
    for path, expected in record["policy_source_sha256"].items():
        require(digest(path) == expected, "Changed code/key")
    for name, expected in record["outputs"].items():
        require(digest(HERE / name) == expected, "Changed isolated output")
    boxes = pd.read_csv(bindings["bbox"]["path"])
    lower = boxes[[f"{axis}_lo_q005" for axis in "XYZ"]].to_numpy(float) * .25
    upper = boxes[[f"{axis}_hi_q995" for axis in "XYZ"]].to_numpy(float) * .25
    names = boxes.sub_region.to_numpy()
    anchors = pd.read_csv(bindings["anchors"]["path"])
    thresholds = dict(zip(anchors.sub_region, anchors.median_nn * .25 * 3))
    reference = pd.read_csv(bindings["reference_neurons"]["path"])
    points = reference[["Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z"]].to_numpy(float)
    points[:, 0] = np.abs(points[:, 0] - 128)
    points *= .25
    reference_by_region = {name: points[reference.Soma_Region_clean.eq(name)] for name in names}
    atlas_key = next(path for path in record["policy_source_sha256"] if Path(path).name == "ARM_key_all.txt")
    atlas = pd.read_csv(atlas_key, sep="\t")
    keep = {"IAL", "IAPM", "IDD5", "IDM", "IDV", "IDD", "IDI", "G"}
    for label in atlas.loc[atlas.Full_Name.fillna("").str.contains("insula", case=False, regex=False), "Abbreviation"]:
        normalized = token(label)
        keep.add(normalized)
        keep.update(re.split(r"\s*/\s*", normalized))
    keep -= {"RI", "RETROINSULA"}
    rescue = {"PRCO", "UNKNOWN", "UNKNOWN_0", "_UNMAPPED", "INSULAUNKNOWN", "", "UNMAPPED"}
    discovery_path = HERE.parent / "discovery_folded_20261009_v2/discovery_scan_per_neuron.csv"
    discovery = pd.read_csv(discovery_path, dtype={"SampleID": str, "NeuronID": str}).fillna("")
    discovery_by_uid = {(r.SampleID, r.NeuronID): r for r in discovery.itertuples()}
    seen, kept, by_sample, counts = set(), [], {}, Counter()
    for source in record["sources"]:
        require(source["status"] == "read" and digest(source["path"]) == source["sha256"], "Changed/missing source workbook")
        sid = source["sample_id"]
        original = pd.read_excel(source["path"], sheet_name="Summary")
        output = pd.read_excel(HERE / f"{sid}_INS_HE_coord_inferred.xlsx", sheet_name="Summary")
        require(len(original) == len(output) == source["rows"], "Dropped source rows")
        pd.testing.assert_frame_equal(original, output[original.columns], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        keeper_ids = []
        for row, saved in zip(original.to_dict("records"), output.to_dict("records")):
            uid = (sid, row["NeuronID"])
            require(uid not in seen and uid in discovery_by_uid, "Duplicate/missing discovery identity")
            seen.add(uid)
            normalized = token(row["Soma_Region"])
            xyz = np.array([row[f"Soma_NII_{axis}"] for axis in "XYZ"], dtype=float)
            xyz[0] = abs(xyz[0] - 128)
            xyz *= .25
            strict = names[np.all((xyz >= lower) & (xyz <= upper), axis=1)].tolist()
            padded = names[np.all((xyz >= lower - 2) & (xyz <= upper + 2), axis=1)].tolist()
            require(set(filter(None, discovery_by_uid[uid].bbox_match_subregion.split(";"))) == set(padded),
                    f"Discovery/refinement geometry differs: {uid}")
            expected_region, match, distance = "", "", None
            if normalized in keep:
                expected_region = match = normalized
                status = "auto_atlas_insula"
            elif normalized not in rescue and not re.fullmatch(r"UNKNOWN_\d+", normalized) or not np.isfinite(xyz).all():
                status = "not_rescue_candidate"
            else:
                if len(strict) == 1:
                    expected_region, tier = strict[0], "strict"
                elif len(padded) == 1:
                    expected_region, tier = padded[0], "padded"
                elif len(strict) > 1 or len(padded) > 1:
                    status, match = "excluded_ambiguous_bbox", ";".join(strict or padded)
                else:
                    status = "excluded_outside_bbox"
                if expected_region:
                    match = expected_region
                    distance = float(np.linalg.norm(reference_by_region[expected_region] - xyz, axis=1).min())
                    status = f"coord_inferred_from_251637_{tier}" + ("_distant" if distance > thresholds[expected_region] else "")
            require(saved["Soma_Region_Source"] == status, f"Candidate policy differs: {uid}")
            require(("" if pd.isna(saved["Soma_Region_Refined"]) else saved["Soma_Region_Refined"]) == expected_region,
                    f"Candidate region differs: {uid}")
            require(("" if pd.isna(saved["Soma_Region_Match"]) else saved["Soma_Region_Match"]) == match,
                    f"Match evidence differs: {uid}")
            actual_distance = saved["Distance_to_nearest_251637_neuron_mm"]
            if distance is None:
                require(pd.isna(actual_distance), f"Unexpected nearest distance: {uid}")
            else:
                np.testing.assert_allclose(actual_distance, distance, rtol=1e-12, atol=1e-12)
            require(saved["PAD_TOL_MM"] == 2 and saved["PAD_TOL_VOX"] == 8
                    and saved["Reference_Geometry_Mode"] == "folded-mm"
                    and saved["Anatomical_Acceptance"] == "not_assessed", "Geometry/acceptance record differs")
            counts[status] += 1
            if status.startswith(("auto_atlas_insula", "coord_inferred_from_251637")):
                keeper_ids.append(row["NeuronID"])
                kept.append(uid)
        keeper_sheet = pd.read_excel(HERE / f"{sid}_INS_HE_coord_inferred.xlsx", sheet_name="Insula_keepers")
        require(keeper_sheet.NeuronID.tolist() == keeper_ids, f"Keeper sheet identities differ: {sid}")
        by_sample[sid] = {"source_rows": len(original), "candidate_keepers": len(keeper_ids)}
    require(seen == set(discovery_by_uid), "Source scope differs from validated discovery")
    consolidated = pd.read_csv(HERE / "all_refined_neurons.csv", dtype={"SampleID": str, "NeuronID": str})
    require(list(zip(consolidated.SampleID, consolidated.NeuronID)) == kept, "Consolidated identity/order differs")
    for source in record["sources"]:
        require(digest(source["path"]) == source["sha256"], "Source changed during readback")
    report = {"status": "passed", "reviewed_utc": datetime.now(timezone.utc).isoformat(),
              "reader_sha256": digest(__file__), "refinement_provenance_sha256": digest(record_path),
              "discovery_csv_sha256": digest(discovery_path), "rows_checked": len(seen), "keeper_rows_checked": len(kept),
              "source_workbooks": len(record["sources"]), "source_bytes_unchanged": True,
              "original_summary_fields_preserved": True, "geometry_divergences_from_validated_discovery": 0,
              "candidate_policy_divergences": 0, "distance_divergences": 0,
              "method": "Independent normalized label sets and NumPy physical-mm bounds/Euclidean distances; refiner not imported",
              "by_sample": by_sample, "source_counts": dict(counts), "anatomical_acceptance": "not_assessed",
              "canonical_promotion": False}
    with (HERE / "independent_readback_20261009.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("status", "rows_checked", "keeper_rows_checked", "geometry_divergences_from_validated_discovery")}))


if __name__ == "__main__":
    main()
