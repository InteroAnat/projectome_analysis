"""Independent read-only ARM462 raw-root/label/coordinate-policy check.

No production lookup/preparation/graph routines are imported. Optional receipt
is created exclusively beside this test; bound prepared artifacts stay intact.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DELIVERY = ROOT / "group_analysis/evolution_20261008/projection_inputs/arm_labels_20261009"
RECEIPT = Path(__file__).with_name("arm_actual462_resource_bound_readback_20261009.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


class ActualARMManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance_path = DELIVERY / "preparation_provenance.json"
        cls.provenance = json.loads(cls.provenance_path.read_text(encoding="utf-8"))
        cls.rows = csv_rows(DELIVERY / "combined/projection_manifest.csv")
        cls.labels = {r["Subregion"]: r for r in csv_rows(DELIVERY / "combined/label_map.csv")}
        cls.key = {int(r["Index"]): r for r in csv_rows(cls.provenance["inputs"]["atlas_key"]["path"], "\t")}
        cls.atlas = np.asanyarray(nib.load(cls.provenance["inputs"]["atlas"]["path"]).dataobj)[:, :, :, 0, 5]
        cls.mask = np.asanyarray(nib.load(cls.provenance["inputs"]["hemisphere_mask"]["path"]).dataobj)
        cls.roots = {}
        cls.source_hashes = {}
        for row in cls.rows:
            path = Path(row["SWCPath"]).resolve(strict=True)
            if not path.is_relative_to(ROOT):
                raise AssertionError("Selected source escaped workspace")
            data = path.read_bytes()
            cls.source_hashes[row["uid"]] = hashlib.sha256(data).hexdigest()
            roots = []
            for line in data.decode("utf-8-sig").splitlines():
                fields = line.split("#", 1)[0].split()
                if not fields:
                    continue
                if len(fields) != 7:
                    raise AssertionError("Non-seven-column source graph")
                parent = float(fields[6])
                if not parent.is_integer():
                    raise AssertionError("Nonintegral parent ID")
                if parent == -1:
                    node = float(fields[0])
                    if not node.is_integer():
                        raise AssertionError("Nonintegral root node ID")
                    roots.append((int(node), [float(x) for x in fields[2:5]]))
            if len(roots) != 1:
                raise AssertionError("Source does not have exactly one raw root")
            cls.roots[row["uid"]] = roots[0]
        cls.checked = []

    def test_all_bound_inputs_outputs_and_exact_union(self):
        for binding in self.provenance["inputs"].values():
            self.assertEqual(digest(binding["path"]), binding["sha256"])
        for relative, expected in self.provenance["outputs"].items():
            self.assertEqual(digest(DELIVERY / relative), expected)
        self.assertEqual(len(self.rows), 462)
        actual = {row["uid"]: row for row in self.rows}
        self.assertEqual(len(actual), 462)
        expected = {}
        for cohort in ("main", "preview"):
            before = csv_rows(self.provenance["inputs"][cohort + "_manifest"]["path"])
            self.assertEqual(len(before), 436 if cohort == "main" else 26)
            for row in before:
                uid = row["SampleID"] + "|" + row["NeuronID"]
                self.assertNotIn(uid, expected)
                expected[uid] = row
                self.assertEqual(actual[uid]["SourceCohort"], cohort)
                for field in ("AnimalID", "SampleID", "NeuronID", "SWCPath", "SWCSHA256",
                              "ReferenceSHA256", "CoordinateFrame", "IndexScaleUm",
                              "AnatomyStatus", "RegistrationStatus", "OriginalSourceLabel"):
                    self.assertEqual(actual[uid][field], row[field], (uid, field))
                self.assertEqual(actual[uid]["EvidenceSourceGroup"], row["Subregion"])
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(len({row["AnimalID"] for row in self.rows}), 8)

    def test_raw_root_primary_and_alternative_labels(self):
        differences = Counter()
        statuses = Counter()
        groups = Counter()
        for row in self.rows:
            uid = row["uid"]
            self.assertEqual(uid, row["SampleID"] + "|" + row["NeuronID"])
            self.assertEqual(self.source_hashes[uid], row["SWCSHA256"])
            self.assertEqual(self.source_hashes[uid], row["map_selected_sha256"])
            for field, binding_name in (("AtlasSHA256", "atlas"),
                                        ("AtlasKeySHA256", "atlas_key"),
                                        ("ReferenceSHA256", "reference")):
                self.assertEqual(row[field], self.provenance["inputs"][binding_name]["sha256"])
            for field, binding_name in (("AtlasPath", "atlas"), ("AtlasKeyPath", "atlas_key")):
                self.assertEqual(Path(row[field]).resolve(),
                                 Path(self.provenance["inputs"][binding_name]["path"]).resolve())
            node, xyz = self.roots[uid]
            self.assertEqual(node, int(row["SourceRootNodeID"]))
            np.testing.assert_array_equal(xyz, json.loads(row["SourceRootXYZ"]))
            self.assertEqual(row["CoordinateFrame"], "atlas_index_um")
            scale = np.array([float(x) for x in row["IndexScaleUm"].split(";")])
            index = np.array(xyz) / scale
            np.testing.assert_array_equal(index, json.loads(row["SourceRootIndexXYZ"]))
            self.assertEqual(row["SourceRootLookupPolicy"], "current_rint_zero_center")
            primary = np.rint(index).astype(int)
            # Independent half-open cell boundaries, not floor(index+.5).
            halfopen = np.array([np.searchsorted(np.arange(n + 1) - .5, value,
                                                 side="right") - 1
                                 for n, value in zip(self.atlas.shape, index)])
            edge = np.ceil(index).astype(int) - 1
            outcomes = {}
            for name, voxel in (("primary", primary), ("halfopen_floor_zero_center", halfopen),
                                ("published_ceil_edge_one_based_to_zero", edge)):
                self.assertTrue(np.all((voxel >= 0) & (voxel < self.atlas.shape)), (uid, name))
                scalar = int(self.atlas[tuple(voxel)])
                keyrow = self.key.get(scalar)
                abbreviation = keyrow["Abbreviation"] if keyrow else ""
                full_name = keyrow["Full_Name"] if keyrow else "Atlas background"
                side = {1: "R", 2: "L"}[int(self.mask[tuple(voxel)])]
                status = "mapped" if scalar else "zero_unassigned"
                outcomes[name] = (scalar, abbreviation, full_name, side, status, voxel.tolist())
                prefix = "" if name == "primary" else name + "_"
                fields = ("ARMIndex", "ARMAbbreviation", "ARMFullName", "Hemisphere", "SourceARMStatus")
                for field, value in zip(fields, outcomes[name][:5]):
                    self.assertEqual(row[prefix + field], str(value), (uid, prefix + field))
                self.assertEqual(json.loads(row[prefix + "SourceRootVoxelXYZ"]), voxel.tolist())
                if scalar:
                    self.assertEqual(abbreviation[1], side)
                    self.assertEqual(full_name[:3], abbreviation[:3])
            scalar, abbreviation, full_name, side, status, voxel = outcomes["primary"]
            self.assertEqual(row["Subregion"], f"ARM6_{scalar}_{side}")
            self.assertEqual(row["ARMLevel"], "6")
            self.assertEqual(row["SourceARMLabelHemisphere"], abbreviation[1] if scalar else "Unknown")
            self.assertEqual(row["SourceARMHemisphereConflict"], "False")
            self.assertEqual(row["DisplayLabel"], f"{full_name} ({'Left' if side == 'L' else 'Right'})")
            for field in self.provenance["label_schema"]:
                self.assertEqual(row[field], self.labels[row["Subregion"]][field], (uid, field))
            for name in ("halfopen_floor_zero_center", "published_ceil_edge_one_based_to_zero"):
                changed = outcomes[name][0] != scalar
                side_changed = outcomes[name][3] != side
                self.assertEqual(row[name + "_label_differs"], str(changed))
                self.assertEqual(row[name + "_hemisphere_differs"], str(side_changed))
                differences[name] += changed
                differences[name + "_hemisphere"] += side_changed
                differences[name + "_voxel"] += outcomes[name][5] != voxel
            statuses[status] += 1
            groups[row["Subregion"]] += 1
            self.checked.append({"uid": uid, "source_sha256": self.source_hashes[uid],
                                 "root_node_id": node, "root_source_xyz": xyz,
                                 "primary_index": scalar, "primary_voxel": voxel,
                                 "halfopen_index": outcomes["halfopen_floor_zero_center"][0],
                                 "halfopen_voxel": outcomes["halfopen_floor_zero_center"][5],
                                 "published_edge_index": outcomes["published_ceil_edge_one_based_to_zero"][0],
                                 "published_edge_voxel": outcomes["published_ceil_edge_one_based_to_zero"][5]})
        self.assertEqual(statuses, {"mapped": 410, "zero_unassigned": 52})
        self.assertEqual(differences["halfopen_floor_zero_center"], 0)
        self.assertEqual(differences["published_ceil_edge_one_based_to_zero"], 60)
        self.assertEqual(groups, self.provenance["selection"]["combined"]["groups"])
        type(self).differences = dict(differences)
        type(self).statuses = dict(statuses)
        type(self).groups = dict(groups)

    def test_rounding_policies_are_not_generically_equivalent(self):
        centres = np.array([.5, 1.5, 2.5])
        rint = np.rint(centres).astype(int).tolist()
        halfopen = [int(np.searchsorted(np.arange(5) - .5, v, side="right") - 1)
                    for v in centres]
        self.assertEqual(rint, [0, 2, 2])
        self.assertEqual(halfopen, [1, 2, 3])
        self.assertNotEqual(rint, halfopen)


if __name__ == "__main__":
    save_receipt = "--write-receipt" in sys.argv
    if save_receipt:
        sys.argv.remove("--write-receipt")
    result = unittest.main(verbosity=2, exit=False).result
    if result.wasSuccessful() and save_receipt:
        cls = ActualARMManifest
        report = {"status": "passed", "tests": result.testsRun, "n_neurons": len(cls.rows),
                  "n_animals": 8, "primary_lookup_policy": "current_rint_zero_center",
                  "coordinate_origin_and_registration_acceptance": "unresolved",
                  "test_sha256": digest(__file__),
                  "preparation_provenance_sha256": digest(cls.provenance_path),
                  "combined_manifest_sha256": digest(DELIVERY / "combined/projection_manifest.csv"),
                  "label_map_sha256": digest(DELIVERY / "combined/label_map.csv"),
                  "differences": cls.differences, "lookup_status_counts": cls.statuses,
                  "source_group_counts": cls.groups, "source_root_readbacks": cls.checked,
                  "bound_prepared_artifacts_modified": False,
                  "limitation": "Actual-cohort label agreement is not general rounding equivalence or anatomical acceptance."}
        with RECEIPT.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print("Created", RECEIPT)
    raise SystemExit(0 if result.wasSuccessful() else 1)
