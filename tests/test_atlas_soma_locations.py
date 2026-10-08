import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "group_analysis/scripts/audit_atlas_soma_locations.py"
SPEC = importlib.util.spec_from_file_location("atlas_soma_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
SENS_SPEC = importlib.util.spec_from_file_location("atlas_origin_sensitivity", SCRIPT.with_name("audit_atlas_coordinate_origin_sensitivity.py"))
sensitivity = importlib.util.module_from_spec(SENS_SPEC)
SENS_SPEC.loader.exec_module(sensitivity)


class AtlasSomaAuditTests(unittest.TestCase):
    def setUp(self):
        self.atlas = np.zeros((4, 4, 4), dtype=int)
        self.plane = np.zeros_like(self.atlas)
        self.brain = np.zeros_like(self.atlas)
        self.atlas[1, 2, 3] = 7
        self.plane[1, 2, 3] = 1
        self.brain[1, 2, 3] = 1
        self.labels = {7: "CR_Ia/Id"}

    def lookup(self, xyz):
        return audit.lookup(xyz, self.atlas, self.plane, self.brain, self.labels,
                            {"CR_Ia/Id": ["CR_INS", "CR_Ia/Id"]})

    def test_encoded_index_um_is_not_world_mm(self):
        self.assertEqual(self.lookup([250, 500, 750])["atlas_label"], "CR_Ia/Id")
        self.assertEqual(self.lookup([250, 500, 750])["index_xyz"], [1, 2, 3])
        # NMT affine translations must not be subtracted from these indices.
        affine = np.diag([.25, .25, .25, 1.])
        affine[:3, 3] = [-31.875, -27.75, -8]
        world = affine @ np.array([1, 2, 3, 1])
        self.assertFalse(np.array_equal(world[:3], [1, 2, 3]))

    def test_half_voxel_ties_even_and_nearby_boundary(self):
        exact = self.lookup([375, 500, 750])  # 1.5 rounds to 2
        below = self.lookup([375 - 1e-5, 500, 750])
        self.assertTrue(exact["half_voxel_boundary"])
        self.assertEqual(exact["rounded_voxel_xyz"][0], 2)
        self.assertEqual(below["atlas_label"], "CR_Ia/Id")
        self.assertEqual(self.lookup([-125, 500, 750])["rounded_voxel_xyz"][0], 0)
        self.assertEqual(self.lookup([-125.01, 500, 750])["lookup_status"], "out_of_bounds")

    def test_official_mask_side_and_label_conflict(self):
        self.assertEqual(self.lookup([250, 500, 750])["mask_side"], "R")
        self.assertFalse(self.lookup([250, 500, 750])["atlas_label_mask_conflict"])
        self.plane[1, 2, 3] = 2
        self.assertEqual(self.lookup([250, 500, 750])["mask_side"], "L")
        self.assertTrue(self.lookup([250, 500, 750])["atlas_label_mask_conflict"])
        self.assertEqual(audit.portal_side("CL_208"), "Unknown")
        self.assertEqual(audit.portal_side("CR_Ig_216"), "R")
        self.assertEqual(audit.portal_side("CL_area_44_76"), "L")

    def test_missing_coordinates_and_sources_are_not_zero(self):
        self.assertEqual(audit.resolve_sources([]), ("missing_unassessed", None))
        for coords in ([], [1, None, 3], [float("nan"), 0, 0]):
            with self.assertRaises((ValueError, TypeError)):
                self.lookup(coords)

    def test_identity_hash_collisions_never_silently_select(self):
        a = {"source_status": "root_checked", "sha256": "abc", "root_xyz_um": [1, 2, 3]}
        b = {**a, "sha256": "def"}
        self.assertEqual(audit.resolve_sources([a, b]), ("source_hash_collision_unresolved", None))
        self.assertEqual(audit.resolve_sources([a, a])[0], "own_atlas_root_checked")
        self.assertEqual(audit.resolve_sources([{**a, "source_status": "source_invalid_unassessed"}])[1], None)
        complete = {**a, "index_xyz": [1, 2, 3], "rounded_voxel_xyz": [1, 2, 3],
                    "lookup_status": "atlas_label", "atlas_label": "CR_Ia/Id", "mask_side": "R"}
        status, consensus = audit.resolve_sources([complete, {**complete, "sha256": "different_bytes"}])
        self.assertEqual(status, "own_atlas_root_consensus_multiple_source_hashes")
        self.assertEqual(consensus["atlas_label"], "CR_Ia/Id")
        self.assertNotIn("sha256", consensus)  # no one source silently chosen
        status, consensus = audit.resolve_sources([complete, {**complete, "sha256": "different_bytes", "root_xyz_um": [4, 5, 6]}])
        self.assertEqual((status, consensus), ("source_hash_collision_unresolved", None))

    def test_fresh_root_does_not_assume_first_row_or_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "112.swc"
            path.write_text("2 3 250 500 750 1 1\n1 1 500 750 1000 1 -1\n")
            root = audit.root_from_swc(path)
            self.assertEqual(root["root_node_id"], 1)
            self.assertEqual(root["root_xyz_um"], [500, 750, 1000])
            path.write_text("1 1 0 0 0 1 -1\n1 1 0 0 0 1 -1\n")
            with self.assertRaises(ValueError):
                audit.root_from_swc(path)

    def test_reviewed_manual_status_requires_exact_sample_neuron_hash(self):
        review = {"sample_id": "251637", "swc_sha256": {"114.swc": "bound"},
                  "annotations": [{"NeuronID": "114.swc", "Soma_Region": "R-IDM", "Soma_Layer": 3,
                                   "status": "provisional IDM selected by user", "Original_Henry_Labels": ["L-IDD5", "L-IDM"]}]}
        sources = [{"source_status": "root_checked", "sha256": "bound"}]
        actual = audit.reviewed_assignment("251637", "114.swc", sources, review)
        self.assertIn("provisional", actual["reviewed_status"])
        self.assertEqual(actual["reviewed_effective_status"], "reviewed_hash_bound")
        self.assertEqual(actual["reviewed_original_labels"], ["L-IDD5", "L-IDM"])
        self.assertEqual(audit.reviewed_assignment("251637_b", "114.swc", sources, review), {})
        self.assertEqual(audit.reviewed_assignment("251637", "114", sources, review), {})
        actual = audit.reviewed_assignment("251637", "114.swc", [{**sources[0], "sha256": "changed"}], review)
        self.assertEqual(actual["reviewed_effective_status"], "reviewed_source_not_verified_unassessed")

    def test_origin_and_tie_policies_are_explicit_sensitivity(self):
        self.assertEqual(sensitivity.indices([250, 500, 750], sensitivity.POLICIES[0]).tolist(), [1, 2, 3])
        self.assertEqual(sensitivity.indices([250, 500, 750], sensitivity.POLICIES[1]).tolist(), [0, 1, 2])
        self.assertEqual(sensitivity.indices([125, 375, 625], sensitivity.POLICIES[0]).tolist(), [0, 2, 2])
        self.assertEqual(sensitivity.indices([125, 375, 625], sensitivity.POLICIES[2]).tolist(), [1, 2, 3])
        self.assertEqual(sensitivity.indices([0, 250, 500], sensitivity.POLICIES[1]).tolist(), [-1, 0, 1])
        self.assertEqual(sensitivity.indices([250.001, 500, 750], sensitivity.POLICIES[1]).tolist(), [1, 1, 2])
        with self.assertRaises(ValueError):
            sensitivity.indices([1, 2, 3], "implicit_default")


if __name__ == "__main__":
    unittest.main()
