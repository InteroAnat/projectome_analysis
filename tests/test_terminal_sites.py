"""Independent small-graph and physical-geometry contracts for candidate tips."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
from projection_maps import ReferenceGrid
from terminal_sites import terminal_site_summary


class TerminalSiteTests(unittest.TestCase):
    def setUp(self):
        self.grid = ReferenceGrid((4, 4, 4), np.eye(4))

    def summarize(self, text, grid=None, **kwargs):
        return terminal_site_summary(text, grid or self.grid, coordinate_frame="atlas_index_um",
                                     index_scale_um=[1, 1, 1], **kwargs)

    def test_unsorted_ids_and_identity_are_preserved(self):
        result = self.summarize("42 2 2 0 0 1 9\n9 2 1 0 0 1 70\n70 1 0 0 0 1 -1\n")
        leaf = result["leaves"][0]
        self.assertEqual((leaf["node_id"], leaf["node_type"], leaf["parent_id"]), (42, 2, 9))
        self.assertEqual(leaf["source_xyz"], [2, 0, 0])
        self.assertEqual(leaf["terminal_edge_length_mm"], 1)
        self.assertEqual(leaf["terminal_branch_length_mm"], 2)
        self.assertEqual(leaf["branch_start_node_id"], 70)
        self.assertEqual(leaf["soma_distance_mm"], 2)
        self.assertEqual(result["neuron_occupancy"], [[2, 0, 0]])
        json.dumps(result, allow_nan=False)

    def test_root_only_never_becomes_endpoint_even_when_type2(self):
        for node_type, state in [(1, "no_type2_nodes"), (2, "no_eligible_axon_leaf")]:
            with self.subTest(node_type=node_type):
                result = self.summarize(f"1 {node_type} 0 0 0 1 -1\n")
                leaf = result["leaves"][0]
                self.assertEqual(leaf["leaf_class"], "root_only")
                self.assertIsNone(leaf["terminal_edge_length_mm"])
                self.assertEqual(leaf["terminal_branch_length_mm"], 0)
                self.assertEqual(result["endpoint_counts"], [])
                self.assertEqual(result["qc"]["candidate_availability"], state)
                self.assertEqual(result["qc"]["biological_innervation_state"], "unassessed")

    def test_repeated_tips_same_voxel_count_twice_occupancy_once(self):
        result = self.summarize("1 1 0 0 0 1 -1\n2 2 1.1 0 0 1 1\n3 2 1.2 0 0 1 1\n")
        self.assertEqual(result["endpoint_counts"], [{"voxel": [1, 0, 0], "count": 2}])
        self.assertEqual(result["neuron_occupancy"], [[1, 0, 0]])
        self.assertEqual(result["qc"]["candidate_axon_endpoint_count"], 2)

    def test_full_graph_prevents_filter_induced_type_transition_tip(self):
        result = self.summarize("1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 3 2 0 0 1 2\n")
        self.assertEqual([leaf["node_id"] for leaf in result["leaves"]], [3])
        self.assertEqual(result["leaves"][0]["leaf_class"], "dendritic_leaf")
        self.assertEqual(result["endpoint_counts"], [])
        self.assertEqual(result["qc"]["candidate_availability"], "no_eligible_axon_leaf")

    def test_custom_compartments_preserved_and_axon_transition_retained(self):
        result = self.summarize("1 1 0 0 0 1 -1\n2 99 1 0 0 1 1\n3 4 0 1 0 1 1\n4 3 0 0 1 1 1\n5 2 1 1 0 1 3\n")
        leaves = {leaf["node_id"]: leaf for leaf in result["leaves"]}
        self.assertEqual(leaves[2]["node_type"], 99)
        self.assertEqual(leaves[2]["leaf_class"], "unresolved_leaf")
        self.assertTrue(leaves[5]["candidate_axon_endpoint"])
        self.assertTrue(leaves[5]["terminal_edge_type_transition"])
        self.assertEqual(leaves[5]["terminal_branch_length_mm"], 2)
        self.assertEqual(leaves[5]["terminal_branch_axon_length_mm"], 1)
        self.assertEqual(result["qc"]["node_type_counts"], {"1": 1, "2": 1, "3": 1, "4": 1, "99": 1})
        self.assertEqual(result["qc"]["leaf_type_counts"], {"2": 1, "3": 1, "99": 1})

    def test_boundary_half_open_assignment_and_outside_accounting(self):
        points = [-.50000001, -.5, .49999999, .5, 3.49999999, 3.5]
        text = "1 1 0 0 0 1 -1\n" + "".join(
            f"{i+2} 2 {point} 0 0 1 1\n" for i, point in enumerate(points))
        result = self.summarize(text)
        self.assertEqual([leaf["voxel"] for leaf in result["leaves"]],
                         [None, [0, 0, 0], [0, 0, 0], [1, 0, 0], [3, 0, 0], None])
        self.assertEqual(result["qc"]["outside_reference_candidate_count"], 2)
        self.assertEqual(result["qc"]["candidate_axon_endpoints"], {"total": 6, "in_reference": 4, "outside_reference": 2})
        self.assertEqual(sum(item["count"] for item in result["endpoint_counts"]), 4)

    def test_immediately_adjacent_float_below_face_stays_in_lower_voxel(self):
        points = [np.nextafter(.5, -np.inf), .5, np.nextafter(.5, np.inf)]
        text = "1 1 0 0 0 1 -1\n" + "".join(
            f"{i+2} 2 {float(point)!r} 0 0 1 1\n"
            for i, point in enumerate(points))
        result = self.summarize(text)
        self.assertEqual([leaf["voxel"] for leaf in result["leaves"]],
                         [[0, 0, 0], [1, 0, 0], [1, 0, 0]])

    def test_zero_length_and_coincident_tips_are_retained(self):
        result = self.summarize("1 1 1 1 1 1 -1\n2 2 1 1 1 1 1\n3 2 1 1 1 1 1\n")
        self.assertEqual(result["qc"]["zero_length_candidate_terminal_edges"], 2)
        self.assertEqual(result["endpoint_counts"][0]["count"], 2)
        for leaf in result["leaves"]:
            self.assertEqual(leaf["coincident_leaf_count"], 2)
            self.assertEqual(leaf["soma_distance_mm"], 0)
            self.assertEqual(leaf["image_review_state"], "unresolved")

    def test_all_candidates_outside_is_distinct_from_no_candidates(self):
        result = self.summarize("1 1 0 0 0 1 -1\n2 2 10 0 0 1 1\n")
        self.assertEqual(result["endpoint_counts"], [])
        self.assertEqual(result["neuron_occupancy"], [])
        self.assertEqual(result["qc"]["candidate_status"], "candidate_endpoints_present")
        self.assertEqual(result["qc"]["candidate_axon_endpoints"],
                         {"total": 1, "in_reference": 0, "outside_reference": 1})
        self.assertEqual(result["qc"]["biological_innervation_state"], "unassessed")

    def test_densifying_terminal_branch_does_not_change_leaf_or_length(self):
        sparse = self.summarize("1 1 0 0 0 1 -1\n9 2 3 0 0 1 1\n")
        dense = self.summarize("1 1 0 0 0 1 -1\n4 2 1 0 0 1 1\n5 2 2 0 0 1 4\n9 2 3 0 0 1 5\n")
        self.assertEqual(sparse["endpoint_counts"], dense["endpoint_counts"])
        self.assertEqual(sparse["leaves"][0]["node_id"], dense["leaves"][0]["node_id"])
        self.assertEqual(sparse["leaves"][0]["terminal_branch_length_mm"],
                         dense["leaves"][0]["terminal_branch_length_mm"])

    def test_terminal_branch_stops_at_full_graph_bifurcation(self):
        result = self.summarize("1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 2 2 0 0 1 2\n4 2 3 0 0 1 3\n5 99 1 1 0 1 2\n")
        leaf = next(leaf for leaf in result["leaves"] if leaf["node_id"] == 4)
        self.assertEqual(leaf["branch_start_node_id"], 2)
        self.assertEqual(leaf["terminal_branch_length_mm"], 2)
        self.assertEqual(leaf["soma_distance_mm"], 3)

    def test_coordinate_equivalence_and_sheared_physical_geometry(self):
        affine = np.array([[2, 1, 0, -5], [0, 3, 0, 8], [0, 0, -4, 2], [0, 0, 0, 1]], dtype=float)
        grid = ReferenceGrid((4, 4, 4), affine)
        indexed = terminal_site_summary("1 1 0 0 0 1 -1\n2 2 250 500 250 1 1\n", grid,
                                        coordinate_frame="atlas_index_um", index_scale_um=[250, 500, 250])
        literal = terminal_site_summary("1 1 -5 8 2 1 -1\n2 2 -2 11 -2 1 1\n", grid,
                                       coordinate_frame="nifti_world_mm")
        for field in ("index_xyz", "world_mm", "soma_distance_mm", "terminal_edge_length_mm"):
            np.testing.assert_allclose(indexed["leaves"][0][field], literal["leaves"][0][field])
        # Direct world-space displacement (3,3,-4) gives sqrt(34), independent
        # of implementation's index-space affine calculation.
        self.assertAlmostEqual(indexed["leaves"][0]["soma_distance_mm"], np.sqrt(34))
        self.assertEqual(indexed["endpoint_counts"], literal["endpoint_counts"])

    def test_soma_anchor_missing_or_ambiguous_is_explicit(self):
        for text, state in [("1 3 0 0 0 1 -1\n2 2 1 0 0 1 1\n", "no_type1_node"),
                            ("1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n3 2 2 0 0 1 2\n", "ambiguous_multiple_type1_nodes")]:
            with self.subTest(state=state):
                result = self.summarize(text)
                self.assertIsNone(result["leaves"][0]["soma_distance_mm"])
                self.assertEqual(result["qc"]["soma_anchor_state"], state)
                self.assertGreater(result["leaves"][0]["root_distance_mm"], 0)

    def test_invalid_graphs_fail_before_measurement(self):
        cases = ["", "1 1 0 0 0 1 -1\n1 2 1 0 0 1 1\n",
                 "1 1 0 0 0 1 -1\n2 2 1 0 0 1 8\n",
                 "1 1 0 0 0 1 -1\n2 2 1 0 0 1 3\n3 2 2 0 0 1 2\n",
                 "1 1 0 0 0 1 -1\n2 2 nan 0 0 1 1\n",
                 "1 1 0 0 0 1 -1\n2 -2 1 0 0 1 1\n"]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.summarize(text)

    def test_coordinate_contract_is_mandatory(self):
        text = "1 1 0 0 0 1 -1\n"
        for kwargs in [dict(coordinate_frame="unknown"), dict(coordinate_frame="atlas_index_um"),
                       dict(coordinate_frame="atlas_index_um", index_scale_um=[1, 0, 1]),
                       dict(coordinate_frame="nifti_world_mm", index_scale_um=[1, 1, 1])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                terminal_site_summary(text, self.grid, **kwargs)


if __name__ == "__main__":
    unittest.main()
