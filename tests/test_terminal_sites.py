"""Independent small-graph and physical-geometry contracts for candidate tips."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
from projection_maps import ReferenceGrid, axon_length_map
from terminal_sites import terminal_site_summary, reconstructed_axon_end_branch_summary


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


class AxonEndBranchTests(unittest.TestCase):
    def setUp(self):
        self.grid = ReferenceGrid((4, 4, 4), np.eye(4))

    def summarize(self, text, grid=None):
        return reconstructed_axon_end_branch_summary(
            text, grid or self.grid, coordinate_frame="atlas_index_um", index_scale_um=[1, 1, 1])

    @staticmethod
    def dense_map(result, shape=(4, 4, 4)):
        data = np.zeros(shape)
        for record in result["voxel_lengths_mm"]:
            data[tuple(record["voxel"])] = record["length_mm"]
        return data

    @staticmethod
    def slab_oracle(start, end, shape, linear):
        """Intersect every cell independently; no production voxel traversal."""
        data = np.zeros(shape)
        delta = np.asarray(end) - start
        length = math.sqrt(sum(float(x) ** 2 for x in linear @ delta))
        for voxel in np.ndindex(shape):
            lower, upper = 0., 1.
            for axis, center in enumerate(voxel):
                if delta[axis] == 0:
                    if not center - .5 <= start[axis] < center + .5:
                        upper = -1.
                        break
                else:
                    a = (center - .5 - start[axis]) / delta[axis]
                    b = (center + .5 - start[axis]) / delta[axis]
                    lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
            if upper > lower:
                data[voxel] = length * (upper - lower)
        return data

    def test_original_chain_is_distributed_along_trajectory_and_densification_invariant(self):
        sparse = self.summarize("1 1 0 0 0 1 -1\n9 2 3 0 0 1 1\n")
        dense = self.summarize("9 2 3 0 0 1 5\n1 1 0 0 0 1 -1\n5 2 2 0 0 1 4\n4 2 1 0 0 1 1\n")
        np.testing.assert_allclose(self.dense_map(sparse), self.dense_map(dense))
        np.testing.assert_allclose(self.dense_map(sparse)[:, 0, 0], [.5, 1, 1, .5])
        self.assertEqual(dense["branches"][0]["selected_child_node_ids"], [9, 5, 4])
        self.assertEqual(dense["branches"][0]["start_node_id"], 1)
        self.assertEqual(dense["qc"]["total_length_mm"], 3)
        self.assertTrue(dense["branches"][0]["final_transition_edge_included"])
        json.dumps(dense, allow_nan=False)

    def test_full_graph_bifurcation_excludes_proximal_trunk_even_with_custom_child(self):
        text = "1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 2 2 0 0 1 2\n4 2 3 0 0 1 3\n5 99 1 1 0 1 2\n"
        result = self.summarize(text)
        self.assertEqual([edge["child_node_id"] for edge in result["selected_edges"]], [3, 4])
        self.assertEqual(result["branches"][0]["start_node_id"], 2)
        self.assertTrue(result["branches"][0]["stop_at_full_graph_bifurcation"])
        self.assertEqual(result["qc"]["total_length_mm"], 2)

    def test_compartment_boundary_edge_retained_but_nonaxon_chain_is_never_bridged(self):
        text = "1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 7 2 0 0 1 2\n4 2 3 0 0 1 3\n"
        result = self.summarize(text)
        self.assertEqual([edge["child_node_id"] for edge in result["selected_edges"]], [4])
        self.assertEqual(result["branches"][0]["stop_reason"], "nonaxon_parent")
        self.assertEqual(result["branches"][0]["start_node_type"], 7)
        self.assertEqual(result["qc"]["final_transition_edge_count"], 1)
        self.assertEqual(result["qc"]["total_length_mm"], 1)
        old = terminal_site_summary(text, self.grid, coordinate_frame="atlas_index_um", index_scale_um=[1, 1, 1])
        self.assertEqual(old["leaves"][0]["terminal_branch_axon_length_mm"], 2)
        self.assertEqual(old["leaves"][0]["terminal_branch_length_mm"], 3)

    def test_no_original_end_is_missing_not_zero_length(self):
        for text in ["1 2 0 0 0 1 -1\n", "1 1 0 0 0 1 -1\n",
                     "1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 3 2 0 0 1 2\n"]:
            result = self.summarize(text)
            self.assertFalse(result["qc"]["computable"])
            self.assertIsNone(result["qc"]["total_length_mm"])
            self.assertIsNone(result["qc"]["in_reference_length_mm"])
            self.assertEqual(result["voxel_lengths_mm"], [])
            self.assertEqual(result["selected_edges"], [])
            self.assertEqual(result["qc"]["biological_innervation_state"], "unassessed")

    def test_zero_length_and_outside_branches_remain_computable(self):
        zero = self.summarize("1 1 0 0 0 1 -1\n2 2 0 0 0 1 1\n")
        self.assertTrue(zero["qc"]["computable"])
        self.assertEqual(zero["qc"]["total_length_mm"], 0)
        self.assertEqual(zero["qc"]["zero_length_selected_edges"], 1)
        self.assertEqual(zero["voxel_lengths_mm"], [])
        outside = self.summarize("1 1 8 0 0 1 -1\n2 2 10 0 0 1 1\n")
        self.assertTrue(outside["qc"]["computable"])
        self.assertEqual(outside["qc"]["in_reference_length_mm"], 0)
        self.assertEqual(outside["qc"]["outside_reference_length_mm"], 2)

    def test_two_outside_endpoints_still_allocate_in_reference_chain(self):
        result = self.summarize("1 1 -2 0 0 1 -1\n2 2 6 0 0 1 1\n")
        np.testing.assert_allclose(self.dense_map(result)[:, 0, 0], [1, 1, 1, 1])
        self.assertEqual(result["qc"]["total_length_mm"], 8)
        self.assertEqual(result["qc"]["in_reference_length_mm"], 4)
        self.assertEqual(result["qc"]["outside_reference_length_mm"], 4)

    def test_shared_leaf_voxel_does_not_duplicate_proximal_edges(self):
        text = "1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 2 2.1 0 0 1 2\n4 2 2.2 0 0 1 2\n"
        result = self.summarize(text)
        self.assertEqual(result["qc"]["branch_count"], 2)
        self.assertEqual([edge["child_node_id"] for edge in result["selected_edges"]], [3, 4])
        self.assertAlmostEqual(result["qc"]["total_length_mm"], 2.3)
        self.assertAlmostEqual(sum(branch["length_mm"] for branch in result["branches"]), 2.3)
        whole, _ = axon_length_map(text, self.grid, coordinate_frame="atlas_index_um", index_scale_um=[1]*3)
        self.assertTrue(np.all(self.dense_map(result) <= whole + 1e-12))

    def test_affine_physical_lengths_and_world_coordinate_equivalence(self):
        affine = np.array([[2, 1, 0, -5], [0, 3, 0, 8], [0, 0, -4, 2], [0, 0, 0, 1]], dtype=float)
        grid = ReferenceGrid((4, 4, 4), affine)
        indexed = reconstructed_axon_end_branch_summary("1 1 0 0 0 1 -1\n2 2 250 500 250 1 1\n", grid,
                    coordinate_frame="atlas_index_um", index_scale_um=[250, 500, 250])
        world = reconstructed_axon_end_branch_summary("1 1 -5 8 2 1 -1\n2 2 -2 11 -2 1 1\n", grid,
                    coordinate_frame="nifti_world_mm")
        self.assertAlmostEqual(indexed["qc"]["total_length_mm"], math.sqrt(34))
        np.testing.assert_allclose(self.dense_map(indexed), self.dense_map(world))

    def test_upper_face_parallel_edge_is_outside_and_lower_face_is_inside(self):
        cases = [(-.5, 1, 0), (3.5, 0, None), (.5, 1, 1),
                 (np.nextafter(.5, -np.inf), 1, 0),
                 (np.nextafter(3.5, -np.inf), 1, 3),
                 (np.nextafter(-.5, -np.inf), 0, None)]
        for x, expected, voxel_x in cases:
            result = self.summarize(f"1 1 {x!r} 0 0 1 -1\n2 2 {x!r} 1 0 1 1\n")
            self.assertEqual(result["qc"]["in_reference_length_mm"], expected)
            if voxel_x is not None:
                self.assertEqual({item["voxel"][0] for item in result["voxel_lengths_mm"]}, {voxel_x})

    def test_seeded_forward_graph_selection_and_independent_cell_slabs(self):
        random = np.random.default_rng(20261009)
        affine = np.array([[.5, .1, 0, 4], [0, .8, 0, -3], [0, 0, -1.2, 5], [0, 0, 0, 1.]])
        grid = ReferenceGrid((4, 4, 4), affine)
        for _ in range(128):
            count = 19
            parents = np.array([-1] + [int(random.integers(1, node)) for node in range(2, count+1)])
            kinds = random.choice([1, 2, 2, 2, 3, 7], count)
            xyz = random.uniform(-1.5, 5, size=(count, 3))
            children = {node: np.flatnonzero(parents == node) + 1 for node in range(1, count+1)}
            # Independent forward criterion: an edge is selected when its axon
            # child can follow exactly one original child at each step through
            # axon nodes to a full-graph leaf, without any bifurcation.
            expected_ids = set()
            for node in range(2, count+1):
                current = node
                while kinds[current-1] == 2 and len(children[current]) == 1:
                    current = children[current][0]
                if kinds[current-1] == 2 and len(children[current]) == 0:
                    expected_ids.add(node)
            rows = [f"{node} {kinds[node-1]} {' '.join(map(repr, xyz[node-1].tolist()))} 1 {parents[node-1]}"
                    for node in range(1, count+1)]
            random.shuffle(rows)
            text = "\n".join(rows)
            result = self.summarize(text, grid)
            actual_ids = [edge["child_node_id"] for edge in result["selected_edges"]]
            self.assertEqual(set(actual_ids), expected_ids)
            self.assertEqual(len(actual_ids), len(set(actual_ids)))
            expected_map = np.zeros(grid.shape)
            expected_total = 0.
            for node in expected_ids:
                start, end = xyz[parents[node-1]-1], xyz[node-1]
                expected_map += self.slab_oracle(start, end, grid.shape, affine[:3, :3])
                expected_total += math.sqrt(sum(float(x)**2 for x in affine[:3, :3] @ (end-start)))
            np.testing.assert_allclose(self.dense_map(result), expected_map, rtol=1e-11, atol=1e-11)
            if expected_ids:
                self.assertAlmostEqual(result["qc"]["total_length_mm"], expected_total, places=10)
                self.assertAlmostEqual(result["qc"]["in_reference_length_mm"], float(expected_map.sum()), places=10)
            whole, _ = axon_length_map(text, grid, coordinate_frame="atlas_index_um", index_scale_um=[1]*3)
            self.assertTrue(np.all(self.dense_map(result) <= whole + 1e-10))

    def test_bad_graph_and_coordinate_contract_fail_closed(self):
        for text in ["", "1 1 0 0 0 1 -1\n2 2 1 0 0 1 99\n",
                     "1 1 0 0 0 1 -1\n2 2 1 0 0 1 3\n3 2 2 0 0 1 2\n"]:
            with self.assertRaises(ValueError):
                self.summarize(text)
        for kwargs in [dict(coordinate_frame="unknown"), dict(coordinate_frame="atlas_index_um"),
                       dict(coordinate_frame="nifti_world_mm", index_scale_um=[1, 1, 1])]:
            with self.assertRaises(ValueError):
                reconstructed_axon_end_branch_summary("1 1 0 0 0 1 -1\n", self.grid, **kwargs)


if __name__ == "__main__":
    unittest.main()
