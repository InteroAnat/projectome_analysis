"""Independent connected-component oracle and scientific accounting guards."""
import math
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import census_connected_axon_sections as method


class CensusValidation(unittest.TestCase):
    def test_random_tree_components_match_sparse_undirected_oracle(self):
        random = np.random.default_rng(20261009)
        for _ in range(80):
            count = 60
            parents = [-1] + [int(random.integers(1, i)) for i in range(2, count+1)]
            kinds = random.choice([0, 1, 2, 2, 2, 3, 7], count)
            xyz = random.normal(size=(count, 3))
            rows = [(i+1, int(kinds[i]), *xyz[i], 1., parents[i]) for i in range(count)]
            labels = dict(zip(range(1, count+1), random.choice([-1, 0, 10, 20], count).tolist()))
            random.shuffle(rows)
            records, membership, _, _, summary = method.analyze(rows, labels, np.eye(3))
            axon_ids = sorted(row[0] for row in rows if row[1] == 2)
            positions = {node: i for i, node in enumerate(axon_ids)}
            links = [(positions[row[0]], positions[row[6]]) for row in rows
                     if row[0] in positions and row[6] in positions and labels[row[0]] == labels[row[6]]]
            adjacency = np.zeros((len(axon_ids), len(axon_ids)), dtype=int)
            for a, b in links:
                adjacency[a, b] = adjacency[b, a] = 1
            if axon_ids:
                _, partition = connected_components(csr_matrix(adjacency), directed=False)
                expected = {frozenset(node for node, label in zip(axon_ids, partition) if label == group)
                            for group in set(partition)}
                self.assertEqual(expected, {frozenset(record["original_node_ids"]) for record in membership})
            parent_set = {row[6] for row in rows}
            ends = {row[0] for row in rows if row[1] == 2 and row[6] != -1 and row[0] not in parent_set}
            self.assertEqual(ends, {node for record in membership for node in record["original_axon_end_ids"]})
            lookup = {row[0]: row for row in rows}
            expected_length = math.fsum(math.dist(row[2:5], lookup[row[6]][2:5]) for row in rows if row[1] == 2 and row[6] != -1)
            self.assertAlmostEqual(summary["total_child_type2_edge_length_NMT_mm"], expected_length, places=11)
            self.assertEqual(sum(record["node_count"] for record in records), len(axon_ids))

    def test_type_transition_and_label_cut_do_not_create_endings(self):
        rows = [(1, 1, 0., 0., 0., 1., -1), (2, 2, 1., 0., 0., 1., 1),
                (3, 2, 2., 0., 0., 1., 2), (4, 7, 3., 0., 0., 1., 3),
                (5, 2, 1., 1., 0., 1., 2), (6, 2, 1., 2., 0., 1., 5)]
        labels = {1: 0, 2: 10, 3: 20, 4: 20, 5: 10, 6: 10}
        records, members, links, _, totals = method.analyze(rows, labels, np.eye(3))
        self.assertEqual(totals["original_axon_end_count"], 1)
        self.assertEqual({node for m in members for node in m["original_axon_end_ids"]}, {6})
        self.assertEqual(next(r for r in records if r["section_root_node_id"] == 3)["nonleaf_nodes_with_no_axon_children_count"], 1)
        self.assertEqual(totals["target_crossing_edge_count"], 1)
        self.assertEqual(totals["type_boundary_edge_count"], 1)
        self.assertEqual({r["crossing_reason"] for r in links}, {"target_label_crossing", "nonaxon_to_axon_type_boundary"})

    def test_missing_native_remains_null_but_observed_uncovered_is_zero(self):
        rows = [(1, 1, 0., 0., 0., 1., -1), (2, 2, 1., 0., 0., 1., 1)]
        labels = {1: 0, 2: 0}
        records, _, _, _, totals = method.analyze(rows, labels, np.eye(3))
        self.assertIsNone(totals["original_axon_ends_with_cached_cube"])
        self.assertIsNone(records[0]["internal_edge_length_native_nominal_um"])
        native = {1: np.array([0., 0., 0.]), 2: np.array([234., 0., 0.])}
        records, _, _, cubes, totals = method.analyze(rows, labels, np.eye(3), native=native, covered_cube_indices=set())
        self.assertEqual(totals["original_axon_ends_with_cached_cube"], 0)
        self.assertEqual(cubes[0]["cube_x"], 1)
        self.assertEqual(records[0]["ending_cube_coverage_fraction"], 0.)

    def test_reference_linear_geometry_and_three_way_length_partition(self):
        rows = [(1, 1, 0., 0., 0., 1., -1), (2, 2, 1., 0., 0., 1., 1),
                (3, 2, 1., 1., 0., 1., 2), (4, 2, 1., 1., 1., 1., 3)]
        labels = {1: 0, 2: 10, 3: 10, 4: 20}
        linear = np.array([[2., 1., 0.], [0., 3., 0.], [0., 0., 4.]])
        _, _, _, _, totals = method.analyze(rows, labels, linear)
        self.assertAlmostEqual(totals["type_boundary_edge_length_NMT_mm"], 2.)
        self.assertAlmostEqual(totals["internal_section_edge_length_NMT_mm"], math.sqrt(10))
        self.assertAlmostEqual(totals["target_crossing_edge_length_NMT_mm"], 4.)

    def test_root_only_and_no_axon_do_not_acquire_an_ending(self):
        for kind, sections in [(2, 1), (1, 0), (7, 0)]:
            result = method.analyze([(1, kind, 0., 0., 0., 1., -1)], {1: 0}, np.eye(3))
            self.assertEqual(result[-1]["original_axon_end_count"], 0)
            self.assertEqual(len(result[0]), sections)

    def test_current_rint_exact_half_ties_and_outside(self):
        rows = [(1, 2, 125., 0., 0., 1., -1), (2, 2, 375., 0., 0., 1., 1),
                (3, 2, -126., 0., 0., 1., 2)]
        self.assertEqual(method.sample_labels(rows, np.arange(4).reshape(4, 1, 1), np.array([250.]*3)), {1: 0, 2: 2, 3: -1})

    def test_existing_output_fails_before_reading_inputs(self):
        with tempfile.TemporaryDirectory(dir=method.HERE) as directory:
            with self.assertRaises(FileExistsError):
                method.build(method.Path(directory))


if __name__ == "__main__":
    unittest.main()
