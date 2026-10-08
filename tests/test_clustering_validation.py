"""Regression checks for identity, missing distances and clustering stability."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
from clustering_validation import (
    evaluate_k, read_fnt_distances, safe_linkage, validate_distance_matrix,
)


class FNTLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.joined = self.root / "joined.fnt"
        # Marker order deliberately differs from lexical and natural sorting.
        self.joined.write_text("0 Neuron m_010\n4 Neuron m_002\n9 Neuron m_001\n")
        self.dist = self.root / "dist.txt"

    def write_pairs(self, pairs):
        self.dist.write_text("I\tJ\tScore\tMatch\tNoMatch\n" + "".join(
            f"{i}\t{j}\t{score}\t10\t0\n" for i, j, score in pairs
        ))

    def triangle(self):
        return [(0, 0, .3), (1, 1, .2), (2, 2, .1),
                (0, 1, 5), (0, 2, 8), (1, 2, 2)]

    def test_triangular_scores_preserve_joined_marker_order(self):
        self.write_pairs(list(reversed(self.triangle())))
        matrix, metadata = read_fnt_distances(self.dist, self.joined)
        self.assertEqual(list(matrix.index), ["m_010", "m_002", "m_001"])
        np.testing.assert_equal(matrix.values, [[0, 5, 8], [5, 0, 2], [8, 2, 0]])
        self.assertAlmostEqual(metadata["raw_self_score_max"], .3)
        self.assertEqual(len(metadata["distance_sha256"]), 64)

    def test_reverse_direction_reduction_is_explicit(self):
        self.write_pairs(self.triangle() + [(1, 0, 7), (2, 0, 4), (2, 1, 6)])
        maximum, meta = read_fnt_distances(self.dist, self.joined)
        mean, _ = read_fnt_distances(self.dist, self.joined, "mean")
        self.assertEqual(maximum.iloc[1, 0], 7)
        self.assertEqual(mean.iloc[1, 0], 6)
        self.assertEqual(meta["max_directional_difference"], 4)

    def test_missing_pair_is_rejected_instead_of_imputed_zero(self):
        self.write_pairs(self.triangle()[:-1])
        with self.assertRaisesRegex(ValueError, "Missing FNT pairs"):
            read_fnt_distances(self.dist, self.joined)

    def test_missing_self_pair_is_rejected(self):
        self.write_pairs(self.triangle()[1:])
        with self.assertRaisesRegex(ValueError, "Missing FNT pairs"):
            read_fnt_distances(self.dist, self.joined)

    def test_duplicate_pair_noninteger_and_bad_score_are_rejected(self):
        for extra in [(0, 1, 5), (.5, 1, 5), (0, 2, float("inf")), (0, 2, -1)]:
            with self.subTest(extra=extra):
                self.write_pairs(self.triangle() + [extra])
                with self.assertRaises(ValueError):
                    read_fnt_distances(self.dist, self.joined)

    def test_index_name_count_and_duplicate_marker_mismatch_are_rejected(self):
        self.write_pairs(self.triangle() + [(0, 3, 1)])
        with self.assertRaisesRegex(ValueError, "neuron count"):
            read_fnt_distances(self.dist, self.joined)
        self.write_pairs(self.triangle())
        self.joined.write_text("0 Neuron a\n1 Neuron a\n2 Neuron c\n")
        with self.assertRaisesRegex(ValueError, "unique neuron markers"):
            read_fnt_distances(self.dist, self.joined)


class DiagnosticTests(unittest.TestCase):
    def distances(self):
        points = np.r_[np.arange(6)[:, None] * .01,
                       10 + np.arange(6)[:, None] * .01]
        return pd.DataFrame(squareform(pdist(points)),
                            index=list("abcdefghijkl"), columns=list("abcdefghijkl"))

    def test_matrix_and_ward_geometry_checks(self):
        matrix = self.distances()
        with self.assertRaisesRegex(ValueError, "Ward"):
            safe_linkage(matrix, "ward")
        for defect in ["asymmetric", "nan", "negative", "diagonal", "identity"]:
            with self.subTest(defect=defect):
                bad = matrix.copy()
                if defect == "identity":
                    bad.columns = list(reversed(bad.columns))
                elif defect == "asymmetric":
                    bad.iloc[0, 1] = 99
                elif defect == "nan":
                    bad.iloc[0, 1] = np.nan
                elif defect == "negative":
                    bad.iloc[0, 1] = -1
                else:
                    bad.iloc[0, 0] = 1
                with self.assertRaises(ValueError):
                    validate_distance_matrix(bad)

    def test_separated_groups_have_stable_reproducible_partition(self):
        matrix = self.distances()
        result = evaluate_k(matrix, [2, 3], repeats=25, seed=14)
        repeated = evaluate_k(matrix, [2, 3], repeats=25, seed=14)
        pd.testing.assert_frame_equal(result[0], repeated[0])
        self.assertEqual(result[0].iloc[0]["realized_k"], 2)
        self.assertGreater(result[0].iloc[0]["silhouette"], .99)
        self.assertEqual(result[0].iloc[0]["subsample_ari_mean"], 1)
        self.assertTrue((result[2][2]["jaccard_mean"] == 1).all())

    def test_identical_distances_do_not_claim_requested_k_or_c_index_optimum(self):
        matrix = np.ones((12, 12)) - np.eye(12)
        diagnostics, labels, _, _ = evaluate_k(matrix, [2, 3], repeats=5)
        self.assertTrue((diagnostics["realized_k"] == 1).all())
        self.assertTrue(diagnostics["silhouette"].isna().all())
        self.assertTrue(diagnostics["c_index"].isna().all())
        self.assertEqual(len(set(labels[3])), 1)

    def test_bad_sampling_parameters_and_group_identity_fail(self):
        matrix = self.distances()
        for kwargs in [{"repeats": 0}, {"fraction": 1}, {"k_values": [9]},
                       {"k_values": [2, 2]}, {"k_values": [2.1]},
                       {"groups": ["a"]}, {"groups": [None] * 12}]:
            args = {"k_values": [2], "repeats": 3, **kwargs}
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    evaluate_k(matrix, **args)
        groups = pd.Series(["a"] * 6 + ["b"] * 6, index=list(reversed(matrix.index)))
        with self.assertRaisesRegex(ValueError, "Group index"):
            evaluate_k(matrix, [2], repeats=3, groups=groups)

    def test_group_holdout_reports_remaining_cohort_sensitivity(self):
        matrix = self.distances()
        groups = pd.Series(["a", "b"] * 6, index=matrix.index)
        diagnostics, _, _, holdouts = evaluate_k(matrix, [2], repeats=3, groups=groups)
        self.assertEqual(set(holdouts["held_out_group"]), {"a", "b"})
        self.assertTrue((holdouts["remaining_n"] == 6).all())
        self.assertTrue((holdouts["ari"] == 1).all())
        self.assertIn("sample_cluster_ari", diagnostics)

    def test_singleton_stability_is_reported_as_unassessed(self):
        points = np.r_[np.arange(11) * .001, 100.0][:, None]
        diagnostics, _, stability, _ = evaluate_k(squareform(pdist(points)), [2], repeats=5)
        self.assertEqual(diagnostics.iloc[0]["jaccard_unassessed_clusters"], 1)
        self.assertEqual(diagnostics.iloc[0]["jaccard_assessed_clusters"], 1)
        self.assertAlmostEqual(diagnostics.iloc[0]["largest_cluster_fraction"], 11 / 12)
        self.assertTrue(stability[2].loc[stability[2].n == 1, "jaccard_mean"].isna().all())


if __name__ == "__main__":
    unittest.main()
