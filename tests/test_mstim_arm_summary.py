"""Counterexamples for common-coverage structural/functional summaries."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "group_analysis/scripts"))
from summarize_mstim_arm import covered_projection, covered_summary


class CommonCoverageTests(unittest.TestCase):
    def test_measured_zero_is_distinct_from_uncovered_region(self):
        labels = np.array([1, 1, 2])
        result = covered_summary(np.zeros(3), np.zeros(3), np.array([1, 0, 0]), labels)
        self.assertEqual(result[1]["mean_signed_contrast"], 0.0)
        self.assertEqual(result[1]["coverage_fraction"], 0.5)
        self.assertIsNone(result[2]["mean_signed_contrast"])
        self.assertIsNone(result[2]["median_T_descriptive_only"])

    def test_signed_response_and_actual_mask_are_preserved(self):
        result = covered_summary(np.array([-4., 2., 99.]), np.array([-5., 3., 99.]),
                                 np.array([1, 1, 0]), np.ones(3, dtype=int))
        self.assertEqual(result[1]["mean_signed_contrast"], -1.0)
        self.assertEqual(result[1]["median_T_descriptive_only"], -1.0)

    def test_nonfinite_inside_coverage_fails_and_outside_is_missing(self):
        for coverage in (np.array([1, 0]), np.array([2, 0])):
            with self.assertRaises(ValueError):
                covered_summary(np.array([np.nan, 1.]), np.zeros(2), coverage, np.ones(2))
        result = covered_summary(np.array([0., np.nan]), np.zeros(2), np.array([1, 0]), np.ones(2))
        self.assertEqual(result[1]["mean_signed_contrast"], 0.)

    def test_additive_projection_sum_uses_common_coverage(self):
        values = covered_projection(np.array([2., 3., 7., 19.]), np.array([1, 1, 2, 2]), np.array([1, 1, 1, 0]))
        np.testing.assert_array_equal(values, [0., 5., 7.])

    def test_projection_invalid_mask_negative_value_and_shape_fail(self):
        for values, coverage in ((np.array([-1., 2.]), np.ones(2)), (np.ones(2), np.array([1, 2]))):
            with self.assertRaises(ValueError):
                covered_projection(values, np.ones(2, dtype=int), coverage)
        with self.assertRaises(ValueError):
            covered_summary(np.zeros(2), np.zeros(2), np.ones(2), np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()
