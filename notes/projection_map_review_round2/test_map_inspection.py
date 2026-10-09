"""Failure cases for independent census and single-plane marker semantics."""
import unittest
import numpy as np
from inspect_projection_maps import check_values, endpoint_indices
from render_endpoint_markers import slice_markers


class MapInspectionTests(unittest.TestCase):
    def test_crop_does_not_create_false_endpoint(self):
        rows = np.array([[1, 1, 0, 0, 0, 1, -1], [2, 2, 1, 1, 1, 1, 1], [3, 2, 20, 1, 1, 1, 2]], dtype=float)
        number, indices = endpoint_indices(rows, np.ones(3), (4, 4, 4))
        self.assertEqual(number, 1)
        self.assertEqual(len(indices), 0)

    def test_nonaxon_child_prevents_axon_leaf(self):
        rows = np.array([[1, 1, 0, 0, 0, 1, -1], [2, 2, 1, 1, 1, 1, 1], [3, 3, 2, 1, 1, 1, 2]], dtype=float)
        self.assertEqual(endpoint_indices(rows, np.ones(3), (4, 4, 4))[0], 0)

    def test_multiple_ends_share_voxel_without_losing_count(self):
        rows = np.array([[1, 1, 0, 0, 0, 1, -1], [2, 2, 1, 1, 1, 1, 1], [3, 2, 1.1, 1, 1, 1, 1]], dtype=float)
        count, indices = endpoint_indices(rows, np.ones(3), (4, 4, 4))
        self.assertEqual(count, 2)
        self.assertEqual(len(np.unique(indices)), 1)

    def test_corrupt_normalization_or_occupancy_fails(self):
        count = np.array([0, 2.0])
        check_values(count, count / .015625, np.array([0, .5]), .015625)
        for density, occupancy in [(count, np.array([0, .5])), (count / .015625, np.array([0, 1.1])), (count / .015625, np.array([.5, .5]))]:
            with self.assertRaises(ValueError):
                check_values(count, density, occupancy, .015625)

    def test_dot_is_single_plane_voxel_centre_no_projection(self):
        values = np.zeros((4, 5, 6))
        values[1, 2, 3] = 5
        values[1, 2, 4] = 10
        affine = np.diag([.25, .25, .25, 1]); affine[:3, 3] = [-1, -2, -3]
        x, y, colour = slice_markers(values, 2, 3, affine)
        np.testing.assert_array_equal(x, [-.75])
        np.testing.assert_array_equal(y, [-1.5])
        np.testing.assert_array_equal(colour, [5])


if __name__ == '__main__':
    unittest.main()
