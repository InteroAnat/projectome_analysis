"""Geometry and full-graph guards for the isolated descriptive pilot."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

import assess_native_terminal_fields as method
import assess_native_terminal_fields_current_rint as current


class PilotGuards(unittest.TestCase):
    def test_cube_boundaries_and_zyx_axes(self):
        self.assertEqual(method.cube_index([234, 468, 270]), (1, 2, 1))
        self.assertEqual(method.cube_index([233.9999, 468, 269.9999]), (0, 2, 0))
        source = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        xyz = source.transpose(2, 1, 0)
        self.assertEqual(xyz[3, 2, 1], source[1, 2, 3])

    def test_compartment_and_target_cuts_do_not_manufacture_endings(self):
        rows = np.array([
            [4, 7, 3, 0, 0, 1, 3], [1, 1, 0, 0, 0, 1, -1],
            [2, 2, 1, 0, 0, 1, 1], [3, 2, 2, 0, 0, 1, 2],
            [5, 2, 1, 1, 0, 1, 2], [6, 2, 1, 2, 0, 1, 5],
        ], dtype=float)
        _, _, leaves = method.graph_information(rows)
        self.assertEqual(leaves, {6})  # Node 3 has an unresolved-type child.
        labels = {1: 0, 2: 10, 3: 10, 4: 10, 5: 20, 6: 10}
        components = method.target_components(rows, labels, 10)
        self.assertEqual({frozenset(c) for c in components}, {frozenset({2, 3}), frozenset({6})})
        self.assertEqual(set(components[0]) & leaves, set())

    def test_pairing_requires_original_parent_and_type(self):
        rows = np.array([[2, 2, 10, 20, 30, 1, 1], [1, 1, 0, 0, 0, 1, -1]], float)
        warped = rows[::-1].copy()
        warped[:, 2:5] += 3000
        self.assertEqual(method.identity_signature(rows), method.identity_signature(warped))
        warped[0, 1] = 2
        self.assertNotEqual(method.identity_signature(rows), method.identity_signature(warped))

    def test_atlas_sampling_non_origin_equivalence_claim(self):
        rows = np.array([[1, 2, 125, 0, 0, 1, -1], [2, 2, -126, 0, 0, 1, 1]], float)
        atlas = np.arange(4).reshape(4, 1, 1)
        labels = method.atlas_labels(rows, atlas)
        self.assertEqual(labels, {1: 1, 2: -1})

    def test_current_policy_retains_ties_to_even_and_records_sensitivity(self):
        rows = np.array([[1, 2, 125, 0, 0, 1, -1], [2, 2, 375, 0, 0, 1, 1]], float)
        atlas = np.arange(4).reshape(4, 1, 1)
        self.assertEqual(current.atlas_labels(rows, atlas), {1: 0, 2: 2})
        self.assertEqual(method.atlas_labels(rows, atlas), {1: 1, 2: 2})

    def test_display_pixel_centres_match_declared_native_index_positions(self):
        origin = np.array([100., 200., 300.])
        with tempfile.TemporaryDirectory(dir=method.HERE) as directory:
            with patch.object(current.plt, "close"):
                current.draw_image_views(np.ones((4, 5, 3)), origin, [], origin,
                                         Path(directory)/"geometry.png", "Geometry validation",
                                         highlight_kind="internal passage node")
                fig = current.plt.gcf()
                extent = fig.axes[0].images[0].get_extent()
                # Independent imshow centre: edge + half one displayed pixel.
                self.assertAlmostEqual(extent[0] + (extent[1]-extent[0])/4/2, origin[0])
                self.assertAlmostEqual(extent[2] + (extent[3]-extent[2])/5/2, origin[1])
                self.assertIn("internal passage node", fig.axes[3].get_title())
            current.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
