"""Validate interval scanner against both pinned rasterizer implementations."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

import audit_whole_axon_boundary_impact as audit

snapshot = Path(__file__).parent / "projection_maps_before_boundary_fix_HEAD.py"
spec = importlib.util.spec_from_file_location("boundary_test_old", snapshot)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


class BoundaryScannerTests(unittest.TestCase):
    def test_scan_allocations_equal_both_rasterizers_for_adversarial_and_random_edges(self):
        shape = np.array([4, 4, 4])
        faces = [np.arange(n-1, dtype=float)+.5 for n in shape]
        random = np.random.default_rng(20261009)
        cases = [(random.uniform(-2, 6, 3), random.uniform(-2, 6, 3)) for _ in range(128)]
        cases += [(np.array([x, 0., 0.]), np.array([x, 1., 0.]))
                  for x in [-.5, .5, np.nextafter(.5, -np.inf),
                            np.nextafter(3.5, -np.inf), 3.5]]
        cases += [(np.array([-2., 0., 0.]), np.array([6., 0., 0.])),
                  (np.array([0., 0., 0.]), np.array([0., 0., 0.]))]
        for start, end in cases:
            scanned = [Counter(), Counter()]
            for a, b, fraction, _ in audit.intervals(start, end, shape, faces):
                for target, voxel in zip(scanned, [a, b]):
                    if audit.valid(voxel, shape):
                        target[tuple(voxel)] += fraction
            for expected, module in zip(scanned, [old, audit.corrected]):
                actual = Counter()
                for voxel, fraction in module.segment_voxels(start, end, shape):
                    actual[voxel] += fraction
                self.assertEqual(expected, actual)

    def test_known_nextafter_parallel_defect_is_detected_and_positive(self):
        shape = np.array([4, 4, 4])
        faces = [np.arange(n-1, dtype=float)+.5 for n in shape]
        x = np.nextafter(.5, -np.inf)
        changed = [(a, b, fraction) for a, b, fraction, _ in audit.intervals(
            np.array([x, 0., 0.]), np.array([x, 1., 0.]), shape, faces)
            if not np.array_equal(a, b)]
        self.assertEqual(sum(fraction for _, _, fraction in changed), 1.)
        self.assertEqual({int(a[0]) for a, _, _ in changed}, {1})
        self.assertEqual({int(b[0]) for _, b, _ in changed}, {0})


if __name__ == "__main__":
    unittest.main()
