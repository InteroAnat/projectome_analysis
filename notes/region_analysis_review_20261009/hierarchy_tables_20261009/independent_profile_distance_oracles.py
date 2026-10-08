"""Closed-form distance checks for the associated descriptive clustering."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "group_analysis/scripts/cluster_arm_projection_profiles.py"
spec = importlib.util.spec_from_file_location("independent_cluster_review", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ClosedFormProfileDistances(unittest.TestCase):
    def test_hellinger_closed_form_and_extent_invariance(self):
        raw = np.array([[1., 0.], [0., 1.], [1., 1.]])
        _, distance, fractions = module.profile_distance(raw)
        half = np.sqrt(1 - 1 / np.sqrt(2))
        np.testing.assert_allclose(distance, [[0, 1, half], [1, 0, half], [half, half, 0]])
        np.testing.assert_allclose(fractions, [[1, 0], [0, 1], [.5, .5]])
        _, scaled, _ = module.profile_distance(raw * np.array([[100], [9], [2]]))
        np.testing.assert_allclose(scaled, distance)

    def test_presence_jaccard_binary_union_intersection(self):
        _, distance, _ = module.profile_distance([[100, 0], [0, 2], [1, 9]], "presence_jaccard")
        np.testing.assert_allclose(distance, [[0, 1, .5], [1, 0, .5], [.5, .5, 0]])

    def test_zero_and_unassessed_profiles_cannot_be_clustered_as_zeros(self):
        for raw in ([[1, 0], [0, 1], [0, 0]], [[1, 0], [0, 1], [np.nan, np.nan]]):
            with self.assertRaises(ValueError):
                module.profile_distance(raw)

    def test_declared_average_linkage_is_not_silently_ward(self):
        with self.assertRaises(ValueError):
            module.profile_distance([[1, 0], [0, 1], [1, 1]], method="ward")


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    if result.wasSuccessful():
        path = Path(__file__).with_name("independent_profile_distance_oracles_20261009.json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"status": "passed", "tests": result.testsRun,
                       "cluster_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                       "oracle_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "scope": "Distance definitions and guards only; no saved clustering or biological-class acceptance",
                       "source_data_modified": False}, stream, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
