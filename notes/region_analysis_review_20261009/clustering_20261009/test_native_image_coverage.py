"""Guard missing-evidence and exact-identity contracts of the coverage join."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_native_image_coverage import checked_join, summarize


class NativeCoverageTests(unittest.TestCase):
    def setUp(self):
        self.assignments = pd.DataFrame([
            {"NeuronUID": "a::1.swc", "AnimalID": "A", "SWCPath": "a/1.swc",
             "SWCSHA256": "first", "ARMFullName": "official_one",
             "Endpoint_candidate_axon_endpoint_count": "5",
             "Henry_coarse_INS_visual_evidence": "True", "all_axon_hellinger_k2": "1",
             "all_endpoint_hellinger_k2": "1"},
            {"NeuronUID": "b::1.swc", "AnimalID": "B", "SWCPath": "b/1.swc",
             "SWCSHA256": "second", "ARMFullName": "official_two",
             "Endpoint_candidate_axon_endpoint_count": "0",
             "Henry_coarse_INS_visual_evidence": "False", "all_axon_hellinger_k2": "",
             "all_endpoint_hellinger_k2": ""}])
        self.coverage = pd.DataFrame([
            {"NeuronUID": "a::1.swc", "AnimalID": "A", "atlas_swc_path": "a/1.swc",
             "atlas_swc_sha256": "first", "SourceARMFullName": "official_one",
             "native_swc_available": "False", "candidate_axon_leaves": "",
             "leaves_with_cached_native_cube": ""},
            {"NeuronUID": "b::1.swc", "AnimalID": "B", "atlas_swc_path": "b/1.swc",
             "atlas_swc_sha256": "second", "SourceARMFullName": "official_two",
             "native_swc_available": "True", "candidate_axon_leaves": "0",
             "leaves_with_cached_native_cube": "0"}])

    def test_missing_native_counts_remain_missing_even_with_atlas_ends(self):
        joined = checked_join(self.assignments, self.coverage).set_index("NeuronUID")
        self.assertTrue(pd.isna(joined.loc["a::1.swc", "NativeOriginalAxonEndCount"]))
        self.assertEqual(joined.loc["b::1.swc", "NativeOriginalAxonEndCount"], 0)
        summary = summarize(joined.reset_index()).set_index(["SummaryType", "SummaryValue"])
        self.assertTrue(pd.isna(summary.loc[("animal", "A"), "NativeEndCountAmongFoundSWCs"]))
        self.assertEqual(summary.loc[("animal", "B"), "NativeEndCountAmongFoundSWCs"], 0)

    def test_reused_bare_neuron_id_never_matches_wrong_sample(self):
        self.coverage.loc[0, "NeuronUID"] = "b::1.swc"
        with self.assertRaises(ValueError):
            checked_join(self.assignments, self.coverage)

    def test_same_identity_with_changed_source_hash_is_rejected(self):
        self.coverage.loc[0, "atlas_swc_sha256"] = "changed"
        with self.assertRaises(ValueError):
            checked_join(self.assignments, self.coverage)

    def test_zero_filling_unknown_counts_is_rejected(self):
        self.coverage.loc[0, ["candidate_axon_leaves", "leaves_with_cached_native_cube"]] = "0"
        with self.assertRaises(ValueError):
            checked_join(self.assignments, self.coverage)


if __name__ == "__main__":
    unittest.main()
