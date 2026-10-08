"""Independent arithmetic oracles for new exporter helper behavior.

Read-only validation of production helpers; source data/output runs untouched.
"""
import importlib.util
import json
from pathlib import Path
import hashlib
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "group_analysis/scripts/export_arm_projection_tables.py"
spec = importlib.util.spec_from_file_location("reviewed_arm_exporter", SOURCE)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class IndependentMeanExamples(unittest.TestCase):
    def test_unequal_animals_selected_vs_eligible_denominators(self):
        summary = pd.DataFrame({"AnimalID": ["A"] * 4 + ["B"], "Subregion": ["source"] * 5})
        counts = pd.DataFrame({"target": [4., 0., 8., np.nan, 10.]})
        length = pd.DataFrame({"target": [1., 2., 3., 4., 10.]})
        eligible = np.array([True, True, True, False, True])
        animal, group = exporter.conditional_means(counts, length, eligible, summary)
        a, b = animal.iloc[0], animal.iloc[1]
        self.assertEqual((a.NSelected, a.NEndpointEligible), (4, 3))
        self.assertEqual((b.NSelected, b.NEndpointEligible), (1, 1))
        self.assertEqual(a.CandidateEndpointMean, 12 / 3)
        self.assertEqual(a.EndpointNeuronFrequency, 2 / 3)
        self.assertEqual(a.AxonTemplateLengthMeanMm, 10 / 4)
        row = group.iloc[0]
        self.assertEqual(row.CandidateEndpointMean, (4 + 10) / 2)
        self.assertAlmostEqual(row.EndpointNeuronFrequency, (2 / 3 + 1) / 2)
        self.assertEqual(row.AxonTemplateLengthMeanMm, (2.5 + 10) / 2)
        self.assertNotEqual(row.CandidateEndpointMean, 22 / 4)
        self.assertNotEqual(row.AxonTemplateLengthMeanMm, 20 / 5)

    def test_no_candidate_group_endpoint_unavailable_axon_available(self):
        summary = pd.DataFrame({"AnimalID": ["A"], "Subregion": ["source"]})
        animal, group = exporter.conditional_means(pd.DataFrame({"target": [np.nan]}),
            pd.DataFrame({"target": [3.]}), np.array([False]), summary)
        self.assertTrue(pd.isna(animal.iloc[0].CandidateEndpointMean))
        self.assertTrue(pd.isna(group.iloc[0].EndpointNeuronFrequency))
        self.assertEqual(group.iloc[0].NEndpointAnimals, 0)
        self.assertEqual(group.iloc[0].NAxonAnimals, 1)
        self.assertEqual(group.iloc[0].AxonTemplateLengthMeanMm, 3)

    def test_all_outside_candidate_remains_eligible_zero_in_region(self):
        summary = pd.DataFrame({"AnimalID": ["A"], "Subregion": ["source"]})
        animal, group = exporter.conditional_means(pd.DataFrame({"inside": [0.], "outside": [5.]}),
            pd.DataFrame({"inside": [0.], "outside": [2.]}), np.array([True]), summary)
        indexed = group.set_index("TargetID")
        self.assertEqual(indexed.loc["inside", "NEndpointEligible"], 1)
        self.assertEqual(indexed.loc["inside", "EndpointNeuronFrequency"], 0)
        self.assertEqual(indexed.loc["outside", "EndpointNeuronFrequency"], 1)
        self.assertEqual(indexed.loc["outside", "CandidateEndpointMean"], 5)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    if result.wasSuccessful():
        out = Path(__file__).with_name("independent_exporter_mean_counterexamples_20261009.json")
        with out.open("x", encoding="utf-8") as stream:
            json.dump({"status": "passed", "tests": result.testsRun,
                       "exporter_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                       "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "scope": "Arithmetic helper counterexamples only, not full saved workbook acceptance",
                       "production_data_modified": False}, stream, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
