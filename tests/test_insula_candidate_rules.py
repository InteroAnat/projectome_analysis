"""Candidate identity tests; no server reads or accepted-workbook writes."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
from region_labels import (normalize_region_label, normalize_portal_region_label,
                           is_explicit_unknown_label)
from insula_label_set import build_insula_label_set


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExactCandidateTests(unittest.TestCase):
    def test_source_specific_normalization_preserves_anatomical_numbers(self):
        self.assertEqual(normalize_region_label(" CL_area_44 "), "AREA_44")
        self.assertEqual(normalize_region_label("L-IDD5"), "IDD5")
        self.assertEqual(normalize_portal_region_label("area_44_76"), "AREA_44")
        self.assertEqual(normalize_portal_region_label("CL_area_44", {"AREA_44"}), "AREA_44")
        self.assertEqual(normalize_portal_region_label("Ia/Id_228"), "IA/ID")
        # Preserve explicit subcortical prefixes to avoid cortical-Pi collision.
        self.assertEqual(normalize_region_label("SL_Pi"), "SL_PI")

    def test_unknown_is_explicit_and_blank_is_separate(self):
        for value in ("Unknown", "Unknown_0", "Unknown_42", "_Unmapped"):
            self.assertTrue(is_explicit_unknown_label(value))
        for value in (None, np.nan, "", "not_unknown_tissue", "Unknown_noise"):
            self.assertFalse(is_explicit_unknown_label(value))

    def test_region_helper_exact_labels_curated_aliases_and_missing_metadata(self):
        fake_ion = types.SimpleNamespace(IONData=lambda: mock.Mock())
        with mock.patch.dict(sys.modules, {"IONData": fake_ion}):
            helper = load_script("candidate_region_helper", "main_scripts/region_analysis/getNeuronListByRegion.py")
        labels = ["Ial_42", "CL_Ial", "CR_Ial", "L-IDM", "R-IDD5", "Ia/Id_228",
                  "L-area_44", "R-area_44", "I", "IAC", "SL_Pi", "SR_Pi_1661",
                  "CL_Pi", "CR_Pi_726", "Unknown_0", None, np.nan, ""]
        helper._iondata.getNeuronListBySampleID.return_value = [
            {"name": f"{i:03d}.swc", "region": label} for i, label in enumerate(labels)]
        atlas = pd.DataFrame({"Abbreviation": ["CL_Ial", "CR_Ial", "CL_Ia/Id", "CL_Pi", "CR_Pi", "SL_Pi", "CL_area_44"],
                              "Full_Name": ["insula"] * 5 + ["pineal_gland", "motor_cortex"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas_key.tsv"
            atlas.to_csv(path, sep="\t", index=False)
            actual = helper.getNeuronListByRegion("251637", "insula", str(path), return_ids_only=True, verbose=False)
            self.assertEqual(actual, [f"{i:03d}.swc" for i in (0, 1, 2, 3, 4, 5, 12, 13)])
            # A narrower requested region does not inherit every curated alias.
            self.assertEqual(helper.getNeuronListByRegion("251637", "pineal", str(path), return_ids_only=True, verbose=False), ["010.swc"])
            helper._iondata.getNeuronListBySampleID.return_value = [{"name": "missing.swc"}]
            self.assertEqual(helper.getNeuronListByRegion("251637", "insula", str(path), return_ids_only=True, verbose=False), [])

    def test_curated_labels_are_candidates_and_prco_is_separate(self):
        labels, rescue = build_insula_label_set()
        self.assertTrue({"IAL", "IAPM", "IDD5", "IDM", "IDV"}.issubset(labels))
        self.assertNotIn("PRCO", labels)
        self.assertIn("UNKNOWN_0", rescue)

    def test_discovery_uses_same_exact_atlas_labels(self):
        scan = load_script("candidate_discovery", "group_analysis/scripts/03_scan_insula_recovery_candidates.py")
        for label in ("CL_Ig", "CR_Iai", "L-IDD5", "R-IDM", "CL_Ia/Id"):
            self.assertEqual(scan.classify_neuron({"Soma_Region": label}, {})[0], "auto_insula")
        for label in ("IAC", "L-area_44", "SL_Pi"):
            self.assertEqual(scan.classify_neuron({"Soma_Region": label}, {})[0], "auto_other_missing_coordinates")

    def test_refinement_keeps_explicit_unknown_and_distant_candidates(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                "os.environ", {"PROJECTOME_RECOVERY_OUT": directory, "PROJECTOME_SAMPLES": "252385"}):
            refine = load_script("candidate_refinement", "group_analysis/scripts/04_refine_soma_region_by_coords.py")
            frame = pd.DataFrame({"NeuronID": ["001.swc", "002.swc", "003.swc"],
                                  "Soma_Region": ["Unknown_0", "Unknown_42", "Unknown_noise"],
                                  "Soma_Side": ["Unknown"] * 3,
                                  "Soma_NII_X": [10.] * 3, "Soma_NII_Y": [10.] * 3, "Soma_NII_Z": [10.] * 3})
            source = Path(directory) / "source.xlsx"
            frame.to_excel(source, sheet_name="Summary", index=False)
            boxes = pd.DataFrame([{"sub_region": "IAL", "X_lo_q005": 9, "X_hi_q995": 11,
                                   "Y_lo_q005": 9, "Y_hi_q995": 11, "Z_lo_q005": 9, "Z_hi_q995": 11}])
            anchors = pd.DataFrame({"sub_region": ["IAL"], "median_nn": [.01]})
            reference = pd.DataFrame({"Soma_Region_clean": ["IAL"], "Soma_NII_X": [9.],
                                      "Soma_NII_Y": [9.], "Soma_NII_Z": [9.]})
            with mock.patch.object(refine, "_load_reference", return_value=(boxes, anchors, reference, "Soma_NII_X")), \
                 mock.patch.object(refine, "find_results_xlsx", return_value=str(source)), \
                 mock.patch.object(refine, "NEW_SAMPLES", ["252385"]):
                # Explicit entrypoint contract; this test isolates label policy
                # from reference parsing using the existing mocked loader.
                ref_dir = Path(directory) / "references"
                ref_dir.mkdir()
                for name in ("boxes.csv", "251637_subregion_anchors.csv", "251637_subregion_neurons.csv"):
                    (ref_dir / name).write_text("fixture", encoding="utf-8")
                output = Path(directory) / "refinement"
                self.assertEqual(refine.main(["--bbox-csv", str(ref_dir / "boxes.csv"),
                                              "--geometry-mode", "raw-legacy", "--pad-mm", "2",
                                              "--samples", "252385", "--step1-dir", str(Path(directory) / "step1"),
                                              "--out-dir", str(output)]), 0)
            result = pd.read_excel(output / "252385_INS_HE_coord_inferred.xlsx", sheet_name="Summary")
            self.assertEqual(result.Soma_Region_Source.tolist(), ["coord_inferred_from_251637_strict_distant"] * 2 + ["not_rescue_candidate"])
            kept = pd.read_excel(output / "252385_INS_HE_coord_inferred.xlsx", sheet_name="Insula_keepers")
            self.assertEqual(kept.NeuronID.tolist(), ["001.swc", "002.swc"])


if __name__ == "__main__":
    unittest.main()
