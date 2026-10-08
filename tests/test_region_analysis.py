"""Region/laterality regressions with explicit independent expected labels."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
from region_analysis.laterality import LateralityParser as P, add_laterality_columns
from region_analysis.classifier import NeuronClassifier
from region_analysis.neuron_analysis import RegionAnalysisPerNeuron
from region_analysis.population import PopulationRegionAnalysis
from region_analysis.hierarchy import add_projection_hierarchy, RegionHierarchy
from region_analysis.hemisphere import AtlasHemisphereReference


class LateralityRegressionTests(unittest.TestCase):
    def test_refined_labels_preserve_explicit_side(self):
        for label, side in [("L-IDM", "L"), ("R-IDD5", "R"), ("L_IAL", "L"),
                            ("R_IDV", "R"), ("SL_Pu", "L"), ("CR_area_44", "R")]:
            with self.subTest(label=label):
                self.assertEqual(P.get_side(label), side)
        self.assertEqual(P.get_base_name("CR_area_44"), "area_44")

    def test_invalid_soma_side_never_becomes_contralateral(self):
        for side in (None, "Unknown", "", "Left", float("nan")):
            self.assertEqual(P.classify_with_soma_side(side, "CR_Ial"), "Unknown")

    def test_documented_112_114_do_not_flip_from_coordinate_distribution(self):
        df = pd.DataFrame({"Soma_Region": ["L-IDM", "L-IDD5", "CL_Ial", "CR_Ial", "Unknown_0"],
                           "Soma_NII_X": [211.5384, 209.5244, 50, 190, 220],
                           "Terminal_Regions": [["CR_Ia/Id", "Unknown_0"]] * 5,
                           "Region_projection_length": [{"CR_Ia/Id": 10}] * 5})
        result = add_laterality_columns(df)
        self.assertEqual(result.Soma_Side.tolist(), ["L", "L", "L", "R", "Unknown"])
        self.assertEqual(result.N_Contralateral.tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(result.loc[4, "N_Laterality_Unknown"], 2)
        self.assertEqual(result.loc[0, "Total_Contralateral_Length"], 10)
        self.assertEqual(result.loc[0, "Laterality_Index"], 1)

    def test_unknown_side_is_invariant_to_other_neurons(self):
        for labels, xs in [(["Unknown_0"], [200]), (["CL_Ial", "CR_Ial", "Unknown_0"], [10, 20, 200])]:
            result = add_laterality_columns(pd.DataFrame({"Soma_Region": labels, "Soma_NII_X": xs,
                                                        "Terminal_Regions": [["CR_Ial"]] * len(xs)}))
            self.assertEqual(result.Soma_Side.iloc[-1], "Unknown")

    def test_reference_requires_alignment_provenance_and_valid_sides(self):
        df = pd.DataFrame({"Soma_Region": ["Unknown_0"], "Terminal_Regions": [["CR_Ial"]]}, index=[4])
        for reference, source in [(pd.Series(["R"], index=[4]), None),
                                  (pd.Series(["R"], index=[5]), "mask:hash"),
                                  (pd.Series(["Right"], index=[4]), "mask:hash")]:
            with self.assertRaises(ValueError):
                add_laterality_columns(df, soma_side_reference=reference, soma_side_reference_source=source)
        result = add_laterality_columns(df, soma_side_reference=pd.Series(["R"], index=[4]), soma_side_reference_source="mask:hash")
        self.assertEqual(result.loc[4, "Soma_Side_Method"], "reference")
        self.assertEqual(result.loc[4, "N_Ipsilateral"], 1)

    def test_conflicting_reference_is_flagged_and_unclassified(self):
        df = pd.DataFrame({"Soma_Region": ["L-IDM"], "Terminal_Regions": [["CR_Ial"]]})
        result = add_laterality_columns(df, soma_side_reference=pd.Series(["R"]), soma_side_reference_source="mask:hash")
        self.assertTrue(result.Soma_Side_Conflict.iloc[0])
        self.assertEqual(result.Soma_Side.iloc[0], "Unknown")
        self.assertEqual(result.N_Laterality_Unknown.iloc[0], 1)

    def test_classifier_uses_the_same_refined_side_contract(self):
        classifier = NeuronClassifier(pd.DataFrame({"Index": [4, 504], "Abbreviation": ["CL_area_32", "CR_area_32"]}))
        self.assertEqual(classifier.classify_single_neuron(["CR_area_32"], "L-IDM"), "ITc")
        self.assertEqual(classifier.classify_single_neuron(["CR_area_32"], "R-IDM"), "ITi")
        self.assertEqual(classifier.classify_single_neuron(["CR_area_32"], "Unknown_0"), "Unclassified")


class HemisphereReferenceTests(unittest.TestCase):
    def fixture(self):
        atlas = np.zeros((4, 2, 2), dtype=int)
        atlas[0] = 4
        atlas[3] = 504
        mask = np.ones_like(atlas)
        mask[:2] = 2  # Official NMT ordering: 1=R, 2=L.
        key = pd.DataFrame({"Index": [4, 504], "Abbreviation": ["CL_A", "CR_A"]})
        return atlas, mask, key

    def test_mask_values_are_calibrated_not_assumed(self):
        atlas, mask, key = self.fixture()
        reference = AtlasHemisphereReference(mask, atlas, key, "declared-grid:hash")
        self.assertEqual(reference.value_to_side, {1: "R", 2: "L"})
        self.assertEqual(reference.sample([0, 1, 1]), "L")
        self.assertEqual(reference.sample([3, 1, 1]), "R")
        self.assertEqual(reference.sample([-1, 1, 1]), "Unknown")

    def test_mixed_mask_hemispheres_are_rejected(self):
        atlas, mask, key = self.fixture()
        mask[3] = 2
        with self.assertRaisesRegex(ValueError, "conflicts"):
            AtlasHemisphereReference(mask, atlas, key, "declared-grid:hash")

    def test_reference_grid_and_provenance_are_required(self):
        atlas, mask, key = self.fixture()
        with self.assertRaisesRegex(ValueError, "3D grid"):
            AtlasHemisphereReference(mask[:2], atlas, key, "source")
        with self.assertRaisesRegex(ValueError, "provenance"):
            AtlasHemisphereReference(mask, atlas, key, "")
        reference = AtlasHemisphereReference(mask, atlas, key, "source")
        with self.assertRaisesRegex(ValueError, "finite XYZ"):
            reference.sample([np.nan, 1, 1])

    def test_population_refinement_recomputes_conflicting_type_and_splits(self):
        atlas, mask, key = self.fixture()
        pop = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
        pop.hemisphere_reference = AtlasHemisphereReference(mask, atlas, key, "declared-grid:hash")
        pop.neurons = {}
        pop.classifier = NeuronClassifier(key)
        pop.hierarchy = pop.hierarchy_table = pop.dual_hierarchy = None
        pop.plot_dataframe = pd.DataFrame({"NeuronID": ["112.swc"], "Neuron_Type": ["ITi"],
            "Soma_Region": ["CR_A"], "Soma_NII_X": [3], "Soma_NII_Y": [1], "Soma_NII_Z": [1],
            "Terminal_Regions": [["CR_A"]], "Region_projection_length": [{"CR_A": 10}]})
        result = pop.apply_soma_labels({"112.swc": "L-IDM"}, "human-workbook:hash")
        self.assertEqual(result.Soma_Region_Auto.iloc[0], "CR_A")
        self.assertEqual(result.Soma_Side_Reference.iloc[0], "R")
        self.assertEqual(result.Soma_Side.iloc[0], "Unknown")
        self.assertTrue(result.Soma_Side_Conflict.iloc[0])
        self.assertEqual(result.Neuron_Type.iloc[0], "Unclassified")
        self.assertEqual(result.Total_Unknown_Laterality_Length.iloc[0], 10)
        ipsi, contra = pop.get_projection_matrix_split()
        self.assertNotIn("A", ipsi)
        self.assertNotIn("A", contra)
        ipsi_strength, contra_strength = pop.get_projection_strength_split()
        self.assertNotIn("A", ipsi_strength)
        self.assertNotIn("A", contra_strength)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "conflict.xlsx"
            pop._write_workbook(str(path))
            lengths = pd.read_excel(path, "Projection_Length_all")
            exported = pd.read_excel(path, "Summary")
            self.assertEqual(lengths.CR_A.iloc[0], 10)
            self.assertEqual(exported.Total_Unknown_Laterality_Length.iloc[0], 10)
            self.assertEqual(exported.Soma_Region.iloc[0], "L-IDM")
            self.assertEqual(exported.Soma_Region_Auto.iloc[0], "CR_A")


class RegionLookupTests(unittest.TestCase):
    def test_arm_first_last_levels_are_not_parent_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "key.txt"
            path.write_text("Index\tAbbreviation\tFull_Name\tFirst_Level\tLast_Level\n"
                            "4\tCL_area_32\tCL_area_32\t4\t6\n", encoding="utf-8")
            hierarchy = RegionHierarchy.from_file(str(path))
            self.assertIsNone(hierarchy.get_at_level("CL_area_32", 1))
            self.assertIsNone(hierarchy.get_at_level("CL_area_32", 2))
            self.assertEqual(hierarchy.get_at_level("CL_area_32", 4), "CL_area_32")
            self.assertEqual(hierarchy.get_at_level("CL_area_32", 6), "CL_area_32")

    def test_shuffled_key_indices_and_outside_endpoints(self):
        atlas = np.zeros((4, 4, 4), dtype=int)
        atlas[1, 1, 1] = 504
        atlas[2, 1, 1] = 4
        key = pd.DataFrame({"Index": [504, 4], "Abbreviation": ["CR_area_32", "CL_area_32"]}, index=[99, 12])
        def node(x):
            return SimpleNamespace(x_nii=x, y_nii=1, z_nii=1)
        root, end, outside = node(1), node(2), node(-1)
        tracer = SimpleNamespace(root=root, terminal_nodes=[end, outside], branches=[[root, end], [root, outside]])
        analysis = RegionAnalysisPerNeuron(tracer, atlas, key)
        analysis.run()
        self.assertEqual(analysis.soma_region, "CR_area_32")
        self.assertEqual(analysis.terminal_regions, [{"region": "CL_area_32", "coords": (2, 1, 1)},
                                                    {"region": "Out_of_Bounds", "coords": (-1, 1, 1)}])
        root.x_nii = -1
        self.assertEqual(analysis._soma_and_terminal_region()[0], "Out_of_Bounds")

    def test_invalid_atlas_selection_and_duplicate_keys_are_rejected(self):
        key = pd.DataFrame({"Index": [1, 1], "Abbreviation": ["CL_A", "CR_A"]})
        with self.assertRaisesRegex(ValueError, "unique"):
            RegionAnalysisPerNeuron(None, np.zeros((2, 2, 2)), key)
        with self.assertRaisesRegex(ValueError, "3D"):
            RegionAnalysisPerNeuron(None, np.zeros((2, 2, 2, 1, 6)), key)

    def test_terminal_hierarchy_accepts_real_list_cells(self):
        df = pd.DataFrame({"Terminal_Regions": [["CL_A", "CR_B"], []]})
        result = add_projection_hierarchy(df, None, max_level=1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.Terminal_Level_1.iloc[1], [])

    def test_actual_workbook_export_preserves_corrected_sides_and_splits(self):
        pop = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
        pop.plot_dataframe = add_laterality_columns(pd.DataFrame({"NeuronID": ["112.swc"], "Neuron_Type": ["ITc"],
            "Soma_Region": ["L-IDM"], "Terminal_Regions": [["CR_Ia/Id"]],
            "Region_projection_length": [{"CR_Ia/Id": 134.274}], "Soma_Level_6": ["L-IDM"]}))
        pop.hierarchy = pop.hierarchy_table = pop.dual_hierarchy = None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.xlsx"
            pop._write_workbook(str(path))
            summary = pd.read_excel(path, "Summary")
            terminals = pd.read_excel(path, "Terminal_Sites")
            contra = pd.read_excel(path, "Projection_Length_contra")
            soma = pd.read_excel(path, "Soma_Hierarchy")
            self.assertEqual(summary.Soma_Side.iloc[0], "L")
            self.assertEqual(terminals.Laterality.iloc[0], "Contralateral")
            self.assertAlmostEqual(contra["Ia/Id"].iloc[0], 134.274)
            self.assertEqual(soma.L6.iloc[0], "L-IDM")


if __name__ == "__main__":
    unittest.main()
