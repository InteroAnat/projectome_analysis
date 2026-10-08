"""Public table-consumer regressions with independently specified values."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(os.environ.get("PROJECTOME_CONSUMER_MODULE_ROOT", ROOT / "main_scripts"))
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(MODULE_ROOT))
from region_analysis.population import PopulationRegionAnalysis
from region_analysis.laterality_projection_analysis import LateralityProjectionAnalyzer
from region_analysis.utils import load_processed_df, projection_column_names
from region_analysis.laterality import LateralityParser, add_laterality_columns
from region_analysis.plotting import plot_laterality_summary_df


def population(frame):
    value = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
    value.plot_dataframe = frame.copy()
    value.hierarchy = value.hierarchy_table = value.dual_hierarchy = None
    return value


def neuron_frame(lengths=None, **extra):
    return pd.DataFrame({"SampleID": ["sample"], "NeuronID": ["001.swc"],
                         "Neuron_Type": ["ITi"], "Soma_Region": ["CL_A"],
                         "Soma_Side": ["L"], "Length_Unit": ["voxel"],
                         "Terminal_Regions": [["CL_A"]],
                         "Region_projection_length": [{"CL_A": 9.0} if lengths is None else lengths],
                         **extra})


class RegionTableConsumerTests(unittest.TestCase):
    def test_serialized_lengths_survive_dataframe_and_csv_reload(self):
        frame = neuron_frame("{'CL_A': 9.0, 'CR_B': 3.0}")
        for source in (frame,):
            pop = population(frame)
            pop.load_processed_dataframe(source)
            self.assertEqual(pop.get_projection_matrix().CL_A.iloc[0], 9.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neurons.csv"
            frame.to_csv(path, index=False)
            pop.load_processed_dataframe(path)
            self.assertEqual(pop.get_projection_matrix().CR_B.iloc[0], 3.0)

    def test_missing_or_invalid_lengths_never_become_silent_zero(self):
        for value in (None, "not a dictionary", "{'CL_A': -1}", {"CL_A": np.nan}, {"CL_A": True}):
            with self.subTest(value=value):
                frame = neuron_frame()
                frame.at[0, "Region_projection_length"] = value
                with self.assertRaises(ValueError):
                    population(frame).load_processed_dataframe(frame)

    def test_multisheet_roundtrip_restores_absolute_lengths_and_targets(self):
        frame = neuron_frame({"CL_A": 9.0, "CR_B": 3.0}, Terminal_Count=[2])
        frame.at[0, "Terminal_Regions"] = ["CL_A", "CR_B"]
        pop = population(frame)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.xlsx"
            pop._write_workbook(str(path))
            loaded = population(pd.DataFrame())
            loaded.load_processed_dataframe(path)
            self.assertEqual(loaded.plot_dataframe.Terminal_Regions.iloc[0], ["CL_A", "CR_B"])
            matrix = loaded.get_projection_matrix()
            self.assertEqual(matrix.CL_A.iloc[0], 9.0)
            self.assertEqual(matrix.CR_B.iloc[0], 3.0)

    def test_multisheet_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.xlsx"
            with pd.ExcelWriter(path) as writer:
                neuron_frame().drop(columns=["Terminal_Regions", "Region_projection_length"]).to_excel(writer, sheet_name="Summary", index=False)
                pd.DataFrame({"NeuronID": ["different.swc"], "Neuron_Type": ["ITi"], "CL_A": [9.]}).to_excel(writer, sheet_name="Projection_Length_all", index=False)
            with self.assertRaises(ValueError):
                population(pd.DataFrame()).load_processed_dataframe(path)

    def test_cortical_and_subcortical_homonyms_remain_distinct(self):
        for name in ("PM", "CM", "RM", "R", "Pi"):
            frame = neuron_frame({f"CL_{name}": 3., f"SL_{name}": 8.})
            pop = population(frame)
            ipsi, _ = pop.get_projection_matrix_split()
            self.assertEqual(ipsi[f"C_{name}"].iloc[0], 3.)
            self.assertEqual(ipsi[f"S_{name}"].iloc[0], 8.)
            strength, _ = pop.get_projection_strength_split()
            self.assertAlmostEqual(strength[f"C_{name}"].iloc[0], round(np.log10(4), 4))
            self.assertAlmostEqual(strength[f"S_{name}"].iloc[0], round(np.log10(9), 4))

    def test_atlas_namespace_is_stable_when_only_one_homonym_is_observed(self):
        pop = population(neuron_frame({"CL_Pi": 3.}))
        pop.atlas_table = pd.DataFrame({"Abbreviation": ["CL_Pi", "CR_Pi", "SL_Pi", "SR_Pi"]})
        ipsi, _ = pop.get_projection_matrix_split()
        self.assertEqual(ipsi.C_Pi.iloc[0], 3.)

    def test_generated_namespace_cannot_collide_with_literal_input(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            projection_column_names(["CL_Pi", "SL_Pi", "C_Pi"])

    def test_finest_only_source_splits_and_strengths_agree(self):
        frame = neuron_frame().rename(columns={"Region_projection_length": "Region_Projection_Length_finest"})
        pop = population(frame)
        ipsi, _ = pop.get_projection_matrix_split()
        strength, _ = pop.get_projection_strength_split()
        self.assertEqual(ipsi.A.iloc[0], 9.)
        self.assertEqual(strength.A.iloc[0], 1.)

    def test_unknown_hemisphere_is_excluded_from_both_splits_but_retained_absolute(self):
        pop = population(neuron_frame({"CL_M1": 77.533, "CR_M1": 996.988}, Soma_Side=["Unknown"]))
        for split in pop.get_projection_matrix_split():
            self.assertNotIn("M1", split)
            self.assertEqual(len(split), 1)
        absolute = pop.get_projection_matrix()
        self.assertAlmostEqual(absolute.CL_M1.iloc[0] + absolute.CR_M1.iloc[0], 1074.521)

    def test_multisample_repeated_ids_survive_roundtrip_with_exact_identity(self):
        first = neuron_frame({"CL_A": 9.}, SampleID=["A"], NeuronUID=["A::001.swc"], Terminal_Count=[1])
        second = neuron_frame({"CL_A": 3.}, SampleID=["B"], NeuronUID=["B::001.swc"], Terminal_Count=[1])
        second.at[0, "Terminal_Regions"] = ["CL_B"]
        pop = population(pd.concat([first, second], ignore_index=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two_animals.xlsx"
            pop._write_workbook(str(path))
            reloaded = population(load_processed_df(path))
            absolute = reloaded.get_projection_matrix()
            self.assertEqual(absolute.NeuronUID.tolist(), ["A::001.swc", "B::001.swc"])
            self.assertEqual(absolute.CL_A.tolist(), [9., 3.])
            self.assertEqual(reloaded.plot_dataframe.Terminal_Regions.tolist(), [["CL_A"], ["CL_B"]])
            strength, _ = reloaded.get_projection_strength_split()
            self.assertEqual(strength.SampleID.tolist(), ["A", "B"])
            self.assertEqual(strength.A.tolist(), [1., round(np.log10(4), 4)])

    def test_duplicate_exact_identity_is_rejected_before_matrix_export(self):
        frame = pd.concat([neuron_frame(), neuron_frame()], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            population(frame).get_projection_matrix()

    def test_mismatched_uid_is_rejected_before_join_or_matrix_export(self):
        frame = neuron_frame(NeuronUID=["foreign::001.swc"])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            population(frame).get_projection_matrix()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.xlsx"
            frame.to_excel(path, sheet_name="Summary", index=False)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_processed_df(path)

    def test_foreign_subject_in_terminal_sheet_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foreign.xlsx"
            with pd.ExcelWriter(path) as writer:
                neuron_frame().to_excel(writer, sheet_name="Summary", index=False)
                pd.DataFrame({"SampleID": ["foreign"], "NeuronID": ["001.swc"],
                              "Terminal_Region": ["CL_B"]}).to_excel(writer, sheet_name="Terminal_Sites", index=False)
            # Remove the literal Terminal_Regions so the long sheet is consumed.
            import openpyxl
            book = openpyxl.load_workbook(path)
            headers = [cell.value for cell in book["Summary"][1]]
            book["Summary"].delete_cols(headers.index("Terminal_Regions") + 1)
            book.save(path)
            book.close()
            with self.assertRaisesRegex(ValueError, "absent"):
                load_processed_df(path)

    def test_split_only_workbook_requires_original_absolute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lossy.xlsx"
            with pd.ExcelWriter(path) as writer:
                neuron_frame().drop(columns=["Region_projection_length"]).to_excel(writer, sheet_name="Summary", index=False)
                pd.DataFrame({"NeuronID": ["001.swc"], "A": [9.]}).to_excel(writer, sheet_name="Projection_Length_ipsi", index=False)
            with self.assertRaisesRegex(ValueError, "Split-only"):
                load_processed_df(path)

    def test_outlier_details_survive_multisheet_roundtrip(self):
        frame = neuron_frame(Outlier_Count=[1], Outlier_Details=[[{"type": "terminal", "region": "outside", "coords": (1, 2, 3)}]])
        pop = population(frame)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outliers.xlsx"
            pop._write_workbook(str(path))
            restored = population(load_processed_df(path))
            pd.testing.assert_frame_equal(restored.get_outlier_df(), pop.get_outlier_df(), check_dtype=False)

    def test_soma_hierarchy_export_does_not_duplicate_or_hide_conflicting_levels(self):
        frame = neuron_frame(Soma_Level_3=["CL_A"], Soma_Region_Hierarchy=[{"L3": "CL_A"}])
        exported = population(frame).get_soma_hierarchy_df()
        self.assertTrue(exported.columns.is_unique)
        frame.at[0, "Soma_Region_Hierarchy"] = {"L3": "CR_B"}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            population(frame).get_soma_hierarchy_df()

    def test_summary_and_hierarchy_sheet_cannot_disagree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hierarchy_conflict.xlsx"
            with pd.ExcelWriter(path) as writer:
                neuron_frame(Soma_Level_3=["CL_A"]).to_excel(writer, sheet_name="Summary", index=False)
                pd.DataFrame({"NeuronID": ["001.swc"], "L3": ["CR_B"]}).to_excel(writer, sheet_name="Soma_Hierarchy", index=False)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                load_processed_df(path)

    def test_unknown_target_controls_remain_unknown_in_every_laterality_consumer(self):
        frame = neuron_frame({"CL_Unknown_0": 5., "CR_Unknown_0": 7., "CL_Ig": 2.})
        frame.at[0, "Terminal_Regions"] = ["CL_Unknown_0", "CR_Unknown_0", "CL_Ig"]
        enriched = add_laterality_columns(frame)
        self.assertEqual(enriched.Total_Unknown_Laterality_Length.iloc[0], 12.)
        self.assertEqual(enriched.N_Laterality_Unknown.iloc[0], 2)
        self.assertEqual(enriched.Total_Ipsilateral_Length.iloc[0], 2.)
        ipsi, contra = population(enriched).get_projection_matrix_split()
        self.assertNotIn("Unknown_0", ipsi)
        self.assertNotIn("Unknown_0", contra)
        result = LateralityProjectionAnalyzer(enriched).analyze()
        self.assertEqual(result["unknown_laterality"].Length.sum(), 12.)
        for side in (None, pd.NA, np.nan, []):
            self.assertEqual(LateralityParser.classify_with_soma_side(side, "CL_Ig"), "Unknown")

    def test_public_laterality_split_decodes_literals_without_silent_missing_zero(self):
        self.assertEqual(LateralityParser.split_projection_lengths_with_soma_side("L", "{'CL_A': 3., 'CR_A': 5.}"),
                         ({"CL_A": 3.}, {"CR_A": 5.}, {}))
        with self.assertRaises(ValueError):
            LateralityParser.split_projection_lengths_with_soma_side("L", None)

    def test_length_pooling_cannot_label_partly_unknown_units_as_voxels(self):
        frame = pd.DataFrame({"N_Ipsilateral": [1, 1], "N_Contralateral": [0, 0],
                              "Length_Unit": ["voxel", None]})
        try:
            with self.assertRaisesRegex(ValueError, "mixed"):
                plot_laterality_summary_df(frame, show=False)
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_standalone_uses_reference_side_and_arbitrary_dataframe_index(self):
        frame = neuron_frame({"CR_A": 9.}, Soma_Region=["Unknown_0"], Soma_Side=["R"])
        frame.index = ["exact-neuron"]
        result = LateralityProjectionAnalyzer(frame).analyze()
        self.assertEqual(result["ipsilateral"].A_length.iloc[0], 9.)
        self.assertTrue(result["contralateral"].empty)

    def test_standalone_does_not_trust_stale_precomputed_splits(self):
        frame = neuron_frame({"CR_A": 9.}, Soma_Side=["Unknown"],
                             Ipsilateral_Projection_Length=[{"CR_A": 9.}],
                             Contralateral_Projection_Length=[{}])
        result = LateralityProjectionAnalyzer(frame).analyze()
        self.assertTrue(result["ipsilateral"].empty)
        self.assertTrue(result["contralateral"].empty)

    def test_standalone_export_has_unique_headers_and_preserves_existing_file(self):
        analyzer = LateralityProjectionAnalyzer(neuron_frame())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "laterality.xlsx"
            analyzer.save_excel(str(path))
            exported = pd.read_excel(path, "Ipsilateral_Length")
            self.assertNotIn("ipsilateral_total_length.1", exported)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                analyzer.save_excel(str(path))
            self.assertEqual(path.read_bytes(), before)

    def test_standalone_cli_can_import_and_exposes_unsupported_levels(self):
        script = MODULE_ROOT / "region_analysis/laterality_projection_analysis.py"
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(script), "--help"],
                                cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.csv", Path(directory) / "cli.xlsx"
            neuron_frame({"CL_A": 3., "CR_A": 5.}).to_csv(source, index=False)
            executed = subprocess.run([sys.executable, "-X", "utf8", "-B", str(script), str(source), "-o", str(output)],
                                      cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            actual = pd.read_excel(output, "Ipsilateral_Length")
            self.assertEqual(actual.A_length.iloc[0], 3.)
            self.assertEqual(actual.NeuronUID.iloc[0], "sample::001.swc")
        with self.assertRaisesRegex(ValueError, "hierarchy"):
            LateralityProjectionAnalyzer(neuron_frame()).analyze(levels=[3])


if __name__ == "__main__":
    with redirect_stdout(io.StringIO()):
        unittest.main()
