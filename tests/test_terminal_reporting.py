"""Known/unresolved reporting regressions; no biological terminal detector."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
from region_analysis.population import PopulationRegionAnalysis
from region_analysis.plotting import (
    plot_terminal_distribution_df, plot_projection_sites_count_df,
)
from region_analysis.utils import (
    is_known_terminal_target, parse_terminal_regions, terminal_target_region_counts,
    terminal_target_status,
)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


class TerminalTargetStatusTests(unittest.TestCase):
    def test_explicit_controls_and_nonstring_values_are_not_known(self):
        cases = {
            "unknown": ["Unknown", "Unknown_0", "unknown_37", "CL_Unknown_0",
                        "SR_Unknown_2", "InsulaUnknown"],
            "outside": ["Out_of_Bounds", " out_of_bounds ", "SR_Out_of_Bounds"],
            "unmapped": ["_Unmapped", "unmapped", "CL__Unmapped", "SL_Unmapped"],
            "absent": ["", "  ", "None", "null", "NaN", None, np.nan,
                       np.float32("nan"), pd.NA, pd.NaT],
            "invalid": [7, 0.0, False, {}, []],
        }
        for status, values in cases.items():
            for value in values:
                with self.subTest(value=value, status=status):
                    self.assertEqual(terminal_target_status(value), status)
                    self.assertFalse(is_known_terminal_target(value))

    def test_literal_anatomical_names_and_digits_remain_known(self):
        labels = ["CL_area_44", "CR_44", "L-IDD5", "SL_Pi", "CL_Pi",
                  "CR_UnknownNeighbor", "UnknownArea", "Unknown_0_border", "NA"]
        for label in labels:
            with self.subTest(label=label):
                self.assertEqual(terminal_target_status(label), "known")
        self.assertEqual(terminal_target_region_counts([labels]).to_dict(), dict.fromkeys(labels, 1))

    def test_counts_conserve_explicit_entries_without_empty_list_phantoms(self):
        values = [[], ["CL_44", "Out_of_Bounds", "", None, 3, np.nan]]
        self.assertEqual(terminal_target_region_counts(values).to_dict(), {"CL_44": 1})
        all_counts = terminal_target_region_counts(values, include_unresolved=True)
        self.assertEqual(int(all_counts.sum()), 6)
        self.assertEqual(all_counts["<absent target>"], 3)
        self.assertEqual(all_counts["<invalid target>"], 1)


class TerminalReportConsumerTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "NeuronID": ["a", "b", "c", "d", "e"],
            "Neuron_Type": ["PT", "ITi", "CT", "ITc", "Unclassified"],
            "Terminal_Regions": [
                ["CR_area_44", "CR_area_44", "Unknown_0", "Out_of_Bounds",
                 "_Unmapped", "", None, 7, np.nan],
                ["SL_Pi", "CL_Pi", "CR_UnknownNeighbor"], [],
                "CL_area_44, CL_area_44, unknown_3", ["unmapped", False],
            ],
            "Terminal_Count": [8, 3, 0, 2, 2],
            "Outlier_Count": [4, 0, 0, 1, 0],
        })
        self.original = self.df.copy(deep=True)

    def tearDown(self):
        plt.close("all")
        pd.testing.assert_frame_equal(self.df, self.original)

    def call_quietly(self, fn, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def assert_terminal_totals(self, report):
        for text in ["Total endpoint-target entries: 15", "Known endpoint-target entries: 5 (",
                     "Unresolved endpoint-target entries: 10 (", "Neurons with unknown regions: 3 (",
                     "Neurons with only known regions: 1 (", "Neurons with no target entries: 1 ("]:
            self.assertIn(text, report)

    def assert_projection_totals(self, report):
        for text in ["Total known endpoint-target entries: 5", "Total unresolved endpoint-target entries excluded: 10",
                     "Mean known endpoint-target regions per neuron: 1.00", "Neurons with unresolved endpoint-target regions: 3 (",
                     "Total outliers: 5"]:
            self.assertIn(text, report)

    def test_terminal_plot_filters_status_and_saves_report_and_figure(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "terminal.png"
            text = Path(folder) / "terminal.txt"
            report = self.call_quietly(plot_terminal_distribution_df, self.df,
                                      top_n=100, show=False, save_path=str(image),
                                      save_report_path=str(text))
            self.assert_terminal_totals(report)
            self.assertIn("Unknown entries excluded from plot: 10", report)
            self.assertIn("Total projection entries (after filtering): 5", report)
            self.assertEqual(text.read_text(encoding="utf-8"), report)
            self.assertTrue(image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        labels = [tick.get_text() for tick in plt.gcf().axes[1].get_xticklabels()]
        self.assertEqual(set(labels), {"CR_area_44", "SL_Pi", "CL_Pi", "CR_UnknownNeighbor", "CL_area_44"})
        self.assertEqual(sum(patch.get_height() for patch in plt.gcf().axes[1].patches), 5)

    def test_include_unresolved_plot_retains_every_entry(self):
        report = self.call_quietly(plot_terminal_distribution_df, self.df, top_n=100,
                                  exclude_unknown=False, show_pie=False, show=False)
        self.assert_terminal_totals(report)
        self.assertIn("Total projection entries (after filtering): 15", report)
        self.assertEqual(sum(patch.get_height() for patch in plt.gcf().axes[0].patches), 15)

    def test_projection_plot_and_population_reports_agree(self):
        report = self.call_quietly(plot_projection_sites_count_df, self.df, show=False)
        self.assert_projection_totals(report)
        reports = {}
        pop = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
        pop.plot_dataframe = self.df
        pop.sample_id = "synthetic"
        pop.save_report = lambda text, name: reports.__setitem__(name, text)
        pop._save_terminal_report()
        pop._save_projection_sites_report()
        self.assert_terminal_totals(reports["terminal_report"])
        self.assert_projection_totals(reports["projection_sites_report"])
        self.assertNotIn("Out_of_Bounds:", reports["terminal_report"])
        self.assertIn("CR_UnknownNeighbor:", reports["terminal_report"])
        # Original unique-order parsing/count contract remains untouched.
        self.assertEqual(self.df.Terminal_Regions.map(parse_terminal_regions).map(len).tolist(),
                         self.df.Terminal_Count.tolist())

    def test_only_unresolved_and_empty_target_lists_render(self):
        for values, total in [([["Out_of_Bounds", "Unknown_0", None]], 3), ([[], []], 0)]:
            frame = pd.DataFrame({"Terminal_Regions": values})
            report = self.call_quietly(plot_terminal_distribution_df, frame, show=False)
            self.assertIn(f"Total endpoint-target entries: {total}", report)
            self.assertIn("Known endpoint-target entries: 0 (", report)
            self.assertIn(f"Unresolved endpoint-target entries: {total} (", report)
            self.assertIn("Total projection entries (after filtering): 0", report)
            self.assertIn("Neurons with only known regions: 0 (", report)
            self.assertTrue(any("No target entries" in text.get_text()
                                for ax in plt.gcf().axes for text in ax.texts))
            projection = self.call_quietly(plot_projection_sites_count_df, frame, show=False)
            self.assertIn("Total known endpoint-target entries: 0", projection)
            self.assertIn(f"Total unresolved endpoint-target entries excluded: {total}", projection)

    def test_empty_dataframe_preserves_no_report_contract(self):
        empty = pd.DataFrame(columns=["Terminal_Regions"])
        self.assertIn("DataFrame is empty", self.call_quietly(plot_terminal_distribution_df, empty))
        self.assertIn("DataFrame is empty", self.call_quietly(plot_projection_sites_count_df, empty))
        pop = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
        pop.plot_dataframe = empty
        pop.save_report = lambda *_: self.fail("Empty input must not save a report")
        pop._save_terminal_report()
        pop._save_projection_sites_report()


if __name__ == "__main__":
    unittest.main()
