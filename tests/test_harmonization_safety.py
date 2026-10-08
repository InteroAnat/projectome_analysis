"""No-clobber and exact full-identity tests on isolated derivative workbooks."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("harmonization_under_test", ROOT / "group_analysis/scripts/06_harmonize_atlas_to_manual.py")
harmonizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harmonizer)


def workbook_sheets():
    summary = pd.DataFrame({"SampleID": [251637, 252385, 252385], "NeuronID": ["001.swc", "001.swc", "002.swc"],
                            "NeuronUID": ["251637::001.swc", "252385::001.swc", "252385::002.swc"],
                            "Soma_Region_Auto": ["CL_Ig", "CR_Ial", "CR_PrCO"],
                            "Soma_Region_Refined": ["IDD5", "Ial", "IAL"],
                            "Soma_Region_Source": ["curated_251637", "auto_atlas_insula", "coord_inferred_from_251637_strict_distant"]})
    quantitative = summary[["SampleID", "NeuronID", "NeuronUID"]].copy()
    quantitative["Ial"] = [1.25, 3.5, 0.]
    sheets = {"Summary": summary,
              "Provenance": pd.DataFrame({"source": ["old"]}),
              "Review_Metadata": pd.DataFrame({"reviewer": ["preserve"], "notes": ["candidate review pending"]})}
    for measure in ("Length", "Strength"):
        for level in ("", "L3_"):
            for side in ("ipsi", "contra"):
                sheets[f"Projection_{measure}_{level}{side}"] = quantitative.copy()
    return sheets


def write_sheets(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


class HarmonizationSafetyTests(unittest.TestCase):
    def test_existing_destination_refused_before_input_read_or_enrichment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.xlsx"
            output.write_bytes(b"protected destination")
            before = hashlib.sha256(output.read_bytes()).hexdigest()
            with mock.patch.object(harmonizer.pd, "ExcelFile", side_effect=AssertionError("must refuse before read")), \
                 mock.patch.object(harmonizer, "_enrich_henry_layers", side_effect=AssertionError("must not enrich")):
                with self.assertRaises(FileExistsError):
                    harmonizer.harmonize(Path(directory) / "nonexistent_input.xlsx", output)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), before)

    def test_same_input_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.xlsx"
            write_sheets(path, workbook_sheets())
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "distinct"):
                harmonizer.harmonize(path, path)
            self.assertEqual(path.read_bytes(), before)

    def test_duplicate_missing_and_reordered_full_identities_are_rejected(self):
        for defect in ("duplicate_summary", "missing_id", "wrong_uid", "missing_projection", "reordered", "different_sample"):
            with self.subTest(defect=defect):
                sheets = workbook_sheets()
                key = "Projection_Length_ipsi"
                if defect == "duplicate_summary":
                    sheets["Summary"].loc[2, ["SampleID", "NeuronID", "NeuronUID"]] = sheets["Summary"].loc[1, ["SampleID", "NeuronID", "NeuronUID"]]
                elif defect == "missing_id":
                    sheets["Summary"].loc[0, "NeuronID"] = None
                elif defect == "wrong_uid":
                    sheets[key].loc[0, "NeuronUID"] = "252385::001.swc"
                elif defect == "missing_projection":
                    sheets[key] = sheets[key].iloc[:-1].copy()
                elif defect == "reordered":
                    sheets[key] = sheets[key].iloc[::-1].copy()
                else:
                    sheets[key].loc[0, ["SampleID", "NeuronUID"]] = [999999, "999999::001.swc"]
                with self.assertRaises(ValueError):
                    harmonizer.validate_workbook_membership(sheets)

    def test_success_preserves_curated_and_distant_rows_all_sheet_membership_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.xlsx", Path(directory) / "derivative.xlsx"
            sheets = workbook_sheets()
            write_sheets(source, sheets)
            original_bytes = source.read_bytes()
            with mock.patch.object(harmonizer, "_enrich_henry_layers") as enrich:
                harmonizer.harmonize(source, output)
                enrich.assert_called_once()
            actual = pd.read_excel(output, sheet_name=None)
            self.assertEqual(harmonizer.validate_workbook_membership(actual), harmonizer.validate_workbook_membership(sheets))
            self.assertEqual(actual["Summary"].Soma_Region_Refined.tolist(), ["IDD5", "IAL", "IAL"])
            self.assertEqual(actual["Summary"].Soma_Region_Source.iloc[0], "curated_251637")
            self.assertEqual(actual["Summary"].Soma_Region_Source.iloc[2], "coord_inferred_from_251637_strict_distant")
            for name in ("Projection_Length_ipsi", "Projection_Strength_L3_contra", "Review_Metadata"):
                pd.testing.assert_frame_equal(actual[name], sheets[name], check_dtype=False)
            note = actual["Mapping_Rule"].set_index("atlas_leaf").loc["Ial", "notes"]
            self.assertTrue(note.startswith("92.1%"))
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertFalse(list(Path(directory).glob(".harmonized_build_*")))

    def test_post_enrichment_membership_change_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.xlsx", Path(directory) / "derivative.xlsx"
            write_sheets(source, workbook_sheets())
            def corrupt(workbook):
                sheets = pd.read_excel(workbook, sheet_name=None)
                sheets["Projection_Length_ipsi"] = sheets["Projection_Length_ipsi"].iloc[:-1]
                write_sheets(workbook, sheets)
            with mock.patch.object(harmonizer, "_enrich_henry_layers", side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, "membership differs"):
                    harmonizer.harmonize(source, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(directory).glob(".harmonized_build_*")))

    def test_destination_created_during_build_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.xlsx", Path(directory) / "derivative.xlsx"
            write_sheets(source, workbook_sheets())
            def create_competing_destination(_):
                output.write_bytes(b"concurrently created protected workbook")
            with mock.patch.object(harmonizer, "_enrich_henry_layers", side_effect=create_competing_destination):
                with self.assertRaises(FileExistsError):
                    harmonizer.harmonize(source, output)
            self.assertEqual(output.read_bytes(), b"concurrently created protected workbook")
            self.assertFalse(list(Path(directory).glob(".harmonized_build_*")))


if __name__ == "__main__":
    unittest.main()
