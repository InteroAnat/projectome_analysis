"""Explicit reference frames, physical padding and honest missing-source counts."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
spec = importlib.util.spec_from_file_location("discovery_geometry", ROOT / "group_analysis/scripts/03_scan_insula_recovery_candidates.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def reference_csv(directory, folded=True):
    row = {"sub_region": "IAL", "n": 2, "X_lo_q005": 40 if folded else 68,
           "X_hi_q995": 60 if folded else 188, "Y_lo_q005": 160, "Y_hi_q995": 168,
           "Z_lo_q005": 80, "Z_hi_q995": 88}
    if folded:
        row.update(x_frame="Soma_NII_X_folded", midline_nii_x=128, nii_voxel_mm=0.25)
    path = Path(directory) / ("folded.csv" if folded else "raw.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def coordinate(x=178, y=164, z=84, region="Unknown_0"):
    return {"Soma_NII_X": x, "Soma_NII_Y": y, "Soma_NII_Z": z, "Soma_Region": region}


class GeometryTests(unittest.TestCase):
    def test_explicit_reference_and_mode_required_without_fallback(self):
        with self.assertRaisesRegex(ValueError, "Explicit"):
            scan.load_bboxes()
        with self.assertRaisesRegex(ValueError, "no geometry fallback"):
            scan.load_bboxes("absent_folded.csv", geometry_mode="folded-mm")
        with tempfile.TemporaryDirectory() as directory:
            raw = reference_csv(directory, False)
            folded = reference_csv(directory)
            with self.assertRaisesRegex(ValueError, "requires explicit"):
                scan.load_bboxes(raw, geometry_mode="folded-mm")
            with self.assertRaisesRegex(ValueError, "cannot consume folded"):
                scan.load_bboxes(folded, geometry_mode="raw-legacy")
            frame = pd.read_csv(folded)
            frame["nii_voxel_mm"] = 0.5
            frame.to_csv(folded, index=False)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                scan.load_bboxes(folded, geometry_mode="folded-mm")

    def test_folded_symmetry_and_two_mm_is_eight_voxels(self):
        with tempfile.TemporaryDirectory() as directory:
            boxes = scan.load_bboxes(reference_csv(directory), geometry_mode="folded-mm")
            self.assertEqual(boxes.pad_mm, 2)
            self.assertEqual(scan.bbox_matches(coordinate(78), boxes), ["IAL"])
            self.assertEqual(scan.bbox_matches(coordinate(178), boxes), ["IAL"])
            self.assertEqual(scan.bbox_matches(coordinate(196, 176, 96), boxes), ["IAL"])
            self.assertEqual(scan.bbox_matches(coordinate(196.004, 176, 96), boxes), [])
            self.assertEqual(scan.bbox_matches(coordinate(128), boxes), [])
            raw = scan.load_bboxes(reference_csv(directory, False), geometry_mode="raw-legacy")
            self.assertEqual(raw.pad_mm, 0)
            self.assertEqual(scan.bbox_matches(coordinate(128), raw), ["IAL"])

    def test_missing_coordinates_are_unknown_and_labels_do_not_promote(self):
        with tempfile.TemporaryDirectory() as directory:
            boxes = scan.load_bboxes(reference_csv(directory), geometry_mode="folded-mm")
            for xyz in ((None, 164, 84), (float("nan"), 164, 84), (178, "bad", 84)):
                row = coordinate(*xyz)
                self.assertIsNone(scan.bbox_matches(row, boxes))
                self.assertEqual(scan.classify_neuron(row, boxes)[0], "auto_other_missing_coordinates")
            self.assertEqual(scan.classify_neuron(coordinate(None, region="PrCO"), boxes)[0], "auto_PrCO_missing_coordinates")
            self.assertEqual(scan.classify_neuron(coordinate(None, region="CL_Ial"), boxes)[0], "auto_insula")
            self.assertEqual(scan.classify_neuron(coordinate(128, region="CL_Ial"), boxes), ("auto_insula", ""))
            self.assertEqual(scan.classify_neuron(coordinate(region="SL_Pi"), boxes)[0], "auto_other_in_insula_bbox")
            self.assertEqual(scan.classify_neuron(coordinate(region="Unknown_0"), boxes)[0], "auto_other_in_insula_bbox")
            self.assertEqual(scan.classify_neuron(coordinate(region="G"), boxes)[0], "auto_other_in_insula_bbox")

    def test_missing_sources_are_not_zero_and_source_memberships_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            boxes = scan.load_bboxes(reference_csv(directory), geometry_mode="folded-mm")
            per, missing = scan.scan_sample("absent", boxes, directory)
            self.assertTrue(per.empty)
            self.assertIsNone(missing["n_total"])
            self.assertIsNone(missing["recoverable"])
            frame = pd.DataFrame([
                {"NeuronID": "001.swc", **coordinate(region="CL_Ial"), "Soma_Region_Source": "curated_251637", "in_combined": True, "exclude": "yes"},
                {"NeuronID": "1.swc", **coordinate(None), "Soma_Region_Source": "auto_atlas", "in_combined": False},
                {"NeuronID": "003.swc", **coordinate(128, region="G"), "Soma_Region_Source": "keep_G_distinct", "in_combined": True}])
            source = Path(directory) / "source.xlsx"
            frame.to_excel(source, sheet_name="Summary", index=False)
            with mock.patch.object(scan, "find_results_xlsx", return_value=str(source)):
                rows, stats = scan.scan_sample("250432ch2", boxes)
            self.assertEqual(rows.NeuronID.tolist(), ["001.swc", "1.swc", "003.swc"])
            self.assertEqual(rows.UID.tolist(), ["250432ch2|001.swc", "250432ch2|1.swc", "250432ch2|003.swc"])
            self.assertEqual(rows.Soma_Region_Source.tolist(), frame.Soma_Region_Source.tolist())
            self.assertEqual(rows.in_combined.tolist(), [True, False, True])
            self.assertEqual(rows.Soma_Region.tolist(), frame.Soma_Region.tolist())
            self.assertEqual(stats["n_total"], 3)
            self.assertEqual(stats["n_excluded"], 1)
            self.assertEqual(stats["candidate_lower_bound"], 0)
            self.assertEqual(stats["n_unassessed"], 1)
            self.assertIsNone(stats["recoverable"])
            self.assertTrue((rows.anatomical_acceptance == "not_assessed").all())

    def test_duplicate_or_missing_ids_remain_rows_but_cannot_be_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            boxes = scan.load_bboxes(reference_csv(directory), geometry_mode="folded-mm")
            frame = pd.DataFrame([{"NeuronID": "001.swc", **coordinate()}, {"NeuronID": "001.swc", **coordinate()}])
            source = Path(directory) / "source.xlsx"
            frame.to_excel(source, sheet_name="Summary", index=False)
            with mock.patch.object(scan, "find_results_xlsx", return_value=str(source)):
                rows, stats = scan.scan_sample("252385", boxes)
            self.assertEqual(len(rows), 2)
            self.assertEqual(stats["candidate_lower_bound"], 0)
            self.assertEqual(stats["n_unassessed"], 2)

    def test_cli_missing_source_is_incomplete_and_refuses_old_output(self):
        allowed = ROOT / "group_analysis/evolution_20261008/validation"
        allowed.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=allowed) as directory:
            path = reference_csv(directory)
            output = Path(directory) / "isolated"
            args = ["--bbox-csv", str(path), "--geometry-mode", "folded-mm", "--samples", "missing", "--step1-dir", directory, "--out-dir", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(scan.main(args), 3)
            report = json.loads((output / "discovery_provenance.json").read_text())
            self.assertIsNone(report["candidate_total"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                scan.main(args)
            args[-1] = str(ROOT / "group_analysis/recovery")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                scan.main(args)


if __name__ == "__main__":
    unittest.main()
