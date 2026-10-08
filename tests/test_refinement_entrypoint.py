"""Explicit geometry, no-clobber and candidate-policy entrypoint regressions."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "group_analysis/scripts/04_refine_soma_region_by_coords.py"
spec = importlib.util.spec_from_file_location("explicit_refinement", SCRIPT)
refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refine)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.step1 = self.root / "step1"
        tables = self.step1 / "sample_20261009_region_analysis/tables"
        tables.mkdir(parents=True)
        self.source = tables / "sample_results_20261009.xlsx"
        self.output = self.root / "derivative"
        self.bbox = self.reference / "folded_boxes.csv"
        self.anchors = self.reference / "251637_subregion_anchors_folded.csv"
        self.neurons = self.reference / "251637_subregion_neurons.csv"
        geometry = dict(x_frame="Soma_NII_X_folded", midline_nii_x=128., nii_voxel_mm=.25)
        bounds = {f"{axis}_{bound}": value for axis in "XYZ" for bound, value in (("lo_q005", 9.), ("hi_q995", 11.))}
        pd.DataFrame([dict(sub_region="IAL", **bounds, **geometry)]).to_csv(self.bbox, index=False)
        pd.DataFrame([dict(sub_region="IAL", median_nn=.01, **geometry)]).to_csv(self.anchors, index=False)
        pd.DataFrame(dict(NeuronID=["ref1.swc", "ref2.swc"], Soma_Region_clean=["IAL", "IAL"],
                          Soma_NII_X=[137., 139.], Soma_NII_Y=[9., 11.], Soma_NII_Z=[9., 11.],
                          Soma_NII_X_folded=[9., 11.])).to_csv(self.neurons, index=False)
        self.frame = pd.DataFrame(dict(NeuronID=["001.swc", "002.swc", "003.swc", "004.swc", "005.swc"],
                                       Soma_Region=["Unknown_0", "Unknown_42", "Unknown_noise", "CR_PrCO", "CL_Iai"],
                                       Soma_Side=["R", "L", "R", "R", "L"],
                                       Soma_NII_X=[138., 118., 138., 147., 200.],
                                       Soma_NII_Y=[10.] * 5, Soma_NII_Z=[10.] * 5))
        self.save_source()

    def save_source(self):
        self.frame.to_excel(self.source, sheet_name="Summary", index=False)

    def argv(self, **overrides):
        options = {"bbox-csv": self.bbox, "geometry-mode": "folded-mm", "out-dir": self.output,
                   "step1-dir": self.step1, "samples": "sample"}
        options.update(overrides)
        return [str(value) for key, value in options.items() for value in ("--" + key, value)]

    def run(self, **overrides):
        return refine.main(self.argv(**overrides))


class RefinementEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stderr = contextlib.redirect_stderr(io.StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_actual_cli_help_and_missing_arguments_do_not_create_default_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "must_not_exist"
            env = {**os.environ, "PROJECTOME_RECOVERY_OUT": str(destination)}
            for arguments, success in ((["--help"], True), ([], False)):
                result = subprocess.run([sys.executable, "-B", str(SCRIPT), *arguments], env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode == 0, success, result.stderr)
                self.assertFalse(destination.exists())
            self.assertIn("--bbox-csv", result.stderr)

    def test_folded_mirrors_padding_distance_and_original_labels_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            before = {path: refine.file_sha256(path) for path in (fixture.bbox, fixture.anchors, fixture.neurons, fixture.source)}
            self.assertEqual(fixture.run(), 0)
            summary = pd.read_excel(fixture.output / "sample_INS_HE_coord_inferred.xlsx", sheet_name="Summary")
            self.assertEqual(summary.NeuronID.tolist(), fixture.frame.NeuronID.tolist())
            self.assertEqual(summary.Soma_Region.tolist(), fixture.frame.Soma_Region.tolist())
            self.assertEqual(summary.Soma_Region_Source.tolist(), ["coord_inferred_from_251637_strict_distant"] * 2 +
                             ["not_rescue_candidate", "coord_inferred_from_251637_padded_distant", "auto_atlas_insula"])
            self.assertEqual(summary.Soma_Region_Refined.iloc[-1], "IAI")
            self.assertAlmostEqual(summary.Distance_to_nearest_251637_neuron_mm.iloc[0],
                                   summary.Distance_to_nearest_251637_neuron_mm.iloc[1])
            self.assertTrue(summary.PAD_TOL_VOX.eq(8).all())
            self.assertTrue(summary.Anatomical_Acceptance.eq("not_assessed").all())
            record = json.loads((fixture.output / "refinement_provenance.json").read_text())
            self.assertEqual(record["geometry_mode"], "folded-mm")
            self.assertEqual(record["candidate_keeper_count"], 4)
            self.assertFalse(record["canonical_promotion"])
            for path, digest in before.items():
                self.assertEqual(refine.file_sha256(path), digest)
            for name, digest in record["outputs"].items():
                self.assertEqual(refine.file_sha256(fixture.output / name), digest)

    def test_existing_even_empty_output_is_refused_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.output.mkdir()
            with self.assertRaises(SystemExit):
                fixture.run()
            self.assertEqual(list(fixture.output.iterdir()), [])
            fixture.output.joinpath("existing.xlsx").write_bytes(b"preserve")
            with self.assertRaises(SystemExit):
                fixture.run()
            self.assertEqual(fixture.output.joinpath("existing.xlsx").read_bytes(), b"preserve")

    def test_no_raw_anchor_or_bbox_fallback_and_explicit_companions_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            raw_anchor = fixture.reference / "251637_subregion_anchors.csv"
            fixture.anchors.rename(raw_anchor)
            with self.assertRaises(SystemExit):
                fixture.run()
            self.assertFalse(fixture.output.exists())
            # Explicit alternate path is accepted only because its metadata
            # still declares folded geometry; its filename has no authority.
            self.assertEqual(fixture.run(**{"anchors-csv": raw_anchor}), 0)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.bbox.rename(fixture.reference / "251637_subregion_bboxes.csv")
            with self.assertRaises(SystemExit):
                fixture.run()
            self.assertFalse(fixture.output.exists())

    def test_wrong_frames_scale_bounds_and_region_coverage_fail_before_output(self):
        for name, column, value in (("bbox", "x_frame", "Soma_NII_X"), ("anchors", "x_frame", "Soma_NII_X"),
                                    ("bbox", "nii_voxel_mm", 1.), ("anchors", "midline_nii_x", 0.),
                                    ("bbox", "X_lo_q005", 99.), ("bbox", "X_hi_q995", float("inf")),
                                    ("anchors", "median_nn", -1.), ("anchors", "sub_region", "IDM"),
                                    ("neurons", "Soma_NII_X_folded", 99.)):
            with self.subTest(name=name, column=column), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                path = getattr(fixture, name)
                table = pd.read_csv(path)
                table[column] = value
                table.to_csv(path, index=False)
                with self.assertRaises(SystemExit):
                    fixture.run()
                self.assertFalse(fixture.output.exists())

    def test_explicit_raw_legacy_works_without_metadata_and_never_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            for path in (fixture.bbox, fixture.anchors):
                pd.read_csv(path).drop(columns=["x_frame", "midline_nii_x", "nii_voxel_mm"]).to_csv(path, index=False)
            raw_anchor = fixture.reference / "251637_subregion_anchors.csv"
            fixture.anchors.rename(raw_anchor)
            fixture.frame.loc[:1, "Soma_NII_X"] = [10., 118.]
            fixture.save_source()
            self.assertEqual(fixture.run(**{"geometry-mode": "raw-legacy"}), 0)
            result = pd.read_excel(fixture.output / "sample_INS_HE_coord_inferred.xlsx", sheet_name="Summary")
            self.assertTrue(result.PAD_TOL_MM.eq(0).all())
            self.assertTrue(result.Soma_Region_Source.iloc[0].startswith("coord_inferred_from_251637_strict"))
            self.assertEqual(result.Soma_Region_Source.iloc[1], "excluded_outside_bbox")

    def test_unique_box_and_ambiguous_rescue_policies_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            for path in (fixture.bbox, fixture.anchors):
                frame = pd.read_csv(path)
                other = frame.copy()
                other["sub_region"] = "IDM"
                pd.concat([frame, other]).to_csv(path, index=False)
            frame = pd.read_csv(fixture.neurons)
            other = frame.copy()
            other["NeuronID"] = ["ref3.swc", "ref4.swc"]
            other["Soma_Region_clean"] = "IDM"
            pd.concat([frame, other]).to_csv(fixture.neurons, index=False)
            fixture.run()
            result = pd.read_excel(fixture.output / "sample_INS_HE_coord_inferred.xlsx", sheet_name="Summary")
            self.assertEqual(result.Soma_Region_Source.iloc[0], "excluded_ambiguous_bbox")
            self.assertEqual(result.Soma_Region_Source.iloc[-1], "auto_atlas_insula")

    def test_missing_sources_are_explicit_and_malformed_sources_fail_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.run(samples="absent")
            record = json.loads((fixture.output / "refinement_provenance.json").read_text())
            self.assertEqual(record["yield_status"], "incomplete_missing_sources")
            self.assertEqual(record["sources"][0]["status"], "missing_source")
            self.assertIsNone(record["sources"][0]["rows"])
        for bad in ("duplicate", "nonfinite", "malformed"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                if bad == "duplicate":
                    fixture.frame.loc[1, "NeuronID"] = fixture.frame.NeuronID.iloc[0]
                else:
                    fixture.frame["Soma_NII_Y"] = fixture.frame.Soma_NII_Y.astype(object)
                    fixture.frame.loc[1, "Soma_NII_Y"] = "inf" if bad == "nonfinite" else "not_a_coordinate"
                fixture.save_source()
                with self.assertRaises(SystemExit):
                    fixture.run()
                self.assertFalse(fixture.output.exists())

    def test_invalid_padding_and_source_directory_outputs_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            for padding in ("NaN", "-1", "Infinity"):
                with self.subTest(padding=padding), self.assertRaises(SystemExit):
                    fixture.run(**{"pad-mm": padding})
            for output in (fixture.reference / "child", fixture.step1 / "child"):
                with self.subTest(output=output), self.assertRaises(SystemExit):
                    fixture.run(**{"out-dir": output})
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
