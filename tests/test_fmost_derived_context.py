"""Source-ramp, coverage, resource, and artifact preservation regressions."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "group_analysis" / "scripts"))
from fmost_derived_context import derive_highres_context
import visual_review_20261002 as review


class DerivedContextTests(unittest.TestCase):
    def test_coordinate_ramp_matches_affine_across_cube_boundary(self):
        # Factor 7 does not divide a source cube; the sampling phase must persist.
        def cube(x, y, z):
            gx = x * 360 + np.arange(360)[None, None, :]
            gy = y * 360 + np.arange(360)[None, :, None]
            gz = z * 90 + np.arange(90)[:, None, None]
            return (gx + 3 * gy + 10 * gz).astype(np.uint16)
        data, report = derive_highres_context([235.3, 234.9, 270.2], cube,
                                              field_um=24, depth_um=12, factor=7)
        self.assertEqual(report["status"], "derived_highres")
        spacing = np.array(report["derived_spacing_xyz_um"])
        origin = np.array(report["origin_xyz_um"])
        z, y, x = np.indices(data.shape)
        expected = ((origin[0] + x * spacing[0]) / .65
                    + 3 * (origin[1] + y * spacing[1]) / .65
                    + 10 * (origin[2] + z * spacing[2]) / 3)
        np.testing.assert_array_equal(data, np.rint(expected).astype(np.uint16))
        self.assertGreater(len(report["loaded"]), 1)

    def test_all_missing_and_missing_root_cube_have_no_usable_array(self):
        native = [234.2, 234.2, 271.0]
        for callback in (lambda *index: None,
                         lambda x, y, z: None if [x, y, z] == [1, 1, 1]
                         else np.ones((90, 360, 360), dtype=np.uint16)):
            data, report = derive_highres_context(native, callback, field_um=24, depth_um=12)
            self.assertIsNone(data)
            self.assertEqual(report["status"], "context_unavailable")
            self.assertFalse(report["central_cube_loaded"])

    def test_missing_neighbor_has_explicit_partial_coverage(self):
        data, report = derive_highres_context(
            [234.2, 234.2, 271.0],
            lambda x, y, z: None if [x, y, z] == [0, 0, 0]
            else np.ones((90, 360, 360), dtype=np.uint16), field_um=24, depth_um=12)
        self.assertIsNotNone(data)
        self.assertEqual(report["status"], "derived_partial")
        self.assertTrue(report["root_pixel_covered"])
        self.assertLess(report["coverage_fraction"], 1)

    def test_malformed_cube_is_rejected(self):
        for volume in (np.ones((4, 40, 40), dtype=np.uint16),
                       np.ones((90, 360, 360), dtype=np.float32)):
            data, report = derive_highres_context([100, 100, 100], lambda *idx: volume,
                                                  field_um=24, depth_um=12)
            self.assertIsNone(data)
            self.assertEqual(len(report["invalid"]), 1)
            self.assertEqual(report["loaded"], [])

    def test_resource_bounds_reject_before_request_or_allocation(self):
        request = mock.Mock(side_effect=AssertionError("no download allowed"))
        for options in ({"factor": 1}, {"max_source_bytes": 1}, {"max_cubes": 1}):
            with mock.patch("fmost_derived_context.np.zeros", side_effect=AssertionError("no allocation allowed")):
                data, report = derive_highres_context([40200, 31689, 19036], request, **options)
            self.assertIsNone(data)
            self.assertEqual(report["status"], "not_run")
        request.assert_not_called()

    def test_runtime_stop_retains_unattempted_source_list(self):
        with mock.patch("fmost_derived_context.time.monotonic", side_effect=[0, 1000, 1000]):
            data, report = derive_highres_context([234.2, 234.2, 271], mock.Mock(),
                                                  field_um=24, depth_um=12, max_seconds=10)
        self.assertIsNone(data)
        self.assertEqual(len(report["unattempted"]), report["cube_count"])

    def test_copied_context_preserved_before_source_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "252384" / "001"
            folder.mkdir(parents=True)
            path = folder / "context.nii.gz"
            path.write_bytes(b"existing copied artifact")
            import pandas as pd
            manifest = pd.DataFrame([{"SampleID": "252384", "NeuronID": "001.swc", "project_id": "Monkey"}])
            with mock.patch.object(review, "load_manifest", return_value=manifest), \
                    mock.patch.object(review, "fetch_raw_swc") as fetch:
                with self.assertRaisesRegex(ValueError, "cannot replace"):
                    review.pilot_derived_context("252384", "001.swc", root)
            fetch.assert_not_called()
            self.assertEqual(path.read_bytes(), b"existing copied artifact")

    def test_missing_plane_does_not_shift_later_array_slot_labels(self):
        acquisition = {"requested": [10, 11, 12, 13], "loaded": [10, 12, 13], "missing": [11]}
        self.assertEqual(review.context_plane_ids(acquisition, 4, [0, 0, 30], [5, 5, 3]),
                         [10, 11, 12, 13])
        self.assertEqual(review.context_plane_ids({}, 4, [0, 0, 30], [5, 5, 3]),
                         [10, 11, 12, 13])

    def test_pilot_exports_matching_pixels_axes_and_trace_planes(self):
        import nibabel as nib
        import pandas as pd
        native = [100.1, 100.3, 99.2]
        rows = review.parse_swc("1 1 100.1 100.3 99.2 1 -1\n2 3 105 102 100 1 1\n")
        canvas, expected = derive_highres_context(native,
            lambda *idx: np.ones((90, 360, 360), dtype=np.uint16), field_um=24, depth_um=12)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = pd.DataFrame([{"SampleID": "252790", "NeuronID": "014.swc", "project_id": "Monkey"}])
            with mock.patch.object(review, "load_manifest", return_value=manifest), \
                    mock.patch.object(review, "fetch_raw_swc", return_value=(rows, "sha", "url", "raw")), \
                    mock.patch.object(review, "derive_highres_context", return_value=(canvas, expected.copy())):
                report = review.pilot_derived_context("252790", "014.swc", root)
            folder = root / "252790" / "014"
            image = nib.load(str(folder / "context.nii.gz"))
            np.testing.assert_array_equal(np.asanyarray(image.dataobj), canvas.transpose(2, 1, 0))
            np.testing.assert_allclose(image.affine[:3, 3], report["origin_xyz_um"])
            self.assertEqual(image.header.get_xyzt_units()[0], "micron")
            meta = json.loads((folder / "context.nii.gz.json").read_text())
            self.assertEqual(meta["acquisition"]["requested"], report["requested"])
            self.assertEqual(meta["acquisition"]["source_plane_ids"], report["source_plane_ids"])
            record = json.loads((folder / "provenance.json").read_text())
            self.assertEqual(record["context_plane_ids"], report["source_plane_ids"])
            for name in ("context_trace.png", "context_stack.png", "context_ortho.png"):
                self.assertTrue((folder / name).is_file())
            self.assertFalse((folder / "section_locator.png").exists())
            self.assertIn("not a full-section", record["section_locator_status"])

    def test_http_size_and_header_guards_precede_pixel_decode(self):
        import Visual_toolkit as visual
        from io import BytesIO
        import tifffile
        payload = BytesIO()
        tifffile.imwrite(payload, np.ones((90, 40, 40), dtype=np.uint16))
        payload.seek(0)
        with mock.patch.object(visual, "_read_tiff") as decode:
            with self.assertRaisesRegex(ValueError, "Expected uint16"):
                visual._read_highres_cube(payload)
            decode.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            toolkit = visual.Visual_toolkit("252790", cache_dir=temp)
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = b"x" * 9
            with mock.patch.object(visual, "MAX_HTTP_BLOCK_BYTES", 8), \
                    mock.patch.object(visual.urllib.request, "urlopen", return_value=response), \
                    mock.patch.object(visual, "_read_highres_cube") as decode, \
                    mock.patch.object(visual.time, "sleep"):
                self.assertIsNone(toolkit._download_http_block(1, 2, 3))
            decode.assert_not_called()
            response.__enter__.return_value.read.assert_called_with(9)
            self.assertFalse((Path(toolkit.cache_http_dir) / "3" / "1_2_3.tif").exists())


if __name__ == "__main__":
    unittest.main()
