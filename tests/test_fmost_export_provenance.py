"""Source provenance must describe the image actually persisted by the exporter."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
import Visual_toolkit as visual


class ExportProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fmost-export-contract-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.toolkit = visual.Visual_toolkit.__new__(visual.Visual_toolkit)
        self.toolkit.sample_id = "fixture"
        self.toolkit.output_dir = str(self.directory)
        self.toolkit.last_high_res_metadata = None
        self.toolkit.last_low_res_metadata = None
        self.toolkit.last_export_metadata = None
        self.volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        self.origin = [100.0, 200.0, 300.0]
        self.spacing = [0.65, 0.65, 3.0]

    def acquisition(self, source, origin=None, spacing=None, shape=None, complete=False):
        return {
            "sample_id": "fixture", "source": source,
            "coordinate_units": "micrometre", "array_axis_order": "ZYX",
            "origin_xyz_um": list(self.origin if origin is None else origin),
            "spacing_xyz_um": list(self.spacing if spacing is None else spacing),
            "shape_zyx": list(self.volume.shape if shape is None else shape),
            "center_xyz_um": [100.65, 201.3, 303.0],
            "requested": [[0, 0, 0]], "loaded": [[0, 0, 0]],
            "missing": [] if complete else [[1, 0, 0]],
            "complete": complete, "source_paths": [source],
        }

    def export(self, volume=None, origin=None, spacing=None, **kwargs):
        with redirect_stdout(StringIO()):
            filename = self.toolkit.export_data(
                self.volume if volume is None else volume,
                self.origin if origin is None else origin,
                self.spacing if spacing is None else spacing,
                "test-neuron", **kwargs)
        return Path(filename), json.loads(Path(filename + ".json").read_text())

    def test_2d_tiff_has_yx_pixels_and_metadata_without_nifti_axes(self):
        pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
        path, metadata = self.export(volume=pixels, suffix="WideField")
        np.testing.assert_array_equal(tifffile.imread(path), pixels)
        self.assertEqual(metadata["array_axis_order"], "YX")
        self.assertEqual(metadata["source_array_axis_order"], "YX")
        self.assertEqual(metadata["shape_yx"], [4, 5])
        self.assertNotIn("shape_zyx", metadata)
        self.assertIsNone(metadata.get("nifti_axis_order"))

    def test_generic_high_export_never_inherits_stale_low_coverage(self):
        self.toolkit.last_high_res_metadata = self.acquisition("matching-high")
        self.toolkit.last_low_res_metadata = self.acquisition(
            "stale-low", origin=[10, 20, 30], spacing=[5, 5, 3],
            shape=[7, 8, 9], complete=True)
        _path, metadata = self.export()
        acquisition = metadata.get("acquisition")
        if acquisition is None:
            self.assertIsNone(metadata["complete"])
        else:
            self.assertEqual(acquisition["source"], "matching-high")
            self.assertFalse(metadata["complete"])
            self.assertEqual(metadata["missing"], [[1, 0, 0]])

    def test_mismatched_high_acquisition_is_unknown_or_rejected(self):
        self.toolkit.last_high_res_metadata = self.acquisition(
            "stale-high", origin=[900, 900, 900], shape=[7, 8, 9])
        try:
            _path, metadata = self.export(suffix="SomaBlock")
        except ValueError:
            return
        self.assertIsNone(metadata.get("acquisition"))
        self.assertIsNone(metadata.get("complete"))

    def test_matching_partial_acquisition_preserves_coverage_and_xyz_voxels(self):
        self.toolkit.last_high_res_metadata = self.acquisition("matching-high")
        root = [100.65, 201.3, 303.0]
        path, metadata = self.export(suffix="SomaBlock", soma_coords=root)
        image = nib.load(path)
        np.testing.assert_array_equal(np.asanyarray(image.dataobj), self.volume.transpose(2, 1, 0))
        np.testing.assert_allclose(image.affine[:3, 3], self.origin)
        np.testing.assert_allclose(np.diag(image.affine)[:3], self.spacing)
        self.assertEqual(image.header.get_xyzt_units()[0], "micron")
        self.assertEqual(metadata["source_array_axis_order"], "ZYX")
        self.assertEqual(metadata["nifti_axis_order"], "XYZ")
        self.assertEqual(metadata["center_xyz_um"], root)
        self.assertFalse(metadata["complete"])
        self.assertEqual(metadata["missing"], [[1, 0, 0]])

    def test_missing_root_and_acquisition_remain_unknown(self):
        _path, metadata = self.export()
        self.assertIsNone(metadata["center_xyz_um"])
        self.assertIsNone(metadata["acquisition"])
        self.assertIsNone(metadata["complete"])

    def test_malformed_or_nonfinite_root_fails_before_creating_files(self):
        for root in ([1, 2], [1, 2, 3, 4], [1, float("nan"), 3], [1, 2, float("inf")]):
            with self.subTest(root=root):
                with self.assertRaises(ValueError):
                    self.export(soma_coords=root)
                self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
