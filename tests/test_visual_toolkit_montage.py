"""Offline checks for the packaged DS00 five-view montage renderer."""
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))

from Visual_toolkit import montage  # noqa: E402


class MontageCoreTests(unittest.TestCase):
    def test_count_text_preserves_missing_and_integer_values(self):
        self.assertEqual(montage.count_text(None), "—")
        self.assertEqual(montage.count_text(""), "—")
        self.assertEqual(montage.count_text(12.0), "12")
        self.assertEqual(montage.count_text("7.5"), "7.5")

    def test_lps_camera_basis_is_orthogonal_for_every_view(self):
        affine = np.diag([-2.0, -3.0, 4.0, 1.0])
        axes = montage.anatomical_axes(affine)
        self.assertTrue(np.allclose(axes["left"], [-1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(axes["posterior"], [0.0, -1.0, 0.0]))
        self.assertTrue(np.allclose(axes["superior"], [0.0, 0.0, 1.0]))
        for view in montage.VIEW_KEYS:
            forward, up = montage.view_forward_up(axes, view)
            self.assertAlmostEqual(float(np.linalg.norm(forward)), 1.0, places=7)
            self.assertAlmostEqual(float(np.linalg.norm(up)), 1.0, places=7)
            self.assertAlmostEqual(float(np.dot(forward, up)), 0.0, places=7)

    def test_sha256_reads_streamed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.bin"
            path.write_bytes(b"visual-toolkit")
            self.assertEqual(
                montage.sha256(path),
                "b1273ca31519677621f1459ddc530080f0fdcfc2d75b634c6288374913992431",
            )

    def test_labels_cli_uses_the_shared_panel_order(self):
        labels = [f"row {index}" for index in range(len(montage.PANELS))]
        stream = io.StringIO()
        with mock.patch.object(montage, "labels_for_panels", return_value=labels), \
                mock.patch.object(sys, "argv", ["montage", "labels"]), \
                redirect_stdout(stream):
            montage.main()
        output = stream.getvalue()
        self.assertIn("LABEL 936 row 0", output)
        self.assertIn("LABEL 747 row 8", output)

    def test_legacy_tool_path_is_a_package_wrapper(self):
        wrapper = ROOT / "tools" / "ch00ds_5view_montage.py"
        spec = importlib.util.spec_from_file_location("legacy_ch00ds_montage", wrapper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.PANELS, montage.PANELS)
        self.assertEqual(module.main.__name__, "main")
        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
