"""Verify the processed-DS sagittal renderer is owned by toolkit.py."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))

import Visual_toolkit as toolkit  # noqa: E402


class SagittalIntegrationTests(unittest.TestCase):
    def test_sagittal_functions_are_defined_by_toolkit(self):
        for name in (
            "SAMPLE_PATHS", "resolve_swc_path", "prepare_sagittal_source",
            "prepare_sagittal_panel", "render_sagittal_context", "SagittalDSContext",
        ):
            self.assertTrue(hasattr(toolkit, name), name)
        self.assertEqual(
            toolkit.prepare_sagittal_panel.__module__,
            "Visual_toolkit.toolkit",
        )

    def test_separate_sagittal_module_is_absent(self):
        self.assertIsNone(importlib.util.find_spec("Visual_toolkit.sagittal"))

    def test_sagittal_marker_uses_shared_review_settings(self):
        self.assertEqual(toolkit._marker_style(), toolkit.REVIEW_SETTINGS["marker"])


if __name__ == "__main__":
    unittest.main()
