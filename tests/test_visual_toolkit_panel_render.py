"""Offline render checks for the VisualToolkit review panels and combined figure.

These exercise the shared panel drawing path, the target marker (cyan box + cross
in every panel type), the title map and the legend path without touching any share
or network source, so a broken panel render is caught before a long bulk export
uses it.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(ROOT / "neuron-vis" / "neuronVis"))

from Visual_toolkit import (  # noqa: E402
    Visual_toolkit, REVIEW_SETTINGS, TARGET_MARKER_COLOR,
    apply_setting_overrides, review_settings_snapshot,
)

CYAN_RGB = tuple(int(TARGET_MARKER_COLOR[i:i + 2], 16) for i in (1, 3, 5))


def cyan_mask(path: str) -> np.ndarray:
    """Boolean mask of the target-marker colour in a rendered PNG."""
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
    distance = np.abs(rgb - np.array(CYAN_RGB)).sum(axis=2)
    return distance < 120


def synthetic_panel(role: str, suffix: str, marker_color: str = "cyan",
                    with_trace: bool = False, label: str = "Target Soma") -> dict:
    """One panel dict in the documented schema; the marker uses the shared style."""
    y, x = np.mgrid[0:48, 0:48]
    display = ((np.sin(x / 6.0) + np.cos(y / 7.0)) + 2.0) / 4.0
    panel = {
        "display": display, "cmap": "gray", "resolution_um": (0.65, 0.65),
        "fov_um": (48 * 0.65, 48 * 0.65), "depth_um": 270.0, "scale_bar_um": 30.0,
        "markers": [(24.0, 24.0, marker_color, 20.0, label)],
        "edge_segments": [], "traces": [], "axis_labels": ("j", "k"),
        "legend": [("patch", "gray", f"{role} display"), ("marker", marker_color, role)],
        "notes": ["synthetic test panel"], "role": role, "suffix": suffix,
    }
    if with_trace:
        panel["traces"] = [(np.array([[4.0, 4.0], [44.0, 40.0]]), "#de2d2c", "dashed", 1.2, 0.9)]
    return panel


class PanelRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="visual_toolkit_panel_")
        self.toolkit = Visual_toolkit("251637", cache_dir=self.tmp)
        self.panels = [
            synthetic_panel("Soma", "SomaBlock", marker_color=TARGET_MARKER_COLOR),
            synthetic_panel("Context", "WideField", marker_color=TARGET_MARKER_COLOR),
            synthetic_panel("Sagittal", "Sagittal", marker_color=TARGET_MARKER_COLOR,
                            with_trace=True),
        ]
        for panel in self.panels:
            panel["title"] = self.toolkit._title_text(
                "003.swc", panel["suffix"], "Ig", (100.0, 200.0, 300.0),
                panel["fov_um"], panel["depth_um"])

    def test_single_sagittal_panel_carries_the_title_and_writes_a_png(self):
        styled = self.toolkit.plot_sagittal_context(
            self.panels[2], "003.swc", soma_region="Ig",
            soma_coords=(100.0, 200.0, 300.0), output_dir=self.tmp)
        self.assertTrue(os.path.isfile(styled["output_png"]))
        self.assertTrue(styled["output_png"].endswith("_Sagittal_Plot.png"))
        self.assertIn("Sagittal Context Plot | src_dataset: DS00 N4", styled["title"])

    def test_combined_figure_offline_legend_band_optional(self):
        # One figure per output directory: the two variants share one filename.
        plain = self.toolkit.plot_tri_plot(
            self.panels, "003.swc", output_dir=os.path.join(self.tmp, "plain"), dpi=40)
        with_legend = self.toolkit.plot_tri_plot(
            self.panels, "003.swc", output_dir=os.path.join(self.tmp, "legend"),
            legend=True, dpi=40)
        self.assertTrue(plain.endswith("_TriPlot.png"))
        self.assertTrue(with_legend.endswith("_TriPlot.png"))
        for path in (plain, with_legend):
            self.assertTrue(os.path.isfile(path))
        # The legend band adds a drawn row, so the two renders cannot be identical.
        self.assertNotEqual(os.path.getsize(plain), os.path.getsize(with_legend))

    def test_two_panel_review_figure_uses_the_pair_suffix(self):
        pair = self.toolkit.plot_tri_plot(
            self.panels[:2], "003.swc", output_dir=self.tmp, suffix="PairPlot", dpi=40)
        self.assertTrue(pair.endswith("_PairPlot.png"))
        self.assertTrue(os.path.isfile(pair))

    def test_every_panel_type_declares_the_shared_cyan_marker(self):
        """The Soma, Context and Sagittal panels must all hand the same colour to the drawer."""
        soma_volume = (np.random.default_rng(0).random((6, 360, 360)) * 400).astype(np.uint16)
        soma_panel = self.toolkit.plot_soma_block(
            soma_volume, [0.0, 0.0, 0.0], [0.65, 0.65, 3.0],
            [360 * 0.65 / 2, 360 * 0.65 / 2, 9.0], "003.swc",
            soma_region="Ig", output_dir=self.tmp)
        context_volume = (np.random.default_rng(1).random((6, 200, 200)) * 900).astype(np.uint16)
        context_panel = self.toolkit.plot_widefield_context(
            context_volume, [0.0, 0.0, 0.0], [5.0, 5.0, 3.0],
            [200 * 5.0 / 2, 200 * 5.0 / 2, 9.0], "003.swc",
            soma_region="Ig", output_dir=self.tmp)
        sagittal_panel = self.panels[2]
        for role, panel in (("Soma", soma_panel), ("Context", context_panel),
                            ("Sagittal", sagittal_panel)):
            with self.subTest(role=role):
                self.assertEqual(panel["markers"][0][2], TARGET_MARKER_COLOR)
                self.assertGreater(REVIEW_SETTINGS["marker"]["crosshair_px"], 0)

    def test_one_helper_draws_every_marker(self):
        """A single marker implementation is what keeps the cross on every panel."""
        source = (ROOT / "main_scripts" / "Visual_toolkit" / "toolkit.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("ax.scatter("), 1)
        self.assertIn("def _draw_target_marker", source)

    def test_rendered_panels_show_the_cyan_box_and_cross(self):
        """Pixel check: a crossed box centred on the target, label-free so only the marker is cyan."""
        output_dir = os.path.join(self.tmp, "marker")
        os.makedirs(output_dir, exist_ok=True)
        y, x = np.mgrid[0:360, 0:360]
        display = ((np.sin(x / 30.0) + np.cos(y / 45.0)) + 2.0) / 4.0
        for role, suffix, resolution, box_um in (
            ("Soma", "SomaBlock", 0.65, 20.0),
            ("Context", "WideField", 5.0, 200.0),
            ("Sagittal", "Sagittal", 0.65, 20.0),
        ):
            with self.subTest(role=role):
                panel = {
                    "display": display, "cmap": "gray",
                    "resolution_um": (resolution, resolution),
                    "fov_um": (360 * resolution, 360 * resolution), "depth_um": 270.0,
                    "scale_bar_um": 30.0,
                    "markers": [(180.0, 180.0, TARGET_MARKER_COLOR, box_um, "")],
                    "edge_segments": [], "traces": [], "axis_labels": None,
                    "legend": [], "notes": [], "role": role, "suffix": suffix,
                    "title": self.toolkit._title_text(
                        "003.swc", suffix, "Ig", None,
                        (360 * resolution, 360 * resolution), 270.0),
                }
                path = self.toolkit._save_panel_plot(
                    panel, "003.swc", suffix, output_dir=output_dir)["output_png"]
                mask = cyan_mask(path)
                ys, xs = np.nonzero(mask)
                self.assertGreater(len(xs), 0, f"{role}: no cyan marker pixels")
                rows = np.bincount(ys, minlength=ys.max() + 1)
                cols = np.bincount(xs, minlength=xs.max() + 1)
                arm_row, arm_col = int(rows.argmax()), int(cols.argmax())
                row_x = np.sort(xs[ys == arm_row])
                col_y = np.sort(ys[xs == arm_col])
                # The cross passes through the target: the widest row and column of
                # marker pixels intersect there and extend both ways.
                arm_half = min(arm_col - row_x[0], row_x[-1] - arm_col,
                               arm_row - col_y[0], col_y[-1] - arm_row)
                self.assertGreater(arm_half, 50, f"{role}: cross arms too short")
                # The arms are symmetric about the target point.
                self.assertLess(abs((arm_col - row_x[0]) - (row_x[-1] - arm_col)),
                                0.1 * arm_half)
                # The box is the closed square around that centre: cyan is present on
                # all four diagonals well inside the arm tips, which a bare cross
                # (or a point marker) cannot produce.
                for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    hit = any(
                        mask[arm_row + dy * t - 8:arm_row + dy * t + 9,
                             arm_col + dx * t - 8:arm_col + dx * t + 9].any()
                        for t in range(10, int(0.6 * arm_half)))
                    self.assertTrue(hit, f"{role}: no box edge along ({dx}, {dy})")


class SettingOverrideTests(unittest.TestCase):
    """The operator handle for tuning review parameters without editing code."""

    def setUp(self):
        self.settings = review_settings_snapshot()

    def test_scalar_overrides_are_typed_and_recorded(self):
        changed = apply_setting_overrides(
            ["widefield.gain=1.6", "context.width_um=6000", "marker.color=#ff8800"],
            settings=self.settings)
        self.assertEqual(changed, {"widefield.gain": 1.6, "context.width_um": 6000,
                                   "marker.color": "#ff8800"})
        self.assertIsInstance(self.settings["widefield"]["gain"], float)
        self.assertIsInstance(self.settings["context"]["width_um"], int)

    def test_unknown_path_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "Unknown review setting"):
            apply_setting_overrides(["marker.colour=#00e5ee"], settings=self.settings)

    def test_non_scalar_and_invalid_values_are_rejected(self):
        for item in ("figure.panel_figsize=10", "marker.color=cyan",
                     "marker.crosshair_px=0", "--set", "marker.crosshair_px="):
            with self.subTest(item=item), self.assertRaises(ValueError):
                apply_setting_overrides([item], settings=self.settings)

    def test_snapshot_is_a_deep_copy(self):
        snapshot = review_settings_snapshot()
        snapshot["marker"]["color"] = "#000000"
        self.assertEqual(REVIEW_SETTINGS["marker"]["color"], TARGET_MARKER_COLOR)


if __name__ == "__main__":
    unittest.main()
