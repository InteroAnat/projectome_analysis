"""Regression checks for bulk-visual panel composition.

The processed-DS sagittal panel does not depend on the 5 um widefield source, so a
sample without one (252383, 252718 verified 2026-10-17) must still get its Soma and
Sagittal panels plus a combined review figure. A missing Context panel is recorded
per neuron under ``panel_errors`` instead of failing the whole neuron.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "main_scripts", ROOT / "neuron-vis" / "neuronVis",
             ROOT / "group_analysis" / "scripts"):
    sys.path.insert(0, str(path))

import bulk_visual_multi_monkey as bvm  # noqa: E402


def _panel(role: str) -> dict:
    return {"role": role, "title": role, "fov_um": (100.0, 100.0), "depth_um": 3.0,
            "display": None, "resolution_um": (0.65, 0.65), "legend": []}


class FakeToolkit:
    """Minimal stand-in recording the panels and figure suffix requested."""

    def __init__(self, widefield_error: str | None = None):
        self.widefield_error = widefield_error
        self.figure_calls = []
        self.last_high_res_metadata = None
        self.last_low_res_metadata = None
        self.cache_http_dir = str(Path(tempfile.gettempdir()) / "panel_decoupling_cache")

    def get_high_res_soma_window(self, soma_xyz, window_px=None):
        return None, [0.0, 0.0, 0.0], [0.65, 0.65, 3.0]

    def export_data(self, *args, **kwargs):
        return str(Path(tempfile.gettempdir()) / "panel_decoupling_soma.nii.gz")

    def plot_soma_block(self, *args, **kwargs):
        return _panel("Soma")

    def get_low_res_widefield(self, *args, **kwargs):
        if self.widefield_error:
            raise RuntimeError(self.widefield_error)
        return None, [0.0, 0.0, 0.0], [5.0, 5.0, 3.0]

    def plot_widefield_context(self, *args, **kwargs):
        return _panel("Context")

    def plot_sagittal_context(self, panel, neuron_id, **kwargs):
        # The toolkit's single-panel plots return the panel dict with output_png added.
        return {**_panel("Sagittal"),
                "output_png": str(Path(tempfile.gettempdir()) / "panel_decoupling_sagittal.png")}

    def plot_tri_plot(self, panels, neuron_id, suffix="TriPlot", **kwargs):
        roles = [panel["role"] for panel in panels if panel]
        self.figure_calls.append((suffix, roles))
        figure = Path(tempfile.gettempdir()) / f"panel_decoupling_{suffix}.png"
        figure.write_bytes(b"figure")
        return str(figure)


class PanelCompositionTests(unittest.TestCase):
    def test_review_figure_suffix_follows_panel_count(self):
        self.assertEqual(
            bvm.review_figure_suffix([_panel("Soma"), None, _panel("Sagittal")]),
            "PairPlot")
        self.assertEqual(bvm.review_figure_suffix(
            [_panel("Soma"), _panel("Context"), _panel("Sagittal")]), "TriPlot")
        self.assertEqual(bvm.review_figure_suffix([None, None, _panel("Sagittal")]), None)
        self.assertEqual(bvm.review_figure_suffix([]), None)

    def _run_neuron(self, widefield_error):
        toolkit = FakeToolkit(widefield_error)
        tree = SimpleNamespace(root=SimpleNamespace(x=1000.0, y=2000.0, z=3000.0))
        dirs = {name: str(Path(tempfile.gettempdir()) / f"panel_decoupling_{name}")
                for name in ("high_img", "high_nii", "low_img", "low_nii", "trio")}
        provenance = mock.Mock(return_value={"neuron_id": "018.swc"})
        with mock.patch.multiple(
            bvm,
            GENERATE_HIGH_RES=True,
            GENERATE_LOW_RES=True,
            prepare_native_sagittal_panel=mock.Mock(
                return_value={"panel": _panel("Sagittal"), "provenance": {}}),
            neuron_visual_provenance=provenance,
        ):
            record = bvm.process_neuron(
                toolkit, None, "252383", "018.swc", "Ig", dirs,
                export_context_volume=False, context_sagittal=True,
                preloaded_tree=tree,
            )
        return toolkit, record, provenance.call_args.kwargs["context_sagittal"]

    def test_missing_context_source_keeps_soma_and_sagittal_panels(self):
        toolkit, record, sagittal = self._run_neuron(
            "Soma's central low-resolution section is unavailable")
        self.assertEqual(record["panel_errors"],
                         {"context": "Soma's central low-resolution section is unavailable"})
        self.assertTrue(sagittal["styled_png"].endswith("sagittal.png"))
        self.assertEqual(toolkit.figure_calls, [("PairPlot", ["Soma", "Sagittal"])])
        self.assertEqual(sagittal["review_figure_suffix"], "PairPlot")
        self.assertTrue(sagittal["tri_plot_sha256"])

    def test_complete_panel_set_still_writes_the_tri_plot(self):
        toolkit, record, sagittal = self._run_neuron(None)
        self.assertNotIn("panel_errors", record)
        self.assertEqual(toolkit.figure_calls,
                         [("TriPlot", ["Soma", "Context", "Sagittal"])])
        self.assertEqual(sagittal["review_figure_suffix"], "TriPlot")


if __name__ == "__main__":
    unittest.main()
