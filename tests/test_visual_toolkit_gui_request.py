"""Regression checks for stable VisualToolkit GUI processing requests."""
import os
import sys
from pathlib import Path
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(ROOT / "neuron-vis" / "neuronVis"))

from Visual_toolkit.gui import NeuroVisGUI, identity_matches, make_processing_request
import Visual_toolkit.gui as gui_module
sys.path.insert(0, str(ROOT / "work" / "visual_qc_251637_115"))
import render_251637_115_toolkit as native_render


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class GuiRequestTests(unittest.TestCase):
    def test_render_runner_rejects_processed_atlas_swc_before_acquisition(self):
        raw = ROOT / "work" / "visual_qc_251637_115" / "source_raw" / "251637" / "115.swc"
        processed = ROOT / "main_scripts" / "processed_neurons" / "251637" / "115.swc"
        self.assertEqual(native_render.verify_native_swc_source(raw, native_render.EXPECTED_RAW_SWC_SHA256), native_render.EXPECTED_RAW_SWC_SHA256)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            native_render.verify_native_swc_source(processed, native_render.EXPECTED_RAW_SWC_SHA256)

    def test_request_is_frozen_and_preserves_native_coordinate_precision(self):
        request = make_processing_request(
            "251637", "115.swc", ("9899.75", "29246.2890625", "25624.23046875"),
            2, "8000", "8000", "30", "out",
        )
        self.assertEqual(request.soma_xyz, (9899.75, 29246.2890625, 25624.23046875))
        with self.assertRaises(AttributeError):
            request.sample_id = "other"
        with self.assertRaisesRegex(ValueError, "positive"):
            make_processing_request("251637", "115.swc", (1, 2, 3), 0, 8, 8, 3, "out")
        with self.assertRaisesRegex(ValueError, "finite"):
            make_processing_request("251637", "115.swc", (float("inf"), 2, 3), 1, 8, 8, 3, "out")
        with self.assertRaisesRegex(ValueError, "finite"):
            make_processing_request("251637", "115.swc", (float("inf"), 2, 3), 1, 8, 8, 3, "out")

    def test_old_async_autofill_identity_does_not_match_new_selection(self):
        self.assertTrue(identity_matches(("251637", "115.swc"), "251637", "115.swc"))
        self.assertFalse(identity_matches(("251637", "115.swc"), "251637", "114.swc"))
        self.assertFalse(identity_matches(("251637", "115.swc"), "252385", "115.swc"))

    def test_identity_edit_clears_coordinates_before_next_request(self):
        gui = NeuroVisGUI.__new__(NeuroVisGUI)
        gui.sample_id = Value("251637")
        gui.neuron_id = Value("115.swc")
        gui.x_coord, gui.y_coord, gui.z_coord = Value("9899.75"), Value("29246.2890625"), Value("25624.23046875")
        gui._selected_identity = ("251637", "115.swc")
        gui._autofill_identity = ("251637", "115.swc")
        gui.neuron_id.set("114.swc")
        gui._on_identity_change()
        self.assertIsNone(gui._autofill_identity)
        self.assertEqual((gui.x_coord.get(), gui.y_coord.get(), gui.z_coord.get()), ("", "", ""))

    def test_late_autofill_result_is_discarded_after_identity_change(self):
        class Root:
            def __init__(self):
                self.callbacks = []

            def after(self, delay, callback):
                self.callbacks.append(callback)

            def update_idletasks(self):
                pass

        class Tree:
            root = type("Soma", (), {"x": 10.25, "y": 20.5, "z": 30.75})()

        gui = NeuroVisGUI.__new__(NeuroVisGUI)
        gui.root = Root()
        gui.sample_id, gui.neuron_id = Value("251637"), Value("115.swc")
        gui.x_coord, gui.y_coord, gui.z_coord = Value(""), Value(""), Value("")
        gui.status_text = Value("")
        gui._operation_running = False
        gui._selected_identity = ("251637", "115.swc")
        gui._autofill_identity = None
        gui._launch_processing = lambda callback: callback()
        client = type("ION", (), {"getRawNeuronTreeByID": lambda self, sample, neuron: Tree()})
        with mock.patch("Visual_toolkit.gui.IT.IONData", client):
            gui.load_and_autofill(silent_success=True)
        gui.neuron_id.set("114.swc")
        gui._on_identity_change()
        gui.root.callbacks[0]()
        self.assertEqual((gui.x_coord.get(), gui.y_coord.get(), gui.z_coord.get()), ("", "", ""))

    def test_current_autofill_result_keeps_fractional_root(self):
        class Root:
            def __init__(self):
                self.callbacks = []

            def after(self, delay, callback):
                self.callbacks.append(callback)

            def update_idletasks(self):
                pass

        class Tree:
            root = type("Soma", (), {"x": 10.25, "y": 20.5, "z": 30.75})()

        gui = NeuroVisGUI.__new__(NeuroVisGUI)
        gui.root = Root()
        gui.sample_id, gui.neuron_id = Value("251637"), Value("115.swc")
        gui.x_coord, gui.y_coord, gui.z_coord = Value(""), Value(""), Value("")
        gui.status_text = Value("")
        gui._operation_running = False
        gui._selected_identity = ("251637", "115.swc")
        gui._autofill_identity = None
        gui._launch_processing = lambda callback: callback()
        client = type("ION", (), {"getRawNeuronTreeByID": lambda self, sample, neuron: Tree()})
        with mock.patch("Visual_toolkit.gui.IT.IONData", client):
            gui.load_and_autofill(silent_success=True)
        gui.root.callbacks[0]()
        self.assertEqual((gui.x_coord.get(), gui.y_coord.get(), gui.z_coord.get()), ("10.25", "20.5", "30.75"))
        self.assertEqual(gui._autofill_identity, ("251637", "115.swc"))

    def test_all_action_workers_use_the_original_frozen_request(self):
        class Root:
            def update_idletasks(self):
                pass

            def after(self, delay, callback):
                callback()

        class Progress:
            def pack(self, *args, **kwargs): pass
            def pack_forget(self): pass
            def start(self): pass
            def stop(self): pass

        calls = []

        class Toolkit:
            def __init__(self, sample_id):
                calls.append(("toolkit", sample_id))

            def get_high_res_block(self, xyz, grid_radius):
                calls.append(("high", tuple(xyz), grid_radius))
                return "high-volume", "high-origin", "high-spacing"

            def get_high_res_soma_window(self, xyz, window_px=None):
                calls.append(("high", tuple(xyz), window_px))
                return "high-volume", "high-origin", "high-spacing"

            def render_sagittal_context_from_ds(self, neuron_id, xyz, **kwargs):
                calls.append(("sagittal", tuple(xyz), neuron_id, kwargs.get("output_dir")))
                return {"panel": {"output_png": "sagittal.png"}, "provenance": {}}

            def plot_tri_plot(self, panels, neuron_id, **kwargs):
                calls.append(("trio", len(panels), neuron_id, kwargs.get("output_dir")))
                return "trio.png"

            def get_low_res_widefield(self, xyz, **kwargs):
                calls.append(("low", tuple(xyz), kwargs))
                return "low-volume", "low-origin", "low-spacing"

            def export_data(self, volume, origin, spacing, neuron_id, **kwargs):
                calls.append(("export", volume, neuron_id, kwargs["output_dir"]))

            def plot_soma_block(self, volume, origin, spacing, xyz, neuron_id, **kwargs):
                calls.append(("plot-high", tuple(xyz), neuron_id, kwargs["output_dir"]))

            def plot_widefield_context(self, volume, origin, spacing, xyz, neuron_id, **kwargs):
                calls.append(("plot-low", tuple(xyz), neuron_id, kwargs["output_dir"]))

            def close(self): pass

        class ION:
            def getRawNeuronTreeByID(self, sample_id, neuron_id):
                calls.append(("raw-tree", sample_id, neuron_id))
                return object()

        for action in ("run_highres", "run_lowres", "run_both"):
            with self.subTest(action=action):
                calls.clear()
                pending = []
                gui = NeuroVisGUI.__new__(NeuroVisGUI)
                gui.root = Root()
                gui.sample_id, gui.neuron_id = Value("251637"), Value("115.swc")
                gui.x_coord, gui.y_coord, gui.z_coord = Value("9899.75"), Value("29246.2890625"), Value("25624.23046875")
                gui.grid_radius = Value(2)
                gui.width_um, gui.height_um, gui.depth_um = Value("8000"), Value("8000"), Value("30")
                gui.output_dir, gui.status_text = Value("out-original"), Value("")
                gui.progress = Progress()
                gui._operation_running = False
                gui.validate_inputs = lambda: True
                gui.prepare_output_dir = lambda: gui.output_dir.get()
                gui._launch_processing = lambda callback: pending.append(callback)
                patches = [
                    mock.patch.object(gui_module, "Visual_toolkit", Toolkit),
                    mock.patch.object(gui_module.IT, "IONData", ION),
                    mock.patch.object(gui_module.messagebox, "showinfo"),
                    mock.patch.object(gui_module.messagebox, "showerror"),
                ]
                if action == "run_both":
                    patches.append(mock.patch.object(gui_module.messagebox, "askyesno", return_value=True))
                with patches[0], patches[1], patches[2], patches[3]:
                    if action == "run_both":
                        with patches[4]:
                            getattr(gui, action)()
                    else:
                        getattr(gui, action)()
                    gui.sample_id.set("252385")
                    gui.neuron_id.set("050.swc")
                    gui.x_coord.set("1")
                    gui.output_dir.set("out-new")
                    pending[0]()
                self.assertIn(("toolkit", "251637"), calls)
                self.assertTrue(any(call[0] in ("high", "low") and call[1] == (9899.75, 29246.2890625, 25624.23046875) for call in calls))
                self.assertTrue(all(not (call[0] == "export" and call[2] == "050.swc") for call in calls))
                self.assertTrue(all(not (call[0] == "export" and call[3] == "out-new") for call in calls))

    def test_busy_gate_does_not_start_a_second_worker(self):
        gui = NeuroVisGUI.__new__(NeuroVisGUI)
        gui._operation_running = True
        called = []
        with mock.patch.object(gui_module.messagebox, "showwarning"):
            gui._launch_processing(lambda: called.append(True))
        self.assertEqual(called, [])

    def test_all_action_workers_use_the_original_frozen_request(self):
        class Root:
            def update_idletasks(self):
                pass

            def after(self, delay, callback):
                callback()

        class Progress:
            def pack(self, *args, **kwargs): pass
            def pack_forget(self): pass
            def start(self): pass
            def stop(self): pass

        calls = []

        class Toolkit:
            def __init__(self, sample_id):
                calls.append(("toolkit", sample_id))

            def get_high_res_block(self, xyz, grid_radius):
                calls.append(("high", tuple(xyz), grid_radius))
                return "high-volume", "high-origin", "high-spacing"

            def get_high_res_soma_window(self, xyz, window_px=None):
                calls.append(("high", tuple(xyz), window_px))
                return "high-volume", "high-origin", "high-spacing"

            def render_sagittal_context_from_ds(self, neuron_id, xyz, **kwargs):
                calls.append(("sagittal", tuple(xyz), neuron_id, kwargs.get("output_dir")))
                return {"panel": {"output_png": "sagittal.png"}, "provenance": {}}

            def plot_tri_plot(self, panels, neuron_id, **kwargs):
                calls.append(("trio", len(panels), neuron_id, kwargs.get("output_dir")))
                return "trio.png"

            def get_low_res_widefield(self, xyz, **kwargs):
                calls.append(("low", tuple(xyz), kwargs))
                return "low-volume", "low-origin", "low-spacing"

            def export_data(self, volume, origin, spacing, neuron_id, **kwargs):
                calls.append(("export", volume, neuron_id, kwargs["output_dir"]))

            def plot_soma_block(self, volume, origin, spacing, xyz, neuron_id, **kwargs):
                calls.append(("plot-high", tuple(xyz), neuron_id, kwargs["output_dir"]))

            def plot_widefield_context(self, volume, origin, spacing, xyz, neuron_id, **kwargs):
                calls.append(("plot-low", tuple(xyz), neuron_id, kwargs["output_dir"]))

            def close(self): pass

        class ION:
            def getRawNeuronTreeByID(self, sample_id, neuron_id):
                calls.append(("raw-tree", sample_id, neuron_id))
                return object()

        for action in ("run_highres", "run_lowres", "run_both"):
            with self.subTest(action=action):
                calls.clear()
                pending = []
                gui = NeuroVisGUI.__new__(NeuroVisGUI)
                gui.root = Root()
                gui.sample_id, gui.neuron_id = Value("251637"), Value("115.swc")
                gui.x_coord, gui.y_coord, gui.z_coord = Value("9899.75"), Value("29246.2890625"), Value("25624.23046875")
                gui.grid_radius = Value(2)
                gui.width_um, gui.height_um, gui.depth_um = Value("8000"), Value("8000"), Value("30")
                gui.output_dir, gui.status_text = Value("out-original"), Value("")
                gui.progress = Progress()
                gui._operation_running = False
                gui.validate_inputs = lambda: True
                gui.prepare_output_dir = lambda: gui.output_dir.get()
                gui._launch_processing = lambda callback: pending.append(callback)
                patches = [
                    mock.patch.object(gui_module, "Visual_toolkit", Toolkit),
                    mock.patch.object(gui_module.IT, "IONData", ION),
                    mock.patch.object(gui_module.messagebox, "showinfo"),
                    mock.patch.object(gui_module.messagebox, "showerror"),
                ]
                if action == "run_both":
                    patches.append(mock.patch.object(gui_module.messagebox, "askyesno", return_value=True))
                with patches[0], patches[1], patches[2], patches[3]:
                    if action == "run_both":
                        with patches[4]:
                            getattr(gui, action)()
                    else:
                        getattr(gui, action)()
                    gui.sample_id.set("252385")
                    gui.neuron_id.set("050.swc")
                    gui.x_coord.set("1")
                    gui.output_dir.set("out-new")
                    pending[0]()
                self.assertIn(("toolkit", "251637"), calls)
                self.assertTrue(any(call[0] in ("high", "low") and call[1] == (9899.75, 29246.2890625, 25624.23046875) for call in calls))
                self.assertTrue(all(not (call[0] == "export" and call[2] == "050.swc") for call in calls))
                self.assertTrue(all(not (call[0] == "export" and call[3] == "out-new") for call in calls))

    def test_busy_gate_does_not_start_a_second_worker(self):
        gui = NeuroVisGUI.__new__(NeuroVisGUI)
        gui._operation_running = True
        called = []
        with mock.patch.object(gui_module.messagebox, "showwarning"):
            gui._launch_processing(lambda: called.append(True))
        self.assertEqual(called, [])

    def test_sagittal_action_uses_the_frozen_request(self):
        """The sagittal action must use the captured request, not later UI edits."""
        class Root:
            def update_idletasks(self):
                pass

            def after(self, delay, callback):
                callback()

        class Progress:
            def pack(self, *args, **kwargs): pass
            def pack_forget(self): pass
            def start(self): pass
            def stop(self): pass

        calls = []

        class Toolkit:
            def __init__(self, sample_id):
                calls.append(("toolkit", sample_id))

            def render_sagittal_context_from_ds(self, neuron_id, xyz, **kwargs):
                calls.append(("sagittal", tuple(xyz), neuron_id, kwargs.get("output_dir")))
                return {"panel": {"output_png": "sagittal.png"}, "provenance": {}}

            def close(self): pass

        pending = []
        gui = NeuroVisGUI.__new__(NeuroVisGUI)
        gui.root = Root()
        gui.sample_id, gui.neuron_id = Value("252527"), Value("048.swc")
        gui.x_coord, gui.y_coord, gui.z_coord = Value("17544.1328125"), Value("31518.39453125"), Value("18892.5546875")
        gui.grid_radius = Value(1)
        gui.width_um, gui.height_um, gui.depth_um = Value("8000"), Value("8000"), Value("30")
        gui.output_dir, gui.status_text = Value("out-original"), Value("")
        gui.progress = Progress()
        gui._operation_running = False
        gui.validate_inputs = lambda: True
        gui.prepare_output_dir = lambda: gui.output_dir.get()
        gui._launch_processing = lambda callback: pending.append(callback)
        with mock.patch.object(gui_module, "Visual_toolkit", Toolkit), \
                mock.patch.object(gui_module.messagebox, "showinfo"), \
                mock.patch.object(gui_module.messagebox, "showerror"):
            gui.run_sagittal()
            gui.sample_id.set("252385")
            gui.neuron_id.set("050.swc")
            gui.output_dir.set("out-new")
            pending[0]()
        self.assertIn(("toolkit", "252527"), calls)
        self.assertIn(("sagittal", (17544.1328125, 31518.39453125, 18892.5546875), "048.swc",
                       "out-original"), calls)
        self.assertTrue(all(not (call[0] == "sagittal" and call[3] == "out-new") for call in calls))

    def test_combined_review_figure_is_opt_in(self):
        """The trio is built only when the review-figure option is ticked."""
        class Root:
            def update_idletasks(self): pass

            def after(self, delay, callback):
                callback()

        class Progress:
            def pack(self, *args, **kwargs): pass
            def pack_forget(self): pass
            def start(self): pass
            def stop(self): pass

        class Toolkit:
            def __init__(self, sample_id): pass

            def get_high_res_soma_window(self, xyz, window_px=None):
                return "vol", [0, 0, 0], [0.65, 0.65, 3.0]

            def plot_soma_block(self, *args, **kwargs):
                return {"output_png": None}

            def get_low_res_widefield(self, xyz, **kwargs):
                return "vol", [0, 0, 0], [5.0, 5.0, 3.0]

            def plot_widefield_context(self, *args, **kwargs):
                return {"output_png": None}

            def render_sagittal_context_from_ds(self, neuron_id, xyz, **kwargs):
                return {"panel": {"output_png": None}, "provenance": {}}

            def export_data(self, *args, **kwargs): pass

            def plot_tri_plot(self, panels, neuron_id, **kwargs):
                calls.append("trio")
                return "trio.png"

            def close(self): pass

        class ION:
            def getRawNeuronTreeByID(self, *args):
                return object()

        calls = []
        for action, make_review, expected in (
            ("run_selected", False, 0),
            ("run_selected", True, 1),
            ("run_all_panels", False, 1),  # Render all always builds the TrioPlot.
        ):
            with self.subTest(action=action, make_review=make_review):
                calls.clear()
                pending = []
                gui = NeuroVisGUI.__new__(NeuroVisGUI)
                gui.root = Root()
                gui.sample_id, gui.neuron_id = Value("251637"), Value("003.swc")
                gui.x_coord, gui.y_coord, gui.z_coord = Value("1"), Value("2"), Value("3")
                gui.grid_radius = Value(1)
                gui.width_um, gui.height_um, gui.depth_um = Value("8000"), Value("8000"), Value("30")
                gui.output_dir, gui.status_text = Value("out"), Value("")
                gui.soma_cubes = Value("1.5")
                gui.plot_soma = gui.plot_context = gui.plot_sagittal = Value(True)
                gui.save_soma_png = gui.save_context_png = Value(True)
                gui.save_sagittal_png = gui.save_soma_nifti = Value(True)
                gui.save_outputs = Value(True)
                gui.make_review_figure = Value(make_review)
                gui.progress = Progress()
                gui._operation_running = False
                gui.validate_inputs = lambda: True
                gui.prepare_output_dir = lambda: gui.output_dir.get()
                gui._launch_processing = lambda callback: pending.append(callback)
                with mock.patch.object(gui_module, "Visual_toolkit", Toolkit), \
                        mock.patch.object(gui_module.IT, "IONData", ION), \
                        mock.patch.object(gui_module.messagebox, "showerror"), \
                        mock.patch.object(gui_module.messagebox, "showwarning"):
                    getattr(gui, action)()
                    pending[0]()
                self.assertEqual(calls.count("trio"), expected)

    def test_cached_figure_check_reports_coordinates_settings_and_files(self):
        """The reload gate checks coordinates, settings key and file hashes."""
        import tempfile
        from PIL import Image

        folder = tempfile.mkdtemp(prefix="gui_cache_check_")
        png = os.path.join(folder, "251637_003.swc_Soma_Plot.png")
        Image.new("RGB", (8, 8), (0, 0, 0)).save(png)

        gui = NeuroVisGUI.__new__(NeuroVisGUI)
        gui.sample_id, gui.neuron_id = Value("251637"), Value("003.swc")
        gui.x_coord, gui.y_coord, gui.z_coord = Value("1"), Value("2"), Value("3")
        gui.soma_cubes = Value("1.5")
        gui.width_um, gui.height_um, gui.depth_um = Value("8000"), Value("8000"), Value("30")
        fingerprint = gui._fingerprint()
        gui._write_receipt(folder, fingerprint, {"Soma": png})
        entry = gui._read_receipt(folder, "003.swc")
        self.assertEqual(gui._evaluate_cached(entry), ("same", []))

        gui.x_coord.set("9")
        state, notes = gui._evaluate_cached(entry)
        self.assertEqual(state, "differs")
        self.assertIn("coordinates changed", notes)

        gui.x_coord.set("1")
        Image.new("RGB", (8, 8), (255, 255, 255)).save(png)
        state, notes = gui._evaluate_cached(entry)
        self.assertEqual(state, "differs")
        self.assertIn("cached files changed or missing", notes)


if __name__ == "__main__":
    unittest.main()
