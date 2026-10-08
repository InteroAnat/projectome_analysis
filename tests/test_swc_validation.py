"""Structural SWC regression checks; no atlas, ION service, or GUI required."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


MAIN_SCRIPTS = Path(__file__).resolve().parents[1] / "main_scripts"
sys.path.insert(0, str(MAIN_SCRIPTS))
from swc_validation import parse_swc


VALID_SWC = "1 1 0 0 0 1 -1\n2 3 250 0 0 0.5 1\n3 2 500 0 0 0 2\n"


class ParseSwcTests(unittest.TestCase):
    def test_unsorted_custom_types_and_dendrite_to_axon_are_preserved(self):
        swc = """   # mixed processes are valid topology
3 2 500 0 0 0 2
1.0 1 0 0 0 1 -1

2 3 250 0 0 0.5 1 # dendrite parent of an axon
4 17 -250 0 0 0 1 99
"""
        rows = parse_swc(swc)
        self.assertEqual([row[0] for row in rows], [3, 1, 2, 4])
        self.assertEqual(rows[0][-1], 2)
        self.assertEqual(rows[-1][1], 17)
        self.assertEqual(rows[-1][2], -250)

    def test_integer_ids_are_not_rounded_through_float(self):
        node_id = 9007199254740993
        rows = parse_swc(f"{node_id} 1 0 0 0 0 -1\n{node_id + 1} 2 1 0 0 0 {node_id}")
        self.assertEqual(rows[0][0], node_id)
        self.assertEqual(rows[1][-1], node_id)

    def test_invalid_numeric_fields_are_rejected_with_source_and_line(self):
        cases = {
            "fractional ID": ("1.5 1 0 0 0 1 -1", "node ID"),
            "nonfinite ID": ("nan 1 0 0 0 1 -1", "node ID"),
            "zero ID": ("0 1 0 0 0 1 -1", "positive"),
            "negative ID": ("-2 1 0 0 0 1 -1", "positive"),
            "fractional type": ("1 1.1 0 0 0 1 -1", "node type"),
            "negative type": ("1 -2 0 0 0 1 -1", "node type"),
            "fractional parent": ("1 1 0 0 0 1 -1.5", "parent ID"),
            "zero parent": ("1 1 0 0 0 1 0", "parent ID"),
            "nonroot negative parent": ("1 1 0 0 0 1 -2", "parent ID"),
            "nan coordinate": ("1 1 nan 0 0 1 -1", "finite"),
            "inf coordinate": ("1 1 0 inf 0 1 -1", "finite"),
            "overflow coordinate": ("1 1 0 0 1e999 1 -1", "finite"),
            "nonfinite radius": ("1 1 0 0 0 -inf -1", "finite"),
            "negative radius": ("1 1 0 0 0 -1 -1", "radius"),
            "nonnumeric coordinate": ("1 1 x 0 0 1 -1", "numeric"),
            "short row": ("1 1 0 0 0 -1", "7 SWC columns"),
        }
        for name, (swc, expected) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, expected) as context:
                    parse_swc(swc, source="fixture.swc")
                self.assertIn("fixture.swc, line 1", str(context.exception))

    def test_invalid_topology_is_rejected(self):
        root = "1 1 0 0 0 1 -1\n"
        cases = {
            "empty": ("# no reconstruction", "found 0"),
            "no root": ("1 1 0 0 0 1 1", "found 0"),
            "multiple roots": (root + "2 2 1 0 0 1 -1", "found 2"),
            "duplicate ID": (root + "1 2 1 0 0 1 1", "duplicate node ID 1"),
            "orphan": (root + "2 2 1 0 0 1 99", "missing parent 99"),
            "self cycle": (root + "2 2 1 0 0 1 2", "cycle"),
            "detached cycle": (root + "2 2 1 0 0 1 3\n3 2 2 0 0 1 2", "cycle"),
        }
        for name, (swc, expected) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, expected):
                    parse_swc(swc)

    def test_long_chain_does_not_require_recursive_validation(self):
        n = 20000
        swc = "\n".join(
            f"{i} 2 {i} 0 0 0 {-1 if i == 1 else i - 1}" for i in range(1, n + 1)
        )
        self.assertEqual(len(parse_swc(swc)), n)


class TracerLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real tracer, replacing only the lab service and GUI backend
        # selection. Tests exercise public process(), branch construction, and
        # CSV export with the installed scientific libraries.
        ion_module = types.ModuleType("IONData")
        ion_module.IONData = mock.Mock()
        spec = importlib.util.spec_from_file_location(
            "neuro_tracer_swc_test", MAIN_SCRIPTS / "neuro_tracer.py"
        )
        module = importlib.util.module_from_spec(spec)
        previous_ion = sys.modules.get("IONData")
        sys.modules["IONData"] = ion_module
        try:
            with mock.patch("matplotlib.use"):
                spec.loader.exec_module(module)
        finally:
            if previous_ion is None:
                sys.modules.pop("IONData", None)
            else:
                sys.modules["IONData"] = previous_ion
        cls.module = module

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.tracer = self.module.neuro_tracer()
        self.tracer.output_dir = str(self.output)
        self.module.IONData.IONData.reset_mock()

    def assert_unloaded(self):
        self.assertEqual(self.tracer.nodes, {})
        self.assertEqual(self.tracer.root, [])
        self.assertEqual(self.tracer.branches, [])
        self.assertIsNone(self.tracer.swc_filename)
        self.assertIsNone(self.tracer.exp_no)

    def test_public_process_valid_input_preserves_geometry_and_edges(self):
        self.tracer.process("fixture", "007.swc", nii_space="monkey",
                            output_dir=str(self.output), swc=VALID_SWC)
        self.assertEqual(self.tracer.root.id, 1)
        self.assertEqual(self.tracer.nodes[3].x_nii, 2.0)
        self.assertEqual([node.id for node in self.tracer.terminal_nodes], [3])
        edges = [(a.id, b.id) for branch in self.tracer.branches
                 for a, b in zip(branch, branch[1:])]
        self.assertEqual(edges, [(1, 2), (2, 3)])
        self.assertTrue((self.output / "fixture_007.csv").exists())
        self.module.IONData.IONData.assert_not_called()

    def test_successful_reload_replaces_old_graph(self):
        self.tracer.process("fixture", "007.swc", nii_space="monkey",
                            output_dir=str(self.output), swc=VALID_SWC)
        replacement = "1 1 0 0 0 1 -1\n4 2 0 250 0 1 1\n5 17 0 0 250 1 1"
        self.tracer.process("fixture", "008.swc", nii_space="monkey",
                            output_dir=str(self.output), swc=replacement)
        self.assertEqual(set(self.tracer.nodes), {1, 4, 5})
        self.assertEqual(len(self.tracer.root.children), 2)
        self.assertEqual(len(self.tracer.branches), 2)
        self.assertEqual({node.id for node in self.tracer.terminal_nodes}, {4, 5})

    def test_invalid_direct_input_creates_no_partial_state_or_csv(self):
        with self.assertRaisesRegex(ValueError, "missing parent"):
            self.tracer.process("fixture", "007.swc", nii_space="monkey",
                                output_dir=str(self.output),
                                swc="1 1 0 0 0 1 -1\n2 2 1 0 0 1 99")
        self.assert_unloaded()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_failed_reload_preserves_complete_previous_graph(self):
        self.tracer._loadSWC("fixture", "007.swc", swc=VALID_SWC)
        previous_nodes = self.tracer.nodes
        previous_root = self.tracer.root
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.tracer._loadSWC("bad", "008.swc", swc=VALID_SWC + "2 2 0 0 0 1 1")
        self.assertIs(self.tracer.nodes, previous_nodes)
        self.assertIs(self.tracer.root, previous_root)
        self.assertEqual(self.tracer.swc_filename, "007.swc")
        self.assertEqual(self.tracer.exp_no, "fixture")

    def test_invalid_cache_is_rejected_without_fetch_or_overwrite(self):
        path = self.output / "007.swc"
        invalid = "1 1 0 0 0 1 -1\n2 2 1 0 0 1 -1"
        path.write_text(invalid)
        with self.assertRaisesRegex(ValueError, "exactly one root"):
            self.tracer._loadSWC("fixture", "007.swc")
        self.assert_unloaded()
        self.assertEqual(path.read_text(), invalid)
        self.module.IONData.IONData.assert_not_called()

    def test_invalid_explicit_file_is_rejected(self):
        path = self.output / "external.swc"
        path.write_text("1 1 nan 0 0 1 -1")
        with self.assertRaisesRegex(ValueError, "external.swc.*finite"):
            self.tracer._loadSWC("fixture", "007.swc", swc=str(path))
        self.assert_unloaded()

    def test_invalid_download_is_not_cached(self):
        self.module.IONData.IONData.return_value.getNeuronByID.return_value = "# empty"
        with self.assertRaisesRegex(ValueError, "ION fixture/007.swc.*found 0"):
            self.tracer._loadSWC("fixture", "007.swc")
        self.assert_unloaded()
        self.assertFalse((self.output / "007.swc").exists())

    def test_valid_download_is_cached_then_reusable_without_fetch(self):
        self.module.IONData.IONData.return_value.getNeuronByID.return_value = VALID_SWC
        self.tracer._loadSWC("fixture", "007.swc")
        self.assertEqual((self.output / "007.swc").read_text(), VALID_SWC)
        self.module.IONData.IONData.reset_mock()
        self.tracer._loadSWC("fixture", "007.swc")
        self.module.IONData.IONData.assert_not_called()
        self.assertEqual(set(self.tracer.nodes), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
