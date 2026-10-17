"""Extensive checks for the ION pyswcloader ground against the Liu NeuroM bridge.

The synthetic neurons are the control. Two optional exemplars, fMOST 251637/007
and one Liu VEN-L SWC, are read when those local files are present.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import tools.pyswcloader_ground as ground


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


class PackageContractTests(unittest.TestCase):
    def test_pinned_wheel_exposes_the_public_modules(self):
        self.assertEqual(ground.installed_version(), "1.1.1")
        ground.ensure_import_path()
        import pyswcloader
        from pyswcloader import brain, cluster, distance, io, projection, swc

        self.assertEqual(Path(pyswcloader.__file__).parent.name, "pyswcloader")
        for module in (brain, cluster, distance, io, projection, swc):
            self.assertTrue(hasattr(module, "__name__"))

    def test_import_loads_the_allen_mouse_ontology(self):
        ground.ensure_import_path()
        from pyswcloader import io

        acronyms = set(io.STL_ACRO_DICT.values())
        self.assertGreater(len(io.ALLEN_BRAIN_TREE.nodes), 1000)
        self.assertIn("root", acronyms)
        self.assertIn("VISp", acronyms)
        self.assertIn("HY", acronyms)

    def test_viewer_camera_is_centered_on_the_allen_box(self):
        ground.ensure_import_path()
        from pyswcloader import visualization

        source = Path(visualization.__file__).read_text(encoding="utf-8")
        self.assertIn("13200/2, 8000/2, 11400/2", source)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.swc = ground.load_swc_module()

    def tearDown(self):
        self.tmp.cleanup()

    def test_comments_blanks_and_scientific_coordinates_load(self):
        path = _write(
            self.dir,
            "sci.swc",
            """
            # macaque-style radius
            1 1 9.8997500000000e+03 2.9200000000000e+04 2.5600000000000e+04 7.5e-01 -1
            """,
        )
        frame = self.swc.read_swc(str(path))
        self.assertEqual(list(frame.columns), ["id", "type", "x", "y", "z", "radius", "parent"])
        self.assertAlmostEqual(frame.loc[1, "x"], 9899.75)

    def test_double_space_tab_and_extra_column_are_rejected(self):
        cases = {
            "double.swc": "1  1  0  0  0  1  -1\n",
            "tab.swc": "1\t1\t0\t0\t0\t1\t-1\n",
            "extra.swc": "1 1 0 0 0 1 -1 99\n",
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                path = _write(self.dir, name, text)
                with self.assertRaises(ValueError):
                    self.swc.read_swc(str(path))

    def test_scientific_node_ids_do_not_keep_exact_integers(self):
        node_id = 9007199254740993
        path = _write(self.dir, "big.swc", f"9.007199254740993e+15 1 0 0 0 1 -1")
        frame = self.swc.read_swc(str(path))
        self.assertNotEqual(float(frame.iloc[0]["id"]), node_id)
        rows = ground.parse_rows(_write(self.dir, "exact.swc", f"{node_id} 1 0 0 0 1 -1"))
        self.assertEqual(rows[0][0], node_id)


class LengthAndTreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.swc = ground.load_swc_module()

    def tearDown(self):
        self.tmp.cleanup()

    def test_total_length_keeps_axon_and_drops_dendrite_types(self):
        path = _write(
            self.dir,
            "parts.swc",
            """
            1 1 0 0 0 1 -1
            2 2 3 0 0 1 1
            3 3 0 4 0 1 1
            4 4 0 0 5 1 1
            """,
        )
        self.assertAlmostEqual(self.swc.total_length(str(path)), 3.0)
        cables = ground.cable_summary(ground.parse_rows(path))
        self.assertAlmostEqual(cables["axon"], 3.0)
        self.assertAlmostEqual(cables["basal"], 4.0)
        self.assertAlmostEqual(cables["apical"], 5.0)
        self.assertAlmostEqual(cables["pyswc_nondendrite_rule"], 3.0)

    def test_unsorted_parents_cannot_build_the_tree(self):
        path = _write(self.dir, "unsort.swc", "2 2 3 0 0 1 1\n1 1 0 0 0 1 -1")
        with self.assertRaises(Exception) as caught:
            self.swc.swc_tree(str(path))
        self.assertIn("Parent", str(caught.exception))

    def test_parent_ids_that_differ_from_row_index_silently_drop_length(self):
        path = _write(
            self.dir,
            "gap.swc",
            """
            1 1 0 0 0 1 -1
            3 2 0 0 0 1 1
            4 2 5 0 0 1 3
            """,
        )
        self.assertAlmostEqual(self.swc.total_length(str(path)), 0.0)
        self.assertAlmostEqual(ground.cable_summary(ground.parse_rows(path))["axon"], 5.0)

    def test_longest_stem_follows_the_longer_axon(self):
        path = _write(
            self.dir,
            "stem.swc",
            """
            1 1 0 0 0 1 -1
            2 2 1 0 0 1 1
            3 2 3 0 0 1 2
            4 2 0 1 0 1 1
            """,
        )
        stem = self.swc.find_longest_stem(str(path))
        self.assertIn(3, set(stem.id.astype(int)))
        self.assertNotIn(4, set(stem.id.astype(int)))

    def test_downsample_requires_removed_dataframe_append(self):
        path = _write(self.dir, "line.swc", "1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 2 2 0 0 1 2")
        with self.assertRaises(AttributeError) as caught:
            self.swc.downsample(str(path), eps=0.5)
        self.assertIn("append", str(caught.exception))


class AllenPreprocessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.swc = ground.load_swc_module()

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_fmost_soma_is_clipped_and_z_flipped(self):
        path = _write(
            self.dir,
            "native.swc",
            "1 1 47541.4 54080.1 20557.8 1 -1\n2 2 47551.4 54080.1 20557.8 1 1",
        )
        risk = ground.allen_risk(self.swc, path, flip=True, dimension=ground.ALLEN_DIMENSION_UM)
        self.assertEqual(risk["preprocessed_first_row_xyz_um"], [13199.0, 7999.0, -9157.8])
        self.assertTrue(risk["z_flipped"])
        self.assertIn("x", risk["axes_changed"])
        self.assertIn("y", risk["axes_changed"])
        self.assertTrue(any("exceeds boundary" in line for line in risk["package_messages"]))

    def test_slice_local_soma_stays_put_under_the_allen_default(self):
        path = _write(self.dir, "slice.swc", "1 1 -0.9 5.8 -0.3 1 -1\n2 3 10 5.8 -0.3 1 1")
        risk = ground.allen_risk(self.swc, path, flip=True, dimension=ground.ALLEN_DIMENSION_UM)
        self.assertEqual(risk["preprocessed_first_row_xyz_um"], [-0.9, 5.8, -0.3])
        self.assertFalse(risk["z_flipped"])
        self.assertEqual(risk["axes_changed"], [])

    def test_flip_can_be_turned_off_and_extra_roots_are_removed(self):
        path = _write(
            self.dir,
            "two.swc",
            """
            1 1 0 0 7000 1 -1
            2 2 1 0 7000 1 1
            3 1 10 0 7000 1 -1
            4 2 11 0 7000 1 3
            5 1 2 0 7000 1 1
            """,
        )
        kept = self.swc.swc_preprocess(str(path), flip=False, check_validity=True, save=False)
        # The extra-root deletion and the extra type-1 relabel are if/elif.
        # Node 5 stays type 1 because a second root already took the first branch.
        self.assertEqual(len(kept), 3)
        self.assertAlmostEqual(float(kept.loc[1, "z"]), 7000.0)
        self.assertNotIn(3, set(kept.id.astype(int)))
        self.assertEqual(int(kept.loc[kept.id == 5, "type"].iloc[0]), 1)
        single_root = _write(self.dir, "types.swc", "1 1 0 0 0 1 -1\n2 1 1 0 0 1 1")
        relabeled = self.swc.swc_preprocess(str(single_root), flip=False, check_validity=True, save=False)
        self.assertEqual(int(relabeled.loc[relabeled.id == 2, "type"].iloc[0]), 0)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        ground.ensure_import_path()
        from pyswcloader import projection

        self.projection = projection

    def tearDown(self):
        self.tmp.cleanup()

    def test_boundary_crossing_axon_is_split_in_half(self):
        import numpy as np

        volume = np.zeros((10, 10, 10), dtype=np.int32)
        volume[1:3, 1:3, 1:3] = 7
        volume[3:6, 1:3, 1:3] = 8
        path = _write(self.dir, "axon.swc", "1 1 15 15 15 1 -1\n2 2 35 15 15 1 1")
        length = self.projection.projection_length(str(path), volume, 10)
        self.assertAlmostEqual(float(length.iloc[0][7]), 10.0)
        self.assertAlmostEqual(float(length.iloc[0][8]), 10.0)
        self.assertNotEqual(length.index[0], "axon")
        self.assertIn("axon", length.index[0])

    def test_dendrite_edges_add_nothing(self):
        import numpy as np

        volume = np.full((10, 10, 10), 7, dtype=np.int32)
        path = _write(self.dir, "dend.swc", "1 1 15 15 15 1 -1\n2 3 35 15 15 1 1")
        length = self.projection.projection_length(str(path), volume, 10)
        self.assertAlmostEqual(float(length.to_numpy().sum()), 0.0)

    def test_flipped_negative_z_wraps_inside_the_annotation_array(self):
        import numpy as np

        volume = np.zeros((4, 4, 5), dtype=np.int32)
        volume[:, :, -1] = 99
        path = _write(
            self.dir,
            "wrap.swc",
            "1 1 150 150 11550 1 -1\n2 2 150 150 11540 1 1",
        )
        length = self.projection.projection_length(str(path), volume, 100)
        self.assertAlmostEqual(float(length.iloc[0][99]), 10.0)

    def test_length_ipsi_follows_the_soma_side_and_terminal_ipsi_uses_low_z(self):
        import numpy as np

        volume = np.zeros((5, 5, 12), dtype=np.int32)
        volume[1, 1, 8] = 7
        volume[1, 1, 9] = 7
        path = _write(
            self.dir,
            "highz.swc",
            "1 1 15 15 80 1 -1\n2 2 15 15 90 1 1",
        )
        ipsi = self.projection.projection_length_ipsi(str(path), volume, 10)
        terminals = self.projection.terminal_count_ipsi(str(path), volume, 10)
        total = self.projection.terminal_count(str(path), volume, 10)
        self.assertAlmostEqual(float(ipsi.to_numpy().sum()), 10.0)
        self.assertAlmostEqual(float(terminals.to_numpy().sum()), 0.0)
        self.assertAlmostEqual(float(total.to_numpy().sum()), 1.0)


class DistanceAndClusterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        ground.ensure_import_path()
        from pyswcloader import cluster, distance

        self.cluster = cluster
        self.distance = distance
        self.paths = []
        for index, shift in enumerate((0, 10, 100, 110)):
            self.paths.append(
                _write(
                    self.dir,
                    f"n{index}.swc",
                    f"1 1 {shift} 0 0 1 -1\n2 2 {shift + 5} 0 0 1 1",
                )
            )

    def tearDown(self):
        self.tmp.cleanup()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_identical_neurons_score_zero_and_separation_increases_the_score(self):
        identical = self.distance.morphology_distance(str(self.paths[0]), str(self.paths[0]))
        near = self.distance.morphology_distance(str(self.paths[0]), str(self.paths[1]))
        far = self.distance.morphology_distance(str(self.paths[0]), str(self.paths[2]))
        self.assertAlmostEqual(identical, 0.0)
        self.assertGreater(near, 0.0)
        self.assertGreater(far, near)

    def test_precomputed_hierarchy_separates_the_two_shifted_pairs(self):
        import pandas as pd

        matrix = pd.DataFrame(0.0, index=self.paths, columns=[str(path) for path in self.paths])
        matrix.index = [str(path) for path in self.paths]
        for i, left in enumerate(matrix.index):
            for right in list(matrix.index)[i + 1 :]:
                value = self.distance.morphology_distance(left, right)
                matrix.loc[left, right] = value
                matrix.loc[right, left] = value
        with mock.patch("matplotlib.pyplot.show"):
            info = self.cluster.cluster(
                n_cluster=2, method="hierarchy", feature="precomputed", matrix=matrix
            )
        labels = list(info.label)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertNotEqual(info.neuron.iloc[0], "n0")
        self.assertIn("n0", info.neuron.iloc[0])

    def test_unknown_method_and_feature_raise(self):
        import pandas as pd

        matrix = pd.DataFrame([[0.0]], index=["a.swc"], columns=["a.swc"])
        with self.assertRaises(ValueError):
            self.cluster.cluster(method="spectral", feature="precomputed", matrix=matrix)
        with self.assertRaises(ValueError):
            self.cluster.cluster(method="hierarchy", feature="atlas", matrix=matrix)


class LiuBridgeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bipolar_neuron_splits_the_liu_question_from_the_package_length(self):
        path = ground.synthetic_bipolar(self.dir / "bipolar.swc")
        receipt = ground.compare_case(path, r_local=800)
        self.assertAlmostEqual(receipt["pyswcloader_total_length_um"], 5000.0)
        self.assertTrue(receipt["pyswcloader_matches_nondendrite_rule"])
        neurom = receipt["neurom"]
        self.assertTrue(neurom["available"])
        self.assertAlmostEqual(neurom["total_dendrite_length"], 1000.0)
        self.assertAlmostEqual(neurom["basal_max_path"], 600.0)
        self.assertAlmostEqual(neurom["apical_max_path"], 400.0)
        self.assertTrue(neurom["complete_morph_labels"])
        self.assertFalse(receipt["questions"]["same_measurement"])
        self.assertFalse(Path("scores_record.txt").exists())

    def test_translating_onto_a_native_soma_preserves_neurom_and_trips_the_allen_box(self):
        origin = ground.synthetic_bipolar(self.dir / "origin.swc")
        shifted = self.dir / "shifted.swc"
        rows = []
        offset = (47541.4, 54080.1, 20557.8)
        for line in origin.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            for axis in range(3):
                parts[2 + axis] = f"{float(parts[2 + axis]) + offset[axis]:.4f}"
            rows.append(" ".join(parts))
        shifted.write_text("\n".join(rows) + "\n", encoding="utf-8")
        origin_features = ground.neurom_features(origin, 800)
        shifted_receipt = ground.compare_case(shifted, r_local=800)
        self.assertAlmostEqual(
            shifted_receipt["neurom"]["total_dendrite_length"],
            origin_features["total_dendrite_length"],
        )
        self.assertAlmostEqual(
            shifted_receipt["neurom"]["basal_max_path"],
            origin_features["basal_max_path"],
        )
        self.assertTrue(shifted_receipt["allen_preprocess"]["z_flipped"])
        self.assertLess(shifted_receipt["allen_preprocess"]["preprocessed_first_row_xyz_um"][2], 0)

    def test_invalid_acronym_lookup_returns_an_exception_object(self):
        ground.ensure_import_path()
        from pyswcloader import brain

        ontology = self.dir / "allen.json"
        ontology.write_text(
            json.dumps(
                {
                    "msg": [
                        {
                            "id": 997,
                            "acronym": "root",
                            "name": "root",
                            "parent_structure_id": None,
                            "children": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        found = brain.find_id_by_acronym("missing", str(ontology))
        self.assertIsInstance(found, ValueError)


class OperatorTests(unittest.TestCase):
    def test_dry_run_names_the_synthetic_case_and_writes_nothing(self):
        before = set(ground.receipts_dir().glob("*")) if ground.receipts_dir().exists() else set()
        code = ground.main(["--dry-run", "--synthetic", "--mode", "compare"])
        after = set(ground.receipts_dir().glob("*")) if ground.receipts_dir().exists() else set()
        self.assertEqual(code, 0)
        self.assertEqual(before, after)

    def test_bad_radius_and_escaped_output_fail_with_a_clear_message(self):
        with self.assertRaises(SystemExit) as radius:
            ground.main(["--synthetic", "--r-local", "0"])
        self.assertIn("r-local", str(radius.exception))
        with self.assertRaises(SystemExit) as output:
            ground.main(["--synthetic", "--output", str(ROOT / "outside.json")])
        self.assertIn("output must stay inside", str(output.exception))

    def test_missing_case_fails_before_a_receipt(self):
        with self.assertRaises(SystemExit) as caught:
            ground.main(["--case", str(ROOT / "missing-neuron.swc")])
        self.assertIn("not found", str(caught.exception))


class ExemplarTests(unittest.TestCase):
    FMOST = ROOT / "processed_neurons" / "251637" / "007.swc"
    VENL = ROOT / "external" / "liu2026" / "swc" / "VENL" / "unSM1133.swc"

    def test_fmost_007_is_outside_the_allen_box_and_keeps_dendritic_neurom_length(self):
        if not self.FMOST.is_file():
            self.skipTest(f"missing {self.FMOST}")
        receipt = ground.compare_case(self.FMOST, r_local=800)
        self.assertGreater(receipt["soma_xyz_um"][2], ground.ALLEN_Z_MIDLINE_UM)
        self.assertTrue(receipt["allen_preprocess"]["z_flipped"])
        self.assertIn("x", receipt["allen_preprocess"]["axes_changed"])
        self.assertTrue(receipt["pyswcloader_matches_nondendrite_rule"])
        dendrite = receipt["cables_um"]["basal"] + receipt["cables_um"]["apical"]
        self.assertGreater(dendrite, 0.0)
        self.assertGreater(receipt["neurom"]["total_dendrite_length"], 0.0)
        self.assertFalse(math.isclose(
            receipt["pyswcloader_total_length_um"],
            receipt["neurom"]["total_dendrite_length"],
            rel_tol=0.05,
            abs_tol=1.0,
        ))

    def test_liu_venl_slice_coordinates_stay_inside_the_allen_box(self):
        if not self.VENL.is_file():
            self.skipTest(f"missing {self.VENL}")
        receipt = ground.compare_case(self.VENL, r_local=800)
        self.assertTrue(all(abs(value) < 100 for value in receipt["soma_xyz_um"]))
        self.assertFalse(receipt["allen_preprocess"]["z_flipped"])
        self.assertEqual(receipt["allen_preprocess"]["axes_changed"], [])
        self.assertTrue(receipt["pyswcloader_matches_nondendrite_rule"])
        self.assertGreater(receipt["cables_um"]["basal"] + receipt["cables_um"]["apical"], 0.0)
        self.assertGreater(receipt["neurom"]["total_dendrite_length"], 0.0)
        self.assertLessEqual(receipt["neurom"]["max_radial_distance_dend"], 800.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
