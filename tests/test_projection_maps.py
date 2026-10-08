"""Analytical line integrals and real CLI output contracts for optional maps."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))
from projection_maps import ReferenceGrid, axon_length_map, animal_mean, segment_voxels, save_map

spec = importlib.util.spec_from_file_location("projection_builder", ROOT / "group_analysis/scripts/build_projection_maps.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def swc(start, end, kind=2):
    return f"1 1 {' '.join(map(str, start))} 1 -1\n2 {kind} {' '.join(map(str, end))} 1 1\n"


class LineIntegralTests(unittest.TestCase):
    def setUp(self):
        self.grid = ReferenceGrid((4, 4, 4), np.eye(4))

    def measure(self, text, grid=None):
        return axon_length_map(text, grid or self.grid, coordinate_frame="atlas_index_um", index_scale_um=[1, 1, 1])

    def test_axis_edge_conserves_length_and_includes_terminal_voxel(self):
        data, metrics = self.measure(swc([0, 0, 0], [3, 0, 0]))
        np.testing.assert_allclose(data[:, 0, 0], [.5, 1, 1, .5])
        self.assertEqual(metrics["selected_axon_length_mm"], 3)
        self.assertEqual(metrics["outside_reference_length_mm"], 0)

    def test_segment_with_both_endpoints_outside_keeps_crossing_length(self):
        data, metrics = self.measure(swc([-2, 0, 0], [6, 0, 0]))
        np.testing.assert_allclose(data[:, 0, 0], 1)
        self.assertEqual(metrics["selected_axon_length_mm"], 8)
        self.assertEqual(metrics["outside_reference_length_mm"], 4)

    def test_diagonal_is_reversal_and_resampling_invariant(self):
        data, _ = self.measure(swc([0, 0, 0], [3, 3, 3]))
        reverse, _ = self.measure(swc([3, 3, 3], [0, 0, 0]))
        dense = "1 1 0 0 0 1 -1\n2 2 1 1 1 1 1\n3 2 2 2 2 1 2\n4 2 3 3 3 1 3\n"
        resampled, _ = self.measure(dense)
        np.testing.assert_allclose(data, reverse)
        np.testing.assert_allclose(data, resampled)
        self.assertAlmostEqual(data.sum(), 3*np.sqrt(3))

    def test_anisotropic_sheared_affine_defines_physical_length(self):
        affine = np.eye(4)
        affine[:3, :3] = [[2, 1, 0], [0, 3, 0], [0, 0, 4]]
        grid = ReferenceGrid((4, 4, 4), affine)
        data, _ = self.measure(swc([0, 0, 0], [1, 1, 0]), grid)
        self.assertAlmostEqual(data.sum(), np.sqrt(18))
        self.assertAlmostEqual(grid.voxel_volume_mm3, 24)

    def test_child_type_selects_transition_and_excludes_dendrite(self):
        text = "1 1 0 0 0 1 -1\n2 3 1 0 0 1 1\n3 2 2 0 0 1 2\n"
        data, metrics = self.measure(text)
        self.assertEqual(data.sum(), 1)
        self.assertEqual(metrics["edge_length_by_child_type_mm"], {"2": 1, "3": 1})

    def test_reordered_graph_and_zero_edges_do_not_change_length(self):
        text = "3 2 1 0 0 1 2\n2 2 0 0 0 1 1\n1 1 0 0 0 1 -1\n"
        data, metrics = self.measure(text)
        self.assertEqual(data.sum(), 1)
        self.assertEqual(metrics["zero_length_axon_edges"], 1)

    def test_fov_upper_face_has_no_volume_and_lower_face_is_included(self):
        self.assertEqual(list(segment_voxels([3.5, 0, 0], [3.5, 2, 0], self.grid.shape)), [])
        data, _ = self.measure(swc([-.5, 0, 0], [-.5, 2, 0]))
        self.assertEqual(data.sum(), 2)

    def test_world_and_index_frames_agree_after_explicit_transform(self):
        affine = np.diag([.25, .25, .25, 1])
        affine[:3, 3] = [-31.875, -27.75, -8]
        grid = ReferenceGrid((4, 4, 4), affine)
        index, _ = self.measure(swc([0, 0, 0], [3, 0, 0]), grid)
        a, b = affine[:3, 3], affine[:3, 3] + [.75, 0, 0]
        world, _ = axon_length_map(swc(a, b), grid, coordinate_frame="nifti_world_mm")
        np.testing.assert_allclose(world, index)
        with self.assertRaises(ValueError):
            axon_length_map(swc(a, b), grid, coordinate_frame="native_microscopy_um")

    def test_bad_graph_and_unknown_reference_units_fail(self):
        with self.assertRaises(ValueError):
            self.measure("1 1 0 0 0 1 -1\n2 2 1 0 0 1 99")
        image = nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
        with self.assertRaises(ValueError):
            ReferenceGrid.from_image(image)

    def test_equal_animal_weights_are_independent_of_neuron_counts(self):
        result = animal_mean([np.ones((2, 2, 2)), np.ones((2, 2, 2))*3])
        np.testing.assert_array_equal(result, 2)
        with self.assertRaises(ValueError):
            animal_mean([])


class BuilderContracts(unittest.TestCase):
    def test_full_saved_maps_and_exact_scope_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
            ref.header.set_xyzt_units("mm")
            nib.save(ref, root / "reference.nii.gz")
            rows = []
            for animal, size in [("a", 1), ("b", 3)]:
                folder = root / animal
                folder.mkdir()
                source = folder / "001.swc"
                source.write_text(swc([0, 0, 0], [size, 0, 0]))
                rows.append(dict(AnimalID=animal, SampleID=animal, NeuronID="001.swc", Subregion="IAL",
                                 SWCPath=str(source), SWCSHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                 ReferenceSHA256=builder.sha256(root / "reference.nii.gz"),
                                 CoordinateFrame="atlas_index_um", IndexScaleUm="1;1;1",
                                 AnatomyStatus="synthetic", RegistrationStatus="synthetic"))
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            with redirect_stdout(io.StringIO()):
                run = builder.build(manifest, root / "reference.nii.gz", root / "result")
            self.assertEqual(run["status"], "software_verified_descriptive")
            entry = run["group_maps"][0]
            data = nib.load(root / "result" / entry["length_path"]).get_fdata()
            self.assertEqual(data.sum(), 2)
            self.assertEqual(entry["n_animals"], 2)
            self.assertEqual(entry["n_neurons"], 2)
            with self.assertRaises(FileExistsError):
                save_map(root / "result" / entry["length_path"], np.zeros((4, 4, 4)),
                         ReferenceGrid((4, 4, 4), np.eye(4)))
            with self.assertRaises(FileExistsError):
                builder.build(manifest, root / "reference.nii.gz", root / "result")
            for labels in [(("a", "x_space-S_region-y"), ("a_space-S_region-x", "y")),
                           (("a", "IAL"), ("A", "IAL")), (("a", "IAL"), ("b", "ial"))]:
                altered = [dict(row) for row in rows]
                for row, (animal, region) in zip(altered, labels):
                    row.update(AnimalID=animal, Subregion=region)
                pd.DataFrame(altered).to_csv(manifest, index=False)
                with self.assertRaisesRegex(ValueError, "collide"):
                    builder.build(manifest, root / "reference.nii.gz", root / "collision", space="S")
                self.assertFalse((root / "collision").exists())
            pd.DataFrame(rows).to_csv(manifest, index=False)
            rows_reused = [dict(row) for row in rows]
            rows_reused[1].update(SWCPath=rows_reused[0]["SWCPath"], SWCSHA256=rows_reused[0]["SWCSHA256"])
            pd.DataFrame(rows_reused).to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "multiple neuron identities"):
                builder.checked_manifest(manifest, root)
            pd.DataFrame(rows + rows[:1]).to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                builder.checked_manifest(manifest, root)
            rows[0]["SWCSHA256"] = "0"*64
            pd.DataFrame(rows).to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                builder.checked_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
