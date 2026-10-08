"""Independent geometric and publication checks for descriptive axon maps.

The allocation oracle clips against each voxel independently. It does not
reuse the production algorithm's global voxel-face cuts or midpoint lookup.
Synthetic validation establishes software properties, not registration QC.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))
import projection_maps as maps

spec = importlib.util.spec_from_file_location(
    "projection_builder_independent", ROOT / "group_analysis/scripts/build_projection_maps.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def independent_voxel_box_fractions(start, end, shape):
    """Intersect a parametric segment separately with every half-open cell."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = end - start
    fractions = np.zeros(shape, dtype=float)
    for cell in itertools.product(*(range(n) for n in shape)):
        entry, exit_time = 0.0, 1.0
        for axis, index in enumerate(cell):
            if direction[axis] == 0:
                if not index - 0.5 <= start[axis] < index + 0.5:
                    exit_time = -1.0
                    break
            else:
                first = (index - 0.5 - start[axis]) / direction[axis]
                second = (index + 0.5 - start[axis]) / direction[axis]
                entry = max(entry, min(first, second))
                exit_time = min(exit_time, max(first, second))
        fractions[cell] = max(0.0, exit_time - entry)
    return fractions


def synthetic_swc(start, end):
    return (f"1 1 {' '.join(map(str, start))} 1 -1\n"
            f"2 2 {' '.join(map(str, end))} 1 1\n")


class IndependentGeometryTests(unittest.TestCase):
    def test_1016_segments_against_independent_voxel_box_oracle(self):
        shape = (4, 3, 5)
        random = np.random.default_rng(31337)
        cases = [(random.uniform(-5, 8, 3), random.uniform(-5, 8, 3))
                 for _ in range(1000)]
        for first, last in itertools.product([-.5, .5, 2.5, 3.5], repeat=2):
            cases.append((np.array([first, 0., 0.]), np.array([last, 2., 4.])))
        self.assertEqual(len(cases), 1016)
        # Include anisotropy, shear and reflection when checking physical length.
        affine = np.array([[-.3, .1, 0., -7.], [0., .2, .05, 9.],
                           [0., 0., .4, -2.], [0., 0., 0., 1.]])
        grid = maps.ReferenceGrid(shape, affine)
        for case, (start, end) in enumerate(cases):
            with self.subTest(case=case):
                expected_fraction = independent_voxel_box_fractions(start, end, shape)
                actual_fraction = np.zeros(shape)
                for voxel, fraction in maps.segment_voxels(start, end, shape):
                    actual_fraction[voxel] += fraction
                np.testing.assert_allclose(actual_fraction, expected_fraction,
                                           rtol=1e-11, atol=1e-12)
                # Physical oracle uses independently transformed endpoints.
                physical_start = affine @ np.append(start, 1.)
                physical_end = affine @ np.append(end, 1.)
                full_length = np.linalg.norm(physical_end[:3] - physical_start[:3])
                actual_map, accounting = maps.axon_length_map(
                    synthetic_swc(start, end), grid, coordinate_frame="atlas_index_um",
                    index_scale_um=[1., 1., 1.])
                np.testing.assert_allclose(actual_map, expected_fraction * full_length,
                                           rtol=1e-10, atol=1e-12)
                self.assertAlmostEqual(accounting["selected_axon_length_mm"], full_length)
                self.assertAlmostEqual(accounting["outside_reference_length_mm"],
                                       full_length * (1. - expected_fraction.sum()))

    def test_mm_micron_meter_references_have_equivalent_lengths_and_densities(self):
        outputs = []
        for unit, spacing in [("mm", .25), ("micron", 250.), ("meter", .00025)]:
            with self.subTest(unit=unit):
                image = nib.Nifti1Image(np.zeros((4, 4, 4)),
                                       np.diag([spacing, spacing, spacing, 1.]))
                image.header.set_xyzt_units(unit)
                grid = maps.ReferenceGrid.from_image(image)
                data, _ = maps.axon_length_map(
                    synthetic_swc([0, 0, 0], [3, 0, 0]), grid,
                    coordinate_frame="atlas_index_um", index_scale_um=[1., 1., 1.])
                np.testing.assert_allclose(data[:, 0, 0], [.125, .25, .25, .125])
                self.assertAlmostEqual(grid.voxel_volume_mm3, .015625)
                outputs.append(data / grid.voxel_volume_mm3)
        for density in outputs[1:]:
            np.testing.assert_allclose(density, outputs[0])


class IndependentBuilderTests(unittest.TestCase):
    def test_unequal_neuron_counts_keep_equal_animal_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
            reference.header.set_xyzt_units("mm")
            reference_path = root / "reference.nii.gz"
            nib.save(reference, reference_path)
            rows = []
            for animal, lengths in [("a", [1]), ("b", [3, 3, 3])]:
                folder = root / animal
                folder.mkdir()
                for index, length in enumerate(lengths):
                    source = folder / f"{index:03d}.swc"
                    source.write_text(synthetic_swc([0, 0, 0], [length, 0, 0]),
                                      encoding="utf-8")
                    rows.append(dict(
                        AnimalID=animal, SampleID=animal, NeuronID=source.name,
                        Subregion="IAL", SWCPath=str(source), SWCSHA256=builder.sha256(source),
                        ReferenceSHA256=builder.sha256(reference_path),
                        CoordinateFrame="atlas_index_um", IndexScaleUm="1;1;1",
                        AnatomyStatus="synthetic", RegistrationStatus="synthetic"))
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            with redirect_stdout(io.StringIO()):
                run = builder.build(manifest, reference_path, root / "output")
            entry = run["group_maps"][0]
            saved = nib.load(root / "output" / entry["length_path"]).get_fdata()
            # Animal a: [.5,.5,0,0]; animal b: [.5,1,1,.5].
            np.testing.assert_allclose(saved[:, 0, 0], [.5, .75, .5, .25])
            self.assertEqual(saved.sum(), 2.)
            self.assertNotEqual(saved.sum(), (1. + 3. + 3. + 3.) / 4.)
            self.assertEqual(entry["n_animals"], 2)
            self.assertEqual(entry["n_neurons"], 4)

    def test_concurrent_destination_creation_preserves_prior_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "map.nii.gz"
            prior = b"concurrently published authoritative bytes\x00\xff"
            real_link = maps.os.link

            def publish_competing_file_then_link(temporary, target):
                self.assertEqual(Path(target), destination)
                # The competitor acts after initial existence checks/readback.
                with destination.open("xb") as stream:
                    stream.write(prior)
                return real_link(temporary, target)

            with patch.object(maps.os, "link", side_effect=publish_competing_file_then_link):
                with self.assertRaises(FileExistsError):
                    maps.save_map(destination, np.ones((2, 2, 2)),
                                  maps.ReferenceGrid((2, 2, 2), np.eye(4)))
            self.assertEqual(destination.read_bytes(), prior)
            self.assertEqual(list(root.iterdir()), [destination])
            with self.assertRaises(FileExistsError):
                maps.save_map(destination, np.zeros((2, 2, 2)),
                              maps.ReferenceGrid((2, 2, 2), np.eye(4)))
            self.assertEqual(destination.read_bytes(), prior)


if __name__ == "__main__":
    unittest.main()
