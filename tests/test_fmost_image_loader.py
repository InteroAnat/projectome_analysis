"""Native crop geometry, missing-data policy, TIFF decoding and export checks."""

from io import BytesIO
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import numpy as np
import nibabel as nib
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main_scripts'))
import Visual_toolkit as visual
from fmost_image_geometry import native_to_index, native_affine, widefield_bounds, clip_segment, percentile_normalize


class GeometryTests(unittest.TestCase):
    def test_native_affine_agrees_with_overlay_and_preserves_z_spacing(self):
        origin, spacing = [100, 200, 300], [.65, .65, 3]
        index = np.array([2, 4, 7])
        point = (native_affine(origin, spacing) @ np.r_[index, 1])[:3]
        np.testing.assert_allclose(point, [101.3, 202.6, 321])
        np.testing.assert_allclose(native_to_index(point, origin, spacing), index)

    def test_depth_has_exact_slice_count_and_exclusive_stop(self):
        start, stop = widefield_bounds([100, 100, 300], [5, 5, 3], 20, 20, 30)
        np.testing.assert_equal(stop-start, [4, 4, 10])
        self.assertTrue(start[2] <= 100 < stop[2])
        with self.assertRaisesRegex(ValueError, 'first configured'):
            widefield_bounds([10, 10, 0], [5, 5, 3], 20, 20, 30)

    def test_clipping_excludes_off_slab_edges_and_retains_crossings(self):
        lower, upper = [0, 0, 0], [10, 10, 2]
        self.assertIsNone(clip_segment([1, 1, 8], [5, 5, 9], lower, upper))
        clipped = clip_segment([5, 5, -1], [5, 5, 3], lower, upper)
        np.testing.assert_equal(clipped, [[5, 5, 0], [5, 5, 2]])

    def test_constant_contrast_is_finite(self):
        for value in [0, 500]:
            normalized = percentile_normalize(np.full((4, 4), value))
            self.assertTrue(np.isfinite(normalized).all())
            self.assertEqual(normalized.max(), 0)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sections = self.root / 'sections'
        self.sections.mkdir()
        self.toolkit = visual.Visual_toolkit('fixture', cache_dir=str(self.root), low_res_dir=str(self.sections))

    def section(self, z):
        image = (np.arange(6)[:, None]*100 + np.arange(8)[None, :] + z*1000).astype(np.uint16)
        path = self.sections / self.toolkit._low_res_slice_filename(z)
        tifffile.imwrite(path, image)
        return image

    def test_low_crop_contains_soma_at_correct_native_pixel(self):
        self.section(9)
        image = self.section(10)
        volume, origin, spacing = self.toolkit.get_low_res_widefield([20, 15, 30], 20, 20, 6)
        self.assertEqual(volume.shape, (2, 4, 4))
        np.testing.assert_equal(origin, [10, 5, 27])
        local = native_to_index([20, 15, 30], origin, spacing).astype(int)
        self.assertEqual(volume[local[2], local[1], local[0]], image[3, 4])
        self.assertEqual(self.toolkit.last_low_res_metadata['requested'], [9, 10])
        self.assertTrue(self.toolkit.last_low_res_metadata['complete'])

    def test_missing_sections_are_explicit_and_central_is_required(self):
        self.section(10)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.toolkit.get_low_res_widefield([20, 15, 30], 20, 20, 6)
        volume, _, _ = self.toolkit.get_low_res_widefield([20, 15, 30], 20, 20, 6, allow_partial=True)
        self.assertEqual(self.toolkit.last_low_res_metadata['missing'], [9])
        self.assertEqual(volume[0].max(), 0)
        self.assertGreater(volume[1].max(), 0)
        (self.sections / self.toolkit._low_res_slice_filename(10)).unlink()
        self.section(9)
        with self.assertRaisesRegex(ValueError, 'central low-resolution'):
            self.toolkit.get_low_res_widefield([20, 15, 30], 20, 20, 6, allow_partial=True)

    def test_image_boundary_is_cropped_not_synthetic_pixel_padding(self):
        self.section(10)
        volume, origin, _ = self.toolkit.get_low_res_widefield([35, 15, 30], 20, 20, 3)
        self.assertEqual(volume.shape, (1, 4, 3))
        self.assertEqual(origin[0], 25)
        self.assertTrue(self.toolkit.last_low_res_metadata['crop_clipped_to_image_bounds'])

    def test_other_sample_never_uses_legacy_ssh_directory(self):
        with mock.patch.object(self.toolkit, '_init_ssh') as connect:
            self.assertIsNone(self.toolkit._get_low_res_slice_path(99))
            connect.assert_not_called()
        self.assertIsNone(self.toolkit.low_res_ssh_base)

    def test_high_cubes_have_correct_zyx_shape_and_required_center(self):
        shape = (2, 4, 3)
        with mock.patch.object(visual, 'BLOCK_SIZE_PIXELS', [3, 4, 2]):
            with mock.patch.object(self.toolkit, '_download_http_block', return_value=np.ones(shape, dtype=np.uint16)):
                volume, origin, spacing = self.toolkit.get_high_res_block([1.3, 1.95, 3], grid_radius=1)
            self.assertEqual(volume.shape, shape)
            np.testing.assert_equal(origin, [0, 0, 0])
            np.testing.assert_allclose(native_to_index([1.3, 1.95, 3], origin, spacing), [2, 3, 1])
            self.assertEqual(self.toolkit.last_high_res_metadata['loaded'], [[0, 0, 0]])
            with mock.patch.object(self.toolkit, '_download_http_block', return_value=None):
                with self.assertRaisesRegex(ValueError, 'central high-resolution'):
                    self.toolkit.get_high_res_block([1.3, 1.95, 3], grid_radius=1, allow_partial=True)

    def test_bad_http_payload_never_becomes_cache(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'<html>unavailable</html>'
        with mock.patch.object(visual.urllib.request, 'urlopen', return_value=response):
            with mock.patch.object(visual.time, 'sleep'):
                self.assertIsNone(self.toolkit._download_http_block(1, 2, 3))
        self.assertFalse((Path(self.toolkit.cache_http_dir) / '3' / '1_2_3.tif').exists())

    def test_pil_fallback_reads_every_tiff_page(self):
        from PIL import Image
        path = self.root / 'pages.tif'
        frames = [Image.fromarray(np.full((4, 5), n, dtype=np.uint8)) for n in [1, 2, 3]]
        frames[0].save(path, save_all=True, append_images=frames[1:])
        previous = Image.MAX_IMAGE_PIXELS
        with mock.patch.object(visual.tifffile, 'imread', side_effect=ValueError('missing LZW codec')):
            image = visual._read_tiff(path)
        self.assertEqual(image.shape, (3, 4, 5))
        np.testing.assert_equal(image[:, 0, 0], [1, 2, 3])
        self.assertEqual(Image.MAX_IMAGE_PIXELS, previous)

    def test_export_persists_native_xyz_affine_micron_units_and_provenance(self):
        volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        path = self.toolkit.export_data(volume, [100, 200, 300], [.65, .65, 3], '007.swc',
                                        suffix='SomaBlock', output_dir=str(self.root / 'export'))
        image = nib.load(path)
        self.assertEqual(image.shape, (4, 3, 2))
        np.testing.assert_allclose(image.affine, native_affine([100, 200, 300], [.65, .65, 3]))
        self.assertEqual(image.header.get_xyzt_units()[0], 'micron')
        np.testing.assert_equal(image.dataobj[1, 2, 0], volume[0, 2, 1])
        self.assertTrue(Path(path + '.json').exists())
        self.assertIn('unverified', self.toolkit.last_export_metadata['coordinate_frame'])

    def test_export_sidecar_keeps_raw_root_nifti_axes_and_coverage(self):
        import json
        volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        # Coverage belongs to a complete acquisition record for this sample/grid.
        self.toolkit.last_high_res_metadata = self.toolkit._acquisition_metadata(
            'fixture native cubes', [110.0, 210.0, 306.0], [100, 200, 300],
            [.65, .65, 3], volume.shape, [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0]], [[1, 0, 0]],
            ['http://fixture.invalid/monkeydata/fixture/cube/0/0_0_0.tif'])
        path = self.toolkit.export_data(
            volume, [100, 200, 300], [.65, .65, 3], '007.swc',
            suffix='SomaBlock', soma_coords=[110.5, 210.5, 306.5],
            output_dir=str(self.root / 'export'))
        meta = json.loads(Path(path + '.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['center_xyz_um'], [110.5, 210.5, 306.5])
        self.assertEqual(meta['origin_xyz_um'], [100, 200, 300])
        self.assertEqual(meta['nifti_axis_order'], 'XYZ')
        self.assertEqual(meta['source_array_axis_order'], 'ZYX')
        self.assertEqual(meta['shape_zyx'], [2, 3, 4])
        self.assertEqual(meta['shape_xyz'], [4, 3, 2])
        self.assertIs(meta['complete'], False)
        self.assertEqual(meta['missing'], [[1, 0, 0]])
        self.assertEqual(Path(meta['output_file']).name, Path(path).name)

    def test_copied_samples_use_own_series_and_never_legacy_ssh(self):
        expected = {
            '252384': '252384_CH1_resample',
            '252385': '252385-CH1_resample',
            '252527': '252527_CH1_resample',
            '252714': '252714-CH1_resample',
        }
        for sample, folder in expected.items():
            toolkit = visual.Visual_toolkit(sample, cache_dir=str(self.root / sample), low_res_slice_cache_limit=1)
            self.assertIn(folder, toolkit.low_res_share_base)
            self.assertNotIn('251637', toolkit.low_res_share_base)
            self.assertIsNone(toolkit.low_res_ssh_base)
        pending = visual.Visual_toolkit('252790', cache_dir=str(self.root / 'pending'), low_res_slice_cache_limit=0)
        self.assertIsNone(pending.low_res_share_base)
        self.assertIsNone(pending.low_res_ssh_base)
        self.assertTrue(all('252790' in path and '251637' not in path for path in pending.low_res_share_candidates))

    def test_partial_neighbor_cubes_stay_explicit(self):
        shape = (2, 4, 3)

        def fake_block(idx_x, idx_y, idx_z):
            if (idx_x, idx_y, idx_z) == (0, 0, 0):
                return np.ones(shape, dtype=np.uint16)
            return None

        with mock.patch.object(visual, 'BLOCK_SIZE_PIXELS', [3, 4, 2]):
            with mock.patch.object(self.toolkit, '_download_http_block', side_effect=fake_block):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    self.toolkit.get_high_res_block([1.3, 1.95, 3], grid_radius=2, allow_partial=False)
                volume, _, _ = self.toolkit.get_high_res_block([1.3, 1.95, 3], grid_radius=2, allow_partial=True)
        self.assertIn([0, 0, 0], self.toolkit.last_high_res_metadata['loaded'])
        self.assertGreater(len(self.toolkit.last_high_res_metadata['missing']), 0)
        self.assertEqual(volume.max(), 1)
        self.assertEqual(volume[0, 0, 0], 0)

    def test_bad_http_payload_retries_are_bounded_and_uncached(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'<html>unavailable</html>'
        with mock.patch.object(visual.urllib.request, 'urlopen', return_value=response) as opened:
            with mock.patch.object(visual.time, 'sleep'):
                self.assertIsNone(self.toolkit._download_http_block(1, 2, 3))
        self.assertEqual(opened.call_count, visual.HTTP_ATTEMPTS)
        self.assertFalse((Path(self.toolkit.cache_http_dir) / '3' / '1_2_3.tif').exists())

    def test_low_res_section_cache_is_bounded(self):
        self.toolkit.low_res_slice_cache_limit = 1
        first = self.section(10)
        second = self.section(11)
        with mock.patch.object(visual, '_read_tiff', side_effect=[first, second]) as reader:
            self.assertIs(self.toolkit._read_cached_section(self.toolkit._get_low_res_slice_path(10)), first)
            self.assertIs(self.toolkit._read_cached_section(self.toolkit._get_low_res_slice_path(10)), first)
            self.assertIs(self.toolkit._read_cached_section(self.toolkit._get_low_res_slice_path(11)), second)
        self.assertEqual(reader.call_count, 2)
        self.assertEqual(len(self.toolkit._section_cache), 1)


if __name__ == '__main__':
    unittest.main()
