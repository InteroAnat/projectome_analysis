"""
Visual_toolkit.py - Macaque Brain Hybrid-Resolution Visualization Toolkit

Version: 1.4.0 (Soma Region & Coordinate Support + Cache)

UPDATE NOTES (v1.4.0):
    - Added soma_region parameter to export and plot methods.
    - Filenames and titles now include soma region and formatted coordinates.
    - Supports cache directory configuration for bulk processing.
"""

import os
import sys
import time
import numpy as np
import urllib.request
import tifffile
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import paramiko
import json
import tempfile
from collections import OrderedDict
from pathlib import Path
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from fmost_image_geometry import (
    triplet, native_to_index, native_affine, widefield_bounds,
    clip_segment, percentile_normalize,
)

# Respect the caller's chosen backend (the bulk runners explicitly use Agg).

# --- PATHS ---
neurovis_path = str(Path(__file__).resolve().parents[1] / "neuron-vis/neuronVis")
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)

import IONData as IT

# --- CONFIGURATION ---

# 1. HIGH RESOLUTION (HTTP Source) - 0.65um
HTTP_HOST = 'http://bap.cebsit.ac.cn'
HTTP_PATH = 'monkeydata'
BLOCK_SIZE_PIXELS = [360, 360, 90]    # [X, Y, Z]
RESOLUTION_HIGH   = [0.65, 0.65, 3.0] # [X, Y, Z] microns
HTTP_ATTEMPTS = 3  # bounded; a failed cube stays missing rather than retrying forever
MAX_HTTP_BLOCK_BYTES = 32 * 1024**2

# 2. LOW RESOLUTION - 5.0um resampled widefield
# Primary: lab SMB share under 5micron_datasets (2026-07). Fallback: legacy SSH.
LOW_RES_SHARE_ROOT = r"\\10.102.8.200\microscopy_data\fMOST\5micron_datasets"
LOW_RES_SHARE_BY_SAMPLE = {
    '251637': os.path.join(LOW_RES_SHARE_ROOT, "251637-CH1_resample", "resample_5um"),
    # Hyphen and underscore differ by sample. These four new copies plus the
    # 251637 reference were confirmed on the 5micron share (2026-10-02).
    '252384': os.path.join(LOW_RES_SHARE_ROOT, "252384_CH1_resample", "resample_5um"),
    '252385': os.path.join(LOW_RES_SHARE_ROOT, "252385-CH1_resample", "resample_5um"),
    '252527': os.path.join(LOW_RES_SHARE_ROOT, "252527_CH1_resample", "resample_5um"),
    '252714': os.path.join(LOW_RES_SHARE_ROOT, "252714-CH1_resample", "resample_5um"),
}
SSH_HOST = "172.20.10.250"
SSH_PORT = 20007
SSH_USER = "binbin"
SSH_PASS = os.environ.get("PROJECTOME_SSH_PASSWORD")
SSH_REMOTE_BASE = "/home/binbin/share/251637CH1_projection/251637-CH1_resample/resample_5um"
RESOLUTION_LOW  = [5.0, 5.0, 3.0]     # [X, Y, Z] microns


def _read_tiff(path):
    """Read fMOST TIFF (LZW); fall back to PIL when imagecodecs is missing."""
    try:
        return tifffile.imread(path)
    except Exception as tiff_error:
        from PIL import Image, ImageSequence
        pixel_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None  # fMOST whole sections are intentionally large.
            with Image.open(path) as image:
                try:
                    pages = [np.array(page) for page in ImageSequence.Iterator(image)]
                finally:
                    image.close()  # Multipage TIFFs can retain a second file handle.
            return pages[0] if len(pages) == 1 else np.stack(pages)
        except Exception as pil_error:
            raise ValueError(f"Could not decode TIFF {path}: {tiff_error}; {pil_error}") from pil_error
        finally:
            Image.MAX_IMAGE_PIXELS = pixel_limit


def _read_highres_cube(source):
    """Validate declared TIFF allocation before decoding a source cube."""
    with tifffile.TiffFile(source) as image:
        if len(image.series) != 1:
            raise ValueError("Expected one TIFF cube series")
        series = image.series[0]
        if series.shape != (90, 360, 360) or series.dtype != np.dtype("uint16"):
            raise ValueError("Expected uint16 high-resolution cube ZYX [90,360,360]")
    if hasattr(source, "seek"):
        source.seek(0)
    block = _read_tiff(source)
    if block.shape != (90, 360, 360) or block.dtype != np.uint16:
        raise ValueError("Decoded cube contradicts TIFF header")
    return block


class Visual_toolkit:
    """
    A unified tool for retrieving Macaque brain data from mixed sources.
    """
    def __init__(self, sample_id='251637', cache_dir=None, low_res_dir=None,
                 low_res_ssh_base=None, low_res_slice_cache_limit=8):
        """
        Initialize toolkit.
        
        Args:
            sample_id: Sample ID (e.g., '251637')
            cache_dir: Optional custom cache directory. If None, uses default project structure.
        """
        self.sample_id = sample_id = str(sample_id)
        if not sample_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in sample_id):
            raise ValueError("sample_id must be a simple sample identifier")
        
        if cache_dir:
            # Use custom cache directory (for bulk processing)
            self.output_dir = os.path.join(cache_dir, 'output')
            self.cache_http_dir = os.path.join(cache_dir, 'cubes', sample_id, 'high_res_http')
            self.cache_ssh_dir = os.path.join(cache_dir, 'cubes', sample_id, 'low_res_ssh')
        else:
            # Use default project structure
            project_root = str(Path(__file__).resolve().parents[1])
            self.output_dir = os.path.join(project_root, 'resource', 'segmented_cubes', sample_id)
            self.cache_http_dir = os.path.join(project_root, 'resource', 'cubes', sample_id, 'high_res_http')
            self.cache_ssh_dir = os.path.join(project_root, 'resource', 'cubes', sample_id, 'low_res_ssh')
        
        for folder in [self.output_dir, self.cache_http_dir, self.cache_ssh_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.ssh_client = None
        self.sftp = None
        self.low_res_share_base = str(low_res_dir) if low_res_dir else LOW_RES_SHARE_BY_SAMPLE.get(sample_id)
        self.low_res_share_candidates = ([self.low_res_share_base] if self.low_res_share_base else [
            os.path.join(LOW_RES_SHARE_ROOT, f"{sample_id}-CH1_resample", "resample_5um"),
            os.path.join(LOW_RES_SHARE_ROOT, f"{sample_id}_CH1_resample", "resample_5um"),
        ])
        # The legacy SSH directory belongs only to 251637. Other samples need
        # an explicitly configured own directory, never another animal's images.
        self.low_res_ssh_base = low_res_ssh_base or (SSH_REMOTE_BASE if sample_id == '251637' else None)
        self.last_high_res_metadata = None
        self.last_low_res_metadata = None
        self.last_export_metadata = None
        self._slice_sources = {}
        self._block_errors = {}
        # Full resampled sections are large. Keep only a few decoded arrays.
        limit = int(low_res_slice_cache_limit)
        if limit < 0:
            raise ValueError("low_res_slice_cache_limit must be nonnegative")
        self.low_res_slice_cache_limit = limit
        self._section_cache = OrderedDict()

    def _low_res_slice_filename(self, z_index):
        return f"{self.sample_id}_{z_index:05d}_CH1_resample.tif"

    def _init_ssh(self):
        if self.sftp: return
        try:
            print(f"[INFO] Connecting to SSH ({SSH_HOST})...")
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=10)
            self.sftp = self.ssh_client.open_sftp()
            print("  > Connection Successful.")
        except Exception as e:
            print(f"  > [ERROR] SSH Connection Failed: {e}")

    def close(self):
        if self.sftp: self.sftp.close()
        if self.ssh_client: self.ssh_client.close()
        print("[INFO] Connections closed.")

    # ==========================================
    # SOURCE 1: HIGH RES (HTTP)
    # ==========================================
    def _download_http_block(self, idx_x, idx_y, idx_z):
        if any(not isinstance(index, (int, np.integer)) or index < 0 for index in (idx_x, idx_y, idx_z)):
            raise ValueError("Cube indices must be nonnegative integers")
        filename = f"{idx_x}_{idx_y}_{idx_z}.tif"
        local_path = os.path.join(self.cache_http_dir, str(idx_z), filename)
        
        if os.path.exists(local_path):
            try:
                if os.path.getsize(local_path) > MAX_HTTP_BLOCK_BYTES:
                    raise ValueError("Cached cube exceeds the response-size bound")
                block = _read_highres_cube(local_path)
                if block.shape != tuple(reversed(BLOCK_SIZE_PIXELS)):
                    raise ValueError(f"Unexpected cube shape {block.shape}")
                self._block_errors.pop((idx_x, idx_y, idx_z), None)
                return block
            except Exception as exc:
                self._block_errors[(idx_x, idx_y, idx_z)] = str(exc)
                # Keep corrupt cache evidence; a failed replacement stays explicit.

        url = f"{HTTP_HOST}/{HTTP_PATH}/{self.sample_id}/cube/{idx_z}/{filename}"
        last_error = None
        for attempt in range(HTTP_ATTEMPTS):
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read(MAX_HTTP_BLOCK_BYTES + 1)
                if len(data) > MAX_HTTP_BLOCK_BYTES:
                    raise ValueError("Cube response exceeds the response-size bound")
                block = _read_highres_cube(BytesIO(data))
                if block.shape != tuple(reversed(BLOCK_SIZE_PIXELS)):
                    raise ValueError(f"Unexpected cube shape {block.shape}")
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=os.path.dirname(local_path), suffix=".part", delete=False) as f:
                        temporary = f.name
                        f.write(data)
                    os.replace(temporary, local_path)
                finally:
                    if temporary and os.path.exists(temporary):
                        os.remove(temporary)
                self._block_errors.pop((idx_x, idx_y, idx_z), None)
                return block
            except Exception as exc:
                last_error = exc
                if attempt + 1 < HTTP_ATTEMPTS:
                    time.sleep(0.2 * (attempt + 1))
        self._block_errors[(idx_x, idx_y, idx_z)] = str(last_error)
        return None

    def _read_cached_section(self, path):
        """Decode one low-resolution section, reusing a small path-keyed cache."""
        key = os.path.abspath(path)
        cached = self._section_cache.get(key)
        if cached is not None:
            self._section_cache.move_to_end(key)
            return cached
        image = _read_tiff(path)
        if self.low_res_slice_cache_limit <= 0:
            return image
        self._section_cache[key] = image
        self._section_cache.move_to_end(key)
        while len(self._section_cache) > self.low_res_slice_cache_limit:
            self._section_cache.popitem(last=False)
        return image

    def get_high_res_block(self, center_um, grid_radius=2, allow_partial=False):
        print(f"\n[ACTION] Acquiring High-Res Soma Block (Radius {grid_radius})")
        
        center_um = triplet(center_um, 'center', nonnegative=True)
        if isinstance(grid_radius, bool) or int(grid_radius) != grid_radius or grid_radius < 1:
            raise ValueError("grid_radius must be a positive integer")
        grid_radius = int(grid_radius)
        bx = int((center_um[0] / RESOLUTION_HIGH[0]) // BLOCK_SIZE_PIXELS[0])
        by = int((center_um[1] / RESOLUTION_HIGH[1]) // BLOCK_SIZE_PIXELS[1])
        bz = int((center_um[2] / RESOLUTION_HIGH[2]) // BLOCK_SIZE_PIXELS[2])
        
        grid_dim = (grid_radius - 1) * 2 + 1
        volume = np.zeros((
            BLOCK_SIZE_PIXELS[2] * grid_dim, 
            BLOCK_SIZE_PIXELS[1] * grid_dim, 
            BLOCK_SIZE_PIXELS[0] * grid_dim
        ), dtype=np.uint16)
        
        count = 0; total = grid_dim**3
        requested, loaded, missing, sources = [], [], [], []
        
        for k in range(1 - grid_radius, grid_radius):
            for j in range(1 - grid_radius, grid_radius):
                for i in range(1 - grid_radius, grid_radius):
                    index = (bx+i, by+j, bz+k)
                    requested.append(list(index))
                    sources.append(f"{HTTP_HOST}/{HTTP_PATH}/{self.sample_id}/cube/{index[2]}/{index[0]}_{index[1]}_{index[2]}.tif")
                    block = self._download_http_block(*index) if min(index) >= 0 else None
                    
                    if block is not None:
                        if np.shape(block) != tuple(reversed(BLOCK_SIZE_PIXELS)):
                            raise ValueError(f"Cube {index} has unexpected ZYX shape {np.shape(block)}")
                        loaded.append(list(index))
                        arr_z = k + grid_radius - 1
                        arr_y = j + grid_radius - 1
                        arr_x = i + grid_radius - 1
                        
                        z_start = arr_z * BLOCK_SIZE_PIXELS[2]
                        y_start = arr_y * BLOCK_SIZE_PIXELS[1]
                        x_start = arr_x * BLOCK_SIZE_PIXELS[0]
                        
                        volume[
                            z_start : z_start + BLOCK_SIZE_PIXELS[2],
                            y_start : y_start + BLOCK_SIZE_PIXELS[1],
                            x_start : x_start + BLOCK_SIZE_PIXELS[0]
                        ] = block
                    else:
                        missing.append(list(index))
                    
                    count += 1
                    print(f"\r  > Downloading Block {count}/{total}", end="")
        
        print("\n  > High-Res Volume Built.")
        
        origin_x = (bx + 1 - grid_radius) * BLOCK_SIZE_PIXELS[0] * RESOLUTION_HIGH[0]
        origin_y = (by + 1 - grid_radius) * BLOCK_SIZE_PIXELS[1] * RESOLUTION_HIGH[1]
        origin_z = (bz + 1 - grid_radius) * BLOCK_SIZE_PIXELS[2] * RESOLUTION_HIGH[2]
        self.last_high_res_metadata = self._acquisition_metadata(
            'HTTP native cubes', center_um, [origin_x, origin_y, origin_z],
            RESOLUTION_HIGH, volume.shape, requested, loaded, missing, sources)
        self.last_high_res_metadata['missing_errors'] = {
            '_'.join(map(str, index)): self._block_errors.get(tuple(index), 'unavailable or outside image bounds')
            for index in missing}
        if [bx, by, bz] not in loaded:
            raise ValueError("Soma's central high-resolution cube is unavailable")
        if missing and not allow_partial:
            raise ValueError(f"High-resolution acquisition incomplete: {len(missing)}/{total} cubes missing")
        
        return volume, [origin_x, origin_y, origin_z], RESOLUTION_HIGH

    # ==========================================
    # SOURCE 2: LOW RES (SMB share, SSH fallback)
    # ==========================================
    def _get_low_res_slice_path(self, z_index):
        """Return path to a resampled z-slice TIFF (UNC share preferred)."""
        filename = self._low_res_slice_filename(z_index)

        for base in self.low_res_share_candidates:
            share_path = os.path.join(base, filename)
            if os.path.isfile(share_path):
                self._slice_sources[z_index] = share_path
                return share_path

        local_path = os.path.join(self.cache_ssh_dir, filename)
        if os.path.isfile(local_path):
            self._slice_sources[z_index] = local_path
            return local_path

        if not self.low_res_ssh_base:
            return None
        self._init_ssh()
        if not self.sftp:
            return None
        try:
            remote_path = f"{self.low_res_ssh_base}/{filename}"
            self.sftp.get(remote_path, local_path)
            self._slice_sources[z_index] = f"ssh://{SSH_HOST}:{SSH_PORT}{remote_path}"
            return local_path if os.path.isfile(local_path) else None
        except Exception:
            return None

    def get_low_res_widefield(self, center_um, width_um=10000, height_um=10000, depth_um=90,
                             allow_partial=False):
        print(f"\n[ACTION] Acquiring Low-Res Wide Field ({width_um}x{height_um} um)")
        if self.low_res_share_base:
            print(f"  > Low-res source: {self.low_res_share_base}")
        
        center_um = triplet(center_um, 'center', nonnegative=True)
        start, stop = widefield_bounds(center_um, RESOLUTION_LOW, width_um, height_um, depth_um)
        min_x, min_y, z_start = map(int, start)
        max_x, max_y, z_stop = map(int, stop)
        z_idx = int(center_um[2] / RESOLUTION_LOW[2])
        stack, loaded, missing, sources = [], [], [], []
        requested = list(range(z_start, z_stop))
        slice_shape = None
        errors = {}
        print(f"  > Fetching Z-Slices: {z_start} to {z_stop - 1}...")
        
        for z in requested:
            f_path = self._get_low_res_slice_path(z)
            th, tw = max_y - min_y, max_x - min_x
            if f_path:
                try:
                    img = self._read_cached_section(f_path)
                except Exception as exc:
                    print(f"  > [WARN] Could not read slice z={z}: {exc}")
                    errors[z] = str(exc)
                    missing.append(z)
                    stack.append(None)
                    continue
                if img.ndim != 2:
                    raise ValueError(f"Low-resolution slice z={z} must be grayscale YX, got {img.shape}")
                if slice_shape is not None and img.shape != slice_shape:
                    raise ValueError(f"Inconsistent low-resolution section shape at z={z}: {img.shape} != {slice_shape}")
                slice_shape = img.shape
                h, w = img.shape
                my = min(max_y, h); mx = min(max_x, w)
                if min_x >= mx or min_y >= my:
                    raise ValueError("Requested soma crop lies outside the native image section")
                crop = img[min_y:my, min_x:mx]
                stack.append(crop)
                loaded.append(z)
                sources.append({'z_index': z, 'path': self._slice_sources.get(z, f_path)})
            else:
                missing.append(z)
                stack.append(None)

        shape = next((crop.shape for crop in stack if crop is not None), (max_y-min_y, max_x-min_x))
        volume = np.stack([crop if crop is not None else np.zeros(shape, dtype=np.uint16) for crop in stack])
        
        origin_x = min_x * RESOLUTION_LOW[0]
        origin_y = min_y * RESOLUTION_LOW[1]
        origin_z = z_start * RESOLUTION_LOW[2]
        self.last_low_res_metadata = self._acquisition_metadata(
            'native resampled TIFF sections', center_um, [origin_x, origin_y, origin_z],
            RESOLUTION_LOW, volume.shape, requested, loaded, missing, sources)
        self.last_low_res_metadata['missing_errors'] = errors
        self.last_low_res_metadata['full_section_shape_yx'] = list(slice_shape) if slice_shape else None
        self.last_low_res_metadata['requested_crop_bounds_xyz'] = [start.tolist(), stop.tolist()]
        wanted_xy = np.floor(center_um[:2] / np.asarray(RESOLUTION_LOW[:2])).astype(int) - (stop-start)[:2] // 2
        self.last_low_res_metadata['crop_clipped_to_image_bounds'] = bool(
            np.any(wanted_xy < 0) or shape != (max_y-min_y, max_x-min_x))
        if z_idx not in loaded:
            raise ValueError("Soma's central low-resolution section is unavailable")
        soma_index = native_to_index(center_um, [origin_x, origin_y, origin_z], RESOLUTION_LOW)
        if not (0 <= soma_index[0] < volume.shape[2] and 0 <= soma_index[1] < volume.shape[1]):
            raise ValueError("Soma lies outside the available low-resolution crop")
        if missing and not allow_partial:
            raise ValueError(f"Low-resolution acquisition incomplete: {len(missing)}/{len(requested)} sections missing")
        
        return volume, [origin_x, origin_y, origin_z], RESOLUTION_LOW

    def _acquisition_metadata(self, source, center, origin, spacing, shape,
                              requested, loaded, missing, source_paths):
        return {
            'sample_id': self.sample_id, 'source': source,
            'coordinate_frame': 'native microscopy acquisition XYZ; anatomical orientation and laterality unverified',
            'coordinate_units': 'micrometre', 'array_axis_order': 'ZYX',
            'center_xyz_um': np.asarray(center).tolist(), 'origin_xyz_um': list(origin),
            'spacing_xyz_um': list(spacing), 'spacing_status': 'nominal repository configuration; not independently calibrated',
            'shape_zyx': list(shape), 'requested': requested, 'loaded': loaded,
            'missing': missing, 'complete': not missing, 'source_paths': source_paths,
            'index_convention': 'XYZ index=(native_um-origin_um)/spacing_um; Z section filename index maps to index*spacing_z; no axis flip',
            'missing_data_policy': 'reject missing central data; partial zero-filled gaps only with explicit allow_partial=True',
        }

    # ==========================================
    # EXPORT & VISUALIZATION (UPDATED WITH SOMA REGION)
    # ==========================================
    def _sanitize_filename(self, name):
        """Sanitize string for use in filename (remove/replace path separators)."""
        if not name:
            return name
        # Replace path separators and other problematic characters
        return name.replace('/', '-').replace('\\', '-').replace(':', '-')

    def _matching_export_acquisition(self, shape, origin, spacing, acquisition=None):
        """Bind provenance to matching sample/grid geometry, never to a suffix."""
        def matches(record):
            if not isinstance(record, dict) or str(record.get('sample_id')) != str(self.sample_id):
                return False
            if record.get('coordinate_units') != 'micrometre':
                return False
            axes = 'ZYX' if len(shape) == 3 else 'YX'
            if record.get('array_axis_order') != axes:
                return False
            shape_key = 'shape_zyx' if len(shape) == 3 else 'shape_yx'
            origin_key = 'origin_xyz_um' if len(origin) == 3 else 'origin_xy_um'
            spacing_key = 'spacing_xyz_um' if len(spacing) == 3 else 'spacing_xy_um'
            try:
                return (
                    tuple(record.get(shape_key, ())) == tuple(shape)
                    and np.asarray(record.get(origin_key), dtype=float).shape == origin.shape
                    and np.asarray(record.get(spacing_key), dtype=float).shape == spacing.shape
                    and np.allclose(record[origin_key], origin, rtol=0, atol=1e-6)
                    and np.allclose(record[spacing_key], spacing, rtol=0, atol=1e-6)
                )
            except (KeyError, TypeError, ValueError):
                return False

        if acquisition is not None:
            if not matches(acquisition):
                raise ValueError('Export acquisition metadata does not match sample, axes or crop geometry')
            candidates = [acquisition]
        else:
            candidates = [record for record in (
                getattr(self, 'last_high_res_metadata', None),
                getattr(self, 'last_low_res_metadata', None),
            ) if matches(record)]
        if len(candidates) != 1:
            return None
        from copy import deepcopy
        return deepcopy(candidates[0])

    def export_data(self, volume, origin, resolution, neuron_id, suffix="Volume", 
                    soma_region=None, soma_coords=None, output_dir=None, acquisition_metadata=None):
        """
        Export volume data with soma region info in filename.
        Coordinates are shown in title but NOT in filename.
        """
        volume = np.asarray(volume)
        if volume.ndim not in (2, 3) or any(size == 0 for size in volume.shape):
            raise ValueError('Export requires a nonempty 2D YX plane or 3D ZYX volume')
        origin = np.asarray(origin, dtype=float)
        resolution = np.asarray(resolution, dtype=float)
        allowed_shapes = {(3,)} if volume.ndim == 3 else {(2,), (3,)}
        if origin.shape not in allowed_shapes or not np.isfinite(origin).all():
            raise ValueError('Export origin must contain finite native coordinates')
        if (resolution.shape not in allowed_shapes or not np.isfinite(resolution).all()
                or np.any(resolution <= 0)):
            raise ValueError('Export spacing must contain finite positive values')
        if soma_coords is not None:
            soma_coords = np.asarray(soma_coords, dtype=float)
            if soma_coords.shape != (3,) or not np.isfinite(soma_coords).all():
                raise ValueError('Export raw root must contain three finite XYZ coordinates')
        acquisition = self._matching_export_acquisition(
            volume.shape, origin, resolution, acquisition_metadata)
        affine = native_affine(origin, resolution) if volume.ndim == 3 else None
        target_dir = output_dir if output_dir else self.output_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        # Build filename with region (coordinates NOT in filename)
        # Sanitize region name to prevent path issues
        if soma_region:
            safe_region = self._sanitize_filename(soma_region)
            filename = f"{self.sample_id}_{neuron_id}_{safe_region}_{suffix}"
        else:
            filename = f"{self.sample_id}_{neuron_id}_{suffix}"
            
        ext = ".nii.gz" if volume.ndim == 3 else ".tif"
        full_path = os.path.join(target_dir, filename + ext)
        
        print(f"  > Exporting: {full_path}")

        if ".nii" in ext:
            vol_xyz = np.transpose(volume, (2, 1, 0))
            image = nib.Nifti1Image(vol_xyz, affine)
            image.header.set_xyzt_units('micron')
            image.header['descrip'] = b'Native microscopy XYZ; anatomical orientation unverified'
            nib.save(image, full_path)
        else:
            tifffile.imwrite(full_path, volume)
        # The persisted NIfTI is XYZ. The in-memory source array remains ZYX.
        # A missing raw root must not be replaced by the crop origin.
        metadata = self._acquisition_metadata(
            'exported native microscopy crop',
            soma_coords if soma_coords is not None else origin,
            origin, resolution, volume.shape, [], [], [], [])
        if soma_coords is None:
            metadata['center_xyz_um'] = None
            metadata['center_status'] = 'raw root was not supplied; origin_xyz_um is the crop origin only'
        else:
            metadata['center_xyz_um'] = [float(value) for value in soma_coords]
        if volume.ndim == 3:
            metadata['source_array_axis_order'] = 'ZYX'
            metadata['nifti_axis_order'] = 'XYZ'
            metadata['array_axis_order'] = 'XYZ'
            metadata['shape_xyz'] = [int(volume.shape[2]), int(volume.shape[1]), int(volume.shape[0])]
        else:
            metadata['source_array_axis_order'] = 'YX'
            metadata['array_axis_order'] = 'YX'
            metadata['shape_yx'] = list(volume.shape)
            metadata.pop('shape_zyx', None)
            metadata['coordinate_frame'] = (
                'native microscopy XY plane; supplied XYZ origin records section position '
                'when available; anatomical orientation and laterality unverified')
            metadata['index_convention'] = (
                'array[Y,X]; XY index=(native_xy-origin_xy)/spacing_xy; '
                'section Z is not inferred from an array axis; no axis flip')
            metadata['origin_xy_um'] = origin[:2].tolist()
            metadata['spacing_xy_um'] = resolution[:2].tolist()
            if origin.size != 3:
                metadata.pop('origin_xyz_um', None)
            if resolution.size != 3:
                metadata.pop('spacing_xyz_um', None)
        metadata['acquisition'] = acquisition if isinstance(acquisition, dict) else None
        if isinstance(acquisition, dict) and acquisition:
            metadata['complete'] = acquisition.get('complete')
            metadata['missing'] = acquisition.get('missing')
            metadata['loaded'] = acquisition.get('loaded')
            metadata['requested'] = acquisition.get('requested')
            metadata['source_paths'] = acquisition.get('source_paths')
        else:
            metadata['complete'] = None
            metadata['missing'] = None
        metadata['output_file'] = full_path
        self.last_export_metadata = metadata
        with open(full_path + '.json', 'w', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2, default=str)
        return full_path

    def plot_soma_block(self, volume_3d, origin, resolution, soma_coords, neuron_id, 
                        suffix="SomaBlock", soma_region=None, output_dir=None):
        """
        Plots Grayscale Anatomy (High Res) using MIP.
        Includes soma region in filename and title.
        """
        print(f"  > Generating Grayscale Plot ({suffix})...")
        
        # --- FIXED: Use MIP (Maximum Intensity Projection) instead of Middle Slice ---
        img = np.max(volume_3d, axis=0)
        
        # Local Soma Pixel
        sx = (soma_coords[0] - origin[0]) / resolution[0]
        sy = (soma_coords[1] - origin[1]) / resolution[1]
        
        # Contrast (Gamma 0.5)
        img_norm = percentile_normalize(img, 0.5, 99.5)
        
        img_final = np.power(img_norm, 0.5)
        
        self._save_plot(img_final, sx, sy, resolution, neuron_id, suffix, 
                       volume_3d.shape[0], cmap='gray', marker_color='cyan', 
                       soma_region=soma_region, soma_coords=soma_coords,
                       output_dir=output_dir, scale_bar_um=30)

    def plot_widefield_context(self, volume_3d, origin, resolution, soma_coords, neuron_id, 
                             suffix="WideField", manual_threshold=100, bg_intensity=0.4, 
                             swc_tree=None, soma_region=None, output_dir=None):
        """
        Plots Green Intensity on Dark Background + SWC Overlay using MIP.
        Includes soma region in filename and title.
        """
        print(f"  > Generating Composite Plot ({suffix})...")
        
        # Uses MIP by default
        mip = np.max(volume_3d, axis=0).astype(float)
        h, w = mip.shape
        
        # 1. Base Gray Layer
        d_max = np.max(mip)
        if d_max < manual_threshold: manual_threshold = d_max * 0.5
        norm_base = np.clip(percentile_normalize(mip) * bg_intensity, 0, 1)
        
        rgb = np.zeros((h, w, 3), dtype=float)
        rgb[..., 0] = norm_base; rgb[..., 1] = norm_base; rgb[..., 2] = norm_base
        
        # 2. Green Signal Layer
        mask = mip > manual_threshold
        if np.sum(mask) > 0:
            sig = mip[mask]
            s_min, s_max = np.min(sig), np.max(sig)
            if s_max > s_min:
                s_norm = (sig - s_min) / (s_max - s_min)
            else:
                s_norm = np.zeros_like(sig)
                
            bright = np.clip(bg_intensity + 0.2 + (0.4 * s_norm), 0, 1)
            
            rgb[mask, 0] *= 0.2; rgb[mask, 2] *= 0.2
            rgb[mask, 1] = bright

        # 3. Setup Figure
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black', dpi=100)
        ax.imshow(rgb, origin='upper')
        
        if swc_tree:
            self._overlay_swc(ax, swc_tree, origin, resolution, h, w, volume_3d.shape[0])

        sx = (soma_coords[0] - origin[0]) / resolution[0]
        sy = (soma_coords[1] - origin[1]) / resolution[1]
        
        self._finalize_plot(fig, ax, sx, sy, resolution, neuron_id, suffix, volume_3d.shape[0], 
                           marker_color='white', soma_region=soma_region, soma_coords=soma_coords,
                           output_dir=output_dir, scale_bar_um=500)

    def _overlay_swc(self, ax, swc_tree, origin, resolution, h, w, z_slices=None):
        if swc_tree:
            print("Overlaying SWC Edges...")
        segments = []
        for edge in swc_tree.edges:
            for p, q in zip(edge.data, edge.data[1:]):
                start = native_to_index([p.x, p.y, p.z], origin, resolution)
                stop = native_to_index([q.x, q.y, q.z], origin, resolution)
                lower_z = -.5 if z_slices is not None else min(start[2], stop[2]) - 1
                upper_z = z_slices-.5 if z_slices is not None else max(start[2], stop[2]) + 1
                clipped = clip_segment(start, stop, [-.5, -.5, lower_z], [w-.5, h-.5, upper_z])
                if clipped is not None:
                    segments.append([clipped[0][:2], clipped[1][:2]])
        if segments:
            ax.add_collection(LineCollection(segments, colors='red', linewidths=.5, alpha=.5))

    def _save_plot(self, img_data, sx, sy, resolution, neuron_id, suffix, z_slices, 
                   cmap, marker_color, soma_region=None, soma_coords=None, output_dir=None, scale_bar_um=30):
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.imshow(img_data, cmap=cmap, origin='upper')
        self._finalize_plot(fig, ax, sx, sy, resolution, neuron_id, suffix, z_slices, 
                           marker_color, soma_region=soma_region, soma_coords=soma_coords,
                           output_dir=output_dir, scale_bar_um=scale_bar_um)

    def _finalize_plot(self, fig, ax, sx, sy, resolution, neuron_id, suffix, z_slices, 
                       marker_color, soma_region=None, soma_coords=None, output_dir=None, scale_bar_um=30):
        h, w = ax.images[0].get_size()
        
        target_dir = output_dir if output_dir else self.output_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        marker_size_um = 20 if resolution[0] < 1.0 else 200
        marker_size_px = marker_size_um / resolution[0]
        marker_s = (marker_size_px * 0.8) ** 2
        
        ax.scatter(sx, sy, s=marker_s, marker='s', facecolors='none', edgecolors=marker_color, 
                   linewidth=0.75, label='Soma', zorder=10)
        
        label_offset_px = marker_size_px 
        ax.text(sx + label_offset_px, sy + label_offset_px, 'Target Soma', 
                color=marker_color, fontsize=9, fontweight='bold')

        bar_px = scale_bar_um / resolution[0]
        bx = w - bar_px - 50; by = h - 50
        ax.plot([bx, bx+bar_px], [by, by], color='white', linewidth=3)
        ax.text(bx+bar_px/2, by-20, f"{scale_bar_um} µm", color='white', ha='center', fontweight='bold')
        
        # Build title with soma region and coordinates
        if soma_region:
            region_line = f"Region: {soma_region}"
            if soma_coords is not None:
                region_line += f" | XYZ: ({int(soma_coords[0])}, {int(soma_coords[1])}, {int(soma_coords[2])})"
            title = f"{self.sample_id} | {neuron_id} | {suffix}\n{region_line}\nFOV: {w*resolution[0]:.0f}x{h*resolution[1]:.0f} µm | Depth: {z_slices*resolution[2]:.0f} µm"
        else:
            title = f"{self.sample_id} | {neuron_id} | {suffix}\nFOV: {w*resolution[0]:.0f}x{h*resolution[1]:.0f} µm | Depth: {z_slices*resolution[2]:.0f} µm"
        
        ax.set_title(title, color='white', fontsize=12)
        ax.axis('off')
        
        # Build filename with soma region (coordinates NOT in filename)
        # Sanitize region name to prevent path issues
        if soma_region:
            safe_region = self._sanitize_filename(soma_region)
            plot_name = f"{self.sample_id}_{neuron_id}_{safe_region}_{suffix}_Plot.png"
        else:
            plot_name = f"{self.sample_id}_{neuron_id}_{suffix}_Plot.png"
            
        save_path = os.path.join(target_dir, plot_name)
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='black', dpi=600)
        print(f"  > Plot saved: {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    toolkit = Visual_toolkit('251637')
    ion = IT.IONData()
    NEURON = '003.swc'
    tree = ion.getRawNeuronTreeByID('251637', NEURON)
    
    if tree:
        try: soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
        except: soma_xyz = tree.root.xyz
        print(f"Target Soma: {soma_xyz}")
        
        # Test with soma region
        test_region = "DI"
        
        # Test 1: High Res Soma Block
        high_res_volume, high_res_origin, high_res_resolution = toolkit.get_high_res_block(soma_xyz, grid_radius=1)
        toolkit.plot_soma_block(high_res_volume, high_res_origin, high_res_resolution, soma_xyz, NEURON, soma_region=test_region)
        
        # Test 2: Low Res Wide Field
        low_res_volume, low_res_origin, low_res_resolution = toolkit.get_low_res_widefield(soma_xyz, width_um=8000, height_um=8000, depth_um=30)
        toolkit.plot_widefield_context(low_res_volume, low_res_origin, low_res_resolution, soma_xyz, NEURON, 
                                       bg_intensity=2.0, swc_tree=tree, soma_region=test_region)
        
    toolkit.close()
