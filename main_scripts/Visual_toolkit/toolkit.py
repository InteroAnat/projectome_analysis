"""
Visual_toolkit.py - Macaque Brain Hybrid-Resolution Visualization Toolkit

Version: 1.9.1 (local/unreleased; package version)
Last updated: 2026-10-17
Change history: CHANGELOG.md (this folder)
Status: candidate (visual review pending)

UPDATE NOTES (v1.4.0):
    - Added soma_region parameter to export and plot methods.
    - Filenames and titles now include soma region and formatted coordinates.
    - Supports cache directory configuration for bulk processing.

2026-10-15 Codex (local, unreleased): corrected 252385 low-resolution sources
require the exhaustive completion record, use an isolated cache, and retain
the shift record hash/parameters in acquisition provenance. Default source
activation follows whole-stack verification; original data remain an override.

2026-10-16 Codex (local, unreleased): WideField default gain is 2.0 to match
legacy GUI/bulk rendering; retain legacy post-normalization gain order so the
green-mask red/blue attenuation also matches. SomaBlock/WideField windows,
gamma, green formula, and threshold remain unchanged.

2026-10-16 Codex (local, unreleased): add provisional per-sample WideField
display profiles after bounded two-crop gain comparisons on six native-5um
samples. Profiles: 251637=1.4; 252384/252385/252527/252714/252790=1.0,
all at threshold 100. Other samples retain the legacy 2.0/100 defaults;
explicit call parameters win. These are display-only candidates, not scientific
acceptance.

2026-10-16 Codex (local, unreleased): the user reviewed the cohort comparison
and selected legacy gain 2.0/threshold 100 for every sample, with no lower-gain
sample defaults. The prior lower-gain candidates are superseded; explicit
per-call parameters remain supported.

2026-10-17 Codex (local, unreleased): add the soma-centred sagittal context
panel and one three-panel review figure (Soma | WideField | Sagittal) with a
single shared legend, so a neuron no longer needs three separate PNG checks.
Single views and tri-plot panels are drawn by one helper; existing SomaBlock/
WideField PNGs, windows, gamma and filenames are unchanged. The A/B raw-SWC
mapping candidates for the processed-DS sagittal remain unresolved.

2026-10-17 Codex (local, unreleased): after review, the sagittal panel shows the
whole constant-i plane with an automatic 1/2/5 mm scale bar, the tri-plot bottom
legend band is off by default (each panel title carries its source dataset), and
the target soma is a white open square plus crosshair centred on the exact
position. Panel titles: `sample | neuron | Soma Plot | src_dataset: 0.65µm_raw`,
`… | Context Plot | src_dataset: 5µm_localserver`,
`… | Sagittal Context Plot | src_dataset: DS00 N4` (field renamed from the earlier
`src:` at the user's request). Filenames keep the legacy suffixes. A/B raw-SWC
mapping remains unresolved.

2026-10-17 Codex (local, unreleased): one target marker for all three panel types —
a cyan/turquoise (`#00e5ee`) open box with a cross on the exact mapped point, drawn
by the single `_draw_target_marker` helper (the legacy single-plot path and the
shared panel path both call it, so no panel can ship a marker without a cross).
`REVIEW_SETTINGS` now holds the shared GUI/runner/adapter settings (marker, context
crop, widefield gain/threshold, scale bars, figure size). The Soma panel is a
soma-centred multi-cube window (`soma_window`, default 1.5 cubes ≈ 351 µm) instead
of a one-cube crop: the largest window step whose cubes all load is used, the centre
offset is recorded, and the window's contiguous cube range now includes middle
columns (the previous two-endpoint set skipped them). This supersedes the earlier
white sagittal marker and one-cube soma window; all outputs are display-only and no
anatomical acceptance is implied.

Change history for the package is consolidated in CHANGELOG.md beside this module.

2026-10-17 Codex (local, unreleased): the processed-DS sagittal implementation was
merged into this module from ``sagittal.py``. Source/SWC resolution, panel setup,
legacy PIL rendering and the shared review-panel preparation now live beside the
Soma/Context renderers, with no separate sagittal module.
"""

import copy
import math
import os
import hashlib
import gzip
import struct
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
from typing import Any
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from .image_geometry import (
    triplet, native_to_index, native_affine, widefield_bounds,
    clip_segment, percentile_normalize,
)

# Respect the caller's chosen backend (the bulk runners explicitly use Agg).

# --- PATHS ---
neurovis_path = str(Path(__file__).resolve().parents[2] / "neuron-vis/neuronVis")
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
BUILTIN_LOW_RES_SHARE_BY_SAMPLE = {
    '251637': os.path.join(LOW_RES_SHARE_ROOT, "251637-CH1_resample", "resample_5um"),
    # Hyphen and underscore differ by sample. These four new copies plus the
    # 251637 reference were confirmed on the 5micron share (2026-10-02).
    '252384': os.path.join(LOW_RES_SHARE_ROOT, "252384_CH1_resample", "resample_5um"),
    # 2026-10-15: fully verified correction activated; original paths preserved.
    '252385': os.path.join(LOW_RES_SHARE_ROOT, "252385-CH1_resample", "resample_5um_z12527_aligned"),
    '252527': os.path.join(LOW_RES_SHARE_ROOT, "252527_CH1_resample", "resample_5um"),
    '252714': os.path.join(LOW_RES_SHARE_ROOT, "252714-CH1_resample", "resample_5um"),
    '252790': os.path.join(LOW_RES_SHARE_ROOT, "252790-CH1_resample", "resample_5um"),
}


def _low_res_sources_from_config() -> dict[str, str]:
    """Per-sample 5 um section folders from Visual_toolkit/visual_toolkit_config.yaml.

    Falls back to the built-in table when the config is absent. Samples without a
    native 5 um dataset are simply absent, so the Context panel reports itself as
    unavailable for that sample while the other panels still render.
    """
    try:
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from .dataset_config import load as _load_dataset_config
        config = _load_dataset_config()
    except Exception:
        return {}
    return {sample_id: str(entry["resample_5um"])
            for sample_id, entry in (config.get("samples") or {}).items()
            if entry.get("resample_5um")}


LOW_RES_SHARE_BY_SAMPLE = _low_res_sources_from_config() or BUILTIN_LOW_RES_SHARE_BY_SAMPLE
SSH_HOST = "172.20.10.250"
SSH_PORT = 20007
SSH_USER = "binbin"
SSH_PASS = os.environ.get("PROJECTOME_SSH_PASSWORD")
SSH_REMOTE_BASE = "/home/binbin/share/251637CH1_projection/251637-CH1_resample/resample_5um"
RESOLUTION_LOW  = [5.0, 5.0, 3.0]     # [X, Y, Z] microns

# --- REVIEW SETTINGS (one source for the GUI, bulk runners and every panel renderer) ---
# Everything the Soma / Context / Sagittal review figures need: what to acquire, how
# to draw it, and the single target marker. Callers read these values instead of
# repeating numbers, and explicit plot/CLI parameters still override them.
REVIEW_SETTINGS = {
    # One cyan/turquoise open box with a cross through the exact point marks the
    # target in every panel type (Soma, Context, Sagittal) and in the review figure.
    'marker': {
        'color': '#00e5ee',
        'min_px': 8.0,          # smallest drawn box side, in display pixels
        'crosshair_px': 60.0,   # cross arm length from the exact centre
        'label_fontsize': 12,
        'label_offset_px': 22.0,
    },
    # Soma panel: soma-centred native window, expressed in cubes. The runner starts
    # at the largest step and uses the largest step whose cubes it can load, so the
    # soma stays centred instead of landing near a cube edge.
    'soma_window': {
        'cubes_xy': 1.5,                    # requested window size, in cubes
        'cubes_xy_steps': (1.5, 1.25, 1.0),  # descending centred alternatives
        'max_cubes_xy': 3.0,                # hard cap: the acquired 3 x 3 cube grid
    },
    'context': {'width_um': 8000, 'height_um': 8000, 'depth_um': 30},
    'widefield': {'gain': 2.0, 'threshold': 100},
    # Drawn scale bars: the Soma/Context panels keep their legacy fixed bars and the
    # sagittal panel picks the largest 1/2/5 mm bar that fits its field of view.
    'scale_bar_um': {'soma': 30.0, 'context': 500.0, 'sagittal': 'auto'},
    # 1/2/5 x 10^k mm scale bars; the drawn bar is the largest one that still reads
    # as a small feature of the panel, so the three review panels stay comparable.
    'scale_bar_candidates_um': (100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0,
                                10000.0, 20000.0, 50000.0),
    'figure': {'panel_figsize': (12, 12), 'panel_dpi': 600,
               'review_figsize': (30, 11.6), 'review_dpi': 200},
}

TARGET_MARKER_COLOR = REVIEW_SETTINGS['marker']['color']

# Uniform legacy display defaults; explicit plot-call overrides remain supported.
WIDEFIELD_DEFAULT_PARAMETERS = (REVIEW_SETTINGS['widefield']['gain'],
                                REVIEW_SETTINGS['widefield']['threshold'])


# Legacy call-site suffixes map to the current review names, which now drive both
# the plot titles and the written filenames (SomaBlock -> Soma, WideField -> Context).
LEGACY_SUFFIX_NAMES = {'SomaBlock': 'Soma', 'WideField': 'Context'}


def _output_name(suffix) -> str:
    """Current review name for a legacy call-site suffix."""
    return LEGACY_SUFFIX_NAMES.get(str(suffix), str(suffix))


SCALE_BAR_CANDIDATES_UM = REVIEW_SETTINGS['scale_bar_candidates_um']


def _auto_scale_bar_um(fov_um, fraction=0.2):
    """Largest decade scale bar no longer than ``fraction`` of the field width."""
    limit = abs(float(fov_um[0])) * float(fraction)
    fits = [value for value in SCALE_BAR_CANDIDATES_UM if value <= limit]
    return fits[-1] if fits else SCALE_BAR_CANDIDATES_UM[0]


def _stamp() -> str:
    """Per-figure timestamp for quick tracking; the role suffix stays last."""
    return time.strftime('%Y%m%d-%H%M%S')


def resolve_widefield_parameters(sample_id, bg_intensity=None, manual_threshold=None):
    """Resolve uniform legacy display defaults, honoring explicit overrides."""
    default_gain, default_threshold = WIDEFIELD_DEFAULT_PARAMETERS
    gain = default_gain if bg_intensity is None else float(bg_intensity)
    threshold = default_threshold if manual_threshold is None else float(manual_threshold)
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("bg_intensity must be finite and non-negative")
    if not np.isfinite(threshold):
        raise ValueError("manual_threshold must be finite")
    return gain, threshold


def review_settings_snapshot() -> dict:
    """Deep copy of the resolved review settings, for run manifests and receipts."""
    return copy.deepcopy(REVIEW_SETTINGS)


# Leaves that must stay strictly positive when overridden from the command line.
_POSITIVE_SETTING_LEAVES = {
    'min_px', 'crosshair_px', 'label_fontsize', 'label_offset_px', 'cubes_xy',
    'max_cubes_xy', 'width_um', 'height_um', 'depth_um', 'gain', 'threshold',
    'soma', 'context', 'panel_dpi', 'review_dpi',
}


def apply_setting_overrides(overrides, settings=None):
    """Apply ``path=value`` overrides to the review settings and return what changed.

    The operator handle for tuning without editing code, e.g.
    ``--set widefield.gain=1.6 --set marker.crosshair_px=80``. Only existing scalar
    leaves can be overridden; unknown paths, non-scalar leaves and invalid values
    raise a clear error so a typo can never silently change a render.
    """
    target = REVIEW_SETTINGS if settings is None else settings
    changed = {}
    for item in overrides or []:
        text = str(item)
        if '=' not in text:
            raise ValueError(f"--set expects PATH=VALUE, got {text!r}")
        path, raw = (part.strip() for part in text.split('=', 1))
        if not path or not raw:
            raise ValueError(f"--set expects PATH=VALUE, got {text!r}")
        node = target
        keys = path.split('.')
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"Unknown review setting {path!r}; see REVIEW_SETTINGS")
            node = node[key]
        leaf = keys[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"Unknown review setting {path!r}; see REVIEW_SETTINGS")
        current = node[leaf]
        if isinstance(current, bool) or isinstance(current, (list, tuple, dict)):
            raise ValueError(f"Review setting {path!r} is not a scalar; edit REVIEW_SETTINGS")
        if isinstance(current, float):
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"Review setting {path!r} needs a finite number")
        elif isinstance(current, int):
            value = int(raw)
        else:
            value = raw
            if leaf == 'color' and not (len(value) == 7 and value.startswith('#')
                                        and all(ch in '0123456789abcdefABCDEF' for ch in value[1:])):
                raise ValueError(f"Review setting {path!r} needs a #rrggbb colour")
        if leaf in _POSITIVE_SETTING_LEAVES and not float(value) > 0:
            raise ValueError(f"Review setting {path!r} must be positive")
        node[leaf] = value
        changed[path] = value
    return changed


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
            project_root = str(Path(__file__).resolve().parents[2])
            self.output_dir = os.path.join(project_root, 'resource', 'segmented_cubes', sample_id)
            self.cache_http_dir = os.path.join(project_root, 'resource', 'cubes', sample_id, 'high_res_http')
            self.cache_ssh_dir = os.path.join(project_root, 'resource', 'cubes', sample_id, 'low_res_ssh')
        
        for folder in [self.output_dir, self.cache_http_dir, self.cache_ssh_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.ssh_client = None
        self.sftp = None
        self.low_res_share_base = str(low_res_dir) if low_res_dir else LOW_RES_SHARE_BY_SAMPLE.get(sample_id)
        self.low_res_correction = None
        if sample_id == '252385' and self.low_res_share_base and Path(self.low_res_share_base).name == 'resample_5um_z12527_aligned':
            record_path = Path(self.low_res_share_base) / 'shift_record.json'
            blob = record_path.read_bytes(); completion = json.loads(blob)
            if (completion.get('state') != 'complete_verified' or completion.get('sample_id') != sample_id
                    or completion.get('low_res_shift_left_down_px') != [653, 782]
                    or completion.get('first_section') != 12527
                    or len(completion.get('precut', [])) != 12526
                    or len(completion.get('sections', {})) != 8262
                    or completion.get('failures') or completion.get('missing_files') or completion.get('unexpected_files')):
                raise ValueError('Corrected 252385 source requires exhaustive complete_verified readback')
            self.low_res_correction = {'record': str(record_path), 'sha256': hashlib.sha256(blob).hexdigest(),
                                      'first_section': 12527, 'shift_left_down_px': [653, 782],
                                      'SWC_transform_applied': False, 'padding': 'zero; excluded source support'}
            self.cache_ssh_dir = str(Path(self.cache_ssh_dir).with_name('low_res_ssh_z12527_aligned'))
            os.makedirs(self.cache_ssh_dir, exist_ok=True)
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

    def get_high_res_soma_window(self, center_um, window_px=None):
        """
        Return a soma-centred native window that is larger than one cube.

        ``get_high_res_block`` returns the cube that happens to contain the soma,
        so the soma can sit anywhere inside it (048 sat 9 px from the right edge).
        This path assembles the cubes needed to cover a window centred on the
        fractional soma position and crops to that window, so the soma marker lands
        at the panel centre while the panel keeps context around it. No resampling:
        the result is a crop of native cubes.

        The window size comes from ``REVIEW_SETTINGS['soma_window']``. The largest
        requested step whose cubes can all be loaded is used, so the soma stays
        exactly centred; only when no centred step is fully covered does the window
        clamp inside the soma's own cube, which records ``window_centered: false``
        with the reason and the achieved centre offset. The soma's own cube is always
        required, so this never introduces zero-filled data.
        """
        print("\n[ACTION] Acquiring High-Res Soma Window (soma-centred)")
        center_um = triplet(center_um, 'center', nonnegative=True)
        block_w, block_h = BLOCK_SIZE_PIXELS[0], BLOCK_SIZE_PIXELS[1]
        settings = REVIEW_SETTINGS['soma_window']
        max_cubes = float(settings['max_cubes_xy'])
        if window_px:
            window_w, window_h = int(window_px[0]), int(window_px[1])
            if window_w <= 0 or window_h <= 0:
                raise ValueError("window_px must be positive")
            if window_w > max_cubes * block_w or window_h > max_cubes * block_h:
                raise ValueError(f"window_px must not exceed {max_cubes:g} cubes "
                                 f"({int(max_cubes * block_w)}x{int(max_cubes * block_h)} px)")
            candidates = [(window_w, window_h)]
        else:
            candidates = [(int(round(cubes * block_w)), int(round(cubes * block_h)))
                          for cubes in settings['cubes_xy_steps']]
        soma_px_x = center_um[0] / RESOLUTION_HIGH[0]
        soma_px_y = center_um[1] / RESOLUTION_HIGH[1]
        bz = int((center_um[2] / RESOLUTION_HIGH[2]) // BLOCK_SIZE_PIXELS[2])
        soma_block = (int(soma_px_x // block_w), int(soma_px_y // block_h), bz)

        requested, loaded, missing, sources, blocks, attempts = [], [], [], [], {}, []
        requested_seen, loaded_seen, missing_seen = set(), set(), set()
        chosen = None
        for window_w, window_h in candidates:
            x0 = max(0, int(round(soma_px_x - window_w / 2)))
            y0 = max(0, int(round(soma_px_y - window_h / 2)))
            # Contiguous cube range: a window wider than one cube can span three
            # columns, and a two-endpoint set would silently skip the middle one.
            wanted_x = list(range(x0 // block_w, (x0 + window_w - 1) // block_w + 1))
            wanted_y = list(range(y0 // block_h, (y0 + window_h - 1) // block_h + 1))
            indices = [(ix, iy, bz) for iy in wanted_y for ix in wanted_x]
            absent = []
            for index in indices:
                if index not in requested_seen:
                    requested_seen.add(index)
                    requested.append(list(index))
                    sources.append(f"{HTTP_HOST}/{HTTP_PATH}/{self.sample_id}/cube/"
                                   f"{index[2]}/{index[0]}_{index[1]}_{index[2]}.tif")
                if index in blocks:
                    continue
                block = self._download_http_block(*index) if min(index) >= 0 else None
                if block is None:
                    absent.append(index)
                    if index not in missing_seen:
                        missing_seen.add(index)
                        missing.append(list(index))
                    continue
                if np.shape(block) != tuple(reversed(BLOCK_SIZE_PIXELS)):
                    raise ValueError(f"Cube {index} has unexpected ZYX shape {np.shape(block)}")
                blocks[index] = block
                if index not in loaded_seen:
                    loaded_seen.add(index)
                    loaded.append(list(index))
            attempts.append({
                'window_px_xy': [window_w, window_h],
                'cubes': [list(index) for index in indices],
                'cubes_unavailable': [list(index) for index in absent],
                'covered': not absent,
            })
            if not absent and soma_block in blocks:
                chosen = (window_w, window_h, x0, y0, wanted_x, wanted_y)
                break

        if chosen is not None:
            window_wpx, window_hpx, x0, y0, wanted_x, wanted_y = chosen
            x_start = wanted_x[0] * block_w
            y_start = wanted_y[0] * block_h
            assembled = np.zeros((BLOCK_SIZE_PIXELS[2], len(wanted_y) * block_h,
                                  len(wanted_x) * block_w), dtype=np.uint16)
            for (ix, iy, _iz), block in blocks.items():
                if ix not in wanted_x or iy not in wanted_y:
                    continue
                ys = (iy - wanted_y[0]) * block_h
                xs = (ix - wanted_x[0]) * block_w
                assembled[:, ys:ys + block_h, xs:xs + block_w] = block
            crop_x, crop_y = x0 - x_start, y0 - y_start
            window = assembled[:, crop_y:crop_y + window_hpx, crop_x:crop_x + window_wpx]
            origin = [(x_start + crop_x) * RESOLUTION_HIGH[0],
                      (y_start + crop_y) * RESOLUTION_HIGH[1],
                      bz * BLOCK_SIZE_PIXELS[2] * RESOLUTION_HIGH[2]]
            centered = True
        else:
            # Last resort only: no centred step is fully covered. Keep the soma in
            # frame inside its own cube, never larger than that cube, and record the
            # achieved centre offset instead of pretending the soma is centred.
            if soma_block not in blocks:
                raise ValueError("Soma's central high-resolution cube is unavailable")
            block = blocks[soma_block]
            window_wpx = min(candidates[-1][0], block.shape[2])
            window_hpx = min(candidates[-1][1], block.shape[1])
            x_start = soma_block[0] * block_w
            y_start = soma_block[1] * block_h
            local_x = int(round(soma_px_x - x_start))
            local_y = int(round(soma_px_y - y_start))
            crop_x = min(max(0, local_x - window_wpx // 2), block.shape[2] - window_wpx)
            crop_y = min(max(0, local_y - window_hpx // 2), block.shape[1] - window_hpx)
            window = block[:, crop_y:crop_y + window_hpx, crop_x:crop_x + window_wpx]
            origin = [(x_start + crop_x) * RESOLUTION_HIGH[0],
                      (y_start + crop_y) * RESOLUTION_HIGH[1],
                      bz * BLOCK_SIZE_PIXELS[2] * RESOLUTION_HIGH[2]]
            x0, y0 = x_start + crop_x, y_start + crop_y
            centered = False
        if window.shape != (BLOCK_SIZE_PIXELS[2], window_hpx, window_wpx):
            raise ValueError(f"Unexpected soma window shape {window.shape}")

        metadata = self._acquisition_metadata(
            'HTTP native cubes (soma-centred window)', center_um, origin,
            RESOLUTION_HIGH, window.shape, requested, loaded, missing, sources)
        metadata['missing_errors'] = {
            '_'.join(map(str, index)): self._block_errors.get(tuple(index), 'unavailable or outside image bounds')
            for index in missing}
        metadata['window_centered'] = bool(centered)
        metadata['window_px_xy'] = [window_wpx, window_hpx]
        metadata['window_requested_px_xy'] = [candidates[0][0], candidates[0][1]]
        metadata['window_cubes_xy'] = round(window_wpx / block_w, 3)
        metadata['window_origin_xy_px'] = [x0, y0]
        metadata['soma_in_window_xy_px'] = [
            round(soma_px_x - origin[0] / RESOLUTION_HIGH[0], 3),
            round(soma_px_y - origin[1] / RESOLUTION_HIGH[1], 3)]
        metadata['window_centre_offset_px'] = [
            round(metadata['soma_in_window_xy_px'][0] - window_wpx / 2, 3),
            round(metadata['soma_in_window_xy_px'][1] - window_hpx / 2, 3)]
        metadata['soma_window_attempts'] = attempts
        metadata['depth_window'] = 'soma z-cube only (one cube deep); depth is not re-centred'
        if not centered:
            metadata['fallback_reason'] = (f"none of the soma-centred window steps "
                                           f"{[list(step) for step in candidates]} could be fully "
                                           "covered by the acquired cubes; window clamped inside "
                                           "the soma's own cube and the centre offset is recorded")
            print(f"  > [warn] soma-centred window unavailable ({len(missing)} neighbour cube(s) missing); "
                  "using the soma's own cube")
        self.last_high_res_metadata = metadata
        return window, origin, RESOLUTION_HIGH

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
        if self.low_res_correction is not None:
            self.last_low_res_metadata['image_correction'] = dict(self.low_res_correction)
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
            filename = f"{self.sample_id}_{neuron_id}_{safe_region}_{_output_name(suffix)}"
        else:
            filename = f"{self.sample_id}_{neuron_id}_{_output_name(suffix)}"
            
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

    # ------------------------------------------------------------------
    # Shared review-panel style (Soma / WideField / Sagittal context and trio)
    # ------------------------------------------------------------------
    # The legacy single-plot look is the reference for every context view:
    # black background, open soma square, white scale bar, three-line title.
    # One helper draws both the single PNG and each panel of the tri-plot, so
    # the combined review figure cannot drift away from the single views.
    #
    # Panel dict schema (one soma-centred display):
    #   display       2-D float array [0,1] for ``cmap``, or HxWx3 RGB in display order
    #   cmap          colormap name for 2-D displays, None for RGB
    #   resolution_um (x, y) physical size of one display pixel
    #   fov_um        (width, height) physical size of the displayed field
    #   depth_um      slab thickness along the panel normal
    #   scale_bar_um  physical length of the white scale bar
    #   markers       [(x_px, y_px, color, size_um or None, label)]
    #   edge_segments [Nx2 float arrays] thin red raw-SWC edge collection
    #   traces        [(Nx2 float array, color, linestyle, linewidth, alpha)]
    #   axis_labels   (xlabel, ylabel) to show indexed axes, else None (legacy off-axes)
    #   legend        [(kind, style, label)] with kind in {patch, line, marker}
    #   notes         [short caveat strings kept out of the image]
    #   role          short role label used by the tri-plot header
    PANEL_FIGSIZE = REVIEW_SETTINGS['figure']['panel_figsize']
    PANEL_DPI = REVIEW_SETTINGS['figure']['panel_dpi']
    TRIO_FIGSIZE = REVIEW_SETTINGS['figure']['review_figsize']
    TRIO_DPI = REVIEW_SETTINGS['figure']['review_dpi']
    TRACE_COLOR_A = '#de2d2c'
    TRACE_COLOR_B = '#1e73de'
    # Review titles: legacy filename suffixes map to the display name and the
    # source dataset the panel is actually drawn from (required in the title).
    TITLE_SUFFIXES = {
        'Soma': ('Soma Plot', '0.65µm_raw'),
        'Context': ('Context Plot', '5µm_localserver'),
        'Sagittal': ('Sagittal Context Plot', 'DS00 N4'),
    }

    def _title_text(self, neuron_id, suffix, soma_region, soma_coords, fov_um, depth_um,
                    hemisphere=None):
        """Legacy three-line panel title; shared by single plots and the trio."""
        name = _output_name(suffix)
        display, dataset = self.TITLE_SUFFIXES.get(name, (name, None))
        head = f"{self.sample_id} | {neuron_id} | {display}"
        if dataset:
            head += f" | src_dataset: {dataset}"
        lines = [head]
        context = []
        if soma_region:
            context.append(f"Region: {soma_region}")
        if soma_coords is not None:
            context.append(f"XYZ: ({int(soma_coords[0])}, {int(soma_coords[1])}, "
                           f"{int(soma_coords[2])})")
        if hemisphere:
            context.append(f"{hemisphere}_hemisphere")
        if context:
            lines.append(" | ".join(context))
        lines.append(f"FOV: {fov_um[0]:.0f}x{fov_um[1]:.0f} µm | Depth: {depth_um:.0f} µm")
        return "\n".join(lines)

    def _draw_target_marker(self, ax, x, y, color=None, size_um=None,
                            label="Target Soma", min_px=None, crosshair_px=None,
                            label_fontsize=None, label_offset_px=None,
                            resolution_um=None):
        """Draw the shared target marker: one cyan/turquoise open box with a cross.

        Every panel type (Soma, Context, Sagittal) and the legacy single-plot path
        use this helper, so the three review panels cannot drift apart. The box and
        the cross are both centred on the exact mapped point; only the drawn size is
        enlarged for visibility, so the intersection is the true position.
        """
        style = REVIEW_SETTINGS['marker']
        color = color or style['color']
        res_x = float((resolution_um or (1.0, 1.0))[0])
        floor_px = float(style['min_px'] if min_px is None else min_px)
        size_px = floor_px if not size_um else max(float(size_um) / res_x, floor_px)
        ax.scatter(x, y, s=(size_px * 0.8) ** 2, marker='s', facecolors='none',
                   edgecolors=color, linewidth=0.9, zorder=10, label=label)
        arms_px = float(style['crosshair_px'] if crosshair_px is None else crosshair_px)
        if arms_px > 0:
            ax.plot([x - arms_px, x + arms_px], [y, y], color=color,
                    linewidth=1.1, alpha=0.9, zorder=11, solid_capstyle='butt')
            ax.plot([x, x], [y - arms_px, y + arms_px], color=color,
                    linewidth=1.1, alpha=0.9, zorder=11, solid_capstyle='butt')
        if label:
            offset_px = float(style['label_offset_px'] if label_offset_px is None
                              else label_offset_px)
            ax.text(x + offset_px, y + offset_px, label, color=color,
                    fontsize=(float(style['label_fontsize']) if label_fontsize is None
                              else float(label_fontsize)),
                    fontweight='bold', zorder=12)
        return {'color': color, 'size_px': size_px, 'crosshair_px': arms_px}

    def _draw_panel(self, ax, panel, marker_min_px=8.0):
        """Draw one styled panel; single plots and the tri-plot share this path."""
        display = panel.get('display')
        if display is None:
            raise ValueError("panel display is missing")
        h, w = display.shape[:2]
        if display.ndim == 2:
            ax.imshow(display, cmap=panel.get('cmap') or 'gray', origin='upper')
        else:
            ax.imshow(display, origin='upper')
        if panel.get('edge_segments'):
            ax.add_collection(LineCollection(panel['edge_segments'], colors='red',
                                             linewidths=.5, alpha=.5))
        for points, color, linestyle, linewidth, alpha in panel.get('traces') or []:
            ax.plot(points[:, 0], points[:, 1], color=color, linestyle=linestyle,
                    linewidth=linewidth, alpha=alpha, solid_capstyle='round')
        res_x = float(panel.get('resolution_um', (1.0, 1.0))[0])
        for marker in panel.get('markers') or []:
            x, y, color, size_um, label = marker[0], marker[1], marker[2], marker[3], marker[4]
            # Optional per-marker display options (see the schema note above):
            # min_px, crosshair_px and label_fontsize enlarge the target marker without
            # moving it; the shared helper always draws the box and the cross on the
            # exact position, so a panel cannot ship a marker without a cross.
            options = marker[5] if len(marker) > 5 and isinstance(marker[5], dict) else {}
            self._draw_target_marker(
                ax, x, y, color=color, size_um=size_um, label=label,
                min_px=options.get('min_px', marker_min_px),
                crosshair_px=options.get('crosshair_px'),
                label_fontsize=options.get('label_fontsize'),
                label_offset_px=options.get('label_offset_px'),
                resolution_um=panel.get('resolution_um', (1.0, 1.0)),
            )
        bar_um = panel.get('scale_bar_um')
        if bar_um is None:
            # Auto bar (used by the sagittal panel so a full-plane and a cropped
            # view both stay readable).
            bar_um = _auto_scale_bar_um(panel['fov_um'])
        bar_um = float(bar_um)
        if bar_um > 0:
            bar_px = bar_um / res_x
            bx, by = w - bar_px - 50, h - 50
            ax.plot([bx, bx + bar_px], [by, by], color='white', linewidth=3)
            bar_label = f"{bar_um:.0f} µm" if bar_um < 1000 else f"{bar_um / 1000:g} mm"
            ax.text(bx + bar_px / 2, by - 20, bar_label, color='white', ha='center',
                    fontweight='bold')
        if panel.get('axis_labels'):
            ax.set_xlabel(panel['axis_labels'][0], color='white', fontsize=9)
            ax.set_ylabel(panel['axis_labels'][1], color='white', fontsize=9)
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('white')
                spine.set_visible(False)
        else:
            ax.axis('off')

    def _save_panel_plot(self, panel, neuron_id, suffix, soma_region=None,
                         soma_coords=None, output_dir=None):
        """Write one legacy-styled panel PNG and return the panel."""
        target_dir = output_dir if output_dir else self.output_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        title = self._title_text(neuron_id, suffix, soma_region, soma_coords,
                                 panel['fov_um'], panel['depth_um'],
                                 hemisphere=panel.get('hemisphere'))
        fig, ax = plt.subplots(figsize=self.PANEL_FIGSIZE, facecolor='black')
        ax.set_facecolor('black')
        self._draw_panel(ax, panel)
        ax.set_title(title, color='white', fontsize=12)
        safe_region = self._sanitize_filename(soma_region) if soma_region else ""
        region_part = f"_{safe_region}" if safe_region else ""
        save_path = os.path.join(target_dir, f"{self.sample_id}_{neuron_id}{region_part}"
                                             f"_{_stamp()}_{_output_name(suffix)}_Plot.png")
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='black',
                    dpi=self.PANEL_DPI)
        plt.close(fig)
        print(f"  > Plot saved: {save_path}")
        return dict(panel, title=title, output_png=save_path)

    def plot_soma_block(self, volume_3d, origin, resolution, soma_coords, neuron_id, 
                        suffix="SomaBlock", soma_region=None, output_dir=None):
        """
        Plots Grayscale Anatomy (High Res) using MIP.
        Includes soma region in filename and title.
        """
        print(f"  > Generating Grayscale Plot ({suffix})...")
        scale_bar_um = REVIEW_SETTINGS['scale_bar_um']['soma']
        
        # --- FIXED: Use MIP (Maximum Intensity Projection) instead of Middle Slice ---
        img = np.max(volume_3d, axis=0)
        
        # Local Soma Pixel
        sx = (soma_coords[0] - origin[0]) / resolution[0]
        sy = (soma_coords[1] - origin[1]) / resolution[1]
        
        # Contrast (Gamma 0.5)
        img_norm = percentile_normalize(img, 0.5, 99.5)
        
        img_final = np.power(img_norm, 0.5)
        
        output_png = self._save_plot(
            img_final, sx, sy, resolution, neuron_id, suffix,
            volume_3d.shape[0], cmap='gray', marker_color=TARGET_MARKER_COLOR,
            soma_region=soma_region, soma_coords=soma_coords,
            output_dir=output_dir, scale_bar_um=scale_bar_um)
        return {
            'display': img_final, 'cmap': 'gray',
            'resolution_um': (float(resolution[0]), float(resolution[1])),
            'fov_um': (img_final.shape[1] * float(resolution[0]),
                       img_final.shape[0] * float(resolution[1])),
            'depth_um': float(volume_3d.shape[0]) * float(resolution[2]),
            'scale_bar_um': float(scale_bar_um),
            'markers': [(sx, sy, TARGET_MARKER_COLOR,
                         20.0 if resolution[0] < 1.0 else 200.0, 'Target Soma')],
            'edge_segments': [], 'traces': [],
            'axis_labels': None,
            'legend': [('patch', 'gray', 'soma grayscale MIP · 0.65 µm'),
                       ('marker', TARGET_MARKER_COLOR,
                        'target soma marker (box + cross, not to scale)')],
            'notes': ['Legacy SomaBlock gamma 0.5; no SWC overlay.'],
            'role': 'Soma (high-res MIP · 0.65 µm)',
            'suffix': suffix,
            'output_png': output_png,
            'title': self._title_text(
                neuron_id, suffix, soma_region, soma_coords,
                (img_final.shape[1] * float(resolution[0]),
                 img_final.shape[0] * float(resolution[1])),
                float(volume_3d.shape[0]) * float(resolution[2])),
        }

    def plot_widefield_context(self, volume_3d, origin, resolution, soma_coords, neuron_id, 
                             suffix="WideField", manual_threshold=None, bg_intensity=None,
                             swc_tree=None, soma_region=None, output_dir=None):
        """
        Plots Green Intensity on Dark Background + SWC Overlay using MIP.
        Includes soma region in filename and title.
        """
        print(f"  > Generating Composite Plot ({suffix})...")
        bg_intensity, manual_threshold = resolve_widefield_parameters(
            getattr(self, "sample_id", None), bg_intensity, manual_threshold)
        scale_bar_um = REVIEW_SETTINGS['scale_bar_um']['context']
        
        # Uses MIP by default
        mip = np.max(volume_3d, axis=0).astype(float)
        h, w = mip.shape
        
        # 1. Base Gray Layer
        d_max = np.max(mip)
        if d_max < manual_threshold: manual_threshold = d_max * 0.5
        # Legacy order: clip the p1-p99.5 normalized base, then apply gain.
        # Keep the finite/constant-image handling in percentile_normalize.
        norm_base = percentile_normalize(mip) * bg_intensity
        
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
        
        segments = []
        if swc_tree:
            segments = self._overlay_swc(ax, swc_tree, origin, resolution, h, w,
                                         volume_3d.shape[0])

        sx = (soma_coords[0] - origin[0]) / resolution[0]
        sy = (soma_coords[1] - origin[1]) / resolution[1]
        
        output_png = self._finalize_plot(
            fig, ax, sx, sy, resolution, neuron_id, suffix, volume_3d.shape[0],
            marker_color=TARGET_MARKER_COLOR, soma_region=soma_region,
            soma_coords=soma_coords, output_dir=output_dir,
            scale_bar_um=scale_bar_um)
        return {
            'display': rgb, 'cmap': None,
            'resolution_um': (float(resolution[0]), float(resolution[1])),
            'fov_um': (w * float(resolution[0]), h * float(resolution[1])),
            'depth_um': float(volume_3d.shape[0]) * float(resolution[2]),
            'scale_bar_um': float(scale_bar_um),
            'markers': [(sx, sy, TARGET_MARKER_COLOR,
                         20.0 if resolution[0] < 1.0 else 200.0, 'Target Soma')],
            'edge_segments': segments, 'traces': [],
            'axis_labels': None,
            'legend': [('patch', 'green', 'widefield green channel · 5 µm'),
                       ('line', 'red', 'raw SWC edges (all depths)'),
                       ('marker', TARGET_MARKER_COLOR,
                        'target soma marker (box + cross, not to scale)')],
            'notes': [f'Legacy WideField display: gain {bg_intensity:g} / threshold {manual_threshold:g}, MIP.'],
            'role': 'WideField context (5 µm)',
            'suffix': suffix,
            'output_png': output_png,
            'title': self._title_text(
                neuron_id, suffix, soma_region, soma_coords,
                (w * float(resolution[0]), h * float(resolution[1])),
                float(volume_3d.shape[0]) * float(resolution[2])),
        }

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
        return segments

    def _save_plot(self, img_data, sx, sy, resolution, neuron_id, suffix, z_slices, 
                   cmap, marker_color, soma_region=None, soma_coords=None, output_dir=None, scale_bar_um=30):
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.imshow(img_data, cmap=cmap, origin='upper')
        return self._finalize_plot(fig, ax, sx, sy, resolution, neuron_id, suffix,
                                   z_slices, marker_color, soma_region=soma_region,
                                   soma_coords=soma_coords, output_dir=output_dir,
                                   scale_bar_um=scale_bar_um)

    def _finalize_plot(self, fig, ax, sx, sy, resolution, neuron_id, suffix, z_slices, 
                       marker_color, soma_region=None, soma_coords=None, output_dir=None, scale_bar_um=30):
        h, w = ax.images[0].get_size()
        
        target_dir = output_dir if output_dir else self.output_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        marker_size_um = 20 if resolution[0] < 1.0 else 200
        # Same target marker as the panel path: cyan/turquoise open box plus a cross
        # through the exact point, so the single PNG and the review figure agree.
        self._draw_target_marker(ax, sx, sy, color=marker_color or TARGET_MARKER_COLOR,
                                 size_um=marker_size_um, label='Target Soma',
                                 resolution_um=resolution)

        bar_px = scale_bar_um / resolution[0]
        bx = w - bar_px - 50; by = h - 50
        ax.plot([bx, bx+bar_px], [by, by], color='white', linewidth=3)
        ax.text(bx+bar_px/2, by-20, f"{scale_bar_um} µm", color='white', ha='center', fontweight='bold')
        
        # Build title with soma region and coordinates (shared with the panel style)
        title = self._title_text(neuron_id, suffix, soma_region, soma_coords,
                                 (w * resolution[0], h * resolution[1]),
                                 z_slices * resolution[2])

        ax.set_title(title, color='white', fontsize=12)
        ax.axis('off')
        
        # Build filename with soma region (coordinates NOT in filename)
        # Sanitize region name to prevent path issues
        review_name = _output_name(suffix)
        stamp = _stamp()
        if soma_region:
            safe_region = self._sanitize_filename(soma_region)
            plot_name = (f"{self.sample_id}_{neuron_id}_{safe_region}_{stamp}"
                         f"_{review_name}_Plot.png")
        else:
            plot_name = f"{self.sample_id}_{neuron_id}_{stamp}_{review_name}_Plot.png"
            
        save_path = os.path.join(target_dir, plot_name)
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='black', dpi=600)
        print(f"  > Plot saved: {save_path}")
        plt.close(fig)
        return save_path

    def plot_sagittal_context(self, panel, neuron_id, suffix="Sagittal", soma_region=None,
                              soma_coords=None, output_dir=None):
        """
        Plot one prepared processed-DS sagittal context panel in the legacy style.

        ``panel`` follows the shared panel schema above and is produced by
        ``prepare_sagittal_panel`` in this module. Rendering only: this
        method never resamples or modifies the DS image or the raw SWC.
        Returns the panel dict, including ``title`` and ``output_png``.
        """
        print(f"  > Generating Sagittal Context Plot ({suffix})...")
        return self._save_panel_plot(panel, neuron_id, suffix, soma_region=soma_region,
                                     soma_coords=soma_coords, output_dir=output_dir)

    def plot_tri_plot(self, panels, neuron_id, output_dir=None, suffix="TriPlot",
                      soma_region=None, soma_coords=None, dpi=None, header=None,
                      caveats=None, legend=False):
        """
        Draw the per-neuron review figure: the panels side by side, one title each.

        ``panels`` is an ordered sequence of panel dicts (Soma, WideField context,
        Sagittal context). Each panel keeps its own title, markers, traces and
        scale bar so the three views no longer have to be opened separately.
        ``legend=True`` adds one shared legend row under the panels.
        """
        panels = [dict(panel) for panel in panels if panel]
        if not panels:
            raise ValueError("plot_tri_plot needs at least one panel")
        target_dir = output_dir if output_dir else self.output_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        figsize = self.TRIO_FIGSIZE if legend else (self.TRIO_FIGSIZE[0], 10.2)
        fig = plt.figure(figsize=figsize, facecolor='black')
        if legend:
            grid = fig.add_gridspec(2, len(panels), height_ratios=(9.2, 1.5),
                                    hspace=0.02, wspace=0.05)
        else:
            grid = fig.add_gridspec(1, len(panels), wspace=0.05)
        for index, panel in enumerate(panels):
            ax = fig.add_subplot(grid[0, index])
            ax.set_facecolor('black')
            self._draw_panel(ax, panel)
            title = panel.get('title') or self._title_text(
                neuron_id, panel.get('suffix', suffix), soma_region, soma_coords,
                panel['fov_um'], panel['depth_um'])
            ax.set_title(title, color='white', fontsize=11)
        if legend:
            legend_ax = fig.add_subplot(grid[1, :])
            legend_ax.axis('off')
            self._draw_shared_legend(legend_ax, panels, neuron_id, header=header,
                                     caveats=caveats)
        safe_region = self._sanitize_filename(soma_region) if soma_region else ""
        region_part = f"_{safe_region}" if safe_region else ""
        save_path = os.path.join(
            target_dir,
            f"{self.sample_id}_{neuron_id}{region_part}_{_stamp()}_{suffix}.png")
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.15, facecolor='black',
                    dpi=dpi or self.TRIO_DPI)
        plt.close(fig)
        print(f"  > Tri-plot saved: {save_path}")
        return save_path

    def _draw_shared_legend(self, ax, panels, neuron_id, header=None, caveats=None):
        """One compact legend for the whole review figure (AGENTS QC style rule)."""
        from matplotlib.lines import Line2D

        entries = []
        for panel in panels:
            for entry in panel.get('legend') or []:
                # Collapse only exact repeats of one displayed role (kind + label),
                # so different marker colours for the same role stay distinguishable.
                if (entry[0], entry[2]) not in [(seen[0], seen[2]) for seen in entries]:
                    entries.append(entry)
        roles = [panel.get('role', '') for panel in panels if panel.get('role')]
        notes = list(panels[0].get('notes') or [])
        for panel in panels[1:]:
            notes.extend(note for note in panel.get('notes') or [] if note not in notes)
        notes.extend(caveats or [])
        ax.text(0.0, 1.02, header or f"{self.sample_id} | {neuron_id} | review panels: "
                + " · ".join(f"{i + 1} {role}" for i, role in enumerate(roles)),
                color='white', fontsize=12, fontweight='bold', va='bottom', ha='left',
                transform=ax.transAxes)
        handles, labels = [], []
        for entry in entries:
            kind, style, label = entry[0], entry[1], entry[2]
            linestyle = entry[3] if len(entry) > 3 else 'solid'
            if kind == 'patch':
                handles.append(Rectangle((0, 0), 1, 1, facecolor=style, edgecolor='0.8'))
            elif kind == 'line':
                handles.append(Line2D([0], [0], color=style, linewidth=2.5,
                                      linestyle=linestyle))
            else:
                handles.append(Line2D([0], [0], color=style, marker='s', linestyle='none',
                                      markerfacecolor='none', markersize=9,
                                      markeredgewidth=1.0))
            labels.append(label)
        if handles:
            legend = ax.legend(handles, labels, loc='upper left', frameon=False,
                               ncol=min(len(labels), 5), fontsize=10, labelcolor='white',
                               handlelength=1.4, columnspacing=1.8, handletextpad=0.6,
                               bbox_to_anchor=(0.0, 0.86), borderaxespad=0.0)
            for text in legend.get_texts():
                text.set_color('white')
        ax.text(0.0, 0.0, "  ".join(notes), color='#dddddd', fontsize=9, va='bottom',
                ha='left', transform=ax.transAxes)

    def render_sagittal_context_from_ds(self, neuron_id, soma_xyz, sample_id=None,
                                        output_dir=None, soma_region=None,
                                        soma_coords=None, **panel_kwargs):
        """
        End-to-end helper: resolve the registered processed-DS sagittal panel for
        one neuron, plot it in the legacy style and return panel plus provenance.

        Requires a sample registered in ``SAMPLE_PATHS`` in this module.
        """
        prepared = prepare_sagittal_panel(sample_id or self.sample_id, neuron_id,
                                          soma_xyz, **panel_kwargs)
        panel = self.plot_sagittal_context(
            prepared['panel'], neuron_id, soma_region=soma_region,
            soma_coords=soma_coords or list(soma_xyz), output_dir=output_dir)
        return {'panel': panel, 'provenance': prepared['provenance']}

# ===== Integrated processed-DS sagittal renderer =====

SHARE_ROOT = Path(r"\\10.102.8.200\microscopy_data\fMOST\CH00DS_datasets")
DATASET_MANIFEST = SHARE_ROOT / "manifest.json"
# Registered selected-cohort samples: monkey id and the cached raw-SWC folder.
# The processed-DS source itself is always resolved from the live dataset manifest
# by sample id, so a row must exist there (and its local copy must hash-match).
SWC_CACHE_ROOT = Path("group_analysis/visual_review_20261002/swc_raw")
BUILTIN_SAMPLE_PATHS = {
    sample: {"monkey": monkey, "swc_rel": SWC_CACHE_ROOT / sample}
    for sample, monkey in (
        ("250432", "747"), ("251637", "936"), ("252383", "605"), ("252384", "948"),
        ("252385", "331"), ("252527", "945"), ("252714", "631"), ("252718", "900"),
        ("252790", "797"),
    )
}


def _sample_paths_from_config() -> dict[str, dict[str, Any]]:
    """Read the per-sample links from Visual_toolkit/visual_toolkit_config.yaml.

    Falls back to the built-in table when the config is absent or unreadable, so older
    checkouts keep working. A sample may carry ``swc_rel = None`` when no SWC cache is
    configured; the sagittal panel then renders image-only.
    """
    try:
        from .dataset_config import load as _load_datasets
    except Exception:
        try:
            from dataset_config import load as _load_datasets
        except Exception:
            return {}
    try:
        config = _load_datasets()
    except Exception as exc:  # a broken config must not break plotting
        print(f"[native_ds_sagittal] dataset config ignored: {exc}")
        return {}
    paths: dict[str, dict[str, Any]] = {}
    for sample_id, entry in (config.get("samples") or {}).items():
        paths[sample_id] = {
            "monkey": entry.get("monkey") or "",
            "swc_rel": entry.get("swc_cache"),
            "ds00_processed": entry.get("ds00_processed"),
            "ds00_sha256": entry.get("ds00_sha256"),
        }
    return paths


SAMPLE_PATHS = _sample_paths_from_config() or BUILTIN_SAMPLE_PATHS
# Only 252527 has an independently verified converted grid; other samples are
# accepted on any 3-D LPS-mm sform grid and record their observed geometry.
EXPECTED_GEOMETRY = {
    "252527": {"shape_ijk": (618, 643, 493), "spacing_mm_ijk": (0.1, 0.09, 0.1),
               "affine_diag": (-0.1, -0.09, 0.1)},
}

# New output indices after tools/nrrd_to_orient_lps.py's permutation:
# i = old i; j = old k; k = reversed old j. NRRD sizes (618,493,643)
# become NIfTI indices (618,643,493).
MAPPINGS = {
    "A": {
        "name": "A_header_RAS_xyz_to_raw_ijk",
        "basis": "SWC x,y,z -> NRRD i,j,k, as implied by the raw RAS header",
        "short_label": "candidate trace A · header RAS x,y,z",
        "axes": (0, 1, 2),
        "equation_output_ijk": "[x_um/100, z_um/90, 492-y_um/100]",
    },
    "B": {
        "name": "B_converter_declared_RSP_xyz_to_raw_ikj",
        "basis": "SWC x,z,y -> NRRD i,j,k, as implied by the converter's documented old-axis convention",
        "short_label": "candidate trace B · converter x,z,y",
        "axes": (0, 2, 1),
        "equation_output_ijk": "[x_um/100, y_um/90, 492-z_um/100]",
    },
}

# Styling shared with the Visual_toolkit panel renderer.
TRACE_COLORS = {"A": "#de2d2c", "B": "#1e73de"}


def _marker_style() -> dict[str, Any]:
    """Target-marker style from the same REVIEW_SETTINGS used by every panel."""
    return REVIEW_SETTINGS["marker"]


def resolve_swc_path(sample_id: str, neuron_id: str, config: dict | None = None,
                     swc_path: str | Path | None = None) -> tuple[Path | None, str]:
    """Resolve the raw SWC: registered cache first, then the legacy raw cache.

    ``resource/swc_raw/<sample>/<neuron>.swc`` is the legacy download cache and
    holds the raw portal bytes (verified for 251637/003: its root equals the portal
    soma coordinate). It is used only when the configured review cache has no file,
    and the returned source label is recorded in the panel provenance.
    """
    repo = Path(__file__).resolve().parents[2]
    if swc_path is not None:
        candidate = Path(swc_path)
        return (candidate, "explicit") if candidate.is_file() else (None, "explicit")
    config = config if config is not None else (SAMPLE_PATHS.get(sample_id) or {})
    if config.get("swc_rel"):
        candidate = repo / Path(config["swc_rel"]) / f"{neuron_id}.swc"
        if candidate.is_file():
            return candidate, "registered_cache"
    legacy = repo / "resource" / "swc_raw" / sample_id / f"{neuron_id}.swc"
    if legacy.is_file():
        return legacy, "legacy_raw_cache"
    return None, "missing"


def _sha256(path: Path, block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _source_record(sample_id: str) -> dict[str, Any]:
    if sample_id not in SAMPLE_PATHS:
        raise ValueError(f"No verified processed-DS source is configured for sample {sample_id}")
    config = SAMPLE_PATHS[sample_id]
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    rows = [r for r in manifest.get("selected_preprocessed_outputs", []) if str(r.get("sample")) == sample_id]
    if len(rows) != 1:
        raise ValueError(f"Live dataset manifest does not uniquely select a preprocessed DS candidate for {sample_id}")
    row = rows[0]
    expected_rel = str(row.get("relative_path", "")).replace("\\", "/")
    server_path = SHARE_ROOT / Path(expected_rel)
    if not server_path.is_file():
        raise FileNotFoundError(f"Live manifest-selected preprocessed DS volume is unavailable: {server_path}")
    source_path = Path(str(row.get("repository_candidate_path", "")))
    if not source_path.is_file():
        raise FileNotFoundError(f"Manifest-bound local copy of selected DS candidate is unavailable: {source_path}")
    stat_before = source_path.stat()
    got_hash = _sha256(source_path)
    stat_after = source_path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise RuntimeError("Preprocessed DS source changed while being read")
    if row.get("sha256", "").lower() != got_hash:
        raise RuntimeError(f"Selected preprocessed DS SHA256 differs from the live manifest: {got_hash}")
    return {
        "path": str(source_path),
        "server_path": str(server_path),
        "relative_path": expected_rel,
        "sha256": got_hash,
        "bytes": stat_after.st_size,
        "mtime_ns_before_after_equal": True,
        "server_file_present": True,
        "manifest_path": str(DATASET_MANIFEST),
        "manifest_sha256": _sha256(DATASET_MANIFEST),
        "manifest_run_id": manifest.get("run_id"),
        "manifest_time": manifest.get("time"),
        "processed_at": row.get("processed_at"),
        "source_historical_path": row.get("source_historical_path"),
        "source_sha256": row.get("source_sha256"),
        "processing_parameters": row.get("processing_parameters"),
        "software_versions": row.get("software_versions"),
        "declared_space": row.get("space"),
        "scientific_status": row.get("status"),
        "source_acquisition_date": "unknown",
    }


def _nifti_metadata(header: bytes, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(header) < 352 or struct.unpack_from("<i", header, 0)[0] != 348:
        raise ValueError("Expected a little-endian NIfTI-1 single-file image")
    dim = struct.unpack_from("<8h", header, 40)
    ndim = dim[0]
    if ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI, got dimensions {dim}")
    shape = tuple(int(x) for x in dim[1:4])
    datatype = struct.unpack_from("<h", header, 70)[0]
    bitpix = struct.unpack_from("<h", header, 72)[0]
    dtype = {16: "<f4", 512: "<u2"}.get(datatype)
    if dtype is None or bitpix != np.dtype(dtype).itemsize * 8:
        raise ValueError(f"Unsupported DS datatype/bit depth {datatype}/{bitpix}")
    vox_offset = int(round(struct.unpack_from("<f", header, 108)[0]))
    slope, intercept = struct.unpack_from("<2f", header, 112)
    sform_code = struct.unpack_from("<h", header, 254)[0]
    if sform_code <= 0:
        raise ValueError("NIfTI sform is missing; cannot place a physical sagittal plane")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray([struct.unpack_from("<4f", header, at) for at in (280, 296, 312)])
    units = header[123] & 0x07
    if units != 2:
        raise ValueError(f"Expected NIfTI spatial units mm (unit code 2), got {units}")
    pixdim = struct.unpack_from("<8f", header, 76)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if expected is not None:
        # Independently verified grid: bind the renderer to this exact geometry.
        if shape != tuple(expected["shape_ijk"]) or not np.allclose(
                spacing, expected["spacing_mm_ijk"], atol=1e-6):
            raise ValueError(f"Unexpected CH00DS geometry: shape={shape}, spacing={spacing.tolist()}")
        if not np.allclose(affine[:3, :3], np.diag(expected["affine_diag"]), atol=1e-6):
            raise ValueError(f"Unexpected CH00DS axis orientation: {affine.tolist()}")
    else:
        # Unverified sample: require a cardinal, non-degenerate LPS-mm grid and
        # record whatever geometry is observed instead of asserting constants.
        if not np.all(np.isfinite(affine)) or np.any(spacing <= 0):
            raise ValueError(f"Non-finite or degenerate DS affine: {affine.tolist()}")
        if not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])), atol=1e-6):
            raise ValueError(f"DS image is not axis-aligned; unsupported for this renderer: {affine.tolist()}")
    return {
        "shape_ijk": shape,
        "spacing_mm_ijk": spacing.tolist(),
        "affine_mm": affine.tolist(),
        "sform_code": sform_code,
        "xyzt_units_byte": header[123],
        "pixdim": pixdim,
        "vox_offset": vox_offset,
        "scl_slope": float(slope),
        "scl_inter": float(intercept),
        "dtype": np.dtype(dtype).name,
        "coordinate_axes": "NIfTI i=nominal left-right, j=nominal anterior-posterior, k=nominal inferior-superior per converter; anatomical orientation unvalidated",
    }


def _read_sagittal_planes(path: Path, indices_i: tuple[int, ...],
                          expected: dict[str, Any] | None = None
                          ) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as stream:
        header = stream.read(352)
        meta = _nifti_metadata(header, expected)
        shape = meta["shape_ijk"]
        indices_i = tuple(sorted(set(int(i) for i in indices_i)))
        if not indices_i or any(not 0 <= i < shape[0] for i in indices_i):
            raise ValueError(f"Sagittal slice indices are invalid for shape {shape}: {indices_i}")
        if meta["vox_offset"] < 352:
            raise ValueError(f"Invalid NIfTI vox_offset {meta['vox_offset']}")
        skip = meta["vox_offset"] - 352
        if skip:
            stream.read(skip)
        itemsize = np.dtype("<f4" if meta["dtype"] == "float32" else "<u2").itemsize
        row_bytes = shape[0] * itemsize
        total_rows = shape[1] * shape[2]
        flat_planes = {i: np.empty(total_rows, dtype=np.float32) for i in indices_i}
        dtype = np.dtype("<f4" if itemsize == 4 else "<u2")
        row_start = 0
        carry = b""
        while row_start < total_rows:
            block = stream.read(16 * 1024 * 1024)
            if not block:
                break
            block = carry + block
            complete = len(block) // row_bytes * row_bytes
            carry = block[complete:]
            if complete == 0:
                continue
            rows = np.frombuffer(block[:complete], dtype=dtype).reshape(-1, shape[0])
            count = rows.shape[0]
            for index_i in indices_i:
                flat_planes[index_i][row_start:row_start + count] = rows[:, index_i]
            row_start += count
        if row_start != total_rows:
            raise ValueError(f"NIfTI voxel payload ended at row {row_start}/{total_rows}")
        slope, intercept = meta["scl_slope"], meta["scl_inter"]
        planes = {i: flat.reshape((shape[1], shape[2]), order="F") for i, flat in flat_planes.items()}
        if slope not in (0.0, 1.0) or intercept != 0.0:
            planes = {i: plane * (slope if slope != 0 else 1.0) + intercept for i, plane in planes.items()}
        return planes, meta


class SagittalDSContext:
    """One-time verified DS source and a batch of soma-i planes for one sample."""

    def __init__(self, sample_id: str, soma_xyz_um: list[tuple[float, float, float]]):
        self.sample_id = str(sample_id)
        self.record = _source_record(self.sample_id)
        self.path = Path(self.record["path"])
        requested = tuple(sorted({int(round(float(xyz[0]) / 100.0)) for xyz in soma_xyz_um}))
        self.planes, self.geometry = _read_sagittal_planes(
            self.path, requested, EXPECTED_GEOMETRY.get(self.sample_id))

    def plane_for_soma(self, soma_xyz_um: tuple[float, float, float]) -> tuple[np.ndarray, int]:
        index_i = int(round(float(soma_xyz_um[0]) / 100.0))
        try:
            return self.planes[index_i], index_i
        except KeyError as exc:
            raise KeyError(f"No preloaded sagittal i={index_i} for sample {self.sample_id}; prepare all requested soma positions first") from exc


def prepare_sagittal_source(sample_id: str, soma_xyz_um: list[tuple[float, float, float]]) -> SagittalDSContext:
    """Verify manifest/source once and extract all requested planes in one sequential pass."""
    if not soma_xyz_um:
        raise ValueError("At least one soma coordinate is required")
    return SagittalDSContext(sample_id, soma_xyz_um)


def _read_swc(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            node = {
                "id": int(parts[0]), "type": int(parts[1]),
                "xyz_um": tuple(float(v) for v in parts[2:5]),
                "radius_um": float(parts[5]), "parent": int(parts[6]),
            }
            nodes.append(node)
            by_id[node["id"]] = node
    if not nodes:
        raise ValueError(f"No SWC nodes found in {path}")
    roots = [n for n in nodes if n["parent"] < 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one raw SWC root/soma match, found {len(roots)}")
    return nodes, by_id


def _out_ijk(xyz_um: tuple[float, float, float], mapping: str,
             geometry: dict[str, Any] | None = None) -> tuple[float, float, float]:
    """Map raw-SWC micrometres onto converted DS indices for one candidate mapping.

    The divisors and the k offset come from the sample's own converted grid
    (``spacing_mm_ijk`` x 1000 µm and ``shape_ijk[2] - 1``), which reproduces the
    originally hardcoded 252527 constants exactly. With no geometry the historical
    252527 constants are used, so the earlier PIL renderer stays reproducible.
    """
    x, y, z = xyz_um
    if geometry is None:
        spacing_um = (100.0, 90.0, 100.0)
        k_offset = 492.0
    else:
        spacing_um = tuple(float(value) * 1000.0 for value in geometry["spacing_mm_ijk"])
        k_offset = float(geometry["shape_ijk"][2]) - 1.0
    dx, dy, dz = spacing_um
    if mapping == "A":
        return x / dx, z / dy, k_offset - y / dz
    if mapping == "B":
        return x / dx, y / dy, k_offset - z / dz
    raise ValueError(mapping)


def _hemisphere_from_index(soma_xyz_um: tuple[float, float, float],
                           geometry: dict[str, Any]) -> str | None:
    """Volume-convention hemisphere of the plane that shows this soma.

    Larger ``i`` lies on the left side of the converted CH00DS volume; this was
    checked against the active workbook's own ``Soma_Side`` for 252527 (R neurons
    at i≈163-175, L neurons at i≈437-447, volume midline 308.5). It describes the
    volume convention, not a visually confirmed anatomical laterality, and a
    declared side from the workbook is carried alongside it for cross-checking.
    """
    spacing_i_um = float(geometry["spacing_mm_ijk"][0]) * 1000.0
    center_i = (float(geometry["shape_ijk"][0]) - 1.0) / 2.0
    index_i = float(soma_xyz_um[0]) / spacing_i_um
    if abs(index_i - center_i) < 1e-6:
        return None
    return "Left" if index_i > center_i else "Right"


def _clip_segment_i(a: tuple[float, float, float], b: tuple[float, float, float], lo: float, hi: float):
    ai, aj, ak = a
    bi, bj, bk = b
    delta = bi - ai
    if abs(delta) < 1e-12:
        if lo <= ai <= hi:
            return (aj, ak), (bj, bk)
        return None
    t0, t1 = (lo - ai) / delta, (hi - ai) / delta
    enter, leave = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
    if enter > leave:
        return None
    return (aj + enter * (bj - aj), ak + enter * (bk - ak)), (aj + leave * (bj - aj), ak + leave * (bk - ak))


def _clip_line_rect(p: tuple[int, int], q: tuple[int, int], xmin: int, xmax: int, ymin: int, ymax: int):
    """Clip a displayed segment to its image panel so it cannot spill into labels."""
    x0, y0 = map(float, p)
    dx, dy = float(q[0] - p[0]), float(q[1] - p[1])
    lo, hi = 0.0, 1.0
    for pval, qval in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(pval) < 1e-12:
            if qval < 0:
                return None
            continue
        t = qval / pval
        if pval < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
        if lo > hi:
            return None
    return (round(x0 + lo * dx), round(y0 + lo * dy)), (round(x0 + hi * dx), round(y0 + hi * dy))


def _render_png(
    plane: np.ndarray,
    geometry: dict[str, Any],
    nodes: list[dict[str, Any]],
    by_id: dict[int, dict[str, Any]],
    soma_xyz_um: tuple[float, float, float],
    output_path: Path,
    *,
    mappings: tuple[str, ...] = ("A", "B"),
    crop_half_width_mm: float = 5.0,
    slab_thickness_mm: float = 0.1,
) -> dict[str, Any]:
    center_i = soma_xyz_um[0] / 100.0
    slice_i = int(round(center_i))
    j0, j1, k0, k1 = 0, plane.shape[0], 0, plane.shape[1]
    W, H = 1400, 1220
    margin = 46
    image_x, image_y, panel_w = 58, 126, 820
    image_h = int(panel_w * 0.9)  # preserve physical scale (j=0.09 mm, k=0.10 mm)
    bg = Image.new("RGB", (W, H), (248, 248, 246))
    draw = ImageDraw.Draw(bg)
    font = ImageFont.load_default()
    draw.text((margin, 18), "252527 / 058 | CH00DS nominal sagittal soma plane", fill=(20, 20, 20), font=font)
    draw.text((margin, 42), f"Constant NIfTI i={slice_i} (SWC x={soma_xyz_um[0]/1000:.3f} mm under both candidate mappings).", fill=(35, 35, 35), font=font)
    draw.text((margin, 60), f"Trace slab: {slab_thickness_mm:.2f} mm total along i; A/B are unresolved y/z mappings.", fill=(35, 35, 35), font=font)
    draw.text((margin, 78), "Source and conversion geometry are verified; SWC origin, anatomical laterality and plane acceptance remain open.", fill=(115, 35, 35), font=font)

    positive = plane[np.isfinite(plane) & (plane > 0)]
    if positive.size == 0:
        raise ValueError("Soma sagittal plane has no positive DS values")
    lo, hi = np.percentile(positive, (1.0, 99.5))
    normalized = np.clip((plane - lo) / max(float(hi - lo), 1.0), 0, 1)
    gray = (255 * np.flipud(normalized.T)).astype(np.uint8)
    panel_img = Image.fromarray(gray, "L").resize((panel_w, image_h), Image.Resampling.BILINEAR).convert("RGB")
    bg.paste(panel_img, (image_x, image_y))
    draw.rectangle((image_x, image_y, image_x + panel_w - 1, image_y + image_h - 1), outline=(50, 50, 50), width=1)

    def to_px(j: float, k: float) -> tuple[int, int]:
        px = image_x + int(j / max(j1 - j0, 1) * panel_w)
        py = image_y + int((k1 - k) / max(k1 - k0, 1) * image_h)
        return px, py

    low_i = center_i - (slab_thickness_mm / 2) / 0.1
    high_i = center_i + (slab_thickness_mm / 2) / 0.1
    panel_meta = []
    for mapping in mappings:
        mapped_nodes = [_out_ijk(node["xyz_um"], mapping) for node in nodes]
        shape_ijk = geometry["shape_ijk"]
        in_bounds_nodes = sum(all(0.0 <= coord[axis] < shape_ijk[axis] for axis in range(3)) for coord in mapped_nodes)
        trace_color = (222, 45, 44) if mapping == "A" else (30, 115, 222)
        edges_drawn = 0
        for node in nodes:
            parent = by_id.get(node["parent"])
            if parent is None:
                continue
            seg = _clip_segment_i(_out_ijk(node["xyz_um"], mapping), _out_ijk(parent["xyz_um"], mapping), low_i, high_i)
            if seg is None:
                continue
            p, q = to_px(*seg[0]), to_px(*seg[1])
            clipped = _clip_line_rect(p, q, image_x, image_x + panel_w - 1, image_y, image_y + image_h - 1)
            if clipped is None:
                continue
            draw.line(clipped, fill=trace_color, width=2)
            edges_drawn += 1
        soma_j, soma_k = _out_ijk(soma_xyz_um, mapping)[1:]
        sx, sy = to_px(soma_j, soma_k)
        soma_inside = 0 <= soma_j < plane.shape[0] and 0 <= soma_k < plane.shape[1]
        soma_plane_intensity = float(plane[min(max(int(round(soma_j)), 0), plane.shape[0] - 1), min(max(int(round(soma_k)), 0), plane.shape[1] - 1)]) if soma_inside else None
        if mapping == "A":
            draw.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), outline=(0, 220, 80), fill=(0, 55, 20), width=2)
        else:
            draw.rectangle((sx - 7, sy - 7, sx + 7, sy + 7), outline=(0, 240, 220), fill=(0, 55, 50), width=2)
        panel_meta.append({
            "mapping": mapping, "mapping_name": MAPPINGS[mapping]["name"], "basis": MAPPINGS[mapping]["basis"],
            "equation_output_ijk": MAPPINGS[mapping]["equation_output_ijk"],
            "sagittal_slice_i_float": center_i, "sagittal_slice_i_nearest": slice_i,
            "slab_thickness_mm_total": slab_thickness_mm, "slab_i_bounds": [low_i, high_i],
            "display_crop_jk": [j0, j1, k0, k1], "soma_output_ijk": list(_out_ijk(soma_xyz_um, mapping)),
            "soma_inside_image_grid": soma_inside, "soma_nearest_voxel_intensity": soma_plane_intensity,
            "swc_nodes_total": len(nodes), "swc_nodes_inside_image_grid": in_bounds_nodes,
            "swc_nodes_outside_image_grid": len(nodes) - in_bounds_nodes,
            "trace_edges_intersecting_slab_and_crop": edges_drawn, "display_percentiles_positive": [1, 99.5],
        })

    # One local inset per hypothesis makes the thin-slab trace around each soma
    # visible while keeping the full sagittal context in the same review figure.
    inset_x, inset_w, inset_h = 930, 410, 355
    half_j = int(math.ceil(crop_half_width_mm / 0.09))
    half_k = int(math.ceil(crop_half_width_mm / 0.1))
    display_array = np.flipud(normalized.T)
    for inset_idx, mapping in enumerate(mappings):
        jcenter, kcenter = _out_ijk(soma_xyz_um, mapping)[1:]
        cj0, cj1 = max(0, int(jcenter) - half_j), min(plane.shape[0], int(jcenter) + half_j + 1)
        ck0, ck1 = max(0, int(kcenter) - half_k), min(plane.shape[1], int(kcenter) + half_k + 1)
        # display_array row 0 is highest k and columns increase with j.
        crop = display_array[plane.shape[1] - ck1:plane.shape[1] - ck0, cj0:cj1]
        inset_img = Image.fromarray((255 * crop).astype(np.uint8), "L").resize((inset_w, inset_h), Image.Resampling.BILINEAR).convert("RGB")
        ix, iy = inset_x, 142 + inset_idx * 398
        bg.paste(inset_img, (ix, iy))
        draw.rectangle((ix, iy, ix + inset_w - 1, iy + inset_h - 1), outline=(50, 50, 50), width=1)
        draw.text((ix, iy - 22), f"Candidate {mapping} soma detail | {MAPPINGS[mapping]['name']}", fill=(20, 20, 20), font=font)

        def inset_px(j: float, k: float) -> tuple[int, int]:
            return ix + int((j - cj0) / max(cj1 - cj0, 1) * inset_w), iy + int((ck1 - k) / max(ck1 - ck0, 1) * inset_h)

        trace_color = (222, 45, 44) if mapping == "A" else (30, 115, 222)
        inset_edges = 0
        for node in nodes:
            parent = by_id.get(node["parent"])
            if parent is None:
                continue
            seg = _clip_segment_i(_out_ijk(node["xyz_um"], mapping), _out_ijk(parent["xyz_um"], mapping), low_i, high_i)
            if seg is None:
                continue
            p, q = inset_px(*seg[0]), inset_px(*seg[1])
            clipped = _clip_line_rect(p, q, ix, ix + inset_w - 1, iy, iy + inset_h - 1)
            if clipped is None:
                continue
            draw.line(clipped, fill=trace_color, width=2)
            inset_edges += 1
        sx, sy = inset_px(jcenter, kcenter)
        if mapping == "A":
            draw.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), outline=(0, 220, 80), fill=(0, 55, 20), width=2)
        else:
            draw.rectangle((sx - 7, sy - 7, sx + 7, sy + 7), outline=(0, 240, 220), fill=(0, 55, 50), width=2)
        scale_px = int(2.0 / (2 * crop_half_width_mm) * inset_w)
        bx, by = ix + inset_w - scale_px - 12, iy + inset_h - 18
        draw.line((bx, by, bx + scale_px, by), fill=(255, 255, 255), width=3)
        draw.text((bx, by - 17), "2 mm", fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        panel_meta[inset_idx]["soma_zoom_crop_jk"] = [cj0, cj1, ck0, ck1]
        panel_meta[inset_idx]["trace_edges_in_soma_zoom"] = inset_edges

    # The coordinate axes are labeled in image-index space; anatomy/directions are not accepted.
    draw.text((image_x + panel_w // 2 - 90, image_y + image_h + 10), "NIfTI axis j | 0.09 mm / voxel", fill=(20, 20, 20), font=font)
    draw.text((image_x, image_y + image_h + 33), "NIfTI axis k | 0.10 mm / voxel", fill=(20, 20, 20), font=font)
    scale_px = int(10.0 / (geometry["spacing_mm_ijk"][1] * plane.shape[0]) * panel_w)
    bx, by = image_x + panel_w - scale_px - 24, image_y + image_h - 24
    draw.line((bx, by, bx + scale_px, by), fill=(255, 255, 255), width=4)
    draw.text((bx, by - 18), "10 mm", fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))

    legend_y = 940
    draw.rectangle((margin, legend_y + 1, margin + 15, legend_y + 16), fill=(150, 150, 150), outline=(20, 20, 20))
    draw.text((margin + 22, legend_y), "Manifest-selected native CH00DS N4 image", fill=(20, 20, 20), font=font)
    draw.line((margin, legend_y + 35, margin + 17, legend_y + 35), fill=(222, 45, 44), width=3)
    draw.text((margin + 22, legend_y + 29), "SWC trace A", fill=(20, 20, 20), font=font)
    draw.line((margin + 142, legend_y + 35, margin + 159, legend_y + 35), fill=(30, 115, 222), width=3)
    draw.text((margin + 164, legend_y + 29), "SWC trace B", fill=(20, 20, 20), font=font)
    draw.ellipse((margin + 270, legend_y + 29, margin + 283, legend_y + 42), outline=(0, 190, 65), fill=(0, 55, 20), width=2)
    draw.text((margin + 290, legend_y + 29), "soma A", fill=(20, 20, 20), font=font)
    draw.rectangle((margin + 360, legend_y + 29, margin + 373, legend_y + 42), outline=(0, 190, 175), fill=(0, 55, 50), width=2)
    draw.text((margin + 380, legend_y + 29), "soma B", fill=(20, 20, 20), font=font)
    draw.text((margin, 1002), "A/B differ in the unresolved raw-SWC y/z mapping; both are review candidates.", fill=(115, 35, 35), font=font)
    draw.text((margin, 1060), f"Candidate A root output j,k={_out_ijk(soma_xyz_um, 'A')[1]:.1f},{_out_ijk(soma_xyz_um, 'A')[2]:.1f}; candidate B root output j,k={_out_ijk(soma_xyz_um, 'B')[1]:.1f},{_out_ijk(soma_xyz_um, 'B')[2]:.1f}.", fill=(35, 35, 35), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(output_path, format="PNG", optimize=True)
    return {"panels": panel_meta, "image_size_px": list(bg.size), "output_png": str(output_path), "output_sha256": _sha256(output_path)}


def render_sagittal_context(
    sample_id: str,
    neuron_id: str,
    soma_xyz_um: tuple[float, float, float],
    output_path: str | Path,
    *,
    processed_path: str | Path | None = None,
    swc_path: str | Path | None = None,
    mappings: tuple[str, ...] = ("A", "B"),
    slab_thickness_mm: float = 0.1,
    crop_half_width_mm: float | None = None,
    source: SagittalDSContext | None = None,
    style: str = "toolkit",
) -> dict[str, Any]:
    """Write one soma-centred sagittal panel and return source/output provenance.

    ``style="toolkit"`` (default) draws the panel with
    ``Visual_toolkit.plot_sagittal_context`` from the shared panel schema, so this
    entry point and the bulk runners / GUI produce the same review style. Titled
    panels, scale bar and marker come from ``REVIEW_SETTINGS``. ``style="legacy"``
    reproduces the 2026-10-16 PIL product and is retained for provenance only.
    """
    sample_id = str(sample_id)
    neuron_id = str(neuron_id).removesuffix(".swc")
    if sample_id not in SAMPLE_PATHS:
        raise ValueError(f"No processed-DS renderer configuration for sample {sample_id}")
    config = SAMPLE_PATHS[sample_id]
    if not mappings or any(x not in MAPPINGS for x in mappings):
        raise ValueError(f"mappings must be a non-empty sequence of {tuple(MAPPINGS)}")
    if style not in ("toolkit", "legacy"):
        raise ValueError("style must be 'toolkit' or 'legacy'")
    if slab_thickness_mm <= 0 or (crop_half_width_mm is not None and crop_half_width_mm <= 0):
        raise ValueError("Slab and crop width must be positive")
    if source is not None and (source.sample_id != sample_id or processed_path is not None):
        raise ValueError("Pass a source prepared for this sample; processed_path cannot override a prepared source")
    if style == "toolkit":
        # One drawing path: the shared panel schema from prepare_sagittal_panel is
        # rendered by Visual_toolkit, so this entry point, the bulk runners and the
        # GUI produce the same review figure.
        prepared = prepare_sagittal_panel(
            sample_id, neuron_id, soma_xyz_um,
            processed_path=processed_path, swc_path=swc_path, mappings=mappings,
            slab_thickness_mm=slab_thickness_mm, crop_half_width_mm=crop_half_width_mm,
            source=source)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        renderer = Visual_toolkit(sample_id)
        styled = renderer.plot_sagittal_context(prepared["panel"], neuron_id, output_dir=str(target.parent))
        produced = Path(styled["output_png"])
        renamed = produced.resolve() != target.resolve()
        if renamed:
            produced.replace(target)
        provenance = dict(prepared["provenance"])
        provenance["status"] = "candidate_geometry_rendered_axes_unvalidated"
        provenance["plot"] = {
            "style": "visual_toolkit.plot_sagittal_context",
            "output_png": str(target),
            "toolkit_default_png": str(produced),
            "renamed_to_requested_path": renamed,
            "settings": "main_scripts/Visual_toolkit/toolkit.py REVIEW_SETTINGS",
        }
        return provenance
    ds_record = source.record if source is not None else _source_record(sample_id)
    ds_path = Path(processed_path) if processed_path is not None else Path(ds_record["path"])
    if str(ds_path) != ds_record["path"]:
        raise ValueError("Custom processed_path is unsupported until a live manifest binding is added")
    swc, swc_source = resolve_swc_path(sample_id, neuron_id, config, swc_path)
    if swc is None:
        raise FileNotFoundError(
            f"Raw SWC is unavailable: no file in the registered cache "
            f"({config.get('swc_rel')}) or the legacy raw cache")
    nodes, by_id = _read_swc(swc)
    roots = [n for n in nodes if n["parent"] < 0]
    root_xyz = roots[0]["xyz_um"]
    if not np.allclose(root_xyz, soma_xyz_um, atol=1e-3, rtol=0):
        raise ValueError(f"SWC root {root_xyz} does not match requested soma {soma_xyz_um}")
    center_i = int(round(root_xyz[0] / 100.0))
    if source is None:
        planes, geometry = _read_sagittal_planes(ds_path, (center_i,), EXPECTED_GEOMETRY.get(sample_id))
        plane = planes[center_i]
    else:
        plane, center_i = source.plane_for_soma(root_xyz)
        geometry = source.geometry
    result_plot = _render_png(plane, geometry, nodes, by_id, root_xyz, Path(output_path), mappings=mappings,
                              crop_half_width_mm=crop_half_width_mm or 5.0,
                              slab_thickness_mm=slab_thickness_mm)
    return {
        "status": "candidate_geometry_rendered_axes_unvalidated",
        "sample_id": sample_id, "monkey_id": config["monkey"], "neuron_id": neuron_id,
        "processed_ds00": ds_record,
        "nifti_geometry": geometry,
        "swc": {"path": str(swc), "sha256": _sha256(swc), "node_count": len(nodes),
                "source": swc_source,
                "coordinate_frame": "raw SWC XYZ, declared micrometres; axes/origin/laterality unvalidated",
                "soma_xyz_um": list(root_xyz)},
        "mapping_hypotheses": list(mappings),
        "plot": result_plot,
        "limits": [
            "The live CH00DS manifest and image hash bind the processed source; this confirms source identity, not anatomical correctness.",
            "A and B reuse the two documented NRRD/SWC axis hypotheses; neither is accepted by in-bounds or signal plausibility alone.",
            "The constant-i plane is a nominal sagittal candidate under the NRRD header and converter; SWC-to-image origin and anatomical laterality remain unvalidated.",
            "Acquisition dates are unknown. No transform, resampling, SWC edit, or scientific acceptance is implied.",
        ],
    }


def prepare_sagittal_panel(
    sample_id: str,
    neuron_id: str,
    soma_xyz_um: tuple[float, float, float],
    *,
    processed_path: str | Path | None = None,
    swc_path: str | Path | None = None,
    mappings: tuple[str, ...] = ("A", "B"),
    slab_thickness_mm: float = 0.1,
    crop_half_width_mm: float | None = None,
    scale_bar_um: float | None = None,
    source: SagittalDSContext | None = None,
    declared_side: str | None = None,
    allow_missing_swc: bool = False,
) -> dict[str, Any]:
    """Prepare one soma-centred sagittal panel for the shared review style.

    Returns ``{"panel": ..., "provenance": ...}``. ``panel`` follows the panel
    schema documented in ``main_scripts/Visual_toolkit/toolkit.py`` and is drawn
    by ``Visual_toolkit.plot_sagittal_context`` / ``plot_tri_plot``; the two
    candidate SWC mappings stay separate traces. This prepares data only: the
    processed DS image, the raw SWC and every mapping remain unchanged. The
    default panel is the full constant-i plane, which keeps both candidate somas
    inside the same view; pass ``crop_half_width_mm`` to zoom on candidate A.
    """
    sample_id = str(sample_id)
    neuron_id = str(neuron_id).removesuffix(".swc")
    if sample_id not in SAMPLE_PATHS:
        raise ValueError(f"No processed-DS renderer configuration for sample {sample_id}")
    config = SAMPLE_PATHS[sample_id]
    if not mappings or any(x not in MAPPINGS for x in mappings):
        raise ValueError(f"mappings must be a non-empty sequence of {tuple(MAPPINGS)}")
    if slab_thickness_mm <= 0:
        raise ValueError("Slab thickness must be positive")
    if crop_half_width_mm is not None and crop_half_width_mm <= 0:
        raise ValueError("Crop width must be positive when set")
    if source is not None and (source.sample_id != sample_id or processed_path is not None):
        raise ValueError("Pass a source prepared for this sample; processed_path cannot override a prepared source")
    ds_record = source.record if source is not None else _source_record(sample_id)
    ds_path = Path(processed_path) if processed_path is not None else Path(ds_record["path"])
    if str(ds_path) != ds_record["path"]:
        raise ValueError("Custom processed_path is unsupported until a live manifest binding is added")
    # Default SWC resolution matches the bulk runner: repository root + registered
    # cache path, never another animal's data. A sample may have no cache configured.
    swc, swc_source = resolve_swc_path(sample_id, neuron_id, config, swc_path)
    if swc is None:
        if not allow_missing_swc:
            raise FileNotFoundError(
                f"Raw SWC is unavailable: no file in the registered cache "
                f"({config.get('swc_rel')}) or the legacy raw cache")
        nodes, by_id = [], {}
        root_xyz = tuple(float(v) for v in soma_xyz_um)
        swc_note = "raw SWC unavailable: no trace overlay, soma taken from the requested coordinate"
    else:
        nodes, by_id = _read_swc(swc)
        roots = [node for node in nodes if node["parent"] < 0]
        root_xyz = roots[0]["xyz_um"]
        if not np.allclose(root_xyz, soma_xyz_um, atol=1e-3, rtol=0):
            raise ValueError(f"SWC root {root_xyz} does not match requested soma {soma_xyz_um}")
        swc_note = None
    center_i = int(round(root_xyz[0] / 100.0))
    if source is None:
        planes, geometry = _read_sagittal_planes(ds_path, (center_i,), EXPECTED_GEOMETRY.get(sample_id))
        plane = planes[center_i]
    else:
        plane, center_i = source.plane_for_soma(root_xyz)
        geometry = source.geometry

    spacing_i_mm, spacing_j_mm, spacing_k_mm = (float(value) for value in geometry["spacing_mm_ijk"])
    shape_j, shape_k = plane.shape
    positive = plane[np.isfinite(plane) & (plane > 0)]
    if positive.size == 0:
        raise ValueError("Sagittal soma plane has no positive DS values")
    lo, hi = np.percentile(positive, (1.0, 99.5))
    normalized = np.clip((plane - lo) / max(float(hi - lo), 1.0), 0, 1)
    # Whole-plane normalization keeps brightness comparable between neurons; the
    # display then flips k so the NIfTI k axis points up in the panel.
    full_display = np.flipud(normalized.T)

    # The two candidate mappings place the soma up to ~18 mm apart, so the default
    # view is the whole constant-i plane. An explicit crop is centred on candidate A.
    candidate_jk = [_out_ijk(root_xyz, mapping, geometry)[1:] for mapping in mappings]
    if crop_half_width_mm is None:
        j0, j1, k0, k1 = 0, shape_j, 0, shape_k
        center_j, center_k = shape_j / 2.0, shape_k / 2.0
        view_note = f"full constant-i plane {shape_j}x{shape_k} voxels"
    else:
        center_j, center_k = float(candidate_jk[0][0]), float(candidate_jk[0][1])
        half_j = int(math.ceil(crop_half_width_mm / spacing_j_mm))
        half_k = int(math.ceil(crop_half_width_mm / spacing_k_mm))
        j0, j1 = max(0, int(center_j) - half_j), min(shape_j, int(center_j) + half_j + 1)
        k0, k1 = max(0, int(center_k) - half_k), min(shape_k, int(center_k) + half_k + 1)
        view_note = f"{crop_half_width_mm:g} mm crop centred on candidate A"
    if j1 <= j0 or k1 <= k0:
        raise ValueError(f"Soma crop is empty for j=[{j0},{j1}) k=[{k0},{k1})")
    display = full_display[shape_k - k1:shape_k - k0, j0:j1]

    low_i = center_i - (slab_thickness_mm / 2) / spacing_i_mm
    high_i = center_i + (slab_thickness_mm / 2) / spacing_i_mm
    markers: list[tuple[float, float, str, float | None, str]] = []
    traces: list[tuple[np.ndarray, str, str, float, float]] = []
    mapping_meta: list[dict[str, Any]] = []
    for mapping in mappings:
        color = TRACE_COLORS.get(mapping, "#ff8c00")
        pieces: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for node in nodes:
            parent = by_id.get(node["parent"])
            if parent is None:
                continue
            segment = _clip_segment_i(_out_ijk(node["xyz_um"], mapping, geometry),
                                      _out_ijk(parent["xyz_um"], mapping, geometry), low_i, high_i)
            if segment is None:
                continue
            (aj, ak), (bj, bk) = segment
            if max(aj, bj) < j0 or min(aj, bj) > j1 or max(ak, bk) < k0 or min(ak, bk) > k1:
                continue
            pieces.append(((aj - j0, (k1 - 1) - ak), (bj - j0, (k1 - 1) - bk)))
        if pieces:
            # One polyline with NaN breaks: a single draw call for the whole trace.
            stacked = np.full((3 * len(pieces) - 1, 2), np.nan)
            for index, (start, stop) in enumerate(pieces):
                stacked[3 * index] = start
                stacked[3 * index + 1] = stop
            # Dashed so the candidate traces stay distinguishable from the solid
            # red raw-SWC edges drawn in the WideField panel of the same figure.
            traces.append((stacked, color, "dashed", 1.2, 0.9))
        soma_j, soma_k = _out_ijk(root_xyz, mapping, geometry)[1:]
        soma_inside = bool(j0 <= soma_j < j1 and k0 <= soma_k < k1)
        if soma_inside and not markers:
            # The same cyan/turquoise target marker as the Soma and Context panels:
            # an open square plus a cross, both centred on the exact mapped position.
            # Only the drawn size grows so the target reads at whole-hemisphere zoom;
            # the unresolved B candidate keeps its dashed trace.
            style = _marker_style()
            markers.append((
                soma_j - j0, (k1 - 1) - soma_k, style["color"], None, "Target Soma",
                {"min_px": style["min_px"] * 2, "crosshair_px": style["crosshair_px"],
                 "label_fontsize": style["label_fontsize"] + 1,
                 "label_offset_px": style["label_offset_px"]},
            ))
        mapping_meta.append({
            "mapping": mapping, "mapping_name": MAPPINGS[mapping]["name"],
            "basis": MAPPINGS[mapping]["basis"],
            "equation_output_ijk": MAPPINGS[mapping]["equation_output_ijk"],
            "soma_output_jk": [round(float(soma_j), 3), round(float(soma_k), 3)],
            "soma_inside_crop": soma_inside,
            "trace_edges_in_crop_and_slab": len(pieces),
        })

    hemisphere = _hemisphere_from_index(root_xyz, geometry)
    declared_name = {"L": "Left", "LEFT": "Left", "R": "Right", "RIGHT": "Right"}.get(
        str(declared_side).strip().upper()) if declared_side else None
    panel = {
        "display": display,
        "cmap": "gray",
        "hemisphere": hemisphere,
        "resolution_um": (spacing_j_mm * 1000.0, spacing_k_mm * 1000.0),
        "fov_um": (display.shape[1] * spacing_j_mm * 1000.0,
                   display.shape[0] * spacing_k_mm * 1000.0),
        "depth_um": slab_thickness_mm * 1000.0,
        "scale_bar_um": None if scale_bar_um is None else float(scale_bar_um),
        "markers": markers,
        "edge_segments": [],
        "traces": traces,
        "axis_labels": (f"NIfTI j | {spacing_j_mm:.2f} mm/voxel",
                        f"NIfTI k | {spacing_k_mm:.2f} mm/voxel"),
        "legend": ([("patch", "#9a9a9a", "DS00 N4 intensity (gray)")]
                   + [("line", TRACE_COLORS.get(mapping, "#ff8c00"),
                       MAPPINGS[mapping].get("short_label", f"trace + soma {mapping}"),
                       "dashed")
                      for mapping in mappings]
                   + [("marker", _marker_style()["color"],
                       "target soma marker: box + cross (at candidate A)")]),
        "notes": [f"Whole-plane p1–p99.5 gray; {view_note}; {slab_thickness_mm:.2f} mm i slab; "
                  "target square and crosshair are centred on the exact candidate-A position "
                  "(drawn larger for visibility, not to scale); candidate B position remains "
                  "unresolved."
                  + (f" Hemisphere label follows the volume i convention ({'large' if hemisphere == 'Left' else 'small'} i = "
                     f"{hemisphere}); workbook Soma_Side comparison is recorded in the manifest, "
                     "and laterality is not visually confirmed." if hemisphere else "")]
                  + ([f"candidate {row['mapping']} soma lies outside this panel"]
                     if not all(row["soma_inside_crop"] for row in mapping_meta) else []),
        "role": "Sagittal context (DS00 N4)",
        "suffix": "Sagittal",
    }
    provenance = {
        "status": "candidate_geometry_rendered_axes_unvalidated",
        "sample_id": sample_id, "monkey_id": config["monkey"], "neuron_id": neuron_id,
        "processed_ds00": ds_record,
        "nifti_geometry": geometry,
        "swc": ({"path": str(swc), "sha256": _sha256(swc), "node_count": len(nodes),
                 "source": swc_source,
                 "coordinate_frame": "raw SWC XYZ, declared micrometres; axes/origin/laterality unvalidated",
                 "soma_xyz_um": list(root_xyz)} if swc is not None else None),
        "sagittal_slice_i": center_i,
        "slab_i_bounds": [low_i, high_i],
        "crop_jk": [j0, j1, k0, k1],
        "crop_center_jk": [round(center_j, 3), round(center_k, 3)],
        "display_percentiles_positive": [1.0, 99.5],
        "mapping_hypotheses": mapping_meta,
        "hemisphere": {
            "derived_from_i": hemisphere,
            "declared_workbook_side": declared_side,
            "declared_normalized": declared_name,
            "agreement": (None if hemisphere is None or declared_name is None
                          else hemisphere == declared_name),
            "basis": ("converted-volume i convention (large i = Left), cross-checked against the "
                      "active workbook Soma_Side for 252527; not a visually confirmed laterality"),
        },
        "panel_style": "Visual_toolkit review panel style (Soma/Context/Sagittal)",
        "limits": [
            "The live CH00DS manifest and image hash bind the processed source; this confirms source identity, not anatomical correctness.",
            "A and B reuse the two documented NRRD/SWC axis hypotheses; neither is accepted by in-bounds or signal plausibility alone.",
            "The constant-i plane is a nominal sagittal candidate; SWC-to-image origin and anatomical laterality remain unvalidated.",
            "Panel markers are enlarged for visibility and are not to scale; the white bar carries the physical scale.",
            "Acquisition dates are unknown. No transform, resampling, SWC edit, or scientific acceptance is implied.",
        ] + ([swc_note] if swc_note else []),
    }
    return {"panel": panel, "provenance": provenance}

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
                                       swc_tree=tree, soma_region=test_region)
        
    toolkit.close()
