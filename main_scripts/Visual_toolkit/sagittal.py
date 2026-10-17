"""Render a raw-SWC soma plane on its candidate native CH00DS image frame.

Version: 1.9.0 (released 636197c; package version)
Last updated: 2026-10-17
Change history: CHANGELOG.md (this folder)
Status: candidate (A/B SWC mapping unresolved)

This small renderer preserves both unresolved 252527 SWC-to-NRRD axis mappings.
It reads the existing converted NIfTI in place and writes only a PNG plus the
returned metadata; it never resamples or modifies source images/SWCs.

Local change history:
- 2026-10-16 Codex (local, unreleased): bind candidate DS00 sagittal PNGs to
  the manifest-selected processed source, verify source/SWC root geometry,
  and add one-pass multi-soma plane extraction. A/B mappings remain explicit
  candidates; no resampling or anatomical acceptance is implied.
- 2026-10-17 Codex (local, unreleased): add ``prepare_sagittal_panel``, which
  returns one soma-centred sagittal panel (display array, candidate traces,
  soma markers) in the shared Visual_toolkit panel schema so the sagittal view
  uses the same legacy style as SomaBlock/WideField and can join the one-figure
  tri-plot. The earlier PIL renderer is retained unchanged as the historical
  full-plane product.
- 2026-10-17 Codex (local, unreleased): ``render_sagittal_context`` now draws
  through ``Visual_toolkit.plot_sagittal_context`` by default (``style="toolkit"``),
  so its output matches the bulk runners and the GUI. Its crop default changed
  from 5 mm to the whole constant-i plane to match the reviewed panel. The PIL
  product remains reachable with ``style="legacy"`` for the 2026-10-16 provenance.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    """Target-marker style owned by the VisualToolkit review settings.

    The fallback keeps this adapter usable when the toolkit is not importable; it
    mirrors the same defaults rather than introducing a second design.
    """
    try:
        from .toolkit import REVIEW_SETTINGS
        return REVIEW_SETTINGS["marker"]
    except Exception:
        try:
            from Visual_toolkit.toolkit import REVIEW_SETTINGS
            return REVIEW_SETTINGS["marker"]
        except Exception:
            return {"color": "#00e5ee", "min_px": 8.0, "crosshair_px": 60.0,
                    "label_fontsize": 12, "label_offset_px": 22.0}


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
        # Same import the bulk runner uses: the package re-exports the class.
        from Visual_toolkit import Visual_toolkit
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
