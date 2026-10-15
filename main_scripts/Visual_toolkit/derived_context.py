"""Bounded native context on the global source-voxel sampling lattice.

Coordinates retain the declared source convention; this does not calibrate
physical origin, orientation, or anatomy. Decimation samples points without
averaging and is intended for exploratory regional context only.
"""

import time

import numpy as np


def context_plan(native_xyz, field_um=4000.0, depth_um=30.0, factor=8,
                 max_cubes=400, max_working_bytes=128 * 1024**2,
                 max_source_bytes=12 * 1024**3):
    native = np.asarray(native_xyz, dtype=float)
    if native.shape != (3,) or not np.isfinite(native).all() or (native < 0).any():
        raise ValueError("native root must be three finite nonnegative coordinates")
    if not all(np.isfinite(v) and v > 0 for v in (field_um, depth_um)):
        raise ValueError("field and depth must be finite and positive")
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    for bound in (max_cubes, max_working_bytes, max_source_bytes):
        if isinstance(bound, bool) or not isinstance(bound, (int, np.integer)) or bound < 1:
            raise ValueError("resource bounds must be positive integers")
    spacing = np.array([0.65, 0.65, 3.0])
    block = np.array([360, 360, 90], dtype=np.int64)
    stride = np.array([factor, factor, 1], dtype=np.int64)
    half = np.array([field_um, field_um, depth_um]) / 2
    # Point samples lie inside [requested lower bound, requested upper bound).
    start = np.maximum(0, np.ceil((native - half) / spacing / stride)).astype(np.int64)
    stop = np.ceil((native + half) / spacing / stride).astype(np.int64)
    shape_xyz = stop - start
    if (shape_xyz <= 0).any():
        raise ValueError("requested field contains no source samples")
    first = start * stride
    last = (stop - 1) * stride
    lo, hi = first // block, last // block
    cube_count = int(np.prod(hi - lo + 1, dtype=object))
    voxels = int(np.prod(shape_xyz, dtype=object))
    cube_bytes = int(np.prod(block)) * 2
    # Canvas/coverage plus bounded compressed payload and decoder page/stack copies.
    # This is an explicit estimate, excluding the Python/library baseline.
    working_bytes = voxels * 3 + 3 * cube_bytes + 32 * 1024**2
    report = {
        "cube_count": cube_count, "max_cubes": int(max_cubes),
        "max_working_bytes": int(max_working_bytes), "working_bytes_estimate": working_bytes,
        "max_source_bytes": int(max_source_bytes), "source_bytes_estimate": cube_count * 32 * 1024**2,
        "derived_spacing_xyz_um": (spacing * stride).tolist(),
        "origin_xyz_um": (first * spacing).tolist(),
        "source_first_index_xyz": first.tolist(), "source_stride_xyz": stride.tolist(),
        "shape_zyx": shape_xyz[::-1].tolist(), "downsample": int(factor),
        "requested_field_um_xyz": [float(field_um), float(field_um), float(depth_um)],
        "requested_bounds_xyz_um": [(native - half).tolist(), (native + half).tolist()],
        "sampling": "global source point decimation; no averaging or maximum aggregation",
        "source_plane_ids": list(range(int(first[2]), int(last[2]) + 1)),
        "central_cube_xyz": (np.floor(native / spacing).astype(np.int64) // block).tolist(),
    }
    if cube_count > max_cubes or working_bytes > max_working_bytes or cube_count * 32 * 1024**2 > max_source_bytes:
        report.update(status="not_run", reason="requested context exceeds a resource bound")
        return None, report
    cubes = [(x, y, z) for z in range(int(lo[2]), int(hi[2]) + 1)
             for y in range(int(lo[1]), int(hi[1]) + 1)
             for x in range(int(lo[0]), int(hi[0]) + 1)]
    center = report["central_cube_xyz"]
    cubes.sort(key=lambda cube: sum(abs(cube[axis] - center[axis]) for axis in range(3)))
    return cubes, report


def derive_highres_context(native_xyz, download, field_um=4000.0, depth_um=30.0,
                           factor=8, max_cubes=400, max_working_bytes=128 * 1024**2,
                           max_source_bytes=12 * 1024**3, max_seconds=900):
    """Download sequentially; require root-cube and root-nearest pixel coverage.

    Runtime is checked between callbacks. The download callback must enforce its
    own request timeout and response-size limit; a running callback is not killed.
    """
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    cubes, report = context_plan(native_xyz, field_um, depth_um, factor, max_cubes,
                                 max_working_bytes, max_source_bytes)
    if cubes is None:
        return None, report
    report["requested"] = [list(cube) for cube in cubes]
    started = time.monotonic()
    first = np.array(report["source_first_index_xyz"])
    stride = np.array(report["source_stride_xyz"])
    shape_xyz = np.array(report["shape_zyx"])[::-1]
    block = np.array([360, 360, 90])
    canvas = np.zeros(report["shape_zyx"], dtype=np.uint16)
    coverage = np.zeros(canvas.shape, dtype=bool)
    loaded, missing, invalid, unattempted = [], [], [], []
    for ordinal, index in enumerate(cubes):
        if time.monotonic() - started >= max_seconds:
            unattempted = [list(cube) for cube in cubes[ordinal:]]
            break
        volume = download(*index)
        if volume is None:
            missing.append(list(index))
            if list(index) == report["central_cube_xyz"]:
                unattempted = [list(cube) for cube in cubes[ordinal + 1:]]
                break
            continue
        if not isinstance(volume, np.ndarray) or volume.shape != (90, 360, 360) or volume.dtype != np.uint16:
            invalid.append({"cube": list(index), "reason": "expected uint16 ZYX [90,360,360]"})
            del volume
            if list(index) == report["central_cube_xyz"]:
                unattempted = [list(cube) for cube in cubes[ordinal + 1:]]
                break
            continue
        cube_start = np.array(index) * block
        # Global stride alignment also works when factor does not divide 360.
        out_lo = np.maximum(0, np.ceil((cube_start - first) / stride).astype(int))
        out_hi = np.minimum(shape_xyz, np.ceil((cube_start + block - first) / stride).astype(int))
        if (out_hi > out_lo).all():
            local = first + out_lo * stride - cube_start
            source = tuple(slice(int(local[a]), int(local[a] + (out_hi[a] - out_lo[a]) * stride[a]),
                                 int(stride[a])) for a in (2, 1, 0))
            target = tuple(slice(int(out_lo[a]), int(out_hi[a])) for a in (2, 1, 0))
            canvas[target] = volume[source]
            coverage[target] = True
        loaded.append(list(index))
        del volume
    local_root = (np.asarray(native_xyz) - report["origin_xyz_um"]) / report["derived_spacing_xyz_um"]
    nearest = np.floor(local_root + 0.5).astype(int)
    inside = ((nearest >= 0) & (nearest < shape_xyz)).all()
    root_covered = bool(inside and coverage[tuple(nearest[::-1])])
    center_loaded = report["central_cube_xyz"] in loaded
    report.update(loaded=loaded, missing=missing, invalid=invalid, unattempted=unattempted,
                  elapsed_seconds=time.monotonic() - started, max_seconds=float(max_seconds),
                  root_nearest_index_xyz=nearest.tolist(), root_pixel_covered=root_covered,
                  central_cube_loaded=center_loaded, coverage_fraction=float(coverage.mean()))
    if not center_loaded or not root_covered:
        report.update(status="context_unavailable", reason="root source coverage is missing or invalid")
        return None, report
    complete = bool(coverage.all())
    report.update(status="derived_highres" if complete else "derived_partial", complete=complete)
    return canvas, report
