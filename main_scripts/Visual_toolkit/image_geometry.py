"""Native microscopy coordinate arithmetic; no anatomical orientation inferred."""

import numpy as np


def triplet(values, name, positive=False, nonnegative=False):
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite XYZ values")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative in the native image frame")
    return result


def native_to_index(point_um, origin_um, spacing_um):
    """Return floating XYZ image indices from native acquisition micrometres."""
    return ((triplet(point_um, "point") - triplet(origin_um, "origin"))
            / triplet(spacing_um, "spacing", positive=True))


def native_affine(origin_um, spacing_um):
    """Map XYZ voxel indices to native acquisition micrometres without flips."""
    affine = np.eye(4)
    affine[:3, :3] = np.diag(triplet(spacing_um, "spacing", positive=True))
    affine[:3, 3] = triplet(origin_um, "origin")
    return affine


def widefield_bounds(center_um, spacing_um, width_um, height_um, depth_um,
                     first_slice_index=1):
    """Return integer XYZ bounds with exclusive stops and exact slice count.

    The existing resampled filename convention uses one-based Z indices.
    The corresponding native Z coordinate is ``index * spacing_z``; this is
    the repository's convention, not an independently verified origin offset.
    """
    center = triplet(center_um, "center", nonnegative=True)
    spacing = triplet(spacing_um, "spacing", positive=True)
    dimensions = triplet([width_um, height_um, depth_um], "field dimensions", positive=True)
    sizes = np.maximum(1, np.floor(dimensions / spacing + 0.5).astype(int))
    index = np.floor(center / spacing).astype(int)
    if index[2] < first_slice_index:
        raise ValueError("Soma Z is below the first configured image slice")
    start = index - sizes // 2
    start[:2] = np.maximum(start[:2], 0)
    start[2] = max(int(start[2]), first_slice_index)
    return start, start + sizes


def clip_segment(start, stop, lower, upper):
    """Clip one XYZ segment to a box, preserving crossings and excluded gaps."""
    start = triplet(start, "segment start")
    stop = triplet(stop, "segment stop")
    lower = triplet(lower, "box lower")
    upper = triplet(upper, "box upper")
    delta = stop - start
    enter, leave = 0.0, 1.0
    for axis in range(3):
        if delta[axis] == 0:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return None
            continue
        limits = sorted(((lower[axis] - start[axis]) / delta[axis],
                         (upper[axis] - start[axis]) / delta[axis]))
        enter, leave = max(enter, limits[0]), min(leave, limits[1])
        if enter > leave:
            return None
    return start + enter * delta, start + leave * delta


def percentile_normalize(image, lower=1, upper=99.5):
    """Finite contrast normalization, including constant or all-zero images."""
    image = np.asarray(image, dtype=float)
    if not image.size or not np.isfinite(image).all():
        raise ValueError("Image must be nonempty and finite")
    lo, hi = np.percentile(image, [lower, upper])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)
