"""Descriptive template-space axon length maps from validated SWC edges.

This optional representation leaves legacy region tables unchanged. SWC type
2 on the child selects an edge, including compartment transitions. Segments
are split at voxel faces; sampling density does not weight the measurement.
Lengths describe the reference template, not native tissue or synapses.
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from swc_validation import parse_swc


@dataclass(frozen=True)
class ReferenceGrid:
    shape: tuple
    affine_mm: np.ndarray

    def __post_init__(self):
        shape = tuple(self.shape)
        affine = np.asarray(self.affine_mm, dtype=np.float64).copy()
        if len(shape) != 3 or any(int(n) != n or n <= 0 for n in shape):
            raise ValueError("Reference must have three positive integer dimensions")
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or not np.allclose(affine[3], [0, 0, 0, 1])
                or abs(np.linalg.det(affine[:3, :3])) < 1e-15):
            raise ValueError("Reference requires a finite invertible spatial affine")
        affine.setflags(write=False)
        object.__setattr__(self, "shape", tuple(int(n) for n in shape))
        object.__setattr__(self, "affine_mm", affine)

    @classmethod
    def from_image(cls, image):
        unit = image.header.get_xyzt_units()[0]
        scales = {"mm": 1.0, "micron": 0.001, "meter": 1000.0}
        if unit not in scales:
            raise ValueError("Reference spatial units must be explicitly mm, micron or meter")
        sform, scode = image.header.get_sform(coded=True)
        qform, qcode = image.header.get_qform(coded=True)
        if not scode and not qcode:
            raise ValueError("Reference must have a coded spatial transform")
        if scode and qcode and not np.allclose(sform, qform, rtol=1e-5, atol=1e-4):
            raise ValueError("Conflicting coded qform and sform require review")
        affine = image.affine.copy()
        affine[:3, :] *= scales[unit]
        return cls(image.shape, affine)

    @property
    def voxel_volume_mm3(self):
        return abs(float(np.linalg.det(self.affine_mm[:3, :3])))


def segment_voxels(start, end, shape):
    """Yield (voxel, fraction of full edge) within the reference FOV.

    Voxel i occupies [i-0.5, i+0.5). First clip the parameter interval to the
    FOV, then split at interior voxel faces. Allocation therefore conserves
    in-FOV length even when both endpoints lie outside. Boundary-only contacts
    have zero length. Work is bounded by grid size, not remote coordinates.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    shape = np.asarray(shape, dtype=int)
    if (start.shape != (3,) or end.shape != (3,) or shape.shape != (3,)
            or not np.isfinite(start).all() or not np.isfinite(end).all()
            or np.any(shape <= 0)):
        raise ValueError("Segment coordinates and reference shape are invalid")
    delta = end - start
    if not np.isfinite(delta).all():
        raise ValueError("Segment coordinate difference overflows")
    lower, upper = 0.0, 1.0
    for axis in range(3):
        if delta[axis] == 0:
            if not -0.5 <= start[axis] < shape[axis] - 0.5:
                return
        else:
            crossings = ((-0.5 - start[axis]) / delta[axis],
                         (shape[axis] - 0.5 - start[axis]) / delta[axis])
            lower = max(lower, min(crossings))
            upper = min(upper, max(crossings))
    if upper <= lower:
        return
    cuts = [lower, upper]
    for axis in range(3):
        if delta[axis] != 0:
            faces = np.arange(shape[axis] - 1, dtype=float) + 0.5
            times = (faces - start[axis]) / delta[axis]
            cuts.extend(times[(times > lower) & (times < upper)])
    cuts = np.unique(cuts)
    for left, right in zip(cuts[:-1], cuts[1:]):
        voxel = np.floor(start + (left + (right-left)/2) * delta + 0.5).astype(int)
        if np.all(voxel >= 0) and np.all(voxel < shape):
            yield tuple(voxel), float(right - left)


def axon_length_map(swc_text, grid, *, coordinate_frame, index_scale_um=None,
                    source="SWC"):
    """Return float64 length (mm/voxel) and explicit edge/coverage accounting.

    ``atlas_index_um`` means XYZ / index_scale_um are voxel indices, as the
    existing macaque tracer declares. ``nifti_world_mm`` means literal reference
    world coordinates. Neither representation is inferred from filenames.
    The source must already be registered; this function performs no warp.
    """
    rows = parse_swc(swc_text, source)
    coords = np.asarray([row[2:5] for row in rows], dtype=float)
    if coordinate_frame == "atlas_index_um":
        scale = np.asarray(index_scale_um, dtype=float)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Index-coordinate SWCs require three positive um/voxel scales")
        coords = coords / scale
    elif coordinate_frame == "nifti_world_mm":
        if index_scale_um is not None:
            raise ValueError("Index scale does not apply to NIfTI world coordinates")
        inverse = np.linalg.inv(grid.affine_mm)
        coords = coords @ inverse[:3, :3].T + inverse[:3, 3]
    else:
        raise ValueError("Declare atlas_index_um or nifti_world_mm coordinates explicitly")
    if not np.isfinite(coords).all():
        raise ValueError("Mapped coordinates must be finite")
    id_to_row = {row[0]: i for i, row in enumerate(rows)}
    children = np.asarray([i for i, row in enumerate(rows) if row[6] != -1], dtype=int)
    parents = np.asarray([id_to_row[rows[i][6]] for i in children], dtype=int)
    types = np.asarray([rows[i][1] for i in children], dtype=int)
    deltas = coords[children] - coords[parents]
    lengths = np.linalg.norm(deltas @ grid.affine_mm[:3, :3].T, axis=1)
    if not np.isfinite(lengths).all():
        raise ValueError("Template-space edge lengths must be finite")
    selected = types == 2
    output = np.zeros(grid.shape, dtype=np.float64)
    starts, ends, weights = coords[parents[selected]], coords[children[selected]], lengths[selected]
    # Dense traces mostly remain within one voxel. Accumulate those edges in
    # one vectorized operation, traversing voxel faces only for the remainder.
    start_voxels = np.floor(starts + 0.5)
    end_voxels = np.floor(ends + 0.5)
    same = (np.all(start_voxels == end_voxels, axis=1)
            & np.all(start_voxels >= 0, axis=1)
            & np.all(start_voxels < grid.shape, axis=1))
    indices = start_voxels[same].astype(int)
    if len(indices):
        np.add.at(output, tuple(indices.T), weights[same])
    for start, end, length in zip(starts[~same], ends[~same], weights[~same]):
        if length:
            for voxel, fraction in segment_voxels(start, end, grid.shape):
                output[voxel] += length * fraction
    total = float(weights.sum())
    inside = float(output.sum())
    if inside > total and not math.isclose(inside, total, rel_tol=1e-10, abs_tol=1e-10):
        raise ArithmeticError("Voxel allocation exceeds selected axon length")
    return output, {
        "node_count": len(rows), "edge_count": len(children),
        "selected_axon_edges": int(selected.sum()),
        "zero_length_axon_edges": int(np.count_nonzero(weights == 0)),
        "edge_length_by_child_type_mm": {str(t): float(lengths[types == t].sum()) for t in np.unique(types)},
        "selected_axon_length_mm": total, "in_reference_length_mm": inside,
        "outside_reference_length_mm": max(0.0, total - inside),
        "fast_same_voxel_edges": int(same.sum()),
        "compartment_policy": "child SWC type == 2; transitions included",
        "length_space": "reference template; native tissue length not established",
        "coordinate_frame": coordinate_frame,
    }


def animal_mean(maps):
    """Equal animal weights; absent animals must not be supplied as zero maps."""
    if not maps:
        raise ValueError("At least one contributing animal map is required")
    result = np.zeros_like(maps[0], dtype=np.float64)
    for data in maps:
        if data.shape != result.shape or not np.isfinite(data).all() or np.any(data < 0):
            raise ValueError("Animal maps must match and contain finite nonnegative length")
        result += data / len(maps)
    return result


def save_map(path, data, grid):
    """Write descriptive float32 NIfTI with the exact mm reference transform."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    stored = np.asarray(data, dtype=np.float32)
    if stored.shape != grid.shape or not np.isfinite(stored).all() or np.any(stored < 0):
        raise ValueError("Map must match the reference and remain finite and nonnegative")
    image = nib.Nifti1Image(stored, grid.affine_mm)
    image.header.set_xyzt_units("mm")
    image.set_sform(grid.affine_mm, code=2)
    # A sheared reference cannot be represented by a quaternion. Retain its
    # exact sform rather than silently stripping shear to create a qform.
    image.set_qform(None, code=0)
    image.header["descrip"] = "Descriptive template axon length; not BOLD or t statistic"
    descriptor, temporary = tempfile.mkstemp(prefix=".axon-map-", suffix=".nii.gz", dir=path.parent)
    os.close(descriptor)
    try:
        nib.save(image, temporary)
        readback = nib.load(temporary)
        if (readback.shape != grid.shape or not np.allclose(readback.affine, grid.affine_mm)
                or readback.header.get_xyzt_units()[0] != "mm"
                or not np.array_equal(np.asanyarray(readback.dataobj), stored)):
            raise IOError("Saved map failed complete pixel/geometry readback")
        # Atomic publication without replacing an existing map, including a
        # destination created concurrently after the initial existence check.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return float(stored.sum(dtype=np.float64))
