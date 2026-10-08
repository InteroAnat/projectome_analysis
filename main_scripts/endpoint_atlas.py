"""Validated ARM-level lookup at already assigned endpoint voxels.

No hierarchy parents, anatomical orientation or biological target acceptance
are inferred. Hemisphere semantics are checked against explicit atlas labels.
"""

from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd

from projection_maps import ReferenceGrid
from region_analysis.laterality import LateralityParser


def coded_mm_grid(image, *, spatial_only=False):
    """Require coded, consistent millimetre geometry, including 5-D ARM files."""
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("All reference/atlas/mask spatial units must explicitly be mm")
    sform, scode = image.header.get_sform(coded=True)
    qform, qcode = image.header.get_qform(coded=True)
    if not scode and not qcode:
        raise ValueError("Reference/atlas/mask requires a coded spatial transform")
    if scode and qcode and not np.allclose(sform, qform, rtol=0, atol=1e-7):
        raise ValueError("Conflicting coded qform and sform require review")
    shape = image.shape[:3] if spatial_only else image.shape
    return ReferenceGrid(shape, image.affine)


def matching_grid(image, grid, *, spatial_only=False):
    other = coded_mm_grid(image, spatial_only=spatial_only)
    if other.shape != grid.shape or not np.allclose(other.affine_mm, grid.affine_mm, rtol=0, atol=1e-7):
        raise ValueError("Atlas/mask geometry must match the selected reference grid")
    return other


class EndpointAtlas:
    """Six actual ARM volumes plus a label-validated hemisphere mask."""

    def __init__(self, atlas_path, key_path, hemisphere_mask, grid):
        image = nib.load(str(atlas_path))
        if image.shape != (*grid.shape, 1, 6):
            raise ValueError("ARM atlas must have reference shape plus singleton dimension and six levels")
        matching_grid(image, grid, spatial_only=True)
        self.data = np.asanyarray(image.dataobj)[..., 0, :]
        if (not np.issubdtype(self.data.dtype, np.integer)
                or np.any(self.data < 0)):
            raise ValueError("ARM atlas values must be nonnegative integer labels")
        table = pd.read_csv(key_path, sep="\t", dtype=str, keep_default_na=False)
        required = {"Index", "Abbreviation", "Full_Name", "First_Level", "Last_Level"}
        if not required <= set(table.columns) or table.empty:
            raise ValueError("ARM key requires Index, Abbreviation, Full_Name, First_Level, Last_Level")
        self.labels = {}
        for row in table.to_dict("records"):
            try:
                index, first, last = (int(row[name]) for name in ("Index", "First_Level", "Last_Level"))
            except ValueError as exc:
                raise ValueError("ARM key label and level indices must be integers") from exc
            if index <= 0 or index in self.labels or not 1 <= first <= last <= 6:
                raise ValueError("ARM key requires unique positive labels and valid level ranges")
            if not row["Abbreviation"].strip() or not row["Full_Name"].strip():
                raise ValueError("ARM key label names must be present")
            self.labels[index] = dict(index=index, abbreviation=row["Abbreviation"],
                                      full_name=row["Full_Name"], first=first, last=last)

        hemisphere_image = nib.load(str(hemisphere_mask))
        matching_grid(hemisphere_image, grid)
        self.hemisphere = np.asanyarray(hemisphere_image.dataobj)
        if (not np.isfinite(self.hemisphere).all() or np.any(self.hemisphere < 0)
                or not np.equal(self.hemisphere, np.floor(self.hemisphere)).all()):
            raise ValueError("Hemisphere mask must contain finite nonnegative integer values")
        # Equivalent to AtlasHemisphereReference's label contingency contract,
        # but uses observed labels instead of allocating by maximum label ID.
        self.value_to_side, self.contingency = {}, {}
        finest = self.data[..., 5]
        for value in np.unique(self.hemisphere):
            if value == 0:
                continue
            labels, counts = np.unique(finest[self.hemisphere == value], return_counts=True)
            sides = Counter()
            for label, count in zip(labels, counts):
                metadata = self.labels.get(int(label))
                side = LateralityParser.get_side(metadata["abbreviation"]) if metadata else "Unknown"
                sides[side] += int(count)
            evidence = {side: sides[side] for side in ("L", "R")}
            if min(evidence.values()) or not max(evidence.values()):
                raise ValueError(f"Hemisphere mask conflicts with finest atlas labels: {value}: {evidence}")
            self.value_to_side[int(value)] = max(evidence, key=evidence.get)
            self.contingency[str(int(value))] = evidence
        if set(self.value_to_side.values()) != {"L", "R"}:
            raise ValueError("Hemisphere reference must independently identify both hemispheres")
        self.volumes = {}
        for level in range(1, 7):
            labels, counts = np.unique(self.data[..., level - 1], return_counts=True)
            self.volumes[level] = {int(label): int(count) * grid.voxel_volume_mm3
                                   for label, count in zip(labels, counts)}

    def lookup(self, voxel):
        """Use the terminal kernel's voxel directly; never round coordinates."""
        if voxel is None:
            return "Unknown", [self.target(level, None, "out_of_FOV") for level in range(1, 7)]
        point = tuple(voxel)
        if (len(point) != 3 or any(not isinstance(value, (int, np.integer)) for value in point)
                or any(value < 0 or value >= self.data.shape[axis] for axis, value in enumerate(point))):
            raise ValueError("Endpoint voxel must be an in-reference integer triple or None")
        side = self.value_to_side.get(int(self.hemisphere[point]), "Unknown")
        targets = []
        for level in range(1, 7):
            index = int(self.data[point + (level - 1,)])
            metadata = self.labels.get(index)
            status = "zero_unassigned" if index == 0 else (
                "unmapped_label" if metadata is None else (
                    "mapped" if metadata["first"] <= level <= metadata["last"] else "key_level_conflict"))
            target = self.target(level, index, status)
            target["hemisphere_conflict"] = (side in ("L", "R") and target["label_side"] in ("L", "R")
                                               and side != target["label_side"])
            targets.append(target)
        return side, targets

    def target(self, level, index, status):
        metadata = self.labels.get(index)
        abbreviation = metadata["abbreviation"] if metadata else None
        return {
            "level": level, "index": index, "abbreviation": abbreviation,
            "full_name": metadata["full_name"] if metadata else None, "target_status": status,
            "label_side": LateralityParser.get_side(abbreviation), "hemisphere_conflict": False,
            "target_volume_mm3": self.volumes[level].get(index) if status == "mapped" else None,
        }
