"""Hemisphere reference validated on the declared atlas voxel grid."""
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from region_analysis.laterality import LateralityParser


class AtlasHemisphereReference:
    def __init__(self, mask, atlas, atlas_table, source):
        if mask.ndim != 3 or atlas.shape != mask.shape:
            raise ValueError("Hemisphere reference and selected atlas must share a 3D grid")
        if not source or not str(source).strip():
            raise ValueError("Hemisphere reference requires source provenance")
        if not np.isfinite(mask).all() or not np.equal(mask, np.rint(mask)).all():
            raise ValueError("Hemisphere mask values must be finite integers")
        if atlas_table.Index.duplicated().any():
            raise ValueError("Atlas label indices must be unique")
        if not np.isfinite(atlas).all() or not np.equal(atlas, np.rint(atlas)).all() or atlas.min() < 0:
            raise ValueError("Atlas values must be finite nonnegative integers")
        codes = np.zeros(int(atlas.max()) + 1, dtype=np.uint8)
        for _, row in atlas_table.iterrows():
            index = int(row.Index)
            if 0 <= index < len(codes):
                codes[index] = {"L": 1, "R": 2, "Unknown": 0}[LateralityParser.get_side(row.Abbreviation)]
        sides = codes[atlas.astype(int)]
        self.value_to_side = {}
        self.contingency = {}
        for value in np.unique(mask):
            if value == 0:
                continue
            counts = {side: int(np.count_nonzero((mask == value) & (sides == code)))
                      for side, code in (("L", 1), ("R", 2))}
            if min(counts.values()) or not max(counts.values()):
                raise ValueError(f"Hemisphere mask conflicts with declared atlas labels: {value}: {counts}")
            self.value_to_side[int(value)] = max(counts, key=counts.get)
            self.contingency[int(value)] = counts
        if set(self.value_to_side.values()) != {"L", "R"}:
            raise ValueError("Reference must independently identify both hemispheres")
        self.mask = mask
        self.source = str(source)

    @classmethod
    def from_files(cls, mask_path, atlas_path, key_path, level=6):
        mask_img, atlas_img = nib.load(mask_path), nib.load(atlas_path)
        if mask_img.shape != atlas_img.shape[:3] or not np.allclose(mask_img.affine, atlas_img.affine, rtol=0, atol=1e-7):
            raise ValueError("Hemisphere mask and atlas shape/affine mismatch")
        atlas = np.asanyarray(atlas_img.dataobj)
        if atlas.ndim == 5 and atlas.shape[3] == 1 and 1 <= level <= atlas.shape[4]:
            atlas = atlas[..., 0, level - 1]
        elif atlas.ndim != 3:
            raise ValueError("Unsupported atlas dimensions or hierarchy level")
        sources = []
        for path in (mask_path, atlas_path, key_path):
            path = Path(path)
            sources.append(f"{path}:sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
        return cls(np.asanyarray(mask_img.dataobj), atlas, pd.read_csv(key_path, sep="\t"), " | ".join(sources))

    def sample(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Soma reference lookup requires finite XYZ atlas voxel coordinates")
        voxel = np.rint(xyz).astype(int)
        if not all(0 <= voxel[d] < self.mask.shape[d] for d in range(3)):
            return "Unknown"
        return self.value_to_side.get(int(self.mask[tuple(voxel)]), "Unknown")
