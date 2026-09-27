"""
Central cohort / geometry constants for multi-monkey insula projectome.

Import the same way as ``insula_label_set`` (SCRIPTS on ``sys.path``).

Env overrides
-------------
PROJECTOME_SAMPLES
    Comma-separated SampleID list replacing ``NEW_SAMPLES``.
"""
from __future__ import annotations

import os

REFERENCE_SAMPLE = "251637"

# NMT v2.1 NII: Soma_Phys = 250 µm × Soma_NII → 0.25 mm / voxel.
# Midline at Phys X = 32000 µm ↔ NII X ≈ 128.
NII_VOXEL_MM = 0.25
MIDLINE_NII_X = 128.0

# New / non-reference samples included in rebuilds. Includes 252527/252718
# (already in canonical combined) plus portal Phase-0 candidates.
_DEFAULT_NEW_SAMPLES = [
    "251730",
    "252383",
    "252384",
    "252385",
    "252527",
    "252718",
    "252790",
    "250432",
    "252714",
]


def _samples_from_env() -> list[str] | None:
    raw = os.environ.get("PROJECTOME_SAMPLES", "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


NEW_SAMPLES: list[str] = _samples_from_env() or list(_DEFAULT_NEW_SAMPLES)

# Atlas leaves kept in the cohort without remapping to manual insula labels.
# Gustatory cortex "G" was historically kept as auto_atlas_insula; leave as G
# (do NOT map to IG — user rule 2026-09-26).
KEEP_ATLAS_LABELS = {"G"}


def fold_nii_x(x: float, midline: float = MIDLINE_NII_X) -> float:
    """Hemisphere-folded ML distance from midline (voxels)."""
    return abs(float(x) - float(midline))


def pad_tol_mm(default: float = 2.0) -> float:
    """Padding tolerance in mm; env ``PROJECTOME_PAD_TOL_MM`` overrides."""
    raw = os.environ.get("PROJECTOME_PAD_TOL_MM", "").strip()
    if not raw:
        return float(default)
    return float(raw)


def pad_tol_vox(default_mm: float = 2.0, voxel_mm: float = NII_VOXEL_MM) -> float:
    """Padding tolerance converted to NII voxels."""
    return pad_tol_mm(default_mm) / float(voxel_mm)
