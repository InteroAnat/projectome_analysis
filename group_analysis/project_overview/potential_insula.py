"""Per-neuron selection reason and potential insula (auto rule based) for the 462 Selected neurons.

Added = Selected neurons that are not atlas insula (``atlas_INS`` false).

Potential insula (auto rule based): an added neuron whose soma is within 2 mm
of an ARM insula voxel. The soma is the neuron's own SWC root
(``own_root_xyz_um``, NMT index = um / 250), looked up on
``ARM_in_NMT_v2.1_sym`` level 6 (slice ``[..., 0, 5]``). Insula labels are the
``build_insula_label_set`` set (Ia/Id, Iai, Ial, Iam/Iapm, Iapl, Ig, Pi, both
sides). Distance is the Euclidean distance transform at 0.25 mm voxels.

The folded 251637 box and the same-sample anchor distance are kept as
reference columns. Selection reason is the ledger ``EvidenceStratum``.
This is a rule flag, not anatomical acceptance.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter

import numpy as np

from .paths import ARM_ATLAS, ARM_KEY, FULL_NEURON_REGION_TABLE, SCRIPTS, SELECTED_LEDGER

ARM_LIMIT_MM = 2.0
ANCHOR_LIMIT_MM = 2.0
VOXEL_MM = 0.25

# EvidenceStratum in the 462 ledger -> plain reason for selection.
SELECTION_REASONS = {
    "HumanINS": "Henry visual insula annotation",
    "AtlasINS": "ARM atlas insula label",
    "Candidate": "spatial candidate (folded 251637 box)",
    "G_exploratory": "gustatory G label, exploratory",
    "OriginCandidate": "insula at one soma-origin lookup only",
    "NearINS": "non-insula label, same-sample insula neuron within 2 mm",
}
REASON_ORDER = list(SELECTION_REASONS)

PER_NEURON_COLUMNS = [
    "AnimalID",
    "SampleID",
    "neuron_id",
    "selection_reason",
    "selection_reason_text",
    "preview_selection_rule",
    "atlas_insula",
    "added",
    "portal_region",
    "soma_ARM_label",
    "ARM_insula_distance_mm",
    "potential_insula_auto",
    "folded_box_hit",
    "nearest_same_sample_insula_mm",
    "nearest_insula_neuron",
]


def _yes(value) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _read(path, delimiter=","):
    csv.field_size_limit(2**31 - 1)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _arm_insula_distance():
    """Return (atlas labels, label names, distance-to-insula map in mm)."""
    import nibabel as nib
    from scipy import ndimage

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from insula_label_set import build_insula_label_set, normalize_label

    insula, _ = build_insula_label_set(str(ARM_KEY))
    names = {int(row["Index"]): row["Abbreviation"] for row in _read(ARM_KEY, delimiter="\t")}
    arm = np.asarray(nib.load(str(ARM_ATLAS)).dataobj)[..., 0, 5]
    ids = [int(i) for i in np.unique(arm) if i and normalize_label(names.get(int(i), "")) in insula]
    if not ids:
        raise ValueError("No ARM insula labels found")
    distance = ndimage.distance_transform_edt(~np.isin(arm, ids), sampling=VOXEL_MM)
    return arm, names, distance


def potential_insula_table() -> dict:
    """Return one row per Selected neuron and counts per (animal, sample)."""
    arm, names, distance = _arm_insula_distance()
    full = {(row["sample"], row["neuron_id"]): row for row in _read(FULL_NEURON_REGION_TABLE)}
    ledger = _read(SELECTED_LEDGER)
    if len(ledger) != 462:
        raise ValueError(f"Selected ledger has {len(ledger)} rows, expected 462")

    rows = []
    counts: dict[tuple[str, str], Counter] = {}
    for neuron in ledger:
        animal = str(neuron["AnimalID"]).strip()
        sample = str(neuron["SampleID"]).strip()
        region = full.get((sample, neuron["neuron_id"]))
        if region is None:
            raise ValueError(f"{sample}|{neuron['neuron_id']} is not in the full neuron region table")
        reason = neuron.get("EvidenceStratum", "")
        if reason not in SELECTION_REASONS:
            raise ValueError(f"{sample}|{neuron['neuron_id']} has unknown EvidenceStratum {reason!r}")
        voxel = np.rint(np.asarray(json.loads(neuron["own_root_xyz_um"]), dtype=float) / 250.0).astype(int)
        if not np.all((voxel >= 0) & (voxel < np.asarray(arm.shape))):
            raise ValueError(f"{sample}|{neuron['neuron_id']} soma is outside the ARM grid")
        arm_mm = float(distance[tuple(voxel)])
        atlas = _yes(neuron["atlas_INS"])
        potential = (not atlas) and arm_mm <= ARM_LIMIT_MM
        raw = str(region.get("nearest_same_exact_sample_INS_anchor_mm", "")).strip()
        anchor = float(raw) if raw not in {"", "None", "nan"} else None
        rows.append({
            "AnimalID": animal,
            "SampleID": sample,
            "neuron_id": neuron["neuron_id"],
            "selection_reason": reason,
            "selection_reason_text": SELECTION_REASONS[reason],
            "preview_selection_rule": neuron.get("PreviewSelectionRule", ""),
            "atlas_insula": atlas,
            "added": not atlas,
            "portal_region": region.get("portal_region", ""),
            "soma_ARM_label": names.get(int(arm[tuple(voxel)]), "0"),
            "ARM_insula_distance_mm": round(arm_mm, 3),
            "potential_insula_auto": potential,
            # bbox_hits is the box alone; potential_INS_candidate is atlas-or-box.
            "folded_box_hit": bool(str(region.get("bbox_hits") or "").strip()),
            "nearest_same_sample_insula_mm": "" if anchor is None else round(anchor, 3),
            "nearest_insula_neuron": region.get("nearest_INS_anchor_id", ""),
        })
        tally = counts.setdefault((animal, sample), Counter())
        tally["added_n"] += not atlas
        tally["potential_insula_auto_n"] += potential
        tally[reason] += 1
    for tally in counts.values():
        tally["selected_by_reason"] = "; ".join(
            f"{name} {tally[name]}" for name in REASON_ORDER if tally[name]
        )
    return {
        "rows": rows,
        "counts": counts,
        "full_neuron_region_table": str(FULL_NEURON_REGION_TABLE),
        "selected_ledger": str(SELECTED_LEDGER),
        "arm_atlas": str(ARM_ATLAS),
        "selection_reasons": SELECTION_REASONS,
        "rule": (
            "Added = Selected minus atlas insula. Potential insula (auto rule based) = added and "
            "own SWC-root soma within 2 mm of an ARM insula voxel (ARM_in_NMT_v2.1_sym level 6, "
            "insula label set, 0.25 mm distance transform). Not anatomical acceptance."
        ),
    }
