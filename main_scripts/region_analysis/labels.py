"""Exact region-label normalization, independent of atlas/network/GUI imports.

Historical atlas labels retain anatomical numbers (``CL_area_44``). Only
portal labels have a declared extra numeric suffix (``area_44_76``).
Normalization identifies candidates; it does not accept an anatomical label.
"""
import re

# These are the cortical/manual soma prefixes used by the insula pipeline.
# Keep SL_/SR_ explicit: cortical Pi and subcortical pineal Pi are distinct.
SOMA_REGION_PREFIXES = ("CL_", "CR_", "L-", "R-", "L_", "R_")
CURATED_INSULA_LABELS = frozenset({"IAL", "IAPM", "IDD5", "IDM", "IDV", "IDD", "IDI"})
EXPLICIT_UNKNOWN_LABELS = frozenset({"UNKNOWN", "_UNMAPPED", "UNMAPPED", "INSULAUNKNOWN"})


def strip_region_prefix(value):
    """Drop one declared side prefix; preserve the remaining anatomical token."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    for prefix in SOMA_REGION_PREFIXES:
        if text.upper().startswith(prefix):
            return text[len(prefix):]
    return text


def normalize_region_label(value):
    """Canonical historical/manual label; numeric anatomy is unchanged."""
    return strip_region_prefix(value).upper().strip()


def normalize_portal_region_label(value, known_labels=()):
    """Normalize a portal region after removing its one numeric index suffix."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    historical = normalize_region_label(text)
    if historical in known_labels:
        return historical
    head, separator, tail = text.rpartition("_")
    if separator and tail.isdigit():
        text = head
    return normalize_region_label(text)


def is_explicit_unknown_label(value):
    """Recognize explicit Unknown sentinels, including the atlas ``Unknown_0``.

    Blank/absent metadata is not an explicit atlas Unknown assignment.
    """
    label = normalize_region_label(value)
    return label in EXPLICIT_UNKNOWN_LABELS or bool(re.fullmatch(r"UNKNOWN_\d+", label))
