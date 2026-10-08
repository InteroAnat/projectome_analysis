"""
Centralized definition of "insula" label set, queried from the ARM atlas
(NMT v2.1 macaque) the same way main_scripts/region_analysis/getNeuronListByRegion.py
does it: any abbreviation whose Full_Name contains "insula", plus a few
neighboring ambiguous labels (Ri retroinsula treated as borderline; PrCO
explicitly excluded but flagged as a rescue candidate).

Returns sets of canonical UPPERCASE base labels (after stripping CL_/CR_).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
import pandas as pd

_MAIN_SCRIPTS = str(Path(__file__).resolve().parents[2] / "main_scripts")
if _MAIN_SCRIPTS not in sys.path:
    sys.path.insert(0, _MAIN_SCRIPTS)
from region_labels import (CURATED_INSULA_LABELS, SOMA_REGION_PREFIXES,
                           strip_region_prefix, normalize_region_label,
                           is_explicit_unknown_label)

ARM_KEY = r"D:\projectome_analysis\atlas\ARM_key_all.txt"


def strip_prefix(s: str) -> str:
    return strip_region_prefix(s)


def build_insula_label_set(atlas_path: str = ARM_KEY,
                           include_retroinsula: bool = False
                           ) -> tuple[set[str], set[str]]:
    """
    Returns (insula_labels_upper, rescue_labels_upper).

    insula_labels_upper: canonical UPPERCASE sub-region abbreviations from
        the atlas (any Full_Name containing 'insula') plus the manually-
        curated finer labels used by the 251637 reference table
        (IAL/IAPM/IDD5/IDM/IDV). These manual labels are retained source
        annotations; their fine anatomical crosswalk to CHARM is unresolved.

    rescue_labels_upper: labels eligible for coordinate-based rescue
        for review, without anatomical acceptance. Includes PrCO, Unknown,
        empty string, and unmapped sentinels.
    """
    atlas = pd.read_csv(atlas_path, sep="\t")
    fn = atlas["Full_Name"].fillna("")
    abbr = atlas["Abbreviation"].fillna("")
    mask = fn.str.contains("insula", case=False, regex=False)
    hits = atlas.loc[mask, "Abbreviation"].dropna().tolist()

    base = set()
    for a in hits:
        s = strip_prefix(a)
        # Split slash-combinations like "Ia/Id" into both members but also
        # keep the combined form (atlas does emit literal "Ia/Id").
        for part in re.split(r"\s*/\s*", s):
            if part:
                base.add(part.upper())
        base.add(s.upper())

    # 251637 manual finer labels, not an established CHARM crosswalk.
    # These are the sub-region names found in
    # neuron_tables_new/251637_INS_HE_inferred.xlsx Summary sheet.
    base.update(CURATED_INSULA_LABELS)

    if not include_retroinsula:
        base.discard("RI")
        base.discard("RETROINSULA")

    rescue = {"PRCO", "UNKNOWN", "UNKNOWN_0", "_UNMAPPED", "INSULAUNKNOWN", ""}
    return base, rescue


def normalize_label(s) -> str:
    """Canonical UPPERCASE label after prefix strip."""
    return normalize_region_label(s)


if __name__ == "__main__":
    insula, rescue = build_insula_label_set()
    print(f"Insula labels ({len(insula)}):")
    for x in sorted(insula):
        print(f"  {x}")
    print(f"\nRescue labels ({len(rescue)}):")
    for x in sorted(rescue):
        print(f"  {x or '<empty>'}")
