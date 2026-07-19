#!/usr/bin/env python3
"""Smoke-check Liu 2026 morphology download layout."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MORPH_ROOT = REPO_ROOT / "external" / "liu2026" / "morph" / "NeuroMorph_upload260215"

EXPECTED = {
    "PatchClamp_morph/VENL": 28,
    "PatchClamp_morph/VENS": 24,
    "PatchClamp_morph/PC-L5_ET": 13,
}


def main() -> int:
    if not MORPH_ROOT.is_dir():
        print(f"Missing morph root: {MORPH_ROOT}")
        print("Run: pwsh external/liu2026/scripts/download_morph.ps1")
        return 1

    ok = True
    for subpath, expected in EXPECTED.items():
        folder = MORPH_ROOT / subpath
        count = len(list(folder.glob("*.ASC"))) if folder.is_dir() else 0
        status = "OK" if count == expected else "MISMATCH"
        if count != expected:
            ok = False
        print(f"{status}: {subpath} — {count} ASC (expected {expected})")

    metadata = MORPH_ROOT / "metadata.csv"
    if metadata.is_file():
        print(f"OK: metadata.csv present")
    else:
        print("WARN: metadata.csv missing")
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
