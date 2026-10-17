"""Compatibility wrapper for the packaged DS00 montage renderer.

Version: 1.9.1 (local/unreleased; VisualToolkit package wrapper)
Last updated: 2026-10-17
Change history: main_scripts/Visual_toolkit/CHANGELOG.md

The implementation now lives in ``Visual_toolkit.montage``. Existing scripts
that import ``ch00ds_5view_montage`` continue to work through this wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MAIN_SCRIPTS = Path(__file__).resolve().parents[1] / "main_scripts"
if str(_MAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_MAIN_SCRIPTS))

from Visual_toolkit.montage import *  # noqa: E402,F401,F403
from Visual_toolkit.montage import main  # noqa: E402


if __name__ == "__main__":
    main()
