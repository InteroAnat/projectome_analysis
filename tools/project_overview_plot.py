#!/usr/bin/env python
"""Refresh the project overview for every cohort group.

This command calls ``project_overview_plot``. That function parses the live
macaque_fMOST Progress Details table, Monkey Data, and the analysis neuron
table, then writes a new table and figure under ``group_analysis/data_progress``.
Usage: tools/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from group_analysis.project_overview.track import main

if __name__ == "__main__":
    main()
