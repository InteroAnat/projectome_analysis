#!/usr/bin/env python
"""Same entry as tools/project_overview_plot.py. Prefer that command."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from group_analysis.project_overview.track import main

if __name__ == "__main__":
    main()
