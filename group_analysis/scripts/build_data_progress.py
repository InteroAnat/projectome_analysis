#!/usr/bin/env python
"""Build data progress table + figure from dataset_status_manifest.csv."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from group_analysis.data_progress.track import main

if __name__ == "__main__":
    main()
