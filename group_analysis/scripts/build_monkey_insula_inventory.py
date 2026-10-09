"""Produce the legacy-overview-backed per-monkey inventory in a fresh directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from group_analysis.data_progress.insula_inventory import main

if __name__ == "__main__":
    main()
