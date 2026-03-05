"""
output_manager.py - Organized output folder management.

Folder structure:
    output/
    └── {sample_id}_{timestamp}/
        ├── plots/
        ├── tables/
        ├── reports/
        └── debug/
"""

from datetime import datetime
from pathlib import Path


class OutputManager:
    """Manages organized output folders with timestamp and sample_id."""

    def __init__(
        self,
        base_path: str = None,
        sample_id: str = None,
        create_timestamp: bool = True,
    ):
        self.base_path = Path(base_path or "./output")
        self.sample_id = sample_id or "analysis"

        if create_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{self.sample_id}_{timestamp}_region_analysis"
        else:
            folder_name = self.sample_id

        self.output_dir = self.base_path / folder_name
        self.plots_dir = self.output_dir / "plots"
        self.tables_dir = self.output_dir / "tables"
        self.reports_dir = self.output_dir / "reports"
        self.debug_dir = self.output_dir / "debug"
        self._created = False

    def create_dirs(self):
        if self._created:
            return
        for d in (
            self.output_dir,
            self.plots_dir,
            self.tables_dir,
            self.reports_dir,
            self.debug_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self._created = True
        print(f"[OUTPUT] Created: {self.output_dir}")

    def get_plot_path(self, name: str, ext: str = "png") -> Path:
        self.create_dirs()
        return self.plots_dir / f"{name}.{ext}"

    def get_table_path(self, name: str, ext: str = "xlsx") -> Path:
        self.create_dirs()
        return self.tables_dir / f"{name}.{ext}"

    def get_report_path(self, name: str, ext: str = "txt") -> Path:
        self.create_dirs()
        return self.reports_dir / f"{name}.{ext}"

    def get_debug_path(self, name: str, ext: str = "png") -> Path:
        self.create_dirs()
        return self.debug_dir / f"{name}.{ext}"

    def print_summary(self):
        print("\n" + "=" * 60)
        print("OUTPUT SUMMARY")
        print("=" * 60)
        print(f"Location: {self.output_dir}")
        print(f"  Tables:  {self.tables_dir}")
        print(f"  Reports: {self.reports_dir}")
        print(f"  Plots:   {self.plots_dir}")
        print(f"  Debug:   {self.debug_dir}")
        print("=" * 60)