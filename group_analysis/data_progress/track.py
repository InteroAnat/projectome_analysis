from __future__ import annotations

from .paths import DOCS, FIGURES, LEGACY_FIGURE_GLOBS, LEGACY_TABLE_NAMES, stamped_outputs
from .plot import plot_progress_table
from .table import append_total_row, build_progress_table


def remove_legacy_outputs() -> list[str]:
    """Drop superseded dataset_status figures and unstamped data_progress files."""
    removed: list[str] = []
    for pattern in LEGACY_FIGURE_GLOBS:
        for path in FIGURES.glob(pattern):
            path.unlink(missing_ok=True)
            removed.append(str(path))
    for name in LEGACY_TABLE_NAMES:
        path = DOCS / name
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def run(write_figure: bool = True, cleanup_legacy: bool = True) -> dict:
    progress, totals = build_progress_table()
    display = append_total_row(progress, totals)

    out_table, out_figure = stamped_outputs(totals["file_stamp"])

    out_table.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(out_table, index=False, encoding="utf-8-sig")

    if write_figure:
        out_figure.parent.mkdir(parents=True, exist_ok=True)
        plot_progress_table(display, totals, out_figure)

    removed = remove_legacy_outputs() if cleanup_legacy else []

    return {
        "table": out_table,
        "figure": out_figure if write_figure else None,
        "totals_dict": totals,
        "removed_legacy": removed,
    }


def main() -> None:
    result = run(write_figure=True)
    t = result["totals_dict"]
    print("Data progress track updated:")
    print(f"  table:  {result['table']}")
    print(f"  figure: {result['figure']}")
    if result["removed_legacy"]:
        print("  removed legacy outputs:")
        for path in result["removed_legacy"]:
            print(f"    - {path}")
    print(
        f"  summary @ {t['generated_at']}: "
        f"insula {t['insula_corrected']}/{t['insula_expected_plan']} corrected; "
        f"ION {t['ion_traced']}/{t['analysis_expected']}; "
        f"combined {t['insula_in_combined']}; "
        f"5µm {t['five_um_ready']}/{t['monkeys']}"
    )


if __name__ == "__main__":
    main()
