from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES

C_HEADER = "#1a1a1a"
C_EVEN = "#ffffff"
C_ODD = "#f5f5f5"
C_GROUP = {
    "meta": ("#eceff1", "#e0e0e0"),
    "counts": ("#e3f2fd", "#bbdefb"),
    "distribution": ("#fff8e1", "#ffe082"),
    "five_um": ("#e8f5e9", "#c8e6c9"),
}
STATUS = {
    "Completed": "#c8e6c9",
    "Pending": "#bbdefb",
    "re-tracking": "#ffe0b2",
    "tracking": "#fff9c4",
}

# (field, header, group)
TABLE_COLS = [
    ("monkey_id", "Monkey", "meta"),
    ("fmost_id", "fMOST", "meta"),
    ("data_status", "Status", "meta"),
    ("ion_n_traced", "ION", "counts"),
    ("insula_corrected_n", "Ins QC", "counts"),
    ("insula_in_combined_n", "Combined", "counts"),
    ("insula_distribution_combined", "Insula distribution", "distribution"),
    ("insula_L_R_combined", "L/R", "distribution"),
    ("five_um_local", "5 µm", "five_um"),
]


def _status_bg(s: str) -> str:
    return STATUS.get(str(s).strip(), "#eeeeee")


def _fmt(val) -> str:
    if pd.isna(val) or val == "":
        return "—"
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def _group_cell_color(group: str, row_idx: int, is_total: bool) -> str:
    if is_total:
        return "#e0e0e0"
    even, odd = C_GROUP[group]
    return even if row_idx % 2 == 0 else odd


def _plot_totals(ax, totals: dict) -> None:
    cats = ["Insula corrected", "All ION traced"]
    exp = [totals["insula_expected_plan"], totals["analysis_expected"]]
    real = [totals["insula_corrected"], totals["ion_traced"]]
    y = np.arange(2)
    h = 0.32
    ax.barh(y + h / 2, exp, height=h, color="#90caf9", label="Expected")
    ax.barh(y - h / 2, real, height=h, color="#ef5350", label="Reality")
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=10)
    ax.set_title("Plan vs reality (8-monkey cohort)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    xmax = max(max(exp), max(real)) * 1.15 + 20
    ax.set_xlim(0, xmax)
    for yi, e, r in zip(y, exp, real):
        ax.text(e, yi + h / 2, f" {e}", va="center", fontsize=9, color="#1565c0")
        ax.text(r, yi - h / 2, f" {r}", va="center", fontsize=9, color="#c62828")


def _plot_distribution(ax, progress: pd.DataFrame) -> None:
    d = progress[progress["monkey_id"] != "TOTAL"].copy()
    d = d[d["insula_in_combined_n"] > 0].sort_values("insula_in_combined_n", ascending=True)
    if d.empty:
        ax.text(0.5, 0.5, "No combined insula rows", ha="center", va="center")
        ax.axis("off")
        return
    labels = d.apply(lambda r: f"{r['monkey_id']}/{_fmt(r['fmost_id'])}", axis=1)
    vals = d["insula_in_combined_n"].astype(int)
    ax.barh(np.arange(len(d)), vals, color="#ffb74d", edgecolor="white")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Neurons in combined table")
    ax.set_title("Insula cohort size (combined xlsx)", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def plot_progress_table(display: pd.DataFrame, totals: dict, out: Path | None = None) -> None:
    """Render integrated progress table figure."""
    if out is None:
        out = FIGURES / "data_progress_table.png"
    view = display.copy()
    n = len(view)
    fig_h = 3.0 + 0.45 * n
    fig = plt.figure(figsize=(20, fig_h), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.22, 0.78], width_ratios=[1, 1], hspace=0.3, wspace=0.25)
    _plot_totals(fig.add_subplot(gs[0, 0]), totals)
    _plot_distribution(fig.add_subplot(gs[0, 1]), view[view["monkey_id"] != "TOTAL"])

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")

    headers = [h for _, h, _ in TABLE_COLS]
    col_groups = [g for _, _, g in TABLE_COLS]
    rows = [[_fmt(r[c]) for c, _, _ in TABLE_COLS] for _, r in view.iterrows()]
    widths = [0.06, 0.07, 0.08, 0.05, 0.05, 0.06, 0.32, 0.06, 0.08]

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.05, 0.98, 0.9],
        colWidths=widths,
    )
    tbl.auto_set_font_size(False)
    dist_i = headers.index("Insula distribution")
    ins_i = headers.index("Ins QC")
    status_i = headers.index("Status")

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        if row == 0:
            cell.set_facecolor(C_HEADER)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            continue
        data = view.iloc[row - 1]
        is_total = data["monkey_id"] == "TOTAL"
        group = col_groups[col]
        cell.set_facecolor(_group_cell_color(group, row, is_total))
        if is_total:
            cell.set_text_props(fontweight="bold", fontsize=9)
        elif col == dist_i:
            cell.get_text().set_ha("left")
            cell.get_text().set_fontsize(7)
        elif col == ins_i:
            cell.set_text_props(fontweight="bold", fontsize=9)
        elif col == status_i and not is_total:
            cell.set_facecolor(_status_bg(data["data_status"]))
        else:
            cell.set_text_props(fontsize=8)

    ts = totals.get("generated_at", "")
    fig.suptitle(
        f"Data progress — insula multi-monkey cohort\nGenerated {ts}",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white", pad_inches=0.35)
    plt.close(fig)
