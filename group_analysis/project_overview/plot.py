"""Draw the insula progress overview.

The command for every cohort group is ``tools/project_overview_plot.py``.
See ``tools/README.md``. ``project_overview_plot`` parses the live table
and the local sources, saves the table, then calls ``plot_progress_table``
on that workbook.
"""
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
# Monkey Data fills. Applied only when the status cell is exactly that word.
STATUS = {
    "Completed": "#C6EFCE",
    "Pending": "#B8CCE4",
    "re-tracking": "#ffe0b2",
    "tracking": "#fff9c4",
}

# (field, header, group) — July progress table, used when insula recount columns are absent.
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

# Human columns for the progress overview once checked insula counts are on the table.
# insula_8_cohort is a cohort mark. It is not a status colour.
OVERVIEW_COLS = [
    ("monkey_id", "Animal", "meta"),
    ("fmost_id", "fMOST", "meta"),
    ("insula_8_cohort", "Cohort", "meta"),
    ("selected_n", "Selected", "counts"),
    ("neurons_screened", "Screened\n(monkey all region)", "counts"),
    ("ARM_insula_n", "ARM\ninsula", "counts"),
    ("added_n", "Added", "counts"),
    ("potential_insula_auto_n", "Potential insula\n(auto rule based)", "counts"),
    ("selected_by_reason", "Selected by", "meta"),
    ("spatial_only_n", "Spatial\nonly", "counts"),
    ("both_n", "ARM and\nspatial", "counts"),
    ("Henry_coarse_INS_n", "Henry", "counts"),
    ("five_um_local", "5 µm", "meta"),
    ("data_status", "Status", "meta"),
]

# Distinct from Completed green and Pending blue.
COHORT_EIGHT = "#D1C4E9"
COHORT_OTHER = "#F5F5F5"
COHORT_EDGE = "#5E35B1"


def _columns_for(display: pd.DataFrame) -> list[tuple[str, str, str]]:
    needed = [name for name, _, _ in OVERVIEW_COLS]
    if all(name in display.columns for name in needed):
        return OVERVIEW_COLS
    return TABLE_COLS


def _status_bg(s: str) -> str | None:
    """Return a status fill only for a known status word. Other text keeps the column colour."""
    return STATUS.get(str(s).strip())


def _use_cjk_font() -> str:
    """Prefer a Windows font that can draw Chinese animal names from the sheet."""
    from matplotlib import font_manager

    for name in ("Microsoft YaHei", "SimHei", "Microsoft JhengHei"):
        if any(font.name == name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return ""


def _fmt(val) -> str:
    if pd.isna(val) or val == "":
        return "—"
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


# Ledger EvidenceStratum -> short figure label. Full text is in selected_neuron_potential_insula.csv.
REASON_SHORT = {
    "HumanINS": "Henry",
    "AtlasINS": "Atlas",
    "Candidate": "Box",
    "G_exploratory": "G",
    "OriginCandidate": "Origin",
    "NearINS": "Near",
}
REASON_KEY = (
    "Selected by: Henry = Henry visual insula; Atlas = ARM insula label; Box = folded 251637 box; "
    "G = gustatory, exploratory; Origin = insula at one soma-origin lookup;\n"
    "Near = same-sample insula neuron within 2 mm. "
    "Added = Selected minus ARM insula. Potential insula (auto rule based) = added soma within 2 mm of an ARM insula voxel."
)


def _short_reasons(value) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "—"
    parts = []
    for part in text.split("; "):
        name, _, count = part.partition(" ")
        parts.append(f"{REASON_SHORT.get(name, name)} {count}".strip())
    if len(parts) > 4:
        return " · ".join(parts[:3]) + "\n" + " · ".join(parts[3:])
    return " · ".join(parts)


def _cohort_mark(value) -> str:
    text = str(value).strip()
    if text == "DEB lab":
        return "DEB lab"
    if text == "Other lab":
        return "Other lab"
    return _fmt(value)


# Attached plan image, 2026-10-12. group_analysis/docs/injection_plan_reference.md
# was checked and does not match these cells (it gives 936 a vAIC R and omits 900 vAIC L).
# Empty cells stay empty. Footer numbers are the sheet's printed totals.
DEB_INJECTION_COLUMNS = [
    "Subject",
    "vAIC L",
    "vAIC R",
    "Cingulate",
    "IDFA L",
    "IDFA R",
    "IDFM L",
    "IDFM R",
    "IDFP L",
    "IDFP R",
]
DEB_INJECTION_ROWS = [
    ["936", "1", "", "2(L+R)", "", "", "", "", "1", "1"],
    ["605", "", "", "", "1", "1", "", "", "1", "1"],
    ["900", "1", "", "", "1", "1", "", "", "1", "1"],
    ["945", "1", "1", "", "", "", "1", "1", "", ""],
    ["797", "1", "", "", "", "", "1", "1", "1", ""],
    ["948", "1", "", "", "", "", "1", "1", "", ""],
    ["331", "1", "", "", "", "", "1", "", "", "1"],
    ["631", "", "", "", "1", "1", "", "", "", ""],
]
DEB_INJECTION_FOOTER = [
    "Plan",
    "14 injections",
    "1050 cells",
    "3% VENs",
    "6 injections",
    "450",
    "6 injections",
    "450",
    "6 injections",
    "450",
]
# Column groups: 0 subject, 1-3 VEN, 4-5 FIDA, 6-7 FIDI, 8-9 FIDP.
_PLAN_HEADER_COLORS = {
    0: "#EEEEEE",
    1: "#A5D6A7",
    2: "#A5D6A7",
    3: "#A5D6A7",
    4: "#F8CBAD",
    5: "#F8CBAD",
    6: "#9BC2E6",
    7: "#9BC2E6",
    8: "#BDD7EE",
    9: "#BDD7EE",
}
_PLAN_CELL_COLORS = {
    1: "#E8F5E9",
    2: "#E8F5E9",
    3: "#E8F5E9",
    4: "#FDE9D9",
    5: "#FDE9D9",
    6: "#DDEBF7",
    7: "#DDEBF7",
    8: "#DEEBF7",
    9: "#DEEBF7",
}


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
    ax.set_title("Plan vs reality (DEB lab)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    xmax = max(max(exp), max(real)) * 1.28
    ax.set_xlim(0, xmax)
    for yi, e, r in zip(y, exp, real):
        ax.text(e, yi + h / 2, f"  {int(e)}", va="center", fontsize=13, color="#1565c0", fontweight="bold")
        ax.text(max(r, 1), yi - h / 2, f"  {int(r)}", va="center", fontsize=13, color="#c62828", fontweight="bold")
    ax.set_xlabel(
        "ION expected 2429 = July plan expectation. Insula expected 2400 = cell target. Reality is the DEB lab recount only.",
        fontsize=11,
        labelpad=8,
    )


def _plot_distribution(ax, progress: pd.DataFrame) -> None:
    d = progress[progress["monkey_id"] != "TOTAL"].copy()
    if "insula_8_cohort" in d.columns:
        d = d[d["insula_8_cohort"].astype(str).eq("DEB lab")]
    # DEB lab ARM counts only. 747 stays off this chart.
    value_col = "cohort_bar_n" if "cohort_bar_n" in d.columns else "insula_in_combined_n"
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
    d = d[d[value_col] > 0].sort_values(value_col, ascending=True)
    if d.empty:
        ax.text(0.5, 0.5, "No insula cohort rows", ha="center", va="center")
        ax.axis("off")
        return
    labels = d.apply(lambda r: f"{r['monkey_id']}/{_fmt(r['fmost_id'])}", axis=1)
    vals = d[value_col].astype(int)
    ax.barh(np.arange(len(d)), vals, color="#ffb74d", edgecolor="white", height=0.7)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(labels, fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(0, max(vals) * 1.18 if len(vals) else 1)
    for yi, value in enumerate(vals):
        ax.text(value, yi, f"  {int(value)}", va="center", fontsize=12, color="#e65100", fontweight="bold")
    ax.set_title("Insula cohort size (DEB lab)", fontsize=16, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    cohort = progress[progress["monkey_id"].astype(str) != "TOTAL"].copy()
    if "insula_8_cohort" in cohort.columns:
        cohort = cohort[cohort["insula_8_cohort"].astype(str).eq("DEB lab")]
    screened = int(pd.to_numeric(cohort.get("neurons_screened"), errors="coerce").fillna(0).sum()) if len(cohort) else 0
    arm = int(pd.to_numeric(cohort.get("ARM_insula_n"), errors="coerce").fillna(0).sum()) if len(cohort) else 0
    ax.set_xlabel(
        f"DEB lab: {len(cohort)} monkeys; screened {screened}; ARM insula {arm}; plan target 2400. 747 is not in this summary.",
        fontsize=12,
        labelpad=8,
    )


# One row per Done monkey. Columns line up with the sheet.
# 605 and 900 have no IDFP mark. 797 has IDFM L only. 331 has IDFM L and IDFP R.
_PLAN_HEADER = ["Subject", "vAIC L", "vAIC R", "Cingulate", "IDFA L", "IDFA R", "IDFM L", "IDFM R", "IDFP L", "IDFP R"]
_PLAN_BODY = [
    ["936", "1", "", "2(L+R)", "", "", "", "", "1", "1"],
    ["605", "", "", "", "1", "1", "", "", "", ""],
    ["900", "1", "", "", "1", "1", "", "", "", ""],
    ["945", "1", "1", "", "", "", "1", "1", "", ""],
    ["797", "1", "", "", "", "", "1", "", "1", ""],
    ["948", "1", "", "", "", "", "1", "1", "", ""],
    ["331", "1", "", "", "", "", "1", "", "", "1"],
    ["631", "", "", "", "1", "1", "", "", "", ""],
    ["Injections", "14", "", "", "6", "", "6", "", "6", ""],
    ["Expected cells", "1050", "", "3% VENs", "450", "", "450", "", "450", ""],
    ["Expected total", "", "", "", "2400", "", "", "", "", ""],
]


def _plot_injection_plan(ax) -> None:
    """Aligned DEB injection grid. Hypothesis text stays above the columns."""
    ax.axis("off")
    ax.set_title(
        "DEB lab injection plan    VEN injections (green)    Interoceptive cortex injections (FIDA / FIDI / FIDP)",
        fontsize=15,
        loc="left",
        fontweight="bold",
        pad=12,
    )
    table = ax.table(
        cellText=_PLAN_BODY,
        colLabels=_PLAN_HEADER,
        loc="upper center",
        cellLoc="center",
        bbox=[0.0, 0.08, 1.0, 0.86],
        colWidths=[0.12, 0.09, 0.09, 0.12, 0.09, 0.09, 0.10, 0.10, 0.10, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    ven = {1, 2, 3}
    fida = {4, 5}
    fidi = {6, 7}
    fidp = {8, 9}
    n_body = len(_PLAN_BODY)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#757575")
        cell.set_height(1 / (n_body + 2))
        if row == 0:
            if col in ven:
                cell.set_facecolor("#2E7D32")
            elif col in fida:
                cell.set_facecolor("#E65100")
            elif col in fidi or col in fidp:
                cell.set_facecolor("#1565C0")
            else:
                cell.set_facecolor("#212121")
            cell.set_text_props(color="white", fontweight="bold", fontsize=12)
            continue
        text = _PLAN_BODY[row - 1][col]
        footer = row > 8
        if col in ven:
            cell.set_facecolor("#C8E6C9" if text else "#F1F8E9")
        elif col in fida:
            cell.set_facecolor("#FFCCBC" if text else "#FFF3E0")
        elif col in fidi:
            cell.set_facecolor("#BBDEFB" if text else "#E3F2FD")
        elif col in fidp:
            cell.set_facecolor("#B3E5FC" if text else "#E1F5FE")
        else:
            cell.set_facecolor("#FFF8E1" if footer else "#FAFAFA")
        cell.set_text_props(fontsize=14 if not footer else 12, fontweight="bold")
    ax.text(
        0.0,
        0.02,
        "Footer as printed on the sheet: 14 VEN injections, 1050 cells, hoping to get 3% of VENs; 6 injections and 450 cells under FIDA, FIDI, and FIDP; expected total 2400.",
        transform=ax.transAxes,
        fontsize=11,
        color="#424242",
    )


def plot_progress_table(display: pd.DataFrame, totals: dict, out: Path | None = None) -> None:
    """Draw the overview from a table already saved by project_overview_plot.

    Insula counts and the insula-eight cohort flag must already be on ``display``.
    They come from ``analysis_table.load_analysis_neuron_table`` (462-neuron
    ledger plus per_animal_insula_summary.csv). A change to that table has
    to be passed through project_overview_plot before this plot is redrawn.
    This function does not read the July combined workbook.
    """
    if out is None:
        out = FIGURES / "data_progress_table.png"
    view = display.copy()
    columns = _columns_for(view)
    _use_cjk_font()
    n = len(view)
    chart_h, plan_h, row_h = 4.8, 6.2, 1.05
    fig_h = chart_h + plan_h + row_h * n + 2.0
    fig_w = 28
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[chart_h, plan_h, row_h * n],
        width_ratios=[1.05, 1],
        hspace=0.38,
        wspace=0.22,
    )
    _plot_totals(fig.add_subplot(gs[0, 0]), totals)
    _plot_distribution(fig.add_subplot(gs[0, 1]), view[view["monkey_id"] != "TOTAL"])
    _plot_injection_plan(fig.add_subplot(gs[1, :]))

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")

    headers = [h for _, h, _ in columns]
    col_groups = [g for _, _, g in columns]
    rows = []
    for _, record in view.iterrows():
        shown = []
        for key, _, _ in columns:
            if key == "insula_8_cohort":
                shown.append(_cohort_mark(record[key]))
            elif key == "selected_by_reason":
                shown.append(_short_reasons(record[key]))
            else:
                shown.append(_fmt(record[key]))
        rows.append(shown)
    if columns is OVERVIEW_COLS:
        widths = [
            0.048, 0.054, 0.058, 0.052, 0.078, 0.054, 0.050, 0.092,
            0.178, 0.052, 0.058, 0.046, 0.072, 0.108,
        ]
    else:
        widths = [0.06, 0.07, 0.08, 0.05, 0.05, 0.06, 0.32, 0.06, 0.08]

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.005, 0.18, 0.99, 0.80],
        colWidths=widths,
    )
    tbl.auto_set_font_size(False)
    tbl.scale(1.0, 2.35)
    header_index = {label: index for index, label in enumerate(headers)}
    dist_i = header_index.get("Insula distribution")
    ins_i = header_index.get("Ins QC")
    status_i = header_index.get("Status")
    five_i = header_index.get("5 µm")
    cohort_i = header_index.get("Cohort")
    reason_i = header_index.get("Selected by")
    animal_i = header_index.get("Animal", header_index.get("Monkey"))

    body_height = tbl.get_celld()[(1, 0)].get_height()
    row_heights: list[float] = []
    for row_idx, row_cells in enumerate(rows):
        reason_text = ""
        if reason_i is not None and reason_i < len(row_cells):
            reason_text = str(row_cells[reason_i])
        scale = 1.25 if "\n" in reason_text else 1.0
        if view.iloc[row_idx]["monkey_id"] == "TOTAL":
            scale = max(scale, 1.1)
        row_heights.append(scale)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        if row == 0:
            # Two-line headers such as "Potential insula (auto rule based)" need a taller row.
            cell.set_height(body_height * 2.4)
            cell.set_facecolor(C_HEADER)
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
            continue
        data = view.iloc[row - 1]
        is_total = data["monkey_id"] == "TOTAL"
        rscale = row_heights[row - 1]
        cell.set_height(body_height * rscale)
        group = col_groups[col]
        cell.set_facecolor(_group_cell_color(group, row, is_total))
        if is_total:
            cell.set_text_props(fontweight="bold", fontsize=12)
        elif dist_i is not None and col == dist_i:
            cell.get_text().set_ha("left")
            cell.get_text().set_fontsize(11)
        elif ins_i is not None and col == ins_i:
            cell.set_text_props(fontweight="bold", fontsize=13)
        elif cohort_i is not None and col == cohort_i and not is_total:
            in_eight = str(data.get("insula_8_cohort", "")) == "DEB lab"
            cell.set_facecolor(COHORT_EIGHT if in_eight else COHORT_OTHER)
            cell.set_text_props(fontsize=13, fontweight="bold" if in_eight else "normal")
        elif animal_i is not None and col == animal_i and not is_total and str(data.get("insula_8_cohort", "")) == "DEB lab":
            cell.set_edgecolor(COHORT_EDGE)
            cell.set_linewidth(1.8)
            cell.set_text_props(fontsize=13, fontweight="bold")
        elif five_i is not None and col == five_i and not is_total:
            if str(data.get("five_um_local", "")).strip() == "Yes":
                cell.set_facecolor("#C8E6C9")
            cell.set_text_props(fontsize=13)
        elif status_i is not None and col == status_i and not is_total:
            paint = _status_bg(data["data_status"])
            if paint:
                cell.set_facecolor(paint)
            cell.set_text_props(fontsize=13)
        else:
            cell.set_text_props(fontsize=12)
        if reason_i is not None and col == reason_i:
            cell.get_text().set_wrap(True)
            cell.set_text_props(fontsize=9 if is_total else 10, linespacing=1.35)
    if reason_i is not None:
        ax.text(0.005, 0.01, REASON_KEY, transform=ax.transAxes, fontsize=9.5, color="#424242", va="bottom")

    ts = totals.get("generated_at", "")
    fig.suptitle(
        f"Data progress — insula multi-monkey cohort\nGenerated {ts}",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white", pad_inches=0.5)
    plt.close(fig)
