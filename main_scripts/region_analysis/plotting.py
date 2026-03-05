"""
plotting.py - All visualization functions.

Each function:
  - Works standalone on any DataFrame
  - Accepts save_path for figure, save_report_path for text
  - Returns figure or report string
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from region_analysis.utils import parse_terminal_regions


NEURON_TYPE_COLORS = {
    "PT": "#d62728",
    "CT": "#2ca02c",
    "ITc": "#9467bd",
    "ITs": "#e377c2",
    "ITi": "#17becf",
    "Unclassified": "#7f7f7f",
}


# ======================================================================
# Soma distribution
# ======================================================================
def plot_soma_distribution_df(
    df: pd.DataFrame,
    title: str = "Summary of Soma Distribution by Region",
    figsize: tuple = (10, 6),
    save_path: str = None,
    show: bool = True,
):
    if df.empty or "Soma_Region" not in df.columns:
        print("DataFrame is empty or missing 'Soma_Region'")
        return None
    soma_counts = df["Soma_Region"].value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    soma_counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Number of Neurons")
    plt.xticks(rotation=45)
    for i, h in enumerate(soma_counts):
        ax.text(i, h, str(h), ha="center", va="bottom")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if show:
        plt.show()
    return fig


# ======================================================================
# Type distribution
# ======================================================================
def plot_type_distribution_df(
    df: pd.DataFrame,
    title: str = None,
    figsize: tuple = (8, 8),
    save_path: str = None,
    show: bool = True,
):
    if df.empty or "Neuron_Type" not in df.columns:
        print("DataFrame is empty or missing 'Neuron_Type'")
        return None
    counts = df["Neuron_Type"].value_counts()
    colors = [NEURON_TYPE_COLORS.get(x, "#333333") for x in counts.index]
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct=lambda p: f"{p:.1f}%\n({int(p / 100.0 * counts.sum())})",
        startangle=140,
        colors=colors,
        pctdistance=0.85,
        explode=[0.02] * len(counts),
    )
    ax.add_artist(plt.Circle((0, 0), 0.70, fc="white"))
    ax.set_title(title or f"Neuron Type Distribution (N={len(df)})", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if show:
        plt.show()
    return fig


# ======================================================================
# Terminal distribution + report
# ======================================================================
def plot_terminal_distribution_df(
    df: pd.DataFrame,
    top_n: int = 20,
    title: str = None,
    exclude_unknown: bool = True,
    show_pie: bool = True,
    figsize: tuple = (16, 6),
    save_path: str = None,
    save_report_path: str = None,
    show: bool = True,
) -> str:
    if df.empty or "Terminal_Regions" not in df.columns:
        msg = "DataFrame is empty or missing 'Terminal_Regions'"
        print(msg)
        return msg

    df = df.copy()
    df["Terminal_Regions"] = df["Terminal_Regions"].apply(parse_terminal_regions)
    df["_known"] = df["Terminal_Regions"].apply(
        lambda x: sum(1 for r in x if "Unknown" not in str(r))
    )
    df["_unk"] = df["Terminal_Regions"].apply(
        lambda x: sum(1 for r in x if "Unknown" in str(r))
    )
    total_known = int(df["_known"].sum())
    total_unknown = int(df["_unk"].sum())
    total_sites = total_known + total_unknown
    n_with_unk = int((df["_unk"] > 0).sum())
    n_only_known = int((df["_unk"] == 0).sum())
    total_n = len(df)

    if show_pie:
        fig, axes = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.5]}
        )
        ax_pie, ax_bar = axes
    else:
        fig, ax_bar = plt.subplots(figsize=(figsize[0] * 0.6, figsize[1]))

    if show_pie:
        vals = [total_known, total_unknown]
        labs = [f"Known\n({total_known})", f"Unknown\n({total_unknown})"]
        wedges, texts, autotexts = ax_pie.pie(
            vals,
            labels=labs,
            autopct=lambda p: f"{p:.1f}%\n({int(p / 100 * sum(vals))})",
            startangle=90,
            colors=["#2E8B57", "#FF6B6B"],
            explode=[0.02, 0.05],
            shadow=True,
            textprops={"fontsize": 10},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
            at.set_fontsize(11)
        ax_pie.set_title(
            f"Site Distribution (N={total_n})", fontsize=11, fontweight="bold", pad=15
        )

    exploded = df.explode("Terminal_Regions")
    if exclude_unknown:
        orig = len(exploded)
        exploded = exploded[
            ~exploded["Terminal_Regions"].str.contains("Unknown", na=False)
        ]
        unk_excl = orig - len(exploded)
    else:
        unk_excl = 0

    counts = exploded["Terminal_Regions"].value_counts().head(top_n)
    counts.plot(kind="bar", ax=ax_bar, color="steelblue", edgecolor="black")
    for i, v in enumerate(counts.values):
        ax_bar.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=10)

    bar_title = title or (
        f"Top {top_n} Terminal Regions (Excl. Unknown)"
        if exclude_unknown
        else f"Top {top_n} Terminal Regions"
    )
    ax_bar.set_title(bar_title, fontsize=12, fontweight="bold")
    ax_bar.set_xlabel("Target Region", fontsize=11)
    ax_bar.set_ylabel("Number of Neurons Projecting", fontsize=11)
    ax_bar.tick_params(axis="x", rotation=45)
    if exclude_unknown and unk_excl > 0:
        ax_bar.text(
            0.5,
            -0.15,
            f'{unk_excl} "Unknown" entries excluded',
            transform=ax_bar.transAxes,
            ha="center",
            fontsize=9,
            style="italic",
            color="red",
        )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if show:
        plt.show()

    # Report
    lines = [
        "=" * 60,
        "TERMINAL REGION DISTRIBUTION REPORT",
        "=" * 60,
        "",
        "--- Known vs Unknown Statistics ---",
        f"  Total projection sites: {total_sites}",
        f"  Known sites: {total_known} ({total_known / max(total_sites, 1) * 100:.1f}%)",
        f"  Unknown sites: {total_unknown} ({total_unknown / max(total_sites, 1) * 100:.1f}%)",
        f"  Neurons with unknown regions: {n_with_unk} ({n_with_unk / total_n * 100:.1f}%)",
        f"  Neurons with only known regions: {n_only_known} ({n_only_known / total_n * 100:.1f}%)",
        "",
        "--- Terminal Region Distribution ---",
        f"  Total unique regions shown: {len(counts)}",
        f"  Total projection entries (after filtering): {int(counts.sum())}",
    ]
    if exclude_unknown:
        lines.append(f"  Unknown entries excluded from plot: {unk_excl}")
    lines.append("")
    lines.append(f"  Top {len(counts)} regions:")
    for i, (region, cnt) in enumerate(counts.items(), 1):
        pct = cnt / max(counts.sum(), 1) * 100
        lines.append(f"    {i:2d}. {region}: {cnt} neurons ({pct:.1f}%)")
    lines += ["", "=" * 60]
    report = "\n".join(lines)
    print(report)
    if save_report_path:
        _write_report(save_report_path, report)
    return report


# ======================================================================
# Projection sites count + report
# ======================================================================
def plot_projection_sites_count_df(
    df: pd.DataFrame,
    title: str = "Projection Sites Count per Neuron",
    figsize: tuple = (12, 5),
    save_path: str = None,
    save_report_path: str = None,
    show: bool = True,
) -> str:
    if df.empty or "Terminal_Regions" not in df.columns:
        msg = "DataFrame is empty or missing 'Terminal_Regions'"
        print(msg)
        return msg

    df = df.copy()
    df["Terminal_Regions"] = df["Terminal_Regions"].apply(parse_terminal_regions)
    df["_psc"] = df["Terminal_Regions"].apply(
        lambda x: sum(1 for r in x if "Unknown" not in str(r))
    )
    df["_usc"] = df["Terminal_Regions"].apply(
        lambda x: sum(1 for r in x if "Unknown" in str(r))
    )
    if "Outlier_Count" not in df.columns:
        df["Outlier_Count"] = 0

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    max_val = df["_psc"].max()
    bins = range(0, max_val + 2)
    axes[0].hist(
        df["_psc"],
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        align="left",
        color="steelblue",
    )
    axes[0].set_xlabel("Projection Sites (Known Only)")
    axes[0].set_ylabel("Number of Neurons")
    axes[0].set_title("Distribution (Excl. Unknown)")
    m = df["_psc"].mean()
    axes[0].axvline(m, color="red", linestyle="--", label=f"Mean: {m:.1f}")
    axes[0].legend()

    if "Neuron_Type" in df.columns:
        order = (
            df.groupby("Neuron_Type")["_psc"]
            .median()
            .sort_values(ascending=False)
            .index
        )
        td = {t: df.loc[df["Neuron_Type"] == t, "_psc"].tolist() for t in order}
        bp = axes[1].boxplot(
            list(td.values()), labels=list(td.keys()), patch_artist=True
        )
        for patch, label in zip(bp["boxes"], td.keys()):
            patch.set_facecolor(NEURON_TYPE_COLORS.get(label, "#333"))
            patch.set_alpha(0.6)
        axes[1].set_xlabel("Neuron Type")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].boxplot(df["_psc"])
    axes[1].set_ylabel("Projection Sites (Known)")
    axes[1].set_title("By Neuron Type")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if show:
        plt.show()

    # Report
    total_n = len(df)
    n_wu = int((df["_usc"] > 0).sum())
    n_wo = int((df["Outlier_Count"] > 0).sum())

    lines = [
        "=" * 60,
        "PROJECTION SITES STATISTICS (KNOWN REGIONS ONLY)",
        "=" * 60,
        "",
        f"Total neurons analyzed: {total_n}",
        "",
        "--- Known Projection Sites (Excluding Unknown) ---",
        f"  Total known sites: {int(df['_psc'].sum())}",
        f"  Mean known sites per neuron: {df['_psc'].mean():.2f}",
        f"  Median: {df['_psc'].median():.1f}",
        f"  Min: {int(df['_psc'].min())}",
        f"  Max: {int(df['_psc'].max())}",
        f"  Std: {df['_psc'].std():.2f}",
        "",
        "  Distribution of known sites per neuron:",
    ]
    known_dist = df["_psc"].value_counts().sort_index()
    for sites, count in known_dist.head(10).items():
        lines.append(
            f"    {int(sites)} site(s): {int(count)} neurons ({count / total_n * 100:.1f}%)"
        )
    lines += [
        "",
        "--- Unknown Projection Sites (Excluded from Plot) ---",
        f"  Neurons with unknown sites: {n_wu} ({n_wu / total_n * 100:.1f}%)",
        f"  Total unknown sites excluded: {int(df['_usc'].sum())}",
        f"  Mean unknown sites per neuron: {df['_usc'].mean():.2f}",
        "",
        "--- Outlier Statistics ---",
        f"  Neurons with outliers: {n_wo} ({n_wo / total_n * 100:.1f}%)",
        f"  Total outliers: {int(df['Outlier_Count'].sum())}",
        f"  Mean outliers per neuron: {df['Outlier_Count'].mean():.2f}",
        "",
        "  Distribution of outlier count per neuron:",
    ]
    outlier_dist = df["Outlier_Count"].value_counts().sort_index()
    for outliers, count in outlier_dist.head(10).items():
        lines.append(
            f"    {int(outliers)} outlier(s): {int(count)} neurons ({count / total_n * 100:.1f}%)"
        )
    lines += ["", "=" * 60]
    report = "\n".join(lines)
    print(report)
    if save_report_path:
        _write_report(save_report_path, report)
    return report


# ======================================================================
# Region distribution at hierarchy level
# ======================================================================
def plot_region_distribution(
    df: pd.DataFrame,
    level: int,
    top_n: int = 15,
    figsize: tuple = (10, 6),
    save_path: str = None,
    show: bool = True,
):
    if "Soma_Region_Hierarchy" in df.columns:
        from region_analysis.hierarchy import extract_soma_level

        data = extract_soma_level(df, level)
    else:
        col = f"Region_L{level}"
        if col not in df.columns:
            print(f"Error: no hierarchy data for level {level}.")
            return None
        data = df[col]

    data = data.dropna()
    if data.empty:
        print(f"No data at level {level}.")
        return None

    counts = data.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", len(counts))
    ax.barh(range(len(counts)), counts.values, color=colors)
    labels = [str(r).replace("CL_", "").replace("CR_", "") for r in counts.index]
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax.text(v + max(counts) * 0.01, i, f"{v}", va="center", fontsize=9)
    ax.set_xlabel("Neuron Count", fontsize=12)
    ax.set_title(
        f"Regional Distribution (Level {level})", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(counts) * 1.12)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if show:
        plt.show()
    return fig


def plot_region_distribution_stacked(
    df: pd.DataFrame,
    levels: list = None,
    top_n: int = 10,
    figsize: tuple = None,
    save_path: str = None,
    show: bool = True,
):
    """
    Create stacked subplots for regional distribution at multiple hierarchy levels.

    Args:
        df: DataFrame with hierarchy data
        levels: List of levels to plot (default: [1, 2, 3, 4, 5, 6])
        top_n: Number of top regions to show per level
        figsize: Figure size (default: auto-calculated based on number of levels)
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Figure object or None
    """
    if levels is None:
        levels = [1, 2, 3, 4, 5, 6]

    # Filter to valid levels
    valid_levels = []
    for lv in levels:
        if "Soma_Region_Hierarchy" in df.columns:
            from region_analysis.hierarchy import extract_soma_level
            test_data = extract_soma_level(df, lv)
            if not test_data.dropna().empty:
                valid_levels.append(lv)
        else:
            col = f"Region_L{lv}"
            if col in df.columns and not df[col].dropna().empty:
                valid_levels.append(lv)

    if not valid_levels:
        print("No hierarchy data available for any level.")
        return None

    n_levels = len(valid_levels)

    # Auto-calculate figure size
    if figsize is None:
        # Width: wider for more levels; Height: scale with number of levels
        figsize = (14, 3 * n_levels)

    fig, axes = plt.subplots(n_levels, 1, figsize=figsize)
    if n_levels == 1:
        axes = [axes]

    for idx, level in enumerate(valid_levels):
        ax = axes[idx]

        if "Soma_Region_Hierarchy" in df.columns:
            from region_analysis.hierarchy import extract_soma_level
            data = extract_soma_level(df, level)
        else:
            data = df[f"Region_L{level}"]

        data = data.dropna()
        counts = data.value_counts().head(top_n)

        colors = sns.color_palette("viridis", len(counts))
        ax.barh(range(len(counts)), counts.values, color=colors)
        labels = [str(r).replace("CL_", "").replace("CR_", "") for r in counts.index]
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()

        for i, v in enumerate(counts.values):
            ax.text(v + max(counts) * 0.01, i, f"{v}", va="center", fontsize=8)

        ax.set_xlabel("Neuron Count", fontsize=10)
        ax.set_title(f"Level {level}", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(counts) * 1.15)

    fig.suptitle("Regional Distribution by Hierarchy Level", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if show:
        plt.show()
    return fig


# ======================================================================
# Laterality summary
# ======================================================================
def plot_laterality_summary_df(
    df: pd.DataFrame,
    figsize: tuple = (14, 5),
    save_path: str = None,
    show: bool = True,
):
    needed = {"N_Ipsilateral", "N_Contralateral"}
    if df.empty or not needed.issubset(df.columns):
        print("Run add_laterality_columns first.")
        return None

    has_length = "Laterality_Index" in df.columns
    n_panels = 3 if has_length else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    total_ipsi = int(df["N_Ipsilateral"].sum())
    total_contra = int(df["N_Contralateral"].sum())
    total_unk = (
        int(df["N_Laterality_Unknown"].sum())
        if "N_Laterality_Unknown" in df.columns
        else 0
    )

    vals = [total_ipsi, total_contra]
    labs = [f"Ipsilateral\n({total_ipsi})", f"Contralateral\n({total_contra})"]
    cols = ["#4CAF50", "#F44336"]
    if total_unk > 0:
        vals.append(total_unk)
        labs.append(f"Unknown\n({total_unk})")
        cols.append("#9E9E9E")

    axes[0].pie(
        vals,
        labels=labs,
        autopct="%1.1f%%",
        startangle=90,
        colors=cols,
        explode=[0.02] * len(vals),
    )
    axes[0].set_title("Terminal Laterality", fontsize=12, fontweight="bold")

    if has_length and "Neuron_Type" in df.columns:
        valid = df.dropna(subset=["Laterality_Index"])
        if not valid.empty:
            order = sorted(valid["Neuron_Type"].unique())
            td = {
                t: valid.loc[valid["Neuron_Type"] == t, "Laterality_Index"].tolist()
                for t in order
            }
            bp = axes[1].boxplot(
                list(td.values()), labels=list(td.keys()), patch_artist=True
            )
            for patch, label in zip(bp["boxes"], td.keys()):
                patch.set_facecolor(NEURON_TYPE_COLORS.get(label, "#999"))
                patch.set_alpha(0.6)
            axes[1].set_ylabel("Laterality Index\n(0=ipsi, 1=contra)")
            axes[1].set_title("Laterality Index by Type", fontsize=12, fontweight="bold")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        else:
            axes[1].text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=axes[1].transAxes,
            )
    elif has_length:
        valid = df["Laterality_Index"].dropna()
        axes[1].hist(valid, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
        axes[1].set_xlabel("Laterality Index")
        axes[1].set_ylabel("Neurons")
        axes[1].set_title("LI Distribution", fontsize=12, fontweight="bold")

    if has_length and n_panels == 3:
        if "Neuron_Type" in df.columns:
            grp = df.groupby("Neuron_Type").agg(
                ipsi=("Total_Ipsilateral_Length", "sum"),
                contra=("Total_Contralateral_Length", "sum"),
            )
            x = range(len(grp))
            w = 0.35
            axes[2].bar(
                [i - w / 2 for i in x],
                grp["ipsi"],
                w,
                label="Ipsilateral",
                color="#4CAF50",
                alpha=0.7,
            )
            axes[2].bar(
                [i + w / 2 for i in x],
                grp["contra"],
                w,
                label="Contralateral",
                color="#F44336",
                alpha=0.7,
            )
            axes[2].set_xticks(list(x))
            axes[2].set_xticklabels(grp.index, rotation=45)
            axes[2].set_ylabel("Total Axon Length (mm)")
            axes[2].set_title(
                "Length by Laterality", fontsize=12, fontweight="bold"
            )
            axes[2].legend()
        else:
            axes[2].bar(
                ["Ipsilateral", "Contralateral"],
                [
                    df["Total_Ipsilateral_Length"].sum(),
                    df["Total_Contralateral_Length"].sum(),
                ],
                color=["#4CAF50", "#F44336"],
                alpha=0.7,
            )
            axes[2].set_ylabel("Total Axon Length (mm)")
            axes[2].set_title("Length", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if show:
        plt.show()
    return fig


# ======================================================================
# Single-neuron projection inspect
# ======================================================================
def plot_neuron_projections(
    region_dict: dict, neuron_id: str, save_path: str = None, show: bool = True
):
    if not region_dict or sum(region_dict.values()) == 0:
        print(f"No projection data for {neuron_id}")
        return None
    stats = pd.Series(region_dict).sort_values(ascending=False)
    stats = stats[stats > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    top_n = stats.head(10)
    sns.barplot(x=top_n.values, y=top_n.index, ax=axes[0], palette="viridis")
    axes[0].set_title(f"Top Projections ({neuron_id})")
    if len(stats) > 6:
        main = stats.head(6)
        other = pd.Series({"Others": stats.iloc[6:].sum()})
        pie_data = pd.concat([main, other])
    else:
        pie_data = stats
    axes[1].pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
    axes[1].set_title("Distribution")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    if show:
        plt.show()
    return fig


# ======================================================================
def _write_report(path: str, text: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Report saved to: {path}]")
    except Exception as e:
        print(f"[Error saving report: {e}]")