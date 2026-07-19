#!/usr/bin/env python3
"""Methods figure + full Liu VEN gallery + side-by-side with 251637 VENs (NeuroM features)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

from morph_features import DEFAULT_R_LOCAL, crop_morphology_to_swc, preprocess, read_swc_table

REPO = Path(__file__).resolve().parents[3]
LIU_ASC = (
    REPO
    / "external/liu2026/morph/NeuroMorph_upload260215/PatchClamp_morph"
)
LIU_SWC = REPO / "external/liu2026/swc"  # fallback for plotting if ASC→temp needed
OUT = REPO / "external/ven_validation/outputs/gallery"
SCORES = REPO / "external/ven_validation/outputs/ven_morph_scores_251637.tsv"
LIU_FEAT = REPO / "external/ven_validation/outputs/liu_ven_features.tsv"
FMOST_FEAT = REPO / "external/ven_validation/outputs/fmost_251637_ven_features.tsv"
CROP_CACHE = REPO / "external/ven_validation/outputs/cropped_swc"

LIU_EXEMPLARS = {
    "VENL": "unSM1139",
    "VENS": "unSM1159",
    "PC-L5_ET": "unSM1107",
}

PAIRINGS = [
    ("007.swc", "VENL", "VEN-L-like (symmetric long basal)"),
    ("036.swc", "VENL", "VEN-L-like"),
    ("056.swc", "VENL", "VEN-L-like"),
    ("026.swc", "VENL", "re-score after NeuroM + VEN polarity"),
    ("028.swc", "VENL", "re-score after NeuroM + VEN polarity"),
    ("055.swc", "VENL", "re-score after NeuroM + VEN polarity"),
]


def resolve_fmost(nid: str) -> Path:
    for p in (
        REPO / "processed_neurons/251637" / nid,
        REPO / "processed_neurons/251637/raw_swcs" / nid,
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(nid)


def resolve_liu(cls: str, nid: str) -> Path:
    for ext in (".ASC", ".asc", ".swc"):
        p = LIU_ASC / cls / f"{nid}{ext}"
        if p.is_file():
            return p
        p2 = LIU_SWC / cls / f"{nid}.swc"
        if p2.is_file():
            return p2
    raise FileNotFoundError(f"{cls}/{nid}")


def cached_crop(path: Path, r_local: float = DEFAULT_R_LOCAL) -> Path | None:
    """Return cropped SWC path, or None if MorphIO write fails (use pandas fallback)."""
    CROP_CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{path.parent.name}_{path.stem}_{int(r_local)}.swc"
    dst = CROP_CACHE / key
    if dst.is_file() and dst.stat().st_mtime >= path.stat().st_mtime:
        return dst
    try:
        crop_morphology_to_swc(path, dst, r_local=r_local)
        return dst
    except Exception as e:
        print("crop write skip", path.name, type(e).__name__, str(e)[:80])
        return None


def segments_xy(path: Path, r_local: float | None = DEFAULT_R_LOCAL):
    d = None
    if r_local is not None:
        cropped = cached_crop(path, r_local)
        if cropped is not None:
            d = read_swc_table(cropped)
        elif path.suffix.lower() == ".swc":
            d = preprocess(read_swc_table(path), r_local)
        else:
            # ASC without writable crop: try preconverted SWC
            alt = LIU_SWC / path.parent.name / f"{path.stem}.swc"
            if alt.is_file():
                d = preprocess(read_swc_table(alt), r_local)
    if d is None:
        if path.suffix.lower() == ".swc":
            d = read_swc_table(path)
        else:
            raise FileNotFoundError(f"cannot plot {path}")
    xyz = {int(r.node_id): (float(r.x), float(r.y)) for r in d.itertuples()}
    segs = []
    for r in d.itertuples():
        pid = int(r.parent_id)
        if pid < 0 or pid not in xyz:
            continue
        segs.append([xyz[pid], (float(r.x), float(r.y))])
    root = d[d["parent_id"] < 0]
    root_xy = (
        (float(root.iloc[0]["x"]), float(root.iloc[0]["y"])) if len(root) else None
    )
    return segs, root_xy, d


def center_translate_segs(segs, root_xy):
    if not root_xy:
        return segs
    ox, oy = root_xy
    return [[(a[0] - ox, a[1] - oy), (b[0] - ox, b[1] - oy)] for a, b in segs]


def draw_neuron_centered(ax, path: Path, color: str, r_local=DEFAULT_R_LOCAL, lw=0.45):
    segs, root_xy, _ = segments_xy(path, r_local)
    segs = center_translate_segs(segs, root_xy)
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=0.9))
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        m = max(abs(min(xs)), abs(max(xs)), abs(min(ys)), abs(max(ys)), 100)
        ax.set_xlim(-m * 1.05, m * 1.05)
        ax.set_ylim(-m * 1.05, m * 1.05)
    ax.scatter([0], [0], c="black", s=18, zorder=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_methods():
    fig = plt.figure(figsize=(11, 7))
    fig.suptitle(
        "Liu 2026 ↔ fMOST morph bridge — NeuroM metrics & preprocess",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    ax0 = fig.add_axes([0.05, 0.55, 0.42, 0.38])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")
    ax0.set_title("A. Preprocess (unify FOV)", loc="left", fontsize=11)
    steps = [
        (0.5, 8.2, "1. Load with NeuroM/MorphIO (Liu ASC, fMOST SWC)"),
        (0.5, 6.4, "2. Drop axon; crop dendrites to soma ball R=800 µm"),
        (0.5, 4.6, "3. NeuroM features: length, path, radial, bifurcations"),
        (0.5, 2.8, "4. Polarity: Liu labels; fMOST infer + auto type-4\n    (_infer_polarity_tags)"),
        (0.5, 0.8, "5. Z-score vs Liu class; Euclidean distance (rank)"),
    ]
    for x, y, t in steps:
        ax0.add_patch(
            FancyBboxPatch(
                (x, y - 0.55),
                9,
                1.4,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor="#f0f0f0",
                edgecolor="#333",
                lw=0.8,
            )
        )
        ax0.text(x + 0.3, y + 0.15, t, fontsize=8.5, va="center", family="monospace")

    ax1 = fig.add_axes([0.52, 0.55, 0.45, 0.38])
    ax1.axis("off")
    ax1.set_title("B. Feature set (scored, NeuroM)", loc="left", fontsize=11)
    metrics = [
        ("basal/apical_max_path", "longest soma→tip path per pole (µm)"),
        ("bipolar_path_symmetry", "short_pole / long_pole path (~1 = bipolar)"),
        ("stem_opposition", "cos(angle) of opposite trunk pair (−1 = 180°)"),
        ("long/short_pole_max_path", "polarity-free pole lengths"),
        ("total_dendrite_length", "Σ dendrite cable in crop (µm)"),
        ("max_radial_distance_dend", "farthest dendrite tip from soma"),
    ]
    y = 0.92
    for name, desc in metrics:
        ax1.text(0.02, y, name, fontsize=8, fontweight="bold", transform=ax1.transAxes, family="monospace")
        ax1.text(0.02, y - 0.045, desc, fontsize=7.5, transform=ax1.transAxes, color="#333")
        y -= 0.14

    ax2 = fig.add_axes([0.08, 0.08, 0.38, 0.38])
    ax2.set_title("C. VEN polarity prior (fMOST)", loc="left", fontsize=11)
    circ = plt.Circle((0, 0), 0.35, color="#222", zorder=5)
    ax2.add_patch(circ)
    ax2.plot([0, 0], [0.35, 2.4], color="#1f77b4", lw=3)
    ax2.plot([0, 0], [-0.35, -3.2], color="#d62728", lw=3)
    ax2.annotate("shorter pole → apical-like", xy=(0, 2.2), xytext=(1.0, 2.0), fontsize=8, color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
    ax2.annotate("longer opposite pole → basal\n(Liu VEN-L prior)", xy=(0, -2.8), xytext=(1.0, -2.4),
                 fontsize=8, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    ax2.set_xlim(-3, 4.5)
    ax2.set_ylim(-3.8, 3.2)
    ax2.set_aspect("equal")
    ax2.axis("off")

    ax3 = fig.add_axes([0.52, 0.08, 0.45, 0.38])
    ax3.axis("off")
    ax3.set_title("D. Liu class priors (NeuroM, this run)", loc="left", fontsize=11)
    liu = pd.read_csv(LIU_FEAT, sep="\t")
    means = liu.groupby("ref_class")[["basal_max_path", "bipolar_path_symmetry", "apical_basal_path_ratio"]].mean()
    rows = [
        ("Class", "basal_max", "path_sym", "apic/bas path"),
        ("VEN-L", f"{means.loc['VENL', 'basal_max_path']:.0f}", f"{means.loc['VENL', 'bipolar_path_symmetry']:.2f}",
         f"{means.loc['VENL', 'apical_basal_path_ratio']:.2f}"),
        ("VEN-S", f"{means.loc['VENS', 'basal_max_path']:.0f}", f"{means.loc['VENS', 'bipolar_path_symmetry']:.2f}",
         f"{means.loc['VENS', 'apical_basal_path_ratio']:.2f}"),
        ("PC-L5_ET", f"{means.loc['PC-L5_ET', 'basal_max_path']:.0f}",
         f"{means.loc['PC-L5_ET', 'bipolar_path_symmetry']:.2f}",
         f"{means.loc['PC-L5_ET', 'apical_basal_path_ratio']:.2f}"),
    ]
    y = 0.85
    for i, row in enumerate(rows):
        w = "bold" if i == 0 else "normal"
        for j, val in enumerate(row):
            ax3.text(0.05 + j * 0.24, y, val, fontsize=8.5, fontweight=w, transform=ax3.transAxes, family="monospace")
        y -= 0.14
    ax3.text(
        0.05, 0.15,
        "Backend: neurom + morphio. Score = ||z − μ_class||₂.\n"
        "Use ranks (prefer_VENL), not absolute d_z.",
        fontsize=8, transform=ax3.transAxes, color="#333",
    )

    out = OUT / "00_methods_metrics.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


def fig_liu_gallery_full():
    """All Liu VEN-L and VEN-S cells (full gallery)."""
    liu = pd.read_csv(LIU_FEAT, sep="\t")
    outs = []
    for cls, color, ncols in [("VENL", "#c0392b", 7), ("VENS", "#2980b9", 6)]:
        sub = liu[liu["ref_class"] == cls].sort_values("basal_max_path").reset_index(drop=True)
        n = len(sub)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.1, nrows * 2.2))
        axes = np.atleast_2d(axes)
        fig.suptitle(
            f"Liu 2026 {cls} — full gallery (n={n}, NeuroM crop R={int(DEFAULT_R_LOCAL)} µm, soma-centered)",
            fontsize=12,
            fontweight="bold",
        )
        for i in range(nrows * ncols):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            if i >= n:
                ax.axis("off")
                continue
            row = sub.iloc[i]
            nid = row["neuron_id"]
            path = resolve_liu(cls, nid)
            draw_neuron_centered(ax, path, color=color, lw=0.35)
            ax.set_title(
                f"{nid}\nbasal={row['basal_max_path']:.0f}  sym={row.get('bipolar_path_symmetry', np.nan):.2f}",
                fontsize=7,
            )
        fig.text(
            0.5, 0.01,
            "Sorted by basal_max_path. Black = soma. Slice-local µm (not NMT).",
            ha="center", fontsize=8, color="#555",
        )
        out = OUT / f"01_liu_{cls}_full_gallery.png"
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", out)
        outs.append(out)
    return outs


def fig_pairwise():
    scores = pd.read_csv(SCORES, sep="\t")
    fmost = pd.read_csv(FMOST_FEAT, sep="\t")
    liu = pd.read_csv(LIU_FEAT, sep="\t")

    fig, axes = plt.subplots(6, 3, figsize=(11, 16))
    fig.suptitle(
        "251637 VENs vs Liu exemplars (NeuroM, dendrite crop R=800 µm)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    for i, (nid, cls, note) in enumerate(PAIRINGS):
        # use nearest class from scores if available
        sc = scores[scores["neuron_id"] == nid].iloc[0]
        nearest = sc["nearest_class"]
        if nearest in ("VENL", "VENS"):
            cls = nearest
        other = "VENS" if cls == "VENL" else "VENL"

        fpath = resolve_fmost(nid)
        lpath = resolve_liu(cls, LIU_EXEMPLARS[cls])
        opath = resolve_liu(other, LIU_EXEMPLARS[other])
        ff = fmost[fmost["neuron_id"] == nid].iloc[0]
        lf = liu[(liu["ref_class"] == cls) & (liu["neuron_id"] == LIU_EXEMPLARS[cls])].iloc[0]
        of = liu[(liu["ref_class"] == other) & (liu["neuron_id"] == LIU_EXEMPLARS[other])].iloc[0]

        ax0, ax1, ax2 = axes[i]
        draw_neuron_centered(ax0, fpath, color="#2c3e50", lw=0.5)
        ax0.set_title(
            f"fMOST {nid}\n"
            f"basal={ff['basal_max_path']:.0f}  apic={ff['apical_max_path']:.0f}  sym={ff['bipolar_path_symmetry']:.2f}\n"
            f"nearest={sc['nearest_class']}  d_z(VENL)={sc['dist_z_VENL']:.1f}",
            fontsize=7.5,
        )
        draw_neuron_centered(ax1, lpath, color="#c0392b" if cls == "VENL" else "#2980b9", lw=0.5)
        ax1.set_title(
            f"Liu {cls} {LIU_EXEMPLARS[cls]}\n"
            f"basal={lf['basal_max_path']:.0f}  sym={lf['bipolar_path_symmetry']:.2f}\n{note}",
            fontsize=7.5,
        )
        draw_neuron_centered(ax2, opath, color="#2980b9" if other == "VENS" else "#c0392b", lw=0.5)
        ax2.set_title(
            f"Liu {other} {LIU_EXEMPLARS[other]} (contrast)\n"
            f"basal={of['basal_max_path']:.0f}  d_z({other})={sc[f'dist_z_{other}']:.1f}",
            fontsize=7.5,
        )

    fig.text(
        0.5, 0.002,
        "NeuroM features after MorphIO crop. Match column follows nearest_class when VENL/VENS.",
        ha="center", fontsize=8, color="#555",
    )
    out = OUT / "02_pairwise_fMOST_vs_Liu.png"
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


def fig_metric_bars():
    scores = pd.read_csv(SCORES, sep="\t")
    fmost = pd.read_csv(FMOST_FEAT, sep="\t")
    liu = pd.read_csv(LIU_FEAT, sep="\t")
    means = liu.groupby("ref_class")[
        ["basal_max_path", "bipolar_path_symmetry", "apical_basal_path_ratio"]
    ].mean()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.suptitle("Key NeuroM metrics: 251637 VENs vs Liu class means", fontsize=12, fontweight="bold")

    order = ["007.swc", "036.swc", "056.swc", "026.swc", "028.swc", "055.swc"]
    x = np.arange(len(order))
    si = scores.set_index("neuron_id")
    fi = fmost.set_index("neuron_id")
    colors = ["#c0392b" if si.loc[n, "prefer_VENL"] else "#2980b9" for n in order]

    ax = axes[0]
    vals = [fi.loc[n, "basal_max_path"] for n in order]
    ax.bar(x, vals, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(means.loc["VENL", "basal_max_path"], color="#c0392b", ls="--", lw=1.2, label="Liu VEN-L mean")
    ax.axhline(means.loc["VENS", "basal_max_path"], color="#2980b9", ls="--", lw=1.2, label="Liu VEN-S mean")
    ax.axhline(500, color="#888", ls=":", lw=1, label="500 µm prior")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(".swc", "") for n in order])
    ax.set_ylabel("basal_max_path (µm)")
    ax.set_title("Basal max path (VEN polarity)")
    ax.legend(fontsize=7)

    ax = axes[1]
    vals = [fi.loc[n, "bipolar_path_symmetry"] for n in order]
    ax.bar(x, vals, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(means.loc["VENL", "bipolar_path_symmetry"], color="#c0392b", ls="--", lw=1.2)
    ax.axhline(means.loc["VENS", "bipolar_path_symmetry"], color="#2980b9", ls="--", lw=1.2)
    ax.axhline(1.0, color="#888", ls=":", lw=1, label="sym=1")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(".swc", "") for n in order])
    ax.set_ylabel("short/long pole path")
    ax.set_title("Bipolar path symmetry")
    ax.legend(fontsize=7)

    ax = axes[2]
    w = 0.35
    dL = [si.loc[n, "dist_z_VENL"] for n in order]
    dS = [si.loc[n, "dist_z_VENS"] for n in order]
    ax.bar(x - w / 2, dL, w, color="#c0392b", label="d_z VEN-L", edgecolor="k", lw=0.3)
    ax.bar(x + w / 2, dS, w, color="#2980b9", label="d_z VEN-S", edgecolor="k", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(".swc", "") for n in order])
    ax.set_ylabel("Euclidean z-distance")
    ax.set_title("Distance to Liu centroids")
    ax.legend(fontsize=7)

    out = OUT / "03_metric_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


def write_html(paths: list[Path]):
    html = OUT.parent / "VEN_MORPH_SHOWCASE.html"
    blocks = []
    for p in paths:
        rel = p.relative_to(OUT.parent).as_posix()
        blocks.append(
            f'<h2>{p.stem}</h2><img src="{rel}" style="max-width:100%;border:1px solid #ccc;margin-bottom:2em"/>'
        )
    html.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        "<title>Liu ↔ fMOST VEN morph showcase (NeuroM)</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em;color:#222}"
        "h1{font-size:1.4rem} h2{font-size:1.05rem;margin-top:2em;color:#444}"
        "code{background:#f4f4f4;padding:0.1em 0.3em}</style></head><body>"
        "<h1>Liu 2026 VEN morphology ↔ 251637 fMOST VENs</h1>"
        "<p>Backend: <code>neurom</code> + MorphIO. Branch <code>test/liu-ven-fmost-bridge</code>. "
        "User VENs: 007, 026, 028, 036, 055, 056. Dendrite-only, R<sub>local</sub>=800 µm. "
        "See <code>docs/METHODS.md</code>.</p>"
        + "\n".join(blocks)
        + "</body></html>",
        encoding="utf-8",
    )
    print("wrote", html)
    return html


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [fig_methods()]
    paths.extend(fig_liu_gallery_full())
    paths.append(fig_pairwise())
    paths.append(fig_metric_bars())
    write_html(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
