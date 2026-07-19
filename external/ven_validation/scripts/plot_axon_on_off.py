#!/usr/bin/env python3
"""Axon ON vs OFF panels (MorphIO type=axon delete) for fMOST or Liu ASC."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
from matplotlib.collections import LineCollection
from morphio import SectionType
from morphio.mut import Morphology as MutMorph

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "external/ven_validation/outputs/gallery"
LIU_ASC = (
    REPO / "external/liu2026/morph/NeuroMorph_upload260215/PatchClamp_morph"
)

AXON_COLOR = "#c0392b"
DEND_COLOR = "#2c3e50"
ZOOM_UM = 1000.0


def _iter_sections(mm: MutMorph):
    stack = list(mm.root_sections)
    while stack:
        s = stack.pop()
        yield s
        stack.extend(list(s.children))


def segs_from_morphio(mm: MutMorph):
    """Soma-centered XY segments split by axon vs dendrite."""
    center = np.asarray(mm.soma.center, dtype=float)
    axon_segs, dend_segs = [], []
    for s in _iter_sections(mm):
        pts = np.asarray(s.points)[:, :3] - center
        segs = [[(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))] for a, b in zip(pts[:-1], pts[1:])]
        if s.type == SectionType.axon:
            axon_segs.extend(segs)
        else:
            dend_segs.extend(segs)
    return axon_segs, dend_segs


def drop_axon(mm: MutMorph) -> MutMorph:
    for s in list(mm.root_sections):
        if s.type == SectionType.axon:
            mm.delete_section(s, recursive=True)
    return mm


def cable_stats(mm: MutMorph) -> dict:
    n = nm.load_morphology(mm)
    axon_len = float(nm.get("total_length", n, neurite_type=nm.AXON) or 0.0)
    dend_len = float(nm.get("total_length", n, neurite_type=nm.BASAL_DENDRITE) or 0.0)
    dend_len += float(nm.get("total_length", n, neurite_type=nm.APICAL_DENDRITE) or 0.0)
    n_neurites = int(nm.get("number_of_neurites", n) or 0)
    return {"axon": axon_len, "dend": dend_len, "neurites": n_neurites}


def _draw(ax, axon_segs, dend_segs, title: str, stats: dict, xlim=None, ylim=None):
    if dend_segs:
        ax.add_collection(
            LineCollection(dend_segs, colors=DEND_COLOR, linewidths=0.55, alpha=0.95, zorder=2)
        )
    if axon_segs:
        ax.add_collection(
            LineCollection(axon_segs, colors=AXON_COLOR, linewidths=0.4, alpha=0.85, zorder=3)
        )
    ax.scatter([0], [0], c="black", s=22, zorder=5)
    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        all_pts = [p for segs in (axon_segs, dend_segs) for s in segs for p in s]
        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            pad = 0.05
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            ax.set_xlim(min(xs) - pad * dx - 10, max(xs) + pad * dx + 10)
            ax.set_ylim(min(ys) - pad * dy - 10, max(ys) + pad * dy + 10)
    ax.set_title(title, fontsize=10, loc="left")
    txt = f"axon={stats['axon']:.0f} µm\ndend={stats['dend']:.0f} µm\nneurites={stats['neurites']}"
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
    )


def fig_axon_on_off(src: Path, out: Path, title: str) -> Path:
    mm_on = MutMorph(str(src))
    stats_on = cable_stats(mm_on)
    axon_on, dend_on = segs_from_morphio(mm_on)

    mm_off = MutMorph(str(src))
    drop_axon(mm_off)
    stats_off = cable_stats(mm_off)
    axon_off, dend_off = segs_from_morphio(mm_off)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(
        f"{title} — axon ON vs OFF\n(soma-centered XY; red=axon, dark=dendrite)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    z = ZOOM_UM
    _draw(axes[0, 0], axon_on, dend_on, "A. Raw (axon ON)", stats_on)
    _draw(axes[0, 1], axon_off, dend_off, "B. Axon deleted (dendrites only)", stats_off)
    _draw(
        axes[1, 0],
        axon_on,
        dend_on,
        f"C. Same as A, zoom ±{int(z)} µm",
        stats_on,
        xlim=(-z, z),
        ylim=(-z, z),
    )
    _draw(
        axes[1, 1],
        axon_off,
        dend_off,
        f"D. Same as B, zoom ±{int(z)} µm",
        stats_off,
        xlim=(-z, z),
        ylim=(-z, z),
    )
    fig.text(
        0.5,
        0.01,
        "Filter rule: MorphIO root sections with type=axon (SWC type 2 / Neurolucida Axon) "
        "deleted recursively. No length heuristic. Top row = full XY extent; "
        f"bottom = local ±{int(z)} µm (near Liu FOV).",
        ha="center",
        fontsize=8,
        color="#444",
        wrap=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    print("  ON ", stats_on)
    print("  OFF", stats_off)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--which",
        choices=("liu", "fmost007", "all"),
        default="liu",
        help="liu = VENL unSM1139 + VENS unSM1159; fmost007 = 007; all = both",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    if args.which in ("liu", "all"):
        for cls, nid, label in (
            ("VENL", "unSM1139", "Liu 2026 VEN-L / unSM1139"),
            ("VENS", "unSM1159", "Liu 2026 VEN-S / unSM1159"),
        ):
            src = LIU_ASC / cls / f"{nid}.ASC"
            if not src.is_file():
                src = LIU_ASC / cls / f"{nid}.asc"
            out = OUT / f"04_{cls}_{nid}_axon_on_vs_off.png"
            paths.append(fig_axon_on_off(src, out, label))
    if args.which in ("fmost007", "all"):
        src = REPO / "processed_neurons/251637/007.swc"
        out = OUT / "04_007_axon_on_vs_off.png"
        paths.append(fig_axon_on_off(src, out, "fMOST 251637 / 007"))
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
