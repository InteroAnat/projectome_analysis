#!/usr/bin/env python3
"""Dendritic morph features via NeuroM (Blue Brain), with local FOV crop.

Stack
-----
- **NeuroM 4.x** (+ MorphIO): load ASC/SWC, extract morphometrics
- **Local crop**: drop axon + keep dendrite within soma ball R_local
- After crop: ``remove_unifurcations()`` then pass MorphIO object to NeuroM
  (avoids SWC write failures on single-child sections)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import neurom as nm
import numpy as np
import pandas as pd
from morphio import SectionType, set_maximum_warnings
from morphio.mut import Morphology as MutMorph

try:
    set_maximum_warnings(0)
except Exception:
    pass

DEFAULT_R_LOCAL = 800.0

FEATURE_SCORE_COLS = [
    "n_dendrite_neurites",
    "total_dendrite_length",
    "basal_max_path",
    "apical_max_path",
    "apical_basal_path_ratio",
    "bipolar_path_symmetry",
    "bipolar_length_symmetry",
    "stem_opposition",
    "long_pole_max_path",
    "short_pole_max_path",
    "max_radial_distance_dend",
]


def read_swc_table(path: Path) -> pd.DataFrame:
    """Minimal SWC reader (gallery plotting)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "node_id": int(float(parts[0])),
                    "label": int(float(parts[1])),
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "z": float(parts[4]),
                    "radius": float(parts[5]),
                    "parent_id": int(float(parts[6])),
                }
            )
    if not rows:
        raise ValueError(f"empty SWC: {path}")
    return pd.DataFrame(rows)


def preprocess(df: pd.DataFrame, r_local: float) -> pd.DataFrame:
    """Drop axon; crop to soma ball (gallery helper)."""
    lab_axon = 2
    d = df[df["label"] != lab_axon].copy()
    roots = d[d["parent_id"] < 0]
    if roots.empty:
        soma = d[d["label"] == 1]
        rid = int((soma if len(soma) else d).iloc[0]["node_id"])
        d.loc[d["node_id"] == rid, "parent_id"] = -1
        roots = d[d["parent_id"] < 0]
    rid = int(roots.iloc[0]["node_id"])
    cx, cy, cz = float(roots.iloc[0]["x"]), float(roots.iloc[0]["y"]), float(roots.iloc[0]["z"])
    dist = np.sqrt((d["x"] - cx) ** 2 + (d["y"] - cy) ** 2 + (d["z"] - cz) ** 2)
    d = d[(dist <= r_local) | (d["node_id"] == rid)].copy()
    ids = set(d["node_id"].astype(int))
    mask = (d["parent_id"] >= 0) & (~d["parent_id"].isin(ids))
    d.loc[mask, "parent_id"] = -1
    roots2 = d[d["parent_id"] < 0]
    if len(roots2) > 1:
        keep = rid if rid in set(roots2["node_id"]) else int(roots2.iloc[0]["node_id"])
        d.loc[(d["parent_id"] < 0) & (d["node_id"] != keep), "parent_id"] = keep
    return d.reset_index(drop=True)


def _iter_sections(morph: MutMorph):
    stack = list(morph.root_sections)
    while stack:
        s = stack.pop()
        yield s
        stack.extend(list(s.children))


def crop_morphology(src: Path, r_local: float = DEFAULT_R_LOCAL) -> MutMorph:
    """Drop axon; truncate/delete dendrites outside soma ball; sanitize unifurcations."""
    mm = MutMorph(str(src))
    for s in list(mm.root_sections):
        if s.type == SectionType.axon:
            mm.delete_section(s, recursive=True)

    center = np.asarray(mm.soma.center, dtype=float)

    changed = True
    while changed:
        changed = False
        for s in list(_iter_sections(mm)):
            p0 = np.asarray(s.points[0][:3], dtype=float)
            if float(np.linalg.norm(p0 - center)) > r_local:
                mm.delete_section(s, recursive=True)
                changed = True
                break

    for s in list(_iter_sections(mm)):
        pts = np.asarray(s.points, dtype=float)
        diams = np.asarray(s.diameters, dtype=float)
        if pts.ndim != 2 or len(pts) == 0:
            continue
        keep_idx: list[int] = []
        cut_at = None
        for i, p in enumerate(pts):
            rd = float(np.linalg.norm(p[:3] - center))
            if rd <= r_local:
                keep_idx.append(i)
            else:
                cut_at = i
                break
        if cut_at is None:
            continue
        if not keep_idx:
            mm.delete_section(s, recursive=True)
            continue
        for c in list(s.children):
            mm.delete_section(c, recursive=True)
        p_in = pts[keep_idx[-1]]
        p_out = pts[cut_at]
        d_in = float(np.linalg.norm(p_in[:3] - center))
        d_out = float(np.linalg.norm(p_out[:3] - center))
        t = float(np.clip((r_local - d_in) / (d_out - d_in + 1e-12), 0.0, 1.0))
        pb = p_in + t * (p_out - p_in)
        db = float(diams[keep_idx[-1]] + t * (diams[cut_at] - diams[keep_idx[-1]]))
        new_pts = np.vstack([pts[keep_idx], pb[None, :]])
        new_d = np.concatenate([diams[keep_idx], [db]])
        s.points = new_pts[:, :3]
        s.diameters = new_d

    # Required before SWC write; also cleans ASC quirks for NeuroM
    try:
        mm.remove_unifurcations()
    except Exception:
        pass
    return mm


def crop_morphology_to_swc(src: Path, dst: Path, r_local: float = DEFAULT_R_LOCAL) -> Path:
    """Crop and write SWC (for gallery)."""
    mm = crop_morphology(src, r_local=r_local)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mm.write(str(dst))
    return dst


def _neurite_max_path(neurite) -> float:
    try:
        pd = np.asarray(nm.features.get("section_path_distances", neurite), dtype=float)
        return float(pd.max()) if pd.size else float(neurite.length)
    except Exception:
        return float(neurite.length)


def _neurite_root_dir(morph, neurite) -> np.ndarray:
    root = np.asarray(neurite.root_node.points[0][:3], dtype=float)
    soma = np.asarray(morph.soma.center, dtype=float)
    v = root - soma
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def _pick_ven_poles(items: list[dict]) -> tuple[dict | None, dict | None, float]:
    """VEN polarity: longest stem = basal-like; most opposite remaining = apical-like.

    Do NOT pick the globally most-opposite pair among all stems — short stubs can
    be nearly antipodal and steal the poles from the true long bipolar arbor.
    """
    if len(items) < 2:
        return (items[0] if items else None), None, np.nan
    ranked = sorted(items, key=lambda s: s["max_path"], reverse=True)
    basal = ranked[0]
    best_b, best_cos = ranked[1], float(np.dot(basal["dir"], ranked[1]["dir"]))
    for b in ranked[1:]:
        cos = float(np.dot(basal["dir"], b["dir"]))
        if cos < best_cos:
            best_b, best_cos = b, cos
    return basal, best_b, best_cos


def _has_complete_morph_labels(mm: MutMorph) -> bool:
    """True when the file already contains apical (type-4) compartment tags."""
    for s in _iter_sections(mm):
        if s.type == SectionType.apical_dendrite:
            return True
    return False


def _dendrite_root_sections(mm: MutMorph) -> list:
    """Soma-rooted dendrite trees (axon already removed by crop)."""
    out = []
    for s in mm.root_sections:
        if s.type == SectionType.axon:
            continue
        out.append(s)
    return out


def _section_cable(section) -> float:
    pts = np.asarray(section.points, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1)))


def _section_max_path(section) -> float:
    pts = np.asarray(section.points, dtype=float)
    here = 0.0
    if pts.ndim == 2 and len(pts) >= 2:
        here = float(np.sum(np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1)))
    if not section.children:
        return here
    return max(_section_max_path(c) for c in section.children) + here


def _section_root_dir(mm: MutMorph, section) -> np.ndarray:
    root = np.asarray(section.points[0][:3], dtype=float)
    soma = np.asarray(mm.soma.center, dtype=float)
    v = root - soma
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def _infer_polarity_tags(mm: MutMorph) -> None:
    """Assign apical (type 4) and basal (type 3) on fMOST-like SWCs lacking type 4.

    Longest dendrite stem → basal; most directionally opposite remaining stem → apical.
    Modifies ``mm`` in memory only.
    """
    roots = _dendrite_root_sections(mm)
    if len(roots) < 2:
        return
    items = [
        {
            "section": s,
            "max_path": _section_max_path(s),
            "dir": _section_root_dir(mm, s),
            "cable": _section_cable(s),
        }
        for s in roots
    ]
    basal, apical, _cos = _pick_ven_poles(items)
    if basal is None or apical is None:
        return
    basal["section"].type = SectionType.basal_dendrite
    apical["section"].type = SectionType.apical_dendrite


def extract_features_neurom(
    path: Path,
    r_local: float = DEFAULT_R_LOCAL,
    source: str = "",
) -> dict:
    """Load with MorphIO, crop, hand to NeuroM, extract features."""
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm = crop_morphology(path, r_local=r_local)
        complete_morph_labels = _has_complete_morph_labels(mm)
        if not complete_morph_labels:
            _infer_polarity_tags(mm)
        morph = nm.load_morphology(mm)

    dend_types = {nm.BASAL_DENDRITE, nm.APICAL_DENDRITE}
    neurites = [n for n in morph.neurites if n.type in dend_types]

    stems = []
    for n in neurites:
        stems.append(
            {
                "type": n.type,
                "cable": float(n.length),
                "max_path": _neurite_max_path(n),
                "dir": _neurite_root_dir(morph, n),
            }
        )

    # Compartment metrics from labels (native Liu ASC or auto-tagged fMOST)
    basal_len = float(
        np.sum(nm.features.get("total_length", morph, neurite_type=nm.BASAL_DENDRITE))
    )
    apical_len = float(
        np.sum(nm.features.get("total_length", morph, neurite_type=nm.APICAL_DENDRITE))
    )
    pd_b = np.asarray(
        nm.features.get("section_path_distances", morph, neurite_type=nm.BASAL_DENDRITE),
        dtype=float,
    )
    pd_a = np.asarray(
        nm.features.get("section_path_distances", morph, neurite_type=nm.APICAL_DENDRITE),
        dtype=float,
    )
    basal_max = float(pd_b.max()) if pd_b.size else 0.0
    apical_max = float(pd_a.max()) if pd_a.size else 0.0
    total_len = basal_len + apical_len

    if len(stems) >= 2:
        _, _, stem_opposition = _pick_ven_poles(stems)
    else:
        stem_opposition = np.nan

    pole_paths = sorted([apical_max, basal_max], reverse=True)
    pole_cables = sorted([apical_len, basal_len], reverse=True)
    long_path = float(pole_paths[0]) if pole_paths else 0.0
    short_path = float(pole_paths[1]) if len(pole_paths) > 1 else 0.0
    long_cable = float(pole_cables[0]) if pole_cables else 0.0
    short_cable = float(pole_cables[1]) if len(pole_cables) > 1 else 0.0
    path_sym = short_path / long_path if long_path > 1e-6 else np.nan
    len_sym = short_cable / long_cable if long_cable > 1e-6 else np.nan
    ratio_len = apical_len / basal_len if basal_len > 1e-6 else np.nan
    ratio_path = apical_max / basal_max if basal_max > 1e-6 else np.nan

    rd_b = np.asarray(
        nm.features.get("section_radial_distances", morph, neurite_type=nm.BASAL_DENDRITE),
        dtype=float,
    )
    rd_a = np.asarray(
        nm.features.get("section_radial_distances", morph, neurite_type=nm.APICAL_DENDRITE),
        dtype=float,
    )
    rd_d = np.concatenate([x for x in (rd_b, rd_a) if x.size]) if (rd_b.size or rd_a.size) else np.array([])
    max_rd = float(rd_d.max()) if rd_d.size else 0.0

    n_bif = int(
        nm.features.get("number_of_bifurcations", morph, neurite_type=nm.BASAL_DENDRITE)
    ) + int(nm.features.get("number_of_bifurcations", morph, neurite_type=nm.APICAL_DENDRITE))

    return {
        "path": str(path),
        "source": source,
        "backend": f"neurom-{getattr(nm, '__version__', '?')}",
        "r_local_um": r_local,
        "units_um": True,
        "complete_morph_labels": bool(complete_morph_labels),
        "n_dendrite_neurites": int(len(stems)),
        "n_stems": int(len(stems)),
        "total_dendrite_length": float(total_len),
        "apical_length": float(apical_len),
        "basal_length": float(basal_len),
        "apical_max_path": float(apical_max),
        "basal_max_path": float(basal_max),
        "apical_basal_length_ratio": float(ratio_len) if ratio_len == ratio_len else np.nan,
        "apical_basal_path_ratio": float(ratio_path) if ratio_path == ratio_path else np.nan,
        "long_pole_max_path": long_path,
        "short_pole_max_path": short_path,
        "bipolar_path_symmetry": float(path_sym) if path_sym == path_sym else np.nan,
        "bipolar_length_symmetry": float(len_sym) if len_sym == len_sym else np.nan,
        "stem_opposition": float(stem_opposition) if stem_opposition == stem_opposition else np.nan,
        "max_radial_distance_dend": max_rd,
        "n_bifurcations_dend": n_bif,
        "basal_gt_500": bool(basal_max > 500),
    }


extract_features = extract_features_neurom


def features_table(paths: list[Path], source: str, r_local: float) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            rows.append(extract_features_neurom(p, r_local=r_local, source=source))
        except Exception as e:
            rows.append({"path": str(p), "source": source, "error": str(e)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="NeuroM morph features with R_local crop")
    ap.add_argument("morph", nargs="+", type=Path, help="ASC or SWC paths")
    ap.add_argument("--r-local", type=float, default=DEFAULT_R_LOCAL)
    ap.add_argument("--source", default="")
    ap.add_argument("-o", type=Path, required=True)
    args = ap.parse_args()
    df = features_table(args.morph, args.source, args.r_local)
    args.o.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.o, sep="\t", index=False)
    print(f"wrote {args.o} n={len(df)} backend=neurom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
