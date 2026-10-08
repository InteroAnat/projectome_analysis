"""
utils.py - Shared helpers: parsing, debug snapshots, file I/O.
"""

import ast
import os

import numpy as np
import pandas as pd


def parse_terminal_regions(x) -> list:
    """Parse Terminal_Regions from various formats to unique-order list."""
    if isinstance(x, (list, tuple)):
        return _dedupe(x)
    if isinstance(x, str):
        try:
            if x.startswith("[") and x.endswith("]"):
                return _dedupe(ast.literal_eval(x))
            if "," in x:
                parts = [p.strip().strip("'\"") for p in x.split(",")]
                return _dedupe(parts)
            return [x.strip().strip("'\"")]
        except Exception:
            return [x]
    return []


def _dedupe(seq) -> list:
    seen, out = set(), []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def save_debug_snapshot(
    voxel_coords,
    neuron_name: str,
    template_img,
    point_type: str,
    folder: str = "../resource/debug_outliers",
):
    if template_img is None:
        return
    os.makedirs(folder, exist_ok=True)

    import nibabel.affines
    from nilearn import plotting

    world_coords = nibabel.affines.apply_affine(template_img.affine, voxel_coords)
    try:
        display = plotting.plot_anat(
            template_img,
            cut_coords=world_coords,
            draw_cross=True,
            title=f"Outlier {point_type}: {neuron_name} -> {voxel_coords}",
        )
        fname = (
            f"{neuron_name}_{point_type}_"
            f"{int(voxel_coords[0])}_{int(voxel_coords[1])}_{int(voxel_coords[2])}.png"
        )
        full_path = os.path.join(folder, fname)
        display.savefig(full_path)
        display.close()
        print(f"     [DEBUG] Snapshot saved: {full_path}")
    except Exception as e:
        print(f"     [DEBUG] Failed: {e}")


def load_processed_df(path: str) -> pd.DataFrame:
    """Load saved Excel/CSV with correct list-column parsing."""
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("File must be .xlsx or .csv")

    if "Terminal_Regions" in df.columns:
        df["Terminal_Regions"] = df["Terminal_Regions"].apply(parse_terminal_regions)

    proj_cols = [c for c in df.columns if c.startswith("Proj_")]
    for col in proj_cols:
        df[col] = df[col].apply(parse_terminal_regions)

    print(f"Loaded {len(df)} neurons from {path}")
    return df