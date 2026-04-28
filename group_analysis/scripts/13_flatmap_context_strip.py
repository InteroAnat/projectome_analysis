#!/usr/bin/env python3
"""
P13 (plan v2): annotate a cached flatmap PNG with a compact L/R sample strip from
combined_primary_v2 context CSV. Does not require soma-to-flatmap coordinates.

Usage (from repo root):
  python group_analysis/scripts/13_flatmap_context_strip.py
"""
from __future__ import annotations

import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "group_analysis/R_analysis/outputs/combined_primary_v2"
CTX = OUT / "stats/00_context_per_subregion_n.csv"
# Prefer pipeline flatmap; fallback to figures_charts conservative insula panel
BASE_CANDIDATES = [
    PROJECT / "group_analysis/R_analysis/outputs/figures/flatmap/flatmap_all_monkeys_combined.png",
    PROJECT / "figures_charts/gou_flatmap_conservative/insula/insula_soma_LR_combined.png",
]
OUT_DIR = OUT / "flatmap_overlays"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed; skip flatmap strip. pip install pillow")
        return

    base = next((p for p in BASE_CANDIDATES if p.is_file()), None)
    if base is None:
        print("No base flatmap PNG found; skip.")
        return
    if not CTX.is_file():
        print("Missing context CSV:", CTX)
        return

    rows = []
    with CTX.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    lines = ["L/R soma counts (harmonized subregions)", ""]
    for r in rows:
        lines.append(
            f"  {r['Region']}: L={r['L']}  R={r['R']}  (total {r['total']})"
        )

    img = Image.open(base).convert("RGBA")
    w, h = img.size
    strip_h = min(220, 36 + 18 * len(lines))
    out_img = Image.new("RGBA", (w, h + strip_h), (255, 255, 255, 255))
    out_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    y = h + 8
    for line in lines:
        draw.text((8, y), line, fill=(20, 20, 20), font=font)
        y += 18

    out_path = OUT_DIR / "P13_flatmap_with_LR_context_strip.png"
    out_img.convert("RGB").save(out_path, dpi=(220, 220))
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
