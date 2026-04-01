#!/usr/bin/env python3
"""
Run the system `tree` command and save output as PNG.

Usage examples:
  python tree_to_png.py .
  python tree_to_png.py D:\\projectome_analysis --output repo_tree.png --files
  python tree_to_png.py . --depth 3 --font-size 14
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import shutil
from pathlib import Path


def build_tree_command(target_dir: Path, include_files: bool, depth: int | None) -> list[str]:
    system_name = platform.system().lower()

    if "windows" in system_name:
        # /A uses ASCII characters to avoid font/encoding issues in images.
        cmd = ["tree", str(target_dir), "/A"]
        if include_files:
            cmd.append("/F")
        return cmd

    # Linux/macOS tree flags
    cmd = ["tree", "-a"]
    if include_files:
        cmd.append("-f")
    if depth is not None:
        cmd.extend(["-L", str(depth)])
    cmd.append(str(target_dir))
    return cmd


def run_tree_command(target_dir: Path, include_files: bool, depth: int | None) -> str:
    cmd = build_tree_command(target_dir, include_files, depth)
    commands_to_try: list[list[str]] = [cmd]

    # Windows fallback: `tree` may be available via cmd even if not directly resolvable.
    if platform.system().lower().startswith("win"):
        win_cmd = ["cmd", "/c"] + cmd
        if win_cmd not in commands_to_try:
            commands_to_try.append(win_cmd)

        tree_exe = shutil.which("tree.com") or shutil.which("tree.exe")
        if tree_exe:
            direct_cmd = [tree_exe, str(target_dir), "/A"]
            if include_files:
                direct_cmd.append("/F")
            if direct_cmd not in commands_to_try:
                commands_to_try.append(direct_cmd)

    last_error: Exception | None = None
    for command in commands_to_try:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return result.stdout
        except FileNotFoundError as exc:
            last_error = exc
            continue
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "No stderr provided."
            raise RuntimeError(f"`tree` command failed: {stderr}") from exc

    raise RuntimeError(
        "Could not execute `tree` from Python. "
        "Try running this from Command Prompt, or pass a full path like "
        "`C:\\Windows\\System32\\tree.com` in your PATH."
    ) from last_error


def render_text_to_png(text: str, output_png: Path, font_size: int, line_spacing: int, margin: int) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise RuntimeError(
            "Pillow is required for PNG export. Install with: pip install pillow"
        ) from None

    lines = text.splitlines() or [""]

    # Prefer a monospaced font if available; fall back to PIL default.
    try:
        font = ImageFont.truetype("consola.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("cour.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    # Use a temporary canvas to measure text.
    temp_img = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(temp_img)

    max_width = 0
    line_height = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_width = max(max_width, width)
        line_height = max(line_height, height)

    content_height = len(lines) * (line_height + line_spacing)
    img_width = max_width + margin * 2
    img_height = content_height + margin * 2

    image = Image.new("RGB", (max(1, img_width), max(1, img_height)), "white")
    draw = ImageDraw.Draw(image)

    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height + line_spacing

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export directory tree to PNG.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to visualize.")
    parser.add_argument(
        "-o",
        "--output",
        default="tree.png",
        help="Output PNG path (default: tree.png).",
    )
    parser.add_argument(
        "--text-output",
        default="tree.txt",
        help="Optional plain-text output path (default: tree.txt).",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="Include files in the tree output (default: directories only).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Max depth (works on Linux/macOS `tree`; ignored on Windows built-in tree).",
    )
    parser.add_argument("--font-size", type=int, default=14, help="Font size in PNG.")
    parser.add_argument("--line-spacing", type=int, default=4, help="Extra spacing between lines.")
    parser.add_argument("--margin", type=int, default=20, help="Image margin in pixels.")

    args = parser.parse_args()

    target_dir = Path(args.directory).resolve()
    output_png = Path(args.output).resolve()
    output_txt = Path(args.text_output).resolve() if args.text_output else None

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory does not exist: {target_dir}", file=sys.stderr)
        return 1

    try:
        tree_text = run_tree_command(target_dir, args.files, args.depth)
        render_text_to_png(
            text=tree_text,
            output_png=output_png,
            font_size=args.font_size,
            line_spacing=args.line_spacing,
            margin=args.margin,
        )
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if output_txt:
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(tree_text, encoding="utf-8")
        print(f"Tree text saved: {output_txt}")

    print(f"Tree image saved: {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
