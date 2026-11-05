#!/usr/bin/env python3
"""Lay an exact 1-inch square grid over a map, for printing and putting figures on.

Why: image models can't draw an even grid, so the grid is drawn in code (AVATAR_PLAN
§8). The map is scaled and cropped to fill the page, and a line lands every --dpi
pixels, so at 100% print scale every square is exactly one inch. Writes
<image>-grid.png and a PDF that keeps the size.

Usage:
    .venv/bin/python tools/art/print_grid.py map.png [--squares 8x10] [--dpi 300] [--line 3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def grid_page(image: Image.Image, across: int, down: int, dpi: int = 300, line: int = 3) -> Image.Image:
    """``image`` filling across x down inches, with a dark line on every inch and round the edge."""
    width, height = across * dpi, down * dpi
    page = ImageOps.fit(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(page, "RGBA")
    for x in range(0, width + 1, dpi):
        draw.rectangle((x - line // 2, 0, x - line // 2 + line - 1, height), fill=(0, 0, 0, 170))
    for y in range(0, height + 1, dpi):
        draw.rectangle((0, y - line // 2, width, y - line // 2 + line - 1), fill=(0, 0, 0, 170))
    return page


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--squares", default="8x10", help="across x down; 8x10 fits US Letter with margins")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--line", type=int, default=3)
    args = ap.parse_args()
    across, down = (int(v) for v in args.squares.split("x"))
    with Image.open(args.image) as source:
        page = grid_page(source, across, down, args.dpi, args.line)
    out = args.image.with_name(f"{args.image.stem}-grid")
    page.save(out.with_suffix(".png"), dpi=(args.dpi, args.dpi))
    page.save(out.with_suffix(".pdf"), resolution=float(args.dpi))
    print(out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
