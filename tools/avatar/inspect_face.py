#!/usr/bin/env python3
"""Render the face look headless for every pose (visemes, blink, states) into a contact sheet.

Why: tuning the face (sprite size, brightness, teeth, aura hair) and judging
viseme shapes needs eyes on real renders of the real asset, repeatably, without
the window or the voice daemon. Writes one PNG per pose plus contact.png to
~/homeai-bench/face-render-<time>/ and prints the folder.

Usage:
    .venv/bin/python tools/avatar/inspect_face.py [--version v1] [--size 480]
        [--pose AA --pose blink ...] [--no-hair] [--captions] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from homeai.avatar import assets  # noqa: E402
from homeai.avatar.tour import FACE_TOUR, render_face  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402


def contact(images: list[tuple[str, np.ndarray]], cols: int = 6) -> Image.Image:
    h, w = images[0][1].shape[:2]
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 18)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (name, rgb) in enumerate(images):
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        sheet.paste(Image.fromarray(rgb), (x, y + 18))
        draw.text((x + 4, y + 3), name, fill=(220, 220, 220))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="asset version (default: current)")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--pose", action="append", choices=FACE_TOUR)
    ap.add_argument("--no-hair", action="store_true")
    ap.add_argument("--captions", action="store_true")
    ap.add_argument("--crop", help="x0,y0,x1,y1 as fractions of the render, e.g. 0.3,0.35,0.7,0.5 for the eyes")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    crop = None
    if args.crop:
        try:
            crop = tuple(int(float(f) * args.size) for f in args.crop.split(","))
            if len(crop) != 4 or crop[0] >= crop[2] or crop[1] >= crop[3]:
                raise ValueError
        except ValueError:
            print("--crop needs x0,y0,x1,y1 fractions with x0<x1, y0<y1", file=sys.stderr)
            return 2
    try:
        asset = assets.load(assets.asset_root() / args.version) if args.version else assets.load_current()
    except (OSError, ValueError) as e:
        print(f"no face asset: {e}", file=sys.stderr)
        return 1
    if asset is None:
        print("no face asset: none chosen (homeai-avatar use vN)", file=sys.stderr)
        return 1
    out = args.out or bench_path("face-render")
    out.mkdir(parents=True, exist_ok=True)
    images = list(render_face(asset, args.pose or FACE_TOUR, (args.size, args.size),
                              hair=not args.no_hair, captions=args.captions))
    if crop:
        images = [(name, np.asarray(Image.fromarray(rgb).crop(crop))) for name, rgb in images]
    for name, rgb in images:
        Image.fromarray(rgb).save(out / f"{name}.png")
    contact(images).save(out / "contact.png")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
