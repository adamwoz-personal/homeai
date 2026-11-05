#!/usr/bin/env python3
"""Draw where marked hair would root (scalp or neck), over the art, for Adam to approve.

Why: Adam wants hair growing from the dark channels between the scalp's four
raised plates, never from the plates. Whether facegl.SCALP_CHANNELS covers the
right places is a judgement on the picture, so this marks the spots in red on the
dots actually drawn and on the source art, zoomed on the crown, and writes
~/homeai-bench/scalp-roots-<time>/overlay.png, with a grid in crop px for marking
areas. ``--area neck`` does the same for the slots beside the neck wires. Nothing
live changes.

Usage:
    .venv/bin/python tools/avatar/probe_scalp_roots.py [--version v1] [--zoom 2] [--area scalp|neck]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar import assets, facebuild, facegl  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"


def mark(image: np.ndarray, gaps: np.ndarray, rows: tuple[int, int], zoom: int) -> np.ndarray:
    """``image`` cut to ``rows`` and enlarged, with each gap as a red dot and a 50 px grid."""
    import cv2

    crop = np.ascontiguousarray(cv2.resize(image[rows[0]:rows[1]], None, fx=zoom, fy=zoom,
                                           interpolation=cv2.INTER_LINEAR))
    for x in range(0, image.shape[1], 50):
        cv2.line(crop, (x * zoom, 0), (x * zoom, crop.shape[0]), (70, 70, 110), 1)
        cv2.putText(crop, str(x), (x * zoom + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    for y in range(-(-rows[0] // 50) * 50, rows[1], 50):
        cv2.line(crop, (0, (y - rows[0]) * zoom), (crop.shape[1], (y - rows[0]) * zoom), (70, 70, 110), 1)
        cv2.putText(crop, str(y), (2, (y - rows[0]) * zoom - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                    cv2.LINE_AA)
    for x, y in gaps:
        cv2.circle(crop, (int(x * zoom), int((y - rows[0]) * zoom)), 3, (255, 40, 40), -1, cv2.LINE_AA)
    return crop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="asset version (default: current)")
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--area", choices=("scalp", "neck"), default="scalp")
    args = ap.parse_args()
    root = assets.asset_root() / args.version if args.version else assets.asset_root() / "current"
    asset = assets.load(root)
    from PIL import Image

    table, count = {"scalp": (facegl.SCALP_CHANNELS, facegl.HAIR_SCALP),
                    "neck": (facegl.NECK_CHANNELS, facegl.HAIR_NECK)}[args.area]
    quads = table.get(asset.build["inputs"]["image_sha256"], ())
    gaps = facegl.scalp_spots(quads, count)
    rows = (0, 245) if args.area == "scalp" else (540, 768)
    print(f"{len(quads)} channels, {len(gaps)} root spots")
    side = float(asset.image_size[0])
    dots = facebuild.render_points(asset.xy, asset.size, asset.colour, int(side), out=int(side))
    art = np.asarray(Image.open(root / "source.png").convert("RGB"))
    out = BENCH_DIR / f"scalp-roots-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    sheet = np.vstack([mark(dots, gaps, rows, args.zoom), mark(art, gaps, rows, args.zoom)])
    Image.fromarray(sheet).save(out / "overlay.png")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
