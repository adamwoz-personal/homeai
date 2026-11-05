#!/usr/bin/env python3
"""Draw where scalp hair would root, over the art, for Adam to approve.

Why: Adam wants hair growing from the dark channels between the scalp's four
raised plates, never from the plates. Whether facegl.SCALP_CHANNELS covers the
right places is a judgement on the picture, so this marks the spots in red on the
dots actually drawn and on the source art, zoomed on the crown, and writes
~/homeai-bench/scalp-roots-<time>/overlay.png. Nothing live changes.

Usage:
    .venv/bin/python tools/avatar/probe_scalp_roots.py [--version v1] [--zoom 3]
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


def mark(image: np.ndarray, gaps: np.ndarray, scale: float, zoom: int) -> np.ndarray:
    """``image`` cut to the crown and enlarged, with each gap as a red dot."""
    import cv2

    rows = int(0.32 * image.shape[0])
    crop = cv2.resize(image[:rows], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    for x, y in gaps * scale * zoom:
        cv2.circle(crop, (int(x), int(y)), 3, (255, 40, 40), -1, cv2.LINE_AA)
    return crop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="asset version (default: current)")
    ap.add_argument("--zoom", type=int, default=2)
    args = ap.parse_args()
    root = assets.asset_root() / args.version if args.version else assets.asset_root() / "current"
    asset = assets.load(root)
    from PIL import Image

    quads = facegl.SCALP_CHANNELS.get(asset.build["inputs"]["image_sha256"], ())
    gaps = facegl.scalp_spots(quads, facegl.HAIR_SCALP)
    print(f"{len(quads)} channels, {len(gaps)} root spots")
    side = float(asset.image_size[0])
    dots = facebuild.render_points(asset.xy, asset.size, asset.colour, int(side), out=int(side))
    art = np.asarray(Image.open(root / "source.png").convert("RGB"))
    out = BENCH_DIR / f"scalp-roots-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    sheet = np.vstack([mark(dots, gaps, 1.0, args.zoom), mark(art, gaps, 1.0, args.zoom)])
    Image.fromarray(sheet).save(out / "overlay.png")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
