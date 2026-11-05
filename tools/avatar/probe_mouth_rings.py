#!/usr/bin/env python3
"""Show the rings beside the mouth on the art, and measure whether they keep their shape.

Why: Adam saw the circles at the mouth corners turn oval when the mouth opens.
They have to be marked on the art before they can move as rigid pieces, so this
zooms the source art and the dots on the mouth, with the lip corner landmarks in
red, marked rings (facegl.MOUTH_RINGS) in yellow and a grid in crop px, and writes
~/homeai-bench/mouth-rings-<time>/overlay.png. It also prints each ring's height
and width at rest and at full jaw, bound to the mesh and rigid, and renders the
AA mouth both ways to before-after.png. Nothing live changes.

Usage:
    .venv/bin/python tools/avatar/probe_mouth_rings.py [--version v2] [--zoom 4]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar import assets, facebuild, facegl, rig, tour  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"
CORNERS = (61, 291, 78, 308)


def crop_marked(image: np.ndarray, box: tuple[int, int, int, int], marks: np.ndarray, zoom: int) -> np.ndarray:
    """``image`` cut to ``box`` (x0, y0, x1, y1) and enlarged, with a 10 px grid and each mark in red."""
    import cv2

    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(cv2.resize(image[y0:y1, x0:x1], None, fx=zoom, fy=zoom,
                                           interpolation=cv2.INTER_CUBIC))
    for x in range(-(-x0 // 10) * 10, x1, 10):
        cv2.line(crop, ((x - x0) * zoom, 0), ((x - x0) * zoom, crop.shape[0]), (70, 70, 110), 1)
        if x % 50 == 0:
            cv2.putText(crop, str(x), ((x - x0) * zoom + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                        cv2.LINE_AA)
    for y in range(-(-y0 // 10) * 10, y1, 10):
        cv2.line(crop, (0, (y - y0) * zoom), (crop.shape[1], (y - y0) * zoom), (70, 70, 110), 1)
        if y % 50 == 0:
            cv2.putText(crop, str(y), (2, (y - y0) * zoom - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                        cv2.LINE_AA)
    for x, y in marks:
        cv2.circle(crop, (int((x - x0) * zoom), int((y - y0) * zoom)), 3, (255, 40, 40), -1, cv2.LINE_AA)
    return crop


def ring_sizes(asset, rings) -> list[dict]:
    """Width and height of each ring's dots at rest and at full jaw, bound to the mesh and rigid."""
    rest = np.asarray(asset.verts, dtype=np.float64)
    opened = rig.deform(rest, rig.default_fields(rest), {"jaw_open": 1.0})
    table, facegl.MOUTH_RINGS = facegl.MOUTH_RINGS, {}
    loose = facegl.face_buffers(asset)
    facegl.MOUTH_RINGS = table
    tight = facegl.face_buffers(asset)
    rows = []
    for centre, radii in rings:
        pick = np.flatnonzero((((asset.xy - centre) / radii) ** 2).sum(axis=1) <= 1.0)
        row = {"points": len(pick)}
        for name, buffers, verts in (("rest", tight, rest), ("mesh_open", loose, opened), ("rigid_open", tight, opened)):
            row[name] = np.round(np.ptp(facegl.face_positions(*buffers, verts)[pick], axis=0), 1).tolist()
        rows.append(row)
    return rows


def before_after(asset, corners: np.ndarray, size: int = 2048) -> np.ndarray:
    """The AA mouth, bound to the mesh on top and rigid below, cut from a real render."""
    side = float(asset.image_size[0])
    px = facegl.to_pixels(corners, side, (size, size), facegl.ZOOM)
    (x0, y0), (x1, y1) = (px.min(axis=0) - 0.09 * size).astype(int), (px.max(axis=0) + 0.11 * size).astype(int)
    table = facegl.MOUTH_RINGS
    facegl.MOUTH_RINGS = {}
    mesh = dict(tour.render_face(asset, ("AA",), (size, size), hair=False))["AA"]
    facegl.MOUTH_RINGS = table
    rigid = dict(tour.render_face(asset, ("AA",), (size, size), hair=False))["AA"]
    return np.vstack([mesh[y0:y1, x0:x1], rigid[y0:y1, x0:x1]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="asset version (default: current)")
    ap.add_argument("--zoom", type=int, default=4)
    args = ap.parse_args()
    root = assets.asset_root() / (args.version or "current")
    asset = assets.load(root)
    from PIL import Image

    v = np.asarray(asset.verts, dtype=np.float64)
    marks = v[list(CORNERS)]
    print("corners:", {i: tuple(np.round(v[i], 1).tolist()) for i in CORNERS})
    pad = 45
    x0, y0 = (marks.min(axis=0) - pad).astype(int)
    x1, y1 = (marks.max(axis=0) + pad).astype(int)
    box = (int(x0), int(y0), int(x1), int(y1))
    side = int(asset.image_size[0])
    dots = facebuild.render_points(asset.xy, asset.size, asset.colour, side, out=side)
    art = np.asarray(Image.open(root / "source.png").convert("RGB"))
    out = BENCH_DIR / f"mouth-rings-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    rings = facegl.MOUTH_RINGS.get(asset.build["inputs"]["image_sha256"], ())
    sheet = np.vstack([crop_marked(art, box, marks, args.zoom), crop_marked(dots, box, marks, args.zoom)])
    import cv2

    for (cx, cy), (rx, ry) in rings:
        for top in (0, sheet.shape[0] // 2):
            cv2.ellipse(sheet, (int((cx - box[0]) * args.zoom), int((cy - box[1]) * args.zoom) + top),
                        (int(rx * args.zoom), int(ry * args.zoom)), 0, 0, 360, (255, 220, 0), 1, cv2.LINE_AA)
    Image.fromarray(sheet).save(out / "overlay.png")
    for row in ring_sizes(asset, rings):
        print(row)
    Image.fromarray(before_after(asset, marks)).save(out / "before-after.png")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
