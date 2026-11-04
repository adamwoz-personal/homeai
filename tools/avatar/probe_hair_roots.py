"""Print where the aura hair roots land on an asset, row by row.

Why: the hair must hang from just outside the head's silhouette; this shows
the face oval's x range against each strand's root so a bad silhouette walk
(roots inside the cheeks, or out at the image edge) is obvious without a render.

Usage: .venv/bin/python tools/avatar/probe_hair_roots.py [--version v1] [--strands 20]
"""
from __future__ import annotations

import argparse

import numpy as np

from homeai.avatar import assets, facegl, rig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default=None)
    ap.add_argument("--strands", type=int, default=20)
    ap.add_argument("--crown", type=int, default=12)
    args = ap.parse_args()
    root = assets.asset_root()
    asset = assets.load(root / args.version) if args.version else assets.load_current(root)
    spacing = facegl.point_spacing(asset.xy)
    lit = facegl.lit_points(asset.xy, asset.colour)
    oval = asset.verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"])]
    print(f"image {asset.image_size} spacing {spacing:.1f} lit {len(lit)}/{len(asset.xy)} "
          f"oval x {oval[:, 0].min():.0f}..{oval[:, 0].max():.0f} centre {oval[:, 0].mean():.0f}")
    layout = facegl.hair_layout(lit, asset.verts, spacing, strands=args.strands, crown=args.crown)
    print(f"hair points {len(layout)} (layout at the given counts)")
    seen = set()
    for x, y, side, length, _s, _phase, _size, ax, ay, droop in layout:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        print(f"side {side:+.0f}  y {y:6.0f}  root x {x:6.0f}  len {length:5.0f}  "
              f"aim {ax:+.2f},{ay:+.2f}  droop {droop:.2f}")


if __name__ == "__main__":
    main()
