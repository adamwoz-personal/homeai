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
    args = ap.parse_args()
    root = assets.asset_root()
    asset = assets.load(root / args.version) if args.version else assets.load_current(root)
    spacing = facegl.point_spacing(asset.xy)
    lit = facegl.lit_points(asset.xy, asset.colour)
    oval = asset.verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"])]
    print(f"image {asset.image_size} spacing {spacing:.1f} lit {len(lit)}/{len(asset.xy)} "
          f"oval x {oval[:, 0].min():.0f}..{oval[:, 0].max():.0f} centre {oval[:, 0].mean():.0f}")
    layout = facegl.hair_layout(lit, asset.verts, spacing, strands=args.strands,
                                points=args.strands * 8)
    seen = set()
    for x, y, side, length, *_ in layout:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        print(f"side {side:+.0f}  y {y:6.0f}  root x {x:6.0f}  len {length:5.0f}")


if __name__ == "__main__":
    main()
