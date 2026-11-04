#!/usr/bin/env python3
"""Show where resting hair crosses into the head's outline (over the face instead of beside it).

Why: the hair is drawn additively behind the face, so a strand that falls
across the face muddies it. ``facegl.hair_rest`` replays the shader's path on
the CPU and ``facegl.inside_outline`` tests it against the same silhouette
walks the roots use; this prints the share inside per strand group and per
height band, so a bad reach or curl is a number, not a squint at a render.

Usage:
    .venv/bin/python tools/avatar/probe_hair_paths.py [--version v1] [--synthetic]
        [--crown 160] [--strands 320]
--synthetic uses the test fixture's landmarks plus a skull dome (what the unit test checks).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from homeai.avatar import assets, facegl, rig  # noqa: E402


def load(args) -> tuple[np.ndarray, np.ndarray]:
    if args.synthetic:
        from tests.avatar_face import head_with_skull, mesh
        verts = mesh()[0]
        return head_with_skull(verts), verts
    root = assets.asset_root()
    asset = assets.load(root / args.version) if args.version else assets.load_current(root)
    return facegl.lit_points(asset.xy, asset.colour), asset.verts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--crown", type=int, default=facegl.HAIR_CROWN)
    ap.add_argument("--strands", type=int, default=facegl.HAIR_STRANDS)
    args = ap.parse_args()
    xy, verts = load(args)
    spacing = facegl.point_spacing(xy)
    h = rig.face_height(verts)
    top = float(min(verts[105, 1], verts[334, 1]))
    layout = facegl.hair_layout(xy, verts, spacing, strands=args.strands, crown=args.crown)
    rest = facegl.hair_rest(layout)
    inside = facegl.inside_outline(xy, verts, spacing, rest)
    crown = layout[:, 1] < top - 1e-3
    print(f"face height {h:.0f}  brow y {top:.0f}  points {len(layout)}")
    for name, mask in (("crown", crown), ("side", ~crown)):
        print(f"{name:5s} inside {inside[mask].mean():6.1%} of {mask.sum()}")
    print("crown points inside, by rest height (x face height from the brows) and root angle:")
    rel = (rest[:, 1] - top) / h
    for lo in np.arange(-0.6, 1.6, 0.2):
        band = crown & (rel >= lo) & (rel < lo + 0.2)
        if band.any():
            print(f"  {lo:+.1f}..{lo + 0.2:+.1f}  {inside[band].mean():6.1%} of {band.sum()}")
    roots_x = (layout[:, 0] - verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean()) / h
    for lo in np.arange(-0.6, 0.6, 0.15):
        band = crown & (roots_x >= lo) & (roots_x < lo + 0.15)
        if band.any():
            print(f"  root x {lo:+.2f}..{lo + 0.15:+.2f}  {inside[band].mean():6.1%} of {band.sum()}")


if __name__ == "__main__":
    main()
