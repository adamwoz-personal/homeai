#!/usr/bin/env python3
"""Measure whether the aura hair moves like one breeze: connected, not mirrored.

Why: Adam said the hair moved like a sine wave, mirrored left and right, and
asked for a gentle breeze. That is two numbers. Over a minute of idle motion,
the sideways movement of neighbouring tips should agree (near 1), while a left
tip and its mirror image on the right should not move as reflections (a mirror
gives 1). Also renders frames a few seconds apart to
~/homeai-bench/hair-wind-<time>/ to look at. No voice, daemon or window is used.

Usage:
    .venv/bin/python tools/avatar/probe_hair_wind.py [--version v1] [--seconds 60] [--size 480]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar import assets, facegl, tour  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"
IDLE_SPEED = 0.6            # hair_phase rad/s at idle (face.FACE_STATES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="asset version (default: current)")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--size", type=int, default=480)
    args = ap.parse_args()
    asset = assets.load(assets.asset_root() / args.version) if args.version else assets.load_current()
    if asset is None:
        print("no face asset: none chosen (homeai-avatar use vN)", file=sys.stderr)
        return 2
    from PIL import Image

    spacing = facegl.point_spacing(asset.xy)
    layout = facegl.asset_hair(asset, spacing)
    face_h = facegl.head_geometry(asset.verts)["face_h"]
    pivot_x = float(facegl.head_geometry(asset.verts)["pivot"][0])
    result = tour.wind_coherence(facegl.hair_rest(layout), face_h, pivot_x, args.seconds * IDLE_SPEED)
    print(json.dumps(result))
    out = BENCH_DIR / f"hair-wind-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    shots = tour.render_wind(asset, (0.0, 12.0, 24.0, 36.0), (args.size, args.size))
    Image.fromarray(np.hstack(shots)).save(out / "contact.png")
    (out / "results.json").write_text(json.dumps(result, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
