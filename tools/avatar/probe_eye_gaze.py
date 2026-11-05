#!/usr/bin/env python3
"""Show the eyes at the far ends of the gaze, and measure the iris's shape and overlap with the lids.

Why: with eye-follow Adam saw the eyes overlap the eyelids, and break up when
looking far to the side. This renders both eyes close up at the gaze extremes to
~/homeai-bench/eye-gaze-<time>/contact.png and prints, per gaze, how much the
iris's dots stretch and how many land outside the eye opening. No voice, daemon
or window is used.

Usage:
    .venv/bin/python tools/avatar/probe_eye_gaze.py [--version v2] [--size 2048]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar import assets, facegl, gl, rig, tour  # noqa: E402
from homeai.avatar.facegl import FaceRenderer  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"
GAZES = ((0.0, 0.0), (0.9, 0.0), (-0.9, 0.0), (0.0, 0.9), (0.0, -0.9), (0.9, 0.9))


def measure(asset, gaze: tuple[float, float]) -> list[dict]:
    """Per eye: size of the iris's dots, the share hidden behind the lids, and how far the rest of the eye moved."""
    rest = np.asarray(asset.verts, dtype=np.float64)
    moved = rig.deform(rest, rig.default_fields(rest), {"gaze_x": gaze[0], "gaze_y": gaze[1]})
    idx, attrs = facegl.face_buffers(asset)
    at_rest, now = facegl.face_positions(idx, attrs, rest), facegl.face_positions(idx, attrs, moved)
    rows = []
    for (region, _), eye, window in zip(facegl.IRIS_EYES, facegl.EYE_LAYERS.values(), facegl.eye_windows(rest, asset.xy, attrs[:len(asset.xy), 8])):
        pick = attrs[:, 8] == eye + facegl.IRIS_LAYER
        reach = np.linalg.norm((now[pick] - window[:2]) / window[2:], axis=1)
        white = attrs[:, 8] == eye
        rows.append({"dots": int(pick.sum()), "w": round(float(np.ptp(now[pick, 0])), 1),
                     "h": round(float(np.ptp(now[pick, 1])), 1), "hidden": round(float((reach >= 1.0).mean()), 2),
                     "fading": round(float(((reach > 0.9) & (reach < 1.0)).mean()), 2),
                     "white_moved": round(float(np.abs(now[white] - at_rest[white]).max()), 2)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="asset version (default: current)")
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()
    asset = assets.load(assets.asset_root() / args.version) if args.version else assets.load_current()
    if asset is None:
        print("no face asset: none chosen (homeai-avatar use vN)", file=sys.stderr)
        return 2
    from PIL import Image

    size = (args.size, args.size)
    ctx, fbo = gl.headless(size)
    renderer = FaceRenderer(ctx, asset, hair=False)
    side = float(asset.image_size[0])
    px = facegl.to_pixels(np.asarray(asset.verts, dtype=np.float64)[list(rig.REGIONS["eye_r"] + rig.REGIONS["eye_l"])],
                          side, size, facegl.ZOOM)
    (x0, y0), (x1, y1) = (px.min(axis=0) - 40).astype(int), (px.max(axis=0) + 40).astype(int)
    rows = []
    for gaze in GAZES:
        print(gaze, measure(asset, gaze))
        scene, t = tour.staged("idle", t0=100.0)
        frame = scene.frame(t + 0.016)
        frame.params.update({"gaze_x": gaze[0], "gaze_y": gaze[1], "head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0,
                             "blink_l": 0.0, "blink_r": 0.0})
        fbo.use()
        renderer.draw(frame, t, size)
        rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(size[1], size[0], 3)[::-1]
        rows.append(rgb[y0:y1, x0:x1])
    out = BENCH_DIR / f"eye-gaze-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    Image.fromarray(np.vstack(rows)).save(out / "contact.png")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
