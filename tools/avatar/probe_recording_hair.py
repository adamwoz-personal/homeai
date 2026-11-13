#!/usr/bin/env python3
"""Look at the "recording the room" hair: the real face, idle, recording off and on, side by side.

Why: interactive mode step 0 records the room, and Adam asked for white hair while it does (2026-11-13). This
renders both headless with the current artwork so the look can be checked by eye, and prints the mean colour of
the brightest pixels outside the face box (the hair) for both, so "it's white" is a number too.

Usage: .venv/bin/python tools/avatar/probe_recording_hair.py
Writes ~/homeai-bench/recording-hair-<stamp>.png (left: off, right: on). Needs headless GL (EGL).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar import assets, gl, tour  # noqa: E402
from homeai.avatar.facegl import FaceRenderer  # noqa: E402

SIZE = (480, 480)


def render(asset, recording: bool) -> np.ndarray:
    ctx, fbo = gl.headless(SIZE)
    renderer = FaceRenderer(ctx, asset, hair=True)
    try:
        scene, t = tour.staged("idle")
        if recording:
            scene.apply({"type": "recording", "data": {"on": True}})
        for _ in range(120):                       # two seconds: the colour eases in
            t += 1 / 60
            frame = scene.frame(t)
        renderer.draw(frame, t, SIZE)
        rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(SIZE[1], SIZE[0], 3)
        return np.flipud(rgb).copy()
    finally:
        renderer.release()
        ctx.release()


def hair_colour(rgb: np.ndarray) -> tuple[int, int, int]:
    """Mean colour of the brightest 5 % of pixels in the top fifth (hair above the head)."""
    top = rgb[: rgb.shape[0] // 5].reshape(-1, 3).astype(float)
    lum = top.sum(axis=1)
    bright = top[lum >= np.percentile(lum, 95)]
    return tuple(int(v) for v in bright.mean(axis=0))


def main() -> int:
    asset = assets.load_current()
    if asset is None:
        print("no face artwork installed")
        return 2
    off, on = render(asset, False), render(asset, True)
    out = Path.home() / "homeai-bench" / f"recording-hair-{time.strftime('%Y%m%d-%H%M%S')}.png"
    from PIL import Image
    Image.fromarray(np.concatenate([off, on], axis=1)).save(out)
    print(f"hair off: {hair_colour(off)}  on: {hair_colour(on)}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
