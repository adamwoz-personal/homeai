#!/usr/bin/env python3
"""GPU time per frame of the face look as the aura hair grows.

Why: avatar_profile measures CPU and GPU memory, but hair points cost GPU draw
time (fill rate on the iGPU), which neither shows. Before pushing the hair
budget, measure what each extra strand actually costs. Renders the real face
asset headless (EGL) at the window size, with no hair and with the hair layout
scaled by each factor (crown and side strands), and reports the GPU timer-query
median per frame. Writes JSON to ~/homeai-bench/bench_hair_gpu-<time>.json.

Usage:
    .venv/bin/python tools/avatar/bench_hair_gpu.py [--size 720] [--frames 120]
        [--scale 1 --scale 1.5 --scale 2 ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar import assets, facegl, gl  # noqa: E402
from homeai.avatar.tour import staged_face  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402


def measure(asset, size: int, frames: int, hair: bool, scale: float) -> dict:
    """Median GPU and wall ms per frame, and hair points, for one configuration."""
    original = facegl.hair_layout

    def scaled(xy, verts, spacing, **kw):
        return original(xy, verts, spacing, strands=round(facegl.HAIR_STRANDS * scale),
                        crown=round(facegl.HAIR_CROWN * scale), **kw)

    facegl.hair_layout = scaled
    ctx, _fbo = gl.headless((size, size))
    try:
        renderer = facegl.FaceRenderer(ctx, asset, hair=hair)
        points = renderer.hair_points if hair and hasattr(renderer, "hair_points") else None
        scene, t0 = staged_face("AA")
        gpu, wall = [], []
        for i in range(frames + 10):
            t = t0 + i / 60.0
            frame = scene.frame(t)
            q = ctx.query(time=True)
            start = time.perf_counter()
            with q:
                renderer.draw(frame, t, (size, size))
            ctx.finish()
            if i >= 10:                                     # skip warm-up (shader compile, uploads)
                wall.append((time.perf_counter() - start) * 1e3)
                gpu.append(q.elapsed / 1e6)
        renderer.release()
    finally:
        ctx.release()
        facegl.hair_layout = original
    return {"hair": hair, "scale": scale, "hair_points": points,
            "gpu_ms": round(statistics.median(gpu), 3), "wall_ms": round(statistics.median(wall), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=720, help="square window px (the live window is 720)")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--scale", type=float, action="append", help="hair strand multiplier (default 1, 1.5, 2, 3)")
    args = ap.parse_args()
    asset = assets.load_current()
    if asset is None:
        print("no face asset chosen (homeai-avatar use vN)", file=sys.stderr)
        return 1
    rows = [measure(asset, args.size, args.frames, False, 0.0)]
    for s in args.scale or [1.0, 1.5, 2.0, 3.0]:
        rows.append(measure(asset, args.size, args.frames, True, s))
    for r in rows:
        print(json.dumps(r))
    out = bench_path("bench_hair_gpu", ".json")
    out.write_text(json.dumps({"size": args.size, "frames": args.frames, "rows": rows}, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
