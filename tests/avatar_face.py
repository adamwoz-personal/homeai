"""Shared helpers for the avatar face tests: a real detected face and a small built asset.

The landmarks are MediaPipe's on candidate 3341dd3a (tools/avatar/build_face.py
--export-landmarks), so tests run on a real face shape without MediaPipe.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent / "fixtures" / "avatar" / "landmarks_3341.json"


def landmarks() -> tuple[np.ndarray, int]:
    data = json.loads(FIXTURE.read_text())
    return np.array(data["landmarks"], dtype=np.float64), int(data["image_size"][0])


def mesh():
    """(verts float32, triangles, side) as the build makes them."""
    from homeai.avatar import facebuild

    lm, side = landmarks()
    verts = np.vstack([lm, facebuild.boundary_ring(lm, side)]).astype(np.float32)
    return verts, facebuild.build_mesh(verts), side


def make_asset(out_dir: Path, n: int = 600, seed: int = 1) -> Path:
    """Write a small but complete version directory (no images, no MediaPipe)."""
    from homeai.avatar import facebuild, rig

    verts, tris, side = mesh()
    rng = np.random.default_rng(seed)
    xy = rng.uniform(5, side - 5, (n, 2))
    tri, bary = facebuild.bind(xy, verts, tris)
    facebuild.write_asset(
        out_dir, side=side, verts=verts, tris=tris, fields=rig.default_fields(verts), xy=xy,
        size=np.ones(n, np.float32), colour=np.full((n, 3), 0.5, np.float32), tri=tri, bary=bary,
        inputs={"image": "test"}, params={"seed": seed}, started=time.monotonic())
    return out_dir
