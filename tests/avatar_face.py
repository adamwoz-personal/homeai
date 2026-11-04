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


def head_with_skull(verts):
    """The face landmarks plus a dome of dots above the brows, as real art has (landmarks stop
    at the hairline): the head's outline for hair tests, without the mesh's frame ring."""
    from homeai.avatar import rig

    lm = landmarks()[0].astype(np.float64)
    h = rig.face_height(verts)
    top = min(verts[105, 1], verts[334, 1])
    cx = verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean()
    half = (lm[:, 0].max() - lm[:, 0].min()) / 2
    g = np.stack(np.meshgrid(np.linspace(-1, 1, 40), np.linspace(0, 1, 25)), -1).reshape(-1, 2)
    g = g[np.hypot(g[:, 0], g[:, 1]) <= 1.0]
    return np.vstack([lm, np.column_stack([cx + g[:, 0] * half, top - g[:, 1] * 0.55 * h])])


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
