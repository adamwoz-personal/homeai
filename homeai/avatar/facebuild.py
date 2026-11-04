"""Build a face asset from art: landmarks, mesh, stipple, binding, preview (AVATAR_PLAN §5).

Build time only (cv2, scipy, mediapipe; the ``avatar-build`` extra). Driven by
``tools/avatar/build_face.py``; the output is read by ``assets.load``.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from . import rig, stipple
from .assets import FORMAT, HASHED, sha256

MODEL_PATH = Path.home() / ".local/share/homeai/models/face_landmarker.task"
DETECT_BLUR = 9             # px; the Phase 0 probe found all 478 landmarks on every candidate with this
RING_SCALE = 1.35           # boundary ellipse around the face oval
RING_POINTS = 24
BORDER_STEPS = 8
MIN_AREA = 0.5              # px^2
MIN_ANGLE = 2.0             # degrees
BIND_CANDIDATES = 12
DEFAULTS = {"points": 14000, "seed": 7, "iterations": 30, "blur_sigma": 2.0, "gamma": 0.5}
POINT_RANGE = (8000, 20000)  # §4.6


class BuildError(RuntimeError):
    pass


# ---- geometry (pure) -----------------------------------------------------------------

def square_crop(width: int, height: int, centre_x: float) -> tuple[int, int, int]:
    """(x0, y0, side): the largest square around ``centre_x`` that fits in the image."""
    side = min(width, height)
    x0 = int(round(min(max(centre_x - side / 2, 0), width - side)))
    return x0, (height - side) // 2, side


def boundary_ring(face: np.ndarray, size: int) -> np.ndarray:
    """Fixed vertices around the face: an ellipse beyond the oval, plus the crop border."""
    centre = face.mean(axis=0)
    half = (face.max(axis=0) - face.min(axis=0)) / 2 * RING_SCALE
    a = np.linspace(0, 2 * np.pi, RING_POINTS, endpoint=False)
    ellipse = centre + np.column_stack([np.cos(a), np.sin(a)]) * half
    ellipse = np.clip(ellipse, 1, size - 1)
    edge = np.linspace(0, size, BORDER_STEPS + 1)
    border = np.unique(np.concatenate([
        np.column_stack([edge, np.zeros_like(edge)]), np.column_stack([edge, np.full_like(edge, size)]),
        np.column_stack([np.zeros_like(edge), edge]), np.column_stack([np.full_like(edge, size), edge])]),
        axis=0)
    return np.vstack([ellipse, border])


def triangle_quality(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(area px^2, smallest angle in degrees) per triangle."""
    a, b, c = (verts[tris[:, i]].astype(np.float64) for i in range(3))
    area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 2
    angles = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u, v = q - p, r - p
        cos = (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
        angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return area, np.min(angles, axis=0)


def build_mesh(verts: np.ndarray) -> np.ndarray:
    """Delaunay over all vertices, minus triangles spanning the mouth slit and slivers."""
    from scipy.spatial import Delaunay

    tris = Delaunay(verts).simplices.astype(np.int32)
    regions = rig.face_regions(verts)
    slit = np.array([rig.spans_mouth(t, regions) for t in tris])
    area, angle = triangle_quality(verts, tris)
    keep = ~slit & (area >= MIN_AREA) & (angle >= MIN_ANGLE)
    return tris[keep]


def barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(..., 3) weights of ``p`` in triangles (a, b, c); exact even outside the triangle."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], axis=-1)


def bind(points: np.ndarray, verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangle and barycentric weights per point: the containing triangle, or the nearest one.

    Points in a dropped triangle (mouth slit, sliver) get slightly negative
    weights on a neighbour, which still reconstructs them exactly at rest.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    v = verts.astype(np.float64)
    centroids = v[tris].mean(axis=1)
    k = min(BIND_CANDIDATES, len(tris))
    _, cand = cKDTree(centroids).query(pts, k=k, workers=-1)
    cand = cand.reshape(len(pts), k)
    tv = v[tris[cand]]                                       # (N, k, 3, 2)
    w = barycentric(pts[:, None, :], tv[..., 0, :], tv[..., 1, :], tv[..., 2, :])
    best = np.argmax(w.min(axis=-1), axis=1)                 # inside, or least outside
    rows = np.arange(len(pts))
    return cand[rows, best].astype(np.int32), w[rows, best].astype(np.float32)


# ---- preview (cv2) ---------------------------------------------------------------------

def render_points(xy: np.ndarray, size: np.ndarray, colour: np.ndarray, image_size: int,
                  out: int = 1080, glow: float = 0.35, dot: float = 0.3) -> np.ndarray:
    """RGB uint8 still of glowing dots, ``out`` px square: what Adam judges the look by.

    ``dot`` is the radius as a share of the typical point spacing (0.5 = touching).
    """
    import cv2
    from scipy.spatial import cKDTree

    scale = out / image_size
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])) if len(xy) > 1 else 4.0
    radius = dot * spacing * scale
    canvas = np.zeros((out, out, 3), dtype=np.float32)
    ss = 4                                                   # sub-pixel shift bits for cv2.circle
    for (x, y), s, c in zip(xy * scale, size, colour):
        cv2.circle(canvas, (int(x * (1 << ss)), int(y * (1 << ss))), max(1, int(radius * s * (1 << ss))),
                   tuple(float(v) for v in c), -1, cv2.LINE_AA, ss)
    halo = cv2.GaussianBlur(canvas, (0, 0), max(1.0, radius * 3))
    img = np.clip(canvas + glow * halo, 0, 1) ** (1 / 1.2)
    return (img * 255).astype(np.uint8)


def draw_mesh(rgb: np.ndarray, verts: np.ndarray, tris: np.ndarray, n_face: int = rig.N_FACE) -> np.ndarray:
    """The crop with the mesh (grey), face landmarks (red) and rig regions (yellow) drawn on."""
    import cv2

    img = rgb.copy()
    for t in tris:
        cv2.polylines(img, [verts[t].astype(np.int32)], True, (90, 90, 90), 1, cv2.LINE_AA)
    rigged = {i for idx in rig.face_regions(verts).values() for i in idx}
    for i, (x, y) in enumerate(verts):
        colour = (255, 220, 0) if i in rigged else (255, 40, 40) if i < n_face else (80, 160, 255)
        cv2.circle(img, (int(x), int(y)), 2, colour, -1, cv2.LINE_AA)
    return img


# ---- detection (mediapipe) -------------------------------------------------------------

def detect_landmarks(rgb: np.ndarray, model: Path = MODEL_PATH) -> np.ndarray:
    """(478, 2) pixel landmarks found on the blurred art; BuildError if no face."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

    if not model.exists():
        raise BuildError(f"{model} missing; run tools/avatar/landmarks_probe.py --download")
    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)), num_faces=1,
                                    min_face_detection_confidence=0.1, min_face_presence_confidence=0.1)
    blurred = np.ascontiguousarray(cv2.GaussianBlur(rgb, (DETECT_BLUR, DETECT_BLUR), 0))
    with FaceLandmarker.create_from_options(options) as landmarker:
        found = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=blurred))
    if not found.face_landmarks:
        raise BuildError("no face found; place the landmarks with tools/avatar/landmark_editor.py")
    h, w = rgb.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in found.face_landmarks[0]], dtype=np.float64)
    if len(pts) != rig.N_FACE:
        raise BuildError(f"expected {rig.N_FACE} landmarks, got {len(pts)}")
    return pts


def apply_overrides(landmarks: np.ndarray, overrides: dict) -> np.ndarray:
    """Editor corrections ``{"landmarks": {"13": [x, y], ...}}`` (crop pixels) over the detected ones."""
    out = landmarks.copy()
    for key, xy in (overrides.get("landmarks") or {}).items():
        i = int(key)
        if not 0 <= i < len(out) or len(xy) != 2:
            raise BuildError(f"override for landmark {key!r} is invalid")
        out[i] = xy
    return out


# ---- the build -------------------------------------------------------------------------

def load_crop(image: Path) -> tuple[np.ndarray, dict]:
    """The square crop around the face (RGB) and how it was cut."""
    import cv2

    bgr = cv2.imread(str(image))
    if bgr is None:
        raise BuildError(f"{image}: unreadable")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    full = detect_landmarks(rgb)
    x0, y0, side = square_crop(w, h, float(full[:, 0].mean()))
    return np.ascontiguousarray(rgb[y0:y0 + side, x0:x0 + side]), {"x0": x0, "y0": y0, "side": side}


def build(image: Path, out_dir: Path, *, points: int = DEFAULTS["points"], seed: int = DEFAULTS["seed"],
          iterations: int = DEFAULTS["iterations"], overrides: dict | None = None,
          blur_sigma: float = DEFAULTS["blur_sigma"], gamma: float = DEFAULTS["gamma"]) -> dict:
    """Write one version directory; returns build.json's contents."""
    import cv2
    from PIL import Image

    if not POINT_RANGE[0] <= points <= POINT_RANGE[1]:
        raise BuildError(f"points must be within {POINT_RANGE}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise BuildError(f"{out_dir} exists; versions are never overwritten")
    started = time.monotonic()
    crop, cut = load_crop(image)
    side = cut["side"]
    if overrides and overrides.get("crop") not in (None, cut):
        raise BuildError(f"overrides were made on crop {overrides['crop']}, this build cuts {cut}")
    landmarks = apply_overrides(detect_landmarks(crop), overrides or {})
    verts = np.vstack([landmarks, boundary_ring(landmarks, side)]).astype(np.float32)
    tris = build_mesh(verts)
    fields = rig.default_fields(verts)

    density = stipple.density_map(crop, blur_sigma, gamma)
    xy = stipple.stipple(density, points, seed, iterations)
    blurred = cv2.GaussianBlur(crop, (0, 0), blur_sigma).astype(np.float64) / 255.0
    colour = stipple.sample(blurred, xy)
    lit = blurred[density > 0]
    top = float(np.percentile(lit.max(axis=1), 99.5)) if lit.size else 1.0
    colour = np.clip(colour / max(top, 1e-6), 0, 1).astype(np.float32)
    size = stipple.sizes(stipple.sample(density, xy))
    tri, bary = bind(xy, verts, tris)

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_points(xy, size, colour, side)).save(out_dir / "preview.png")
    Image.fromarray(crop).save(out_dir / "source.png")
    return write_asset(
        out_dir, side=side, verts=verts, tris=tris, fields=fields, xy=xy, size=size, colour=colour,
        tri=tri, bary=bary, started=started,
        inputs={"image": str(image), "image_sha256": sha256(image), "crop": cut, "overrides": overrides or {}},
        params={"points": points, "seed": seed, "iterations": iterations, "blur_sigma": blur_sigma,
                "gamma": gamma})


_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
         176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
_EYES = ((33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
         (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466))
_BROWS = ((70, 63, 105, 66, 107, 55, 65, 52, 53, 46), (300, 293, 334, 296, 336, 285, 295, 282, 283, 276))


def synthetic_face(landmarks: np.ndarray, side: int, out_dir: Path, points: int = 6000, seed: int = 1) -> dict:
    """A plain stippled face drawn from landmarks alone (numpy only): test and golden asset.

    Bright oval with dark eyes, bright brows and lips, and a dark mouth slit,
    coloured by height so pose and lid motion show. Independent of any artwork.
    """
    started = time.monotonic()
    lm = np.asarray(landmarks, dtype=np.float64)
    verts = np.vstack([lm, boundary_ring(lm, side)]).astype(np.float32)
    tris = build_mesh(verts)
    rng = np.random.default_rng(seed)
    cand = rng.uniform(0, side, (points * 6, 2))
    dens = np.where(rig._inside(cand, lm[list(_OVAL)]), 0.55, 0.0)
    irises = np.zeros(len(cand), bool)
    for eye in _EYES:
        inside = rig._inside(cand, lm[list(eye)])
        dens[inside] = 0.08
        ring = lm[list(eye)]
        r = 0.5 * float(np.ptp(ring[:, 1]))
        irises |= inside & (np.linalg.norm(cand - ring.mean(axis=0), axis=1) < r)
    dens[irises] = 1.0                                      # a bright iris, so lids and eye tint show
    for brow in _BROWS:
        dens[rig._inside(cand, lm[list(brow)])] = 1.0
    lips = rig._inside(cand, lm[list(rig._LIPS_UPPER_OUTER + rig._LIPS_LOWER_OUTER[::-1])])
    slit = rig._inside(cand, lm[list(rig._LIPS_UPPER_INNER + rig._LIPS_LOWER_INNER[::-1])])
    dens[lips] = 0.95
    dens[slit] = 0.0
    sel = rng.random(len(cand)) < dens
    keep = cand[sel][:points]
    iris = irises[sel][:points]
    if len(keep) < points // 2:
        raise BuildError(f"synthetic face kept only {len(keep)} points")
    d = np.zeros(len(keep))
    for value, mask in ((0.55, rig._inside(keep, lm[list(_OVAL)])),
                        (0.95, rig._inside(keep, lm[list(rig._LIPS_UPPER_OUTER + rig._LIPS_LOWER_OUTER[::-1])]))):
        d[mask] = value
    y = keep[:, 1] / side
    colour = np.column_stack([0.25 + 0.5 * y, 0.95 - 0.3 * y, 0.45 + 0.4 * (1 - y)]).astype(np.float32)
    colour[iris] = 0.9
    d[iris] = 0.9
    tri, bary = bind(keep, verts, tris)
    return write_asset(out_dir, side=side, verts=verts, tris=tris, fields=rig.default_fields(verts), xy=keep,
                       size=stipple.sizes(d).astype(np.float32), colour=colour, tri=tri, bary=bary,
                       inputs={"image": "synthetic"}, params={"points": points, "seed": seed}, started=started)


def write_asset(out_dir: Path, *, side: int, verts: np.ndarray, tris: np.ndarray, fields: dict,
                xy: np.ndarray, size: np.ndarray, colour: np.ndarray, tri: np.ndarray, bary: np.ndarray,
                inputs: dict, params: dict, started: float) -> dict:
    """rig.json, points.npz and build.json (with their hashes, written last); returns build.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rig_json = {"format": FORMAT, "image_size": [side, side], "vertices": verts.round(3).tolist(),
                "triangles": tris.tolist(), "regions": {**{k: list(v) for k, v in rig.face_regions(verts).items()},
                                                        "outer": list(range(rig.N_FACE, len(verts)))},
                "fields": {k: v.round(3).tolist() for k, v in fields.items()}}
    (out_dir / "rig.json").write_text(json.dumps(rig_json))
    np.savez_compressed(out_dir / "points.npz", xy=xy.astype(np.float32), size=size, colour=colour,
                        tri=tri, bary=bary)
    manifest = {
        "format": FORMAT, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "inputs": inputs, "params": params,
        "counts": {"points": len(xy), "vertices": len(verts), "triangles": len(tris)},
        "tools": _tool_versions(), "seconds": round(time.monotonic() - started, 1),
        "files": {name: sha256(out_dir / name) for name in HASHED},
    }
    (out_dir / "build.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _tool_versions() -> dict:
    """Installed versions from package metadata (importing mediapipe would start PortAudio)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "opencv-python", "opencv-python-headless", "mediapipe"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out
