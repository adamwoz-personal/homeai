"""Face rig: landmark regions, default displacement fields, deformation (AVATAR_PLAN §4.5, §5).

Pure numpy, so the avatar window can use it; the build (``facebuild.py``) may
also need scipy/cv2. Coordinates are source-image pixels, y down.

Vertices are the 478 MediaPipe Face Mesh landmarks followed by a boundary ring
(hair, neck, background) that never moves. Each rig parameter is a field: one
(dx, dy) per vertex at parameter value 1, blended linearly. Stipple points
follow through barycentric binding to the mesh triangles.

Left/right are the subject's (MediaPipe's convention): the subject's right eye
is on the image's left.
"""
from __future__ import annotations

import numpy as np

N_FACE = 478

_LIPS_UPPER_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
_LIPS_UPPER_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
_LIPS_LOWER_OUTER = (146, 91, 181, 84, 17, 314, 405, 321, 375)
_LIPS_LOWER_INNER = (95, 88, 178, 87, 14, 317, 402, 318, 324)
EYE_R_UPPER = (246, 161, 160, 159, 158, 157, 173)
EYE_R_LOWER = (33, 7, 163, 144, 145, 153, 154, 155, 133)
EYE_L_UPPER = (466, 388, 387, 386, 385, 384, 398)
EYE_L_LOWER = (263, 249, 390, 373, 374, 380, 381, 382, 362)
BROW_R_INNER = (107, 66, 55, 65)
BROW_L_INNER = (336, 296, 285, 295)

REGIONS: dict[str, tuple[int, ...]] = {
    "jaw": (361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132,
            199, 200, 175, 18, 83, 313, 428, 208),
    "upper_lip": _LIPS_UPPER_OUTER + _LIPS_UPPER_INNER,      # includes the corners
    "lower_lip": _LIPS_LOWER_OUTER + _LIPS_LOWER_INNER,
    "eye_r": EYE_R_UPPER + EYE_R_LOWER,
    "eye_l": EYE_L_UPPER + EYE_L_LOWER,
    "iris_r": (468, 469, 470, 471, 472),
    "iris_l": (473, 474, 475, 476, 477),
    "brow_r": (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    "brow_l": (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    "nose": (1, 2, 4, 5, 6, 168, 197, 195, 98, 327, 49, 279, 129, 358, 64, 294),
    "cheeks": (50, 280, 205, 425, 187, 411, 123, 352, 116, 345, 117, 346),
    "forehead": (10, 338, 297, 332, 284, 251, 389, 356, 454, 234, 127, 162, 21, 54, 103, 67, 109, 151, 9),
}
MOUTH_SLIT = frozenset(_LIPS_UPPER_INNER + _LIPS_LOWER_INNER)
MOUTH_CORNERS = frozenset((61, 291, 78, 308))


def _inside(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for many points."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (crosses & (x < xs)).sum(axis=1) % 2 == 1


def lip_regions(verts: np.ndarray) -> dict[str, tuple[int, ...]]:
    """Upper and lower lip including the mesh's mid-lip rings, found from where they lie.

    The named contours miss the rings between them (e.g. 42, 74); a vertex
    inside the outer lip contour belongs to the lip on its side of the inner line.
    """
    v = np.asarray(verts, dtype=np.float64)[:N_FACE]
    poly = v[list(_LIPS_UPPER_OUTER) + list(reversed(_LIPS_LOWER_OUTER))]
    named = set(REGIONS["upper_lip"]) | set(REGIONS["lower_lip"])
    up = sorted(_LIPS_UPPER_INNER, key=lambda i: v[i, 0])
    lo = sorted(_LIPS_LOWER_INNER + (78, 308), key=lambda i: v[i, 0])
    upper, lower = list(REGIONS["upper_lip"]), list(REGIONS["lower_lip"])
    for i in np.nonzero(_inside(v, poly))[0]:
        if int(i) in named:
            continue
        line = (np.interp(v[i, 0], v[up, 0], v[up, 1]) + np.interp(v[i, 0], v[lo, 0], v[lo, 1])) / 2
        (upper if v[i, 1] < line else lower).append(int(i))
    return {"upper_lip": tuple(upper), "lower_lip": tuple(lower)}


def face_regions(verts: np.ndarray) -> dict[str, tuple[int, ...]]:
    """REGIONS with the lips completed for this face (what the build stores in rig.json)."""
    return {**REGIONS, **lip_regions(verts)}


def spans_mouth(tri, regions: dict) -> bool:
    """True if a triangle joins the upper and lower lips across the gap (it must not exist,
    or opening the jaw would stretch dots over the open mouth). The corners belong to
    both lips, so corner triangles are kept: that is the lip stretching, as it does.
    """
    t = {int(i) for i in tri} - MOUTH_CORNERS
    return bool(t & set(regions["upper_lip"]) and t & set(regions["lower_lip"]))
FIELD_PARAMS = ("jaw_open", "lips_close", "lips_round", "lips_wide", "lip_bite", "blink_r", "blink_l",
                "brow_raise", "brow_frown", "gaze_x", "gaze_y")
MAX_FIELD = 0.25            # largest allowed displacement, as a share of the face height


class RigError(ValueError):
    pass


def face_height(verts: np.ndarray) -> float:
    """Forehead top (10) to chin (152), pixels."""
    return float(np.linalg.norm(verts[10] - verts[152]))


def _lids(verts, upper, lower, close):
    """Upper lid down onto the lower lid (``close`` of the gap), lower lid up the rest."""
    f = np.zeros_like(verts)
    lo = sorted(lower, key=lambda i: verts[i, 0])
    up = sorted(upper + (lower[0], lower[-1]), key=lambda i: verts[i, 0])   # corners close the curve
    for i in upper:
        f[i, 1] = (np.interp(verts[i, 0], verts[lo, 0], verts[lo, 1]) - verts[i, 1]) * close
    for i in lower:
        f[i, 1] = (np.interp(verts[i, 0], verts[up, 0], verts[up, 1]) - verts[i, 1]) * (1 - close)
    return f


def default_fields(verts: np.ndarray) -> dict[str, np.ndarray]:
    """Fields computed from where the landmarks are; Phase 4 may hand-tune them.

    Every field is (V, 2) float32 and zero on the boundary ring.
    """
    verts = np.asarray(verts, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < N_FACE:
        raise RigError(f"need at least {N_FACE} (x, y) vertices, got {verts.shape}")
    h = face_height(verts)
    if h <= 0:
        raise RigError("face height is zero")
    face = verts[:N_FACE]
    x, y = face[:, 0], face[:, 1]
    cx = (verts[61, 0] + verts[291, 0]) / 2
    half_mouth = abs(verts[291, 0] - verts[61, 0]) / 2 or 1.0
    half_jaw = abs(verts[361, 0] - verts[132, 0]) / 2 or 1.0
    lips_by_side = lip_regions(verts)
    upper, lower = np.array(lips_by_side["upper_lip"]), np.array(lips_by_side["lower_lip"])
    lips = np.concatenate([upper, lower])

    def blank():
        return np.zeros_like(verts)

    fields = {}
    f = blank()
    below = np.clip((y - verts[13, 1]) / (0.04 * h), 0, 1)              # skin under the mouth ramps in
    below[lower] = 1.0                                                   # the lower lip is all jaw: the gap opens
    sideways = np.clip(1 - 0.7 * ((x - cx) / half_jaw) ** 2, 0, 1)     # jaw corners move least
    f[:N_FACE, 1] = 0.11 * h * below * sideways
    fields["jaw_open"] = f

    f = blank()
    f[upper, 1], f[lower, 1] = 0.012 * h, -0.012 * h                     # pressed: lips thin
    fields["lips_close"] = f

    f = blank()
    f[lips, 0] = -(verts[lips, 0] - cx) * 0.3
    f[upper, 1], f[lower, 1] = -0.01 * h, 0.01 * h
    fields["lips_round"] = f

    f = blank()
    f[lips, 0] = (verts[lips, 0] - cx) * 0.15
    fields["lips_wide"] = f

    f = blank()
    f[lower, 1] = -0.02 * h
    fields["lip_bite"] = f

    fields["blink_r"] = _lids(verts, EYE_R_UPPER, EYE_R_LOWER, 0.85)
    fields["blink_l"] = _lids(verts, EYE_L_UPPER, EYE_L_LOWER, 0.85)

    f = blank()
    f[list(REGIONS["brow_r"] + REGIONS["brow_l"]), 1] = -0.035 * h
    fields["brow_raise"] = f

    f = blank()
    for inner in (BROW_R_INNER, BROW_L_INNER):
        idx = list(inner)
        f[idx, 1] = 0.015 * h
        f[idx, 0] = -np.sign(verts[idx, 0] - cx) * 0.012 * h
    fields["brow_frown"] = f

    f, g = blank(), blank()
    for iris, corners in (("iris_r", (33, 133)), ("iris_l", (263, 362))):
        eye_w = abs(verts[corners[0], 0] - verts[corners[1], 0])
        f[list(REGIONS[iris]), 0] = 0.3 * eye_w
        g[list(REGIONS[iris]), 1] = -0.12 * eye_w                        # +1 looks up
    fields["gaze_x"], fields["gaze_y"] = f, g
    return {k: v.astype(np.float32) for k, v in fields.items()}


def validate_fields(fields: dict, n_verts: int, h: float) -> None:
    """Raise RigError unless every field is a finite, bounded (n_verts, 2) array, still on the ring."""
    missing = [p for p in FIELD_PARAMS if p not in fields]
    if missing:
        raise RigError(f"fields missing: {', '.join(missing)}")
    for name, f in fields.items():
        f = np.asarray(f)
        if f.shape != (n_verts, 2):
            raise RigError(f"field {name}: shape {f.shape}, want {(n_verts, 2)}")
        if not np.isfinite(f).all():
            raise RigError(f"field {name}: not finite")
        if np.abs(f).max(initial=0) > MAX_FIELD * h:
            raise RigError(f"field {name}: moves a vertex more than {MAX_FIELD:.0%} of the face height")
        if np.abs(f[N_FACE:]).max(initial=0) > 0:
            raise RigError(f"field {name}: moves the boundary ring")


def validate_regions(regions: dict, n_verts: int) -> None:
    for name in REGIONS:
        idx = regions.get(name)
        if not idx:
            raise RigError(f"region {name} has no landmarks")
        if min(idx) < 0 or max(idx) >= n_verts:
            raise RigError(f"region {name}: landmark index out of range")


def deform(rest: np.ndarray, fields: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Vertices for ``params``; parameters without a field (glow, head_*) are ignored here."""
    out = np.array(rest, dtype=np.float32, copy=True)
    for name, value in params.items():
        f = fields.get(name)
        if f is not None and value:
            out += np.float32(value) * f
    return out


def positions(verts: np.ndarray, triangles: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """(N, 2) point positions from bound triangles and barycentric weights."""
    return np.einsum("nk,nkd->nd", bary, verts[triangles[tri]])
