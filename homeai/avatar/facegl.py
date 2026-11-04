"""Draw the stippled face and its aura hair with moderngl (AVATAR_PLAN §4.5, §4.6, Phase 4).

Same contract as gl.SphereRenderer: ``draw(frame, t, size, captions)``, with
every number from scene.Frame (``frame.params`` holds the rig values from
face.FaceMotion). Works on a window or a headless EGL context.

- Face: the rig's vertices are deformed on the CPU (534 x 2 floats,
  rig.Deformer, no allocation) and uploaded as a 1-row float texture; each
  stipple point looks up its three vertices and blends them by its
  barycentric weights in the vertex shader. Head pose (yaw/pitch/roll), a
  per-point shimmer and the art's own colours (dimmed / greyed by state) are
  applied there too.
- Mouth interior: the open mouth is the black background showing through the
  gap (no triangle spans it, see rig.spans_mouth). Teeth are a faint row of
  points bound between the inner lips, visible only as the lips part.
- Aura hair: strands rooted on an ellipse around the skull, flowing outward
  and down, coloured and moved by the state like the sphere (colour, glow,
  ripple = waves run inward, swirl, ring, speech bands).
"""
from __future__ import annotations

import numpy as np

from . import rig
from .gl import Overlay

DOT_SPRITE = 1.5            # sprite diameter / typical point spacing (core is 0.3x spacing in radius)
SHIMMER_PX = 0.25           # image px
HAIR_STRANDS = 320          # side strands from the brows to the jaw angle; the neck gets more at that density
HAIR_CROWN = 240            # long strands from the skull's outline, parted in the middle (Adam 2026-11-04: more)
HAIR_DENSITY = 48           # points per face-height of strand length (was 40 per strand at ~0.84 h)
HAIR_CROWN_LEN = (1.0, 1.6)  # x face height: as long as the side hair (Adam: "longer, like the sides")
HAIR_CROWN_CLEAR = 0.12     # crown strands reach this far (x face height) past the head's edge before falling
HAIR_CROWN_RISE = 1.0       # upward start at the top of the head (1.5 clears more but grows horns)
HAIR_HIGH_DROOP = 0.8       # gravity on hair rooted up high (crown, brows), rising to 1 at the jaw: frames the face
HAIR_LIT = 0.12             # colour above this is part of the head's silhouette
HAIR_GAP = 0.25             # lit dots further than this (x face height) beyond the head are strays
HAIR_SPREAD = 0.38          # Adam 2026-11-04: drift wider, less gravity (was 0.10 out, 1.0 down)
HAIR_DROP = 0.85
HAIR_GRAVITY = 0.45
HAIR_LOCKS = 9.0            # rad of wave phase from the top root to the bottom one
HEAD_DEPTH = 0.25           # head pose shift, share of the face height
ZOOM = 0.86                 # the image's height as a share of the window: room for the aura
LUMA = (0.299, 0.587, 0.114)

_POSE = """
uniform vec2 pivot;
uniform float depth;
uniform vec3 head;          // yaw, pitch, roll (rad)
uniform vec4 view;          // sx, sy, cx, cy: image px -> NDC
vec2 pose(vec2 p) {
    vec2 d = p - pivot;
    d = vec2(d.x * cos(head.x), d.y * cos(head.y)) + vec2(sin(head.x), sin(head.y)) * depth;
    float cr = cos(head.z), sr = sin(head.z);
    return pivot + vec2(d.x * cr - d.y * sr, d.x * sr + d.y * cr);
}
vec4 to_clip(vec2 p) { return vec4((p - view.zw) * vec2(view.x, -view.y), 0.0, 1.0); }
"""
FACE_VERTEX = "#version 330\n" + _POSE + """
uniform sampler2D verts;
uniform float t;
uniform float point_px;
uniform float bright;
uniform float grey;
uniform float glow;
uniform float teeth;
uniform float low;
uniform float high;
uniform float shimmer;
uniform vec2 lids;          // blink_r, blink_l: 1 = shut
uniform vec4 eye_tint;      // rgb, amount: the irises glow the state's colour
in ivec3 idx;
in vec3 w;
in vec3 col;
in float size;
in float phase;
in float layer;
out vec3 c;
vec2 V(int i) { return texelFetch(verts, ivec2(i, 0), 0).xy; }
void main() {
    vec2 p = w.x * V(idx.x) + w.y * V(idx.y) + w.z * V(idx.z);
    p += shimmer * vec2(sin(t * 1.9 + phase), cos(t * 1.6 + phase * 1.7));
    gl_Position = to_clip(pose(p));
    float small = 1.0 - step(1.0, size);
    vec3 g = vec3(dot(col, vec3(0.299, 0.587, 0.114)));
    float k = bright * (0.85 + 0.25 * glow) * (1.0 + 0.6 * high * small);
    vec3 base = col;
    if (layer > 0.5 && layer < 3.5) k *= 0.55 * teeth * (layer > 1.5 ? 0.45 : 1.0);
    if (layer > 3.5) {                                      // eye interior: hidden by the lid, tinted by state
        float shut = layer > 4.5 ? lids.y : lids.x;
        k *= 1.0 - smoothstep(0.25, 0.8, shut);
        base = mix(col, eye_tint.rgb * (0.6 + 0.8 * g.r), eye_tint.a);
        k *= 1.0 + 0.8 * eye_tint.a;
    }
    c = mix(base, g, grey) * k;
    gl_PointSize = point_px * size * (1.0 + 0.25 * low * (1.0 - small));
}
"""
HAIR_VERTEX = "#version 330\n" + _POSE + """
uniform float point_px;
uniform float hair_phase;
uniform float hair_amp;
uniform float swirl;
uniform float ripple;
uniform float ring;
uniform float ring_amp;
uniform float low;
uniform float high;
uniform float glow;
uniform vec3 colour;
uniform float spread;       // outward drift at the tip, x strand length
uniform float drop;         // downward reach at the tip, x strand length
uniform float gravity;      // 0 straight slant .. 1 droops near the tips
in vec2 root;       // image px, just outside the head's silhouette
in float side;      // -1 left, +1 right
in float len;       // strand length, image px (short above the ears, long below)
in float s;         // 0 root .. 1 tip
in float phase;
in float size;
in vec2 aim;        // outward direction x reach: (side, 0) at the sides, up and over on the crown
in float droop;     // how much of the fall applies: 1 at the sides, less on the crown
out vec3 c;
void main() {
    float dir = 1.0 - 2.0 * ripple;                     // listening: waves run up toward the head
    float sway = 0.05 * sin(swirl + phase);             // thinking: strands swing as a whole
    float curl = 0.11 * sin(7.0 * s + phase) * s;          // resting wave: wavy even when still
    float wv = (sin(8.0 * s - hair_phase * dir + phase) * 0.10
              + sin(3.5 * s - 0.6 * hair_phase + phase * 1.7) * 0.07) * hair_amp * s + curl;
    // Out from the head, then drifting down: light gravity, felt most at the tips.
    float out_ = spread * (1.0 - (1.0 - s) * (1.0 - s));
    float fall = (1.0 - gravity) * s + gravity * s * s;
    vec2 p = root + len * (aim * out_ + vec2(side * (wv + sway * s), drop * droop * fall));
    gl_Position = to_clip(pose(p));
    float fade = pow(1.0 - s, 1.1) * smoothstep(0.0, 0.05, s);
    float hit = ring < 0.0 ? 0.0 : 1.0 - smoothstep(0.0, 0.08, abs(s - ring));
    float small = 1.0 - step(1.0, size);
    c = colour * (0.35 + 0.65 * glow) * fade * (1.0 + 2.0 * ring_amp * hit + 1.5 * high * small) * 1.2;
    gl_PointSize = point_px * size * (1.0 + 0.6 * low * (1.0 - small));
}
"""
DOT_FRAGMENT = """
#version 330
in vec3 c;
out vec4 frag;
void main() {
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;
    float core = 1.0 - smoothstep(0.30, 0.42, r);
    float halo = 0.18 * exp(-r * r * 5.0);
    frag = vec4(c * (core + halo), 1.0);
}
"""


# ---- pure layout ---------------------------------------------------------------------------

def point_spacing(xy: np.ndarray, sample: int = 400, seed: int = 0) -> float:
    """Median nearest-neighbour distance (image px) from a sample: numpy only, ~10 ms."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return 4.0
    pick = np.random.default_rng(seed).choice(len(xy), min(sample, len(xy)), replace=False)
    nearest = np.empty(len(pick))
    for a in range(0, len(pick), 32):                       # chunked: a few MB, not ~200 MB
        rows = pick[a:a + 32]
        d = np.linalg.norm(xy[rows, None, :] - xy[None, :, :], axis=-1)
        d[np.arange(len(rows)), rows] = np.inf
        nearest[a:a + 32] = d.min(axis=1)
    return float(np.median(nearest))


def head_geometry(verts: np.ndarray) -> dict:
    """Pivot and size for head pose, from the face oval (image px)."""
    v = np.asarray(verts, dtype=np.float64)
    oval = v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"])]
    return {"pivot": oval.mean(axis=0), "face_h": rig.face_height(v)}


def silhouette_edge(xy: np.ndarray, y: float, side: int, band: float, centre: float | None = None,
                    gap: float | None = None) -> float | None:
    """x where the art ends on ``side`` (-1 left, +1 right) within ``band`` px of row ``y``.

    Walks out from ``centre`` (default: the row's median) and stops at the first
    gap wider than ``gap`` px (default 3 bands), so stray dots and shoulders
    further out don't count as the head.
    """
    row = xy[np.abs(xy[:, 1] - y) <= band, 0]
    if not len(row):
        return None
    c = float(np.median(row)) if centre is None else float(centre)
    gap = 3.0 * band if gap is None else gap
    xs = np.sort(row[row >= c] if side > 0 else -row[row <= c])
    if not len(xs):
        return None
    cut = np.flatnonzero(np.diff(xs) > gap)                 # walk starts at the first lit point
    return float((xs[cut[0]] if len(cut) else xs[-1]) * side)


def lit_points(xy: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """The art's points bright enough to count as the head (its black surround is stippled too)."""
    lit = np.asarray(xy)[np.asarray(colour).max(axis=1) > HAIR_LIT]
    return lit if len(lit) else np.asarray(xy)


def ray_edge(xy: np.ndarray, origin, direction, band: float, gap: float) -> float | None:
    """Distance along the ray from ``origin`` (unit ``direction``) to where the art ends.

    Like ``silhouette_edge`` but along any direction, for the skull's outline:
    points within ``band`` px of the ray, walked out from the origin, stopping
    at the first gap wider than ``gap`` px.
    """
    d = np.asarray(xy, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    u = np.asarray(direction, dtype=np.float64)
    along = d @ u
    across = np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])
    xs = np.sort(along[(across <= band) & (along >= 0)])
    if not len(xs):
        return None
    gaps = np.flatnonzero(np.diff(xs) > gap)
    return float(xs[gaps[0]] if len(gaps) else xs[-1])


def _side_roots(pts, v, h, spacing, strands, rng) -> list:
    """Brows to the bottom of the art: short above the ears, long from the jaw down the neck."""
    top = float(min(v[105, 1], v[334, 1]))                  # brows: about the top of the ears
    jaw = float(max(v[172, 1], v[397, 1]))                  # jaw angle
    bottom = max(jaw, float(pts[:, 1].max()) - 2.0 * spacing)   # the neck, down to where the art ends
    centre = float(v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean())
    per_side = max(1, int(round(strands / 2 * (bottom - top) / (jaw - top))))
    roots = []
    for k in range(2 * per_side):
        side = -1 if k % 2 == 0 else 1
        u = float(np.clip((k // 2 + rng.uniform(-0.4, 0.4)) / max(per_side - 1, 1), 0.0, 1.0))
        y = top + u * (bottom - top)
        t = (y - top) / (jaw - top)                         # 0 brows .. 1 jaw angle .. >1 down the neck
        edge = silhouette_edge(pts, y, side, 2.0 * spacing, centre=centre, gap=HAIR_GAP * h)
        if edge is None:
            continue
        below = (y - jaw) / max(bottom - jaw, 1.0) if y > jaw else 0.0
        taper = 1.0 - 0.7 * below                           # lower down the neck, less is still on screen
        length = h * (0.15 + 1.45 * min(t, 1.0) ** 1.1) * taper * rng.uniform(0.8, 1.2)
        roots.append((edge + side * 1.2 * spacing, y, side, length,
                      HAIR_LOCKS * t + rng.uniform(-0.5, 0.5), float(side), 0.0,
                      HAIR_HIGH_DROOP + (1.0 - HAIR_HIGH_DROOP) * min(t, 1.0)))   # neighbours: locks
    return roots


def _crown_roots(pts, v, h, spacing, strands, rng) -> list:
    """Long strands from just outside the skull's outline, parted in the middle: each rises a
    little, reaches sideways just past the head's edge, then falls like the side hair."""
    top = float(min(v[105, 1], v[334, 1]))
    origin = (float(v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean()), top)
    jaw = float(max(v[172, 1], v[397, 1]))
    centre = origin[0]
    rows = np.linspace(top, jaw, 12)
    reach = {}
    for s in (-1.0, 1.0):                                   # the head's widest point below the brows
        edges = [silhouette_edge(pts, y, int(s), 2.0 * spacing, centre=centre, gap=HAIR_GAP * h) for y in rows]
        edges = [(e - centre) * s for e in edges if e is not None]
        reach[s] = max(edges) if edges else None
    roots = []
    for k in range(strands):
        a = np.radians(90.0) * (2.0 * (k + rng.uniform(0.1, 0.9)) / strands - 1.0)   # 0 up, +-90 brow level
        n = np.array([np.sin(a), -np.cos(a)])
        edge = ray_edge(pts, origin, n, 2.0 * spacing, HAIR_GAP * h)
        if edge is None:
            continue
        side = 1.0 if a > 0 else -1.0
        root = np.asarray(origin) + n * (edge + 1.2 * spacing)
        length = h * rng.uniform(*HAIR_CROWN_LEN)
        width = reach[side] if reach[side] is not None else edge * abs(n[0])
        clear = width + HAIR_CROWN_CLEAR * h - (root[0] - origin[0]) * side   # sideways to clear the head
        aim = (side * max(clear, 0.0) / (HAIR_SPREAD * length), -HAIR_CROWN_RISE * abs(n[1]))
        roots.append((root[0], root[1], side, length,
                      HAIR_LOCKS * abs(a) + rng.uniform(-0.5, 0.5), aim[0], aim[1], HAIR_HIGH_DROOP))
    return roots


def hair_rest(layout: np.ndarray, spread: float = HAIR_SPREAD, drop: float = HAIR_DROP,
              gravity: float = HAIR_GRAVITY) -> np.ndarray:
    """(M, 2) image px of each hair point at rest (no wave, sway or head pose).

    The hair shader's path on the CPU, for tests and probes: keep it in step
    with HAIR_VERTEX.
    """
    a = np.asarray(layout, dtype=np.float64)
    rx, ry, side, ln, s, phase = a[:, :6].T
    ax, ay, droop = a[:, 7:10].T
    curl = 0.11 * np.sin(7.0 * s + phase) * s
    out = spread * (1.0 - (1.0 - s) ** 2)
    fall = (1.0 - gravity) * s + gravity * s * s
    return np.column_stack([rx + ln * (ax * out + side * curl), ry + ln * (ay * out + drop * droop * fall)])


def inside_outline(xy: np.ndarray, verts: np.ndarray, spacing: float, p: np.ndarray) -> np.ndarray:
    """(M,) bool: which of ``p`` lie inside the art's outline by more than a dot, i.e. over
    the head rather than beside it. The same walks the hair roots use: rays from the
    brow-line centre above the brows, rows below."""
    v = np.asarray(verts, dtype=np.float64)
    pts = np.asarray(xy, dtype=np.float64)
    h = rig.face_height(v)
    top = float(min(v[105, 1], v[334, 1]))
    centre = float(v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean())
    origin = np.array([centre, top])
    inside = np.zeros(len(p), bool)
    for i, q in enumerate(np.asarray(p, dtype=np.float64)):
        if q[1] < top:
            d = float(np.linalg.norm(q - origin))
            edge = ray_edge(pts, origin, (q - origin) / max(d, 1e-9), 2.0 * spacing, HAIR_GAP * h)
        else:
            side = 1 if q[0] > centre else -1
            edge = silhouette_edge(pts, q[1], side, 2.0 * spacing, centre=centre, gap=HAIR_GAP * h)
            d = (q[0] - centre) * side
            edge = None if edge is None else (edge - centre) * side
        inside[i] = edge is not None and d < edge - spacing
    return inside


def hair_layout(xy: np.ndarray, verts: np.ndarray, spacing: float, strands: int = HAIR_STRANDS,
                crown: int = HAIR_CROWN, points: int | None = None, seed: int = 5) -> np.ndarray:
    """(M, 10) float32 per hair point: root x, root y, side, length, s, phase, size, aim x, aim y, droop.

    Side strands hang from just outside the head's silhouette from brow height
    (about the top of the ears) down past the jaw to the bottom of the neck:
    ``strands`` of them between the brows and the jaw, more down the neck at
    that spacing. ``crown`` short ones sprout from the skull's outline above.
    Points are shared by length: ``points`` in all, default HAIR_DENSITY per
    face height of strand.
    """
    if strands < 2 or (points is not None and points < (strands + crown) * 4):
        raise ValueError("need at least two strands of four points")
    rng = np.random.default_rng(seed)
    v = np.asarray(verts, dtype=np.float64)
    pts = np.asarray(xy, dtype=np.float64)
    if not len(pts):
        raise ValueError("no silhouette found for the hair")
    h = rig.face_height(v)
    roots = _side_roots(pts, v, h, spacing, strands, rng) + _crown_roots(pts, v, h, spacing, crown, rng)
    if not roots:
        raise ValueError("no silhouette found for the hair")
    total = sum(r[3] for r in roots)
    points = int(round(HAIR_DENSITY * total / h)) if points is None else points
    out = []
    for x, y, side, length, phase, ax, ay, droop in roots:
        n = max(4, int(round(points * length / total)))
        s = (np.arange(n) + rng.random(n) * 0.6) / n
        size = np.where(rng.random(n) < 0.3, rng.uniform(1.0, 1.5, n), rng.uniform(0.6, 0.99, n))
        fixed = np.tile([x, y, side, length], (n, 1))
        out.append(np.column_stack([fixed, s, np.full(n, phase), size, np.tile([ax, ay, droop], (n, 1))]))
    return np.vstack(out).astype(np.float32)


def mouth_points(verts: np.ndarray, per_gap: int = 6, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teeth: (idx (M, 3) int32, weights (M, 3), layer (M,)) bound between the inner lips.

    Upper teeth hang from the upper inner lip (layer 1); a dimmer lower row
    rises from the lower inner lip (layer 2). Each point is a fixed blend of two
    neighbouring lip vertices and the facing vertex on the other lip, so it
    stays between the lips however far they part.
    """
    v = np.asarray(verts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    upper = sorted(rig._LIPS_UPPER_INNER, key=lambda i: v[i, 0])
    lower = sorted(rig._LIPS_LOWER_INNER + (78, 308), key=lambda i: v[i, 0])
    idx, w, layer = [], [], []
    for row, (lip, other, depths, lay) in enumerate(((upper, lower, (0.05, 0.13, 0.21), 1.0),
                                                      (lower, upper, (0.07,), 2.0))):
        for a, b in zip(lip, lip[1:]):
            mid = (v[a, 0] + v[b, 0]) / 2
            c = min(other, key=lambda i: abs(v[i, 0] - mid))
            for u in np.linspace(0.1, 0.9, per_gap):
                for d in depths:
                    d = d + rng.uniform(-0.025, 0.025)
                    u = float(np.clip(u + rng.uniform(-0.04, 0.04), 0.0, 1.0))
                    idx.append((a, b, c))
                    w.append(((1 - d) * (1 - u), (1 - d) * u, d))
                    layer.append(lay)
    return np.array(idx, np.int32), np.array(w, np.float32), np.array(layer, np.float32)


EYE_LAYERS = {"r": 4.0, "l": 5.0}               # face_buffers layer for points inside each eye
EYE_GROW = (1.1, 1.3)                           # eye outline scaled about its centre: painted eyes overhang the lids


def eye_layers(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Layer per point (rest position, image px): 4 / 5 inside the right / left eye's outline, else 0."""
    v = np.asarray(verts, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    out = np.zeros(len(pts), np.float32)
    for (upper, lower), layer in zip(((rig.EYE_R_UPPER, rig.EYE_R_LOWER), (rig.EYE_L_UPPER, rig.EYE_L_LOWER)),
                                     EYE_LAYERS.values()):
        poly = v[list(lower) + list(upper)[::-1]]
        c = poly.mean(axis=0)
        out[rig._inside(pts, c + (poly - c) * EYE_GROW)] = layer
    return out


def face_buffers(asset, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """(idx (N, 3) int32, attrs (N, 9) float32: w3, colour3, size, phase, layer) for face + teeth."""
    rng = np.random.default_rng(seed)
    idx = asset.triangles[asset.tri].astype(np.int32)
    n = len(idx)
    attrs = np.empty((n, 9), np.float32)
    attrs[:, 0:3] = asset.bary
    attrs[:, 3:6] = asset.colour
    attrs[:, 6] = asset.size
    attrs[:, 7] = rng.uniform(0, 2 * np.pi, n)
    attrs[:, 8] = eye_layers(asset.xy, asset.verts)
    t_idx, t_w, t_layer = mouth_points(asset.verts)
    m = len(t_idx)
    teeth = np.empty((m, 9), np.float32)
    teeth[:, 0:3] = t_w
    lip = asset.colour[np.isin(idx, list(rig._LIPS_UPPER_INNER)).any(axis=1)]
    tint = lip.mean(axis=0) if len(lip) else np.array([0.5, 0.5, 0.5])
    teeth[:, 3:6] = 0.35 + 0.4 * tint / max(float(tint.max()), 1e-6)   # pale, in the art's hue
    teeth[:, 6] = 0.6
    teeth[:, 7] = rng.uniform(0, 2 * np.pi, m)
    teeth[:, 8] = t_layer
    return np.vstack([idx, t_idx]), np.vstack([attrs, teeth])


def view(side: float, size: tuple[int, int], zoom: float = 1.0) -> tuple[float, float, float, float]:
    """(sx, sy, cx, cy): image px -> NDC, the image ``zoom`` x the window's height, centred
    across and sitting on the window's bottom edge (shoulders touch it; the room is above)."""
    w, h = size
    sy = 2.0 * zoom / side
    return sy * h / max(w, 1), sy, side / 2, side - 1.0 / sy


def to_pixels(points: np.ndarray, side: float, size: tuple[int, int], zoom: float = 1.0) -> np.ndarray:
    """Image px -> window px (top-left origin), without head pose: for measuring renders."""
    sx, sy, cx, cy = view(side, size, zoom)
    p = np.asarray(points, dtype=np.float64)
    ndc = np.column_stack([(p[:, 0] - cx) * sx, -(p[:, 1] - cy) * sy])
    return np.column_stack([(ndc[:, 0] + 1) / 2 * size[0], (1 - ndc[:, 1]) / 2 * size[1]])


def teeth_visibility(verts: np.ndarray, teeth_show: float, face_h: float) -> float:
    """Teeth glow 0-1: the viseme's teeth_show, faded in as the inner lips part."""
    gap = float(verts[14, 1] - verts[13, 1])
    x = min(max((gap / face_h - 0.008) / 0.025, 0.0), 1.0)
    return float(teeth_show) * x * x * (3 - 2 * x)


# ---- GL -------------------------------------------------------------------------------------

class FaceRenderer:
    def __init__(self, ctx, asset, hair: bool = True, zoom: float = ZOOM) -> None:
        import moderngl

        self.ctx, self._mgl, self.asset, self.zoom = ctx, moderngl, asset, zoom
        ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.deform = rig.Deformer(asset.verts, asset.fields)
        self.geo = head_geometry(asset.verts)
        self.side = float(asset.image_size[0])
        self.spacing = point_spacing(asset.xy)

        self.verts_tex = ctx.texture((len(asset.verts), 1), 2, dtype="f4")
        self.verts_tex.filter = moderngl.NEAREST, moderngl.NEAREST
        self.face_prog = ctx.program(vertex_shader=FACE_VERTEX, fragment_shader=DOT_FRAGMENT)
        idx, attrs = face_buffers(asset)
        self.face_idx = ctx.buffer(idx.tobytes())
        self.face_attrs = ctx.buffer(attrs.tobytes())
        self.face_vao = ctx.vertex_array(self.face_prog, [
            (self.face_idx, "3i", "idx"),
            (self.face_attrs, "3f 3f 1f 1f 1f", "w", "col", "size", "phase", "layer")])
        self.hair_prog = self.hair_vbo = self.hair_vao = None
        if hair:
            self.hair_prog = ctx.program(vertex_shader=HAIR_VERTEX, fragment_shader=DOT_FRAGMENT)
            layout = hair_layout(lit_points(asset.xy, asset.colour), asset.verts, self.spacing)
            self.hair_vbo = ctx.buffer(layout.tobytes())
            self.hair_vao = ctx.vertex_array(self.hair_prog, [
                (self.hair_vbo, "2f 1f 1f 1f 1f 1f 2f 1f", "root", "side", "len", "s", "phase", "size",
                 "aim", "droop")])
            self.hair_points = len(layout)
        self.overlay = Overlay(ctx)
        self.points = len(idx) + (self.hair_points if hair else 0)

    @staticmethod
    def _set(prog, name: str, value) -> None:
        uniform = prog.get(name, None)          # the GLSL compiler drops unused uniforms
        if uniform is not None:
            uniform.value = value

    def draw(self, frame, t: float, size: tuple[int, int], captions: np.ndarray | None = None) -> None:
        w, h = size
        p = frame.params
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.blend_func = self._mgl.ONE, self._mgl.ONE      # additive: overlaps glow
        verts = self.deform(p)
        self.verts_tex.write(verts.tobytes())
        px_per_img = h * self.zoom / self.side
        common = {"pivot": tuple(self.geo["pivot"]), "depth": HEAD_DEPTH * self.geo["face_h"],
                  "head": (p.get("head_yaw", 0.0), p.get("head_pitch", 0.0), p.get("head_roll", 0.0)),
                  "view": view(self.side, size, self.zoom), "low": frame.low, "high": frame.high}
        if self.hair_vao is not None:
            for name, value in {**common, "spread": HAIR_SPREAD, "drop": HAIR_DROP, "gravity": HAIR_GRAVITY,
                                "point_px": max(1.5, 1.6 * px_per_img * self.spacing),
                                "hair_phase": p.get("hair_phase", 0.0), "hair_amp": p.get("hair_amp", 1.0),
                                "swirl": frame.swirl, "ripple": frame.ripple, "ring": frame.ring,
                                "ring_amp": frame.ring_amp, "glow": frame.glow,
                                "colour": tuple(frame.colour)}.items():
                self._set(self.hair_prog, name, value)
            self.hair_vao.render(self._mgl.POINTS)
        teeth = teeth_visibility(verts, p.get("teeth_show", 0.0), self.geo["face_h"])
        for name, value in {**common, "t": t, "point_px": max(1.5, DOT_SPRITE * px_per_img * self.spacing),
                            "bright": p.get("bright", 1.0), "grey": p.get("grey", 0.0),
                            "glow": p.get("glow", 0.0), "teeth": teeth, "shimmer": SHIMMER_PX,
                            "lids": (p.get("blink_r", 0.0), p.get("blink_l", 0.0)),
                            "eye_tint": (*frame.colour, p.get("eye_tint", 0.0))}.items():
            self._set(self.face_prog, name, value)
        self.verts_tex.use(0)
        self._set(self.face_prog, "verts", 0)
        self.face_vao.render(self._mgl.POINTS)
        if captions is not None:
            self.overlay.draw(captions)

    def release(self) -> None:
        for obj in (self.face_vao, self.face_idx, self.face_attrs, self.face_prog, self.verts_tex,
                    self.hair_vao, self.hair_vbo, self.hair_prog):
            if obj is not None:
                obj.release()
        self.overlay.release()
