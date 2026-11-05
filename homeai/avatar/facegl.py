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
  ripple = waves run inward, swirl, ring). Speech starts at the roots and
  travels down each strand (HairPulse), so no point flashes all at once.
"""
from __future__ import annotations

import numpy as np

from . import rig
from .face import HAIR_PHASE_WRAP
from .gl import Overlay

DOT_SPRITE = 1.5            # sprite diameter / typical point spacing (core is 0.3x spacing in radius)
SHIMMER_PX = 0.25           # image px
HAIR_STRANDS = 320          # side strands from the brows to the jaw angle; the neck gets more at that density
HAIR_CROWN = 720            # long strands from the skull's outline, parted in the middle (Adam 2026-11-04: more, twice)
HAIR_LENGTH = 1.6           # all hair (Adam 2026-11-05: longer everywhere)
HAIR_DENSITY = 48           # points per face-height of strand length (was 40 per strand at ~0.84 h)
HAIR_CROWN_LEN = (1.6, 2.6)  # x face height: longer than the side hair (Adam: "longer", three times)
HAIR_CROWN_CLEAR = 0.22     # crown strands reach this far (x face height) past the head's edge before falling
HAIR_CROWN_LIFT = 0.22      # x face height: crown hair grows this far up before gravity turns it (Adam: grow up, then fall)
HAIR_TEMPLE_LEN = 0.43      # crown strands at brow level, share of their length: meets the side hair with no step
HAIR_LIFT_FADE = (85.0, 130.0)  # degrees round from the top: lift fades to none here (Adam: temples fell too fast)
HAIR_CROWN_FAN = 0.5        # x face height: crown strands land across this width; one line piled up into "cat ears"
HAIR_PART = 20.0            # degrees either side of the top where strands mix sides: no hard part line
HAIR_SCALP = 300            # strands from the channels beside the scalp's plates (Adam: not from the plates)
HAIR_SCALP_LEN = (0.9, 1.5)  # x face height: shorter than the crown, so the plates stay visible
# The channels between the four raised plates, per artwork (sha256 of the source image), as quads in crop px.
# Marked by hand on tools/avatar/probe_scalp_roots.py: the stipple evens out dot density, so finding them
# automatically failed on v1 and on the crisper build alike. Any rebuild of the same art keeps them.
SCALP_CHANNELS = {
    "afe7fb9676533ae8daf8770bc06a85d474874e9103ff6d896821adaa84833f5c": (
        ((242, 55), (310, 25), (312, 75), (255, 100)),      # left outer plate to centre left
        ((350, 12), (412, 14), (410, 75), (352, 75)),       # between the two centre plates
        ((452, 25), (527, 50), (515, 100), (447, 75)),      # centre right to right outer
        ((172, 150), (202, 150), (197, 200), (172, 200)),   # beyond the left outer plate
        ((557, 140), (585, 140), (590, 190), (562, 190)),   # beyond the right outer plate
    ),
}
HAIR_NECK = 200             # strands from the dark slots beside the neck wires
# The dark slots between the neck wires and the neck, per artwork, as for SCALP_CHANNELS.
NECK_CHANNELS = {
    "afe7fb9676533ae8daf8770bc06a85d474874e9103ff6d896821adaa84833f5c": (
        ((203, 575), (226, 585), (240, 690), (224, 690)),   # left: outer wire to the jaw and neck
        ((530, 570), (560, 570), (548, 660), (528, 665)),   # right
    ),
}
HAIR_NECK_LEN = (0.4, 0.9)  # x face height
# The hard rings beside the mouth, as ((centre x, y), (radius x, y)) in crop px, marked by hand on the art.
# Bound to the mesh dot by dot they stretch as the jaw opens (Adam: they must stay as drawn).
MOUTH_RINGS = {
    "afe7fb9676533ae8daf8770bc06a85d474874e9103ff6d896821adaa84833f5c": (
        ((281.5, 550.0), (15.0, 19.0)),
        ((480.0, 549.0), (16.0, 21.0)),
    ),
}
HAIR_EAR = 240              # extra strands around the ears (Adam: more there)
HAIR_EAR_LEN = (0.6, 1.3)   # x face height
HAIR_HIGH_DROOP = 0.6       # (Adam: lower, twice) gravity on hair rooted up high (crown, brows), rising to 1 at the jaw: frames the face
HAIR_LIT = 0.12             # colour above this is part of the head's silhouette
HAIR_GAP = 0.25             # lit dots further than this (x face height) beyond the head are strays
HAIR_SPREAD = 0.38          # Adam 2026-11-04: drift wider, less gravity (was 0.10 out, 1.0 down)
HAIR_DROP = 0.85
HAIR_GRAVITY = 0.45
HAIR_LOCKS = 9.0            # rad of wave phase from the top root to the bottom one
HAIR_SIDE_SHIFT = 1.1       # rad of curl phase between the sides, so left doesn't mirror right
HAIR_WIND = 0.14            # breeze at the tip, x strand length (Adam: a gentle breeze, not a sine wave)
# One wind field for all the hair, in face heights and hair_phase: (amount, kx, ky, rate, offset) sines.
# Rates are whole thousandths, so hair_phase's wrap (2 pi x 1000) is seamless. The rate-only terms are the
# slow change of direction, the same on both sides; the rest roll across the hair.
HAIR_WIND_X = ((0.35, 0.0, 0.0, 0.113, 0.5), (0.30, 0.9, 1.3, 0.710, 0.0),
               (0.18, -1.7, 2.9, 1.130, 1.0), (0.09, 3.1, 5.3, 1.910, 2.0))
HAIR_WIND_Y = ((0.12, 0.0, 0.0, 0.089, 2.1), (0.16, 1.2, -0.8, 0.590, 0.7), (0.08, -2.3, 2.1, 1.370, 3.0))
HAIR_GUST = (0.65, 0.35, 0.6, 0.197)    # base, swing, kx, rate: the breeze comes and goes across the head
HAIR_PULSE_HZ = 60          # speech history samples per second
HAIR_PULSE_SAMPLES = 128    # ring length: must outlast the slowest strand (2.1 s)
HAIR_PULSE_SPEED = 2.6      # face heights per second: about 1 s to the longest crown tip (Adam: travel, not flash)
HAIR_SWELL_OMEGA = 4.0      # rad/s: the travelling sway follows speech this slowly (Adam dislikes jumpy hair)
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


def _wind_glsl(terms) -> str:
    return " + ".join(f"{a:.3f} * sin({kx:.3f} * q.x + {ky:.3f} * q.y - {w:.3f} * t + {o:.3f})"
                      for a, kx, ky, w, o in terms)


_WIND = f"""
vec2 wind(vec2 q, float t) {{   // keep in step with hair_wind()
    float gust = {HAIR_GUST[0]:.3f} + {HAIR_GUST[1]:.3f} * sin({HAIR_GUST[2]:.3f} * q.x - {HAIR_GUST[3]:.3f} * t);
    return gust * vec2({_wind_glsl(HAIR_WIND_X)}, {_wind_glsl(HAIR_WIND_Y)});
}}
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
uniform vec4 eye_window_r;  // centre, radii (image px): an iris shows only inside this, as if behind the lids
uniform vec4 eye_window_l;
in ivec3 idx;
in vec3 w;
in vec3 col;
in float size;
in float phase;
in float layer;
in vec2 rigid;              // offset from the anchor the mesh moves: hard parts keep their shape
out vec3 c;
vec2 V(int i) { return texelFetch(verts, ivec2(i, 0), 0).xy; }
void main() {
    vec2 p = w.x * V(idx.x) + w.y * V(idx.y) + w.z * V(idx.z) + rigid;
    vec4 window = layer > 4.5 ? eye_window_l : eye_window_r;
    float behind = smoothstep(0.9, 1.0, length((p - window.xy) / window.zw));
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
        if (fract(layer) > 0.1) k *= 1.0 - behind;          // the iris slides under the lids
        base = mix(col, eye_tint.rgb * (0.6 + 0.8 * g.r), eye_tint.a);
        k *= 1.0 + 0.8 * eye_tint.a;
    }
    c = mix(base, g, grey) * k;
    gl_PointSize = point_px * size * (1.0 + 0.25 * low * (1.0 - small));
}
"""
HAIR_VERTEX = "#version 330\n" + _POSE + _WIND + """
uniform float point_px;
uniform float face_h;       // image px
uniform float hair_wind_amount;
uniform float hair_phase;
uniform float hair_wave;    // travelling-wave phase, direction already integrated (HairWave)
uniform float hair_amp;
uniform float swirl;
uniform float ripple;
uniform float ring;
uniform float ring_amp;
uniform sampler2D pulse;    // recent speech (HairPulse): x low, y high, z openness, w swell
uniform int pulse_head;     // newest sample
uniform float pulse_per_px; // samples of delay per image px along a strand
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
in float bend;      // 0 reaches out from the root, 1 rises first and reaches out later (top of the crown)
out vec3 c;
vec4 speech(float delay) {  // the ring ``delay`` samples ago, blended between samples
    int n = textureSize(pulse, 0).x;
    float d = min(delay, float(n - 2));
    int back = int(d);
    vec4 a = texelFetch(pulse, ivec2((pulse_head - back + n) % n, 0), 0);
    vec4 b = texelFetch(pulse, ivec2((pulse_head - back - 1 + n) % n, 0), 0);
    return mix(a, b, d - float(back));
}
void main() {
    vec4 sp = speech(s * len * pulse_per_px);
    float amp = hair_amp + 0.8 * sp.w;
    float sway = 0.05 * sin(swirl + phase);             // thinking: strands swing as a whole
    float curl = (0.08 * sin(7.0 * s + phase) + 0.045 * sin(12.7 * s + 1.6 * phase)) * s;  // wavy even when still
    float heard = ripple * 0.10 * sin(8.0 * s - hair_wave + phase) * amp * s;   // listening: waves run inward
    // Out from the head, then drifting down: light gravity, felt most at the tips.
    float out_ = spread * (1.0 - (1.0 - s) * (1.0 - s));
    float fall = (1.0 - gravity) * s + gravity * s * s;
    vec2 reach = vec2(mix(out_, spread * s * s * (3.0 - 2.0 * s), bend), out_);
    vec2 rest = root + len * (aim * reach + vec2(side * curl, drop * droop * fall));
    vec2 breeze = hair_wind_amount * amp * pow(s, 1.5) * wind(rest / face_h, hair_phase);
    vec2 p = rest + len * (breeze + vec2(side * heard + sway * s, 0.0));
    gl_Position = to_clip(pose(p));
    float fade = pow(1.0 - s, 1.1) * smoothstep(0.0, 0.05, s);
    float hit = ring < 0.0 ? 0.0 : 1.0 - smoothstep(0.0, 0.08, abs(s - ring));
    float small = 1.0 - step(1.0, size);
    float base = 0.35 + 0.65 * glow;
    float carry = pow(1.0 - s, 0.35) * smoothstep(0.0, 0.05, s);   // pulses outlast the strand's own fade
    c = colour * (base * fade * (1.0 + 2.0 * ring_amp * hit) + carry * (0.1 * sp.z + 0.75 * base * sp.y * small)) * 1.2;
    gl_PointSize = point_px * size * (1.0 + 0.6 * sp.x * (1.0 - small));
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


def smooth(lo: float, hi: float, x: float) -> float:
    """0 below ``lo``, 1 above ``hi``, an S-curve between."""
    u = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def lift_at(turn: float, h: float, length: float, droop: float = HAIR_HIGH_DROOP) -> float:
    """aim y for a strand rooted ``turn`` degrees round from the top: one curve for all the hair."""
    peak = HAIR_CROWN_LIFT * h * max(np.cos(np.radians(min(turn, 90.0))), 0.0) / length
    return -crown_rise(peak, droop=droop) * (1.0 - smooth(*HAIR_LIFT_FADE, turn)) / HAIR_SPREAD


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
        length = HAIR_LENGTH * h * (0.9 + 0.7 * min(t, 1.0) ** 1.1) * taper * rng.uniform(0.8, 1.2)
        droop = HAIR_HIGH_DROOP + (1.0 - HAIR_HIGH_DROOP) * min(t, 1.0)
        roots.append((edge + side * 1.2 * spacing, y, side, length,
                      HAIR_LOCKS * t + side * HAIR_SIDE_SHIFT + rng.uniform(-0.5, 0.5), float(side),
                      lift_at(90.0 + 45.0 * t, h, length, droop), droop, 0.0))   # neighbours: locks
    return roots


def _crown_roots(pts, v, h, spacing, strands, rng) -> list:
    """Long strands from just outside the skull's outline, parted in the middle: each rises a
    little, reaches sideways just past the head's edge, then falls like the side hair."""
    top = float(min(v[105, 1], v[334, 1]))
    origin = (float(v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean()), top)
    jaw = float(max(v[172, 1], v[397, 1]))
    reach = _head_reach(pts, h, spacing, origin[0], top, jaw)
    roots = []
    for k in range(strands):
        a = np.radians(90.0) * (2.0 * (k + rng.uniform(0.1, 0.9)) / strands - 1.0)   # 0 up, +-90 brow level
        n = np.array([np.sin(a), -np.cos(a)])
        edge = ray_edge(pts, origin, n, 2.0 * spacing, HAIR_GAP * h)
        if edge is None:
            continue
        side = 1.0 if a > 0 else -1.0
        if rng.random() < 0.5 * (1.0 - min(abs(a) / np.radians(HAIR_PART), 1.0)):
            side = -side                                    # near the top either way: no hard part line
        root = np.asarray(origin) + n * (edge + 1.2 * spacing)
        turn = float(np.degrees(abs(a)))
        length = HAIR_LENGTH * h * rng.uniform(*HAIR_CROWN_LEN) * (1.0 - (1.0 - HAIR_TEMPLE_LEN) * smooth(40.0, 90.0, turn))
        width = reach[side] if reach[side] is not None else edge * abs(n[0])
        land = width + (HAIR_CROWN_CLEAR + HAIR_CROWN_FAN * rng.random()) * h
        clear = HAIR_LENGTH * (land - (root[0] - origin[0]) * side)   # sideways to clear the head
        aim = (side * max(clear, 0.0) / (HAIR_SPREAD * length), lift_at(turn, h, length))
        roots.append((root[0], root[1], side, length,
                      HAIR_LOCKS * abs(a) + side * HAIR_SIDE_SHIFT + rng.uniform(-0.5, 0.5), aim[0], aim[1],
                      HAIR_HIGH_DROOP, abs(n[1])))
    return roots


def _head_reach(pts, h, spacing, centre, top, jaw) -> dict:
    """The head's widest point below the brows, each side of ``centre``."""
    rows = np.linspace(top, jaw, 12)
    reach = {}
    for s in (-1.0, 1.0):
        edges = [silhouette_edge(pts, y, int(s), 2.0 * spacing, centre=centre, gap=HAIR_GAP * h) for y in rows]
        edges = [(e - centre) * s for e in edges if e is not None]
        reach[s] = max(edges) if edges else None
    return reach


def _scalp_roots(pts, v, h, spacing, quads, strands, rng) -> list:
    """Strands from between the scalp's plates: up and out from the skull like the crown, then falling."""
    spots = scalp_spots(quads, strands)
    if not len(spots):
        return []
    top = float(min(v[105, 1], v[334, 1]))
    origin = np.array([v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean(), top])
    reach = _head_reach(pts, h, spacing, origin[0], top, float(max(v[172, 1], v[397, 1])))
    roots = []
    for x, y in spots:
        d = np.array([x, y]) - origin
        n = d / max(float(np.linalg.norm(d)), 1e-9)
        side = 1.0 if d[0] > 0 else -1.0
        length = HAIR_LENGTH * h * rng.uniform(*HAIR_SCALP_LEN)
        width = reach[side] if reach[side] is not None else abs(d[0])
        land = width + (HAIR_CROWN_CLEAR + HAIR_CROWN_FAN * rng.random()) * h
        clear = HAIR_LENGTH * (land - d[0] * side)
        rise = lift_at(float(np.degrees(np.arctan2(abs(d[0]), -d[1]))), h, length)
        roots.append((x, y, side, length, HAIR_LOCKS * abs(n[0]) + side * HAIR_SIDE_SHIFT + rng.uniform(-0.5, 0.5),
                      side * max(clear, 0.0) / (HAIR_SPREAD * length), rise, HAIR_HIGH_DROOP, abs(n[1])))
    return roots


def _neck_roots(v, h, quads, strands, rng) -> list:
    """Strands from the dark slots beside the neck wires, hanging down and easing outward."""
    spots = scalp_spots(quads, strands, seed=11)
    centre = float(v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean())
    roots = []
    for x, y in spots:
        side = 1.0 if x > centre else -1.0
        roots.append((x, y, side, HAIR_LENGTH * h * rng.uniform(*HAIR_NECK_LEN),
                      rng.uniform(0.0, 2.0 * np.pi), side, 0.0, 1.0, 0.0))
    return roots


def crown_rise(peak: float, drop: float = HAIR_DROP, gravity: float = HAIR_GRAVITY,
               droop: float = HAIR_HIGH_DROOP) -> float:
    """The upward reach (x strand length) that peaks ``peak`` above the root before gravity wins."""
    s = np.linspace(0.0, 1.0, 129)
    up = 1.0 - (1.0 - s) ** 2
    down = drop * droop * ((1.0 - gravity) * s + gravity * s * s)
    lo, hi = 0.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (mid * up - down).max() < peak:
            lo = mid
        else:
            hi = mid
    return hi


def _ear_roots(pts, v, h, spacing, strands, rng) -> list:
    """Extra strands from the head's edge between the brows and the mouth, where the ears sit."""
    top = float(min(v[105, 1], v[334, 1]))
    jaw = float(max(v[172, 1], v[397, 1]))
    bottom = top + 0.7 * (jaw - top)
    centre = float(v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean())
    roots = []
    for k in range(strands):
        side = -1 if k % 2 == 0 else 1
        y = float(rng.uniform(top, bottom))
        edge = silhouette_edge(pts, y, side, 2.0 * spacing, centre=centre, gap=HAIR_GAP * h)
        if edge is None:
            continue
        t = (y - top) / (jaw - top)
        length = HAIR_LENGTH * h * rng.uniform(*HAIR_EAR_LEN)
        droop = HAIR_HIGH_DROOP + (1.0 - HAIR_HIGH_DROOP) * t
        roots.append((edge + side * 1.2 * spacing, y, side, length,
                      HAIR_LOCKS * t + side * HAIR_SIDE_SHIFT + rng.uniform(-0.5, 0.5), float(side),
                      lift_at(90.0 + 45.0 * t, h, length, droop), droop, 0.0))
    return roots


def hair_rest(layout: np.ndarray, spread: float = HAIR_SPREAD, drop: float = HAIR_DROP,
              gravity: float = HAIR_GRAVITY) -> np.ndarray:
    """(M, 2) image px of each hair point at rest (no wave, sway or head pose).

    The hair shader's path on the CPU, for tests and probes: keep it in step
    with HAIR_VERTEX.
    """
    a = np.asarray(layout, dtype=np.float64)
    rx, ry, side, ln, s, phase = a[:, :6].T
    ax, ay, droop, bend = a[:, 7:11].T
    curl = (0.08 * np.sin(7.0 * s + phase) + 0.045 * np.sin(12.7 * s + 1.6 * phase)) * s
    out = spread * (1.0 - (1.0 - s) ** 2)
    fall = (1.0 - gravity) * s + gravity * s * s
    across = out + bend * (spread * s * s * (3.0 - 2.0 * s) - out)
    return np.column_stack([rx + ln * (ax * across + side * curl), ry + ln * (ay * out + drop * droop * fall)])


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


def scalp_spots(quads, count: int, seed: int = 9) -> np.ndarray:
    """(count, 2) image px spread evenly over four-cornered ``quads`` (corners in order round the edge)."""
    q = np.asarray(quads, dtype=np.float64).reshape(-1, 4, 2)
    if not len(q) or count <= 0:
        return np.empty((0, 2))
    x, y = q[..., 0], q[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(q), count, p=area / area.sum())
    u, v = rng.random((2, count, 1))
    a, b, c, d = (q[pick, i] for i in range(4))
    return (1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * d + u * c)


def hair_wind(q: np.ndarray, t: float) -> np.ndarray:
    """(M, 2) breeze at ``q`` (face heights) at hair_phase ``t``: the shader's wind() on the CPU."""
    x, y = np.asarray(q, dtype=np.float64).T

    def blow(terms):
        return sum(a * np.sin(kx * x + ky * y - w * t + o) for a, kx, ky, w, o in terms)

    base, swing, kx, rate = HAIR_GUST
    gust = base + swing * np.sin(kx * x - rate * t)
    return np.column_stack([gust * blow(HAIR_WIND_X), gust * blow(HAIR_WIND_Y)])


def hair_layout(xy: np.ndarray, verts: np.ndarray, spacing: float, strands: int = HAIR_STRANDS,
                crown: int = HAIR_CROWN, points: int | None = None, seed: int = 5,
                ears: int = HAIR_EAR, scalp=(), scalp_strands: int = HAIR_SCALP, neck=(),
                neck_strands: int = HAIR_NECK) -> np.ndarray:
    """(M, 11) float32 per hair point: root x, root y, side, length, s, phase, size, aim x, aim y, droop, bend.

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
    roots = (_side_roots(pts, v, h, spacing, strands, rng) + _crown_roots(pts, v, h, spacing, crown, rng)
             + _ear_roots(pts, v, h, spacing, ears, rng) + _scalp_roots(pts, v, h, spacing, scalp, scalp_strands, rng)
             + _neck_roots(v, h, neck, neck_strands, rng))
    if not roots:
        raise ValueError("no silhouette found for the hair")
    total = sum(r[3] for r in roots)
    points = int(round(HAIR_DENSITY * total / h)) if points is None else points
    out = []
    for x, y, side, length, phase, ax, ay, droop, bend in roots:
        n = max(4, int(round(points * length / total)))
        s = (np.arange(n) + rng.random(n) * 0.6) / n
        size = np.where(rng.random(n) < 0.3, rng.uniform(1.0, 1.5, n), rng.uniform(0.6, 0.99, n))
        fixed = np.tile([x, y, side, length], (n, 1))
        out.append(np.column_stack([fixed, s, np.full(n, phase), size, np.tile([ax, ay, droop, bend], (n, 1))]))
    return np.vstack(out).astype(np.float32)


def asset_hair(asset, spacing: float) -> np.ndarray:
    """The hair layout for a face asset, with scalp hair if its artwork's channels are marked."""
    art = asset.build.get("inputs", {}).get("image_sha256")
    return hair_layout(lit_points(asset.xy, asset.colour), asset.verts, spacing,
                       scalp=SCALP_CHANNELS.get(art, ()), neck=NECK_CHANNELS.get(art, ()))


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
IRIS_LAYER = 0.25                               # added to an eye's layer for the dots of its iris
IRIS_GROW = 1.4                                 # painted iris, x the landmarks' iris radius
WINDOW_CLEAR = 1.14                             # eye window, x the iris's reach at rest
IRIS_EYES = (("iris_r", rig.EYE_R_UPPER + rig.EYE_R_LOWER), ("iris_l", rig.EYE_L_UPPER + rig.EYE_L_LOWER))


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


def iris_disc(verts: np.ndarray, region: str) -> tuple[int, np.ndarray, float]:
    """(centre vertex, its rest position, radius in image px) of the painted iris."""
    v = np.asarray(verts, dtype=np.float64)
    ring = rig.REGIONS[region]
    return ring[0], v[ring[0]], IRIS_GROW * float(np.ptp(v[list(ring)][:, 0])) / 2


def eye_windows(verts: np.ndarray, xy: np.ndarray, layer: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Per eye, (x, y, rx, ry) image px: where its iris can show, from where the lids cut the painted iris at rest.

    ``xy`` and ``layer`` are the face's dots. The window only just clears the iris at rest, so any move toward a lid
    starts to tuck it under; fading starts at 0.9 of the window, hence the margin.
    """
    v = np.asarray(verts, dtype=np.float64)
    out = []
    for (region, lids), eye in zip(IRIS_EYES, EYE_LAYERS.values()):
        _, centre, radius = iris_disc(v, region)
        dots = np.asarray(xy, dtype=np.float64)[layer == eye + IRIS_LAYER]
        contour = v[list(lids)]
        x = float(contour[:, 0].min() + contour[:, 0].max()) / 2
        y = float(dots[:, 1].min() + dots[:, 1].max()) / 2 if len(dots) else float(centre[1])
        half = (float(np.abs(dots[:, 0] - x).max()), float(np.ptp(dots[:, 1])) / 2) if len(dots) else (radius, radius)
        rx, ry = max(WINDOW_CLEAR * half[0], float(np.ptp(contour[:, 0])) / 2), WINDOW_CLEAR * half[1]
        reach = float(np.linalg.norm((dots - (x, y)) / (rx, ry), axis=1).max()) if len(dots) else 0.0
        grow = max(1.0, reach / 0.89)                       # whole at rest, even at the ellipse's narrow shoulders
        out.append((x, y, rx * grow, ry * grow))
    return out


def bind_irises(xy: np.ndarray, idx: np.ndarray, bary: np.ndarray, offset: np.ndarray, layer: np.ndarray,
                verts: np.ndarray) -> None:
    """Gaze moves only the iris vertices, so re-bind in place what touches them (Adam: the eyes broke up to the side).

    An iris's dots follow its centre vertex at fixed offsets and so keep their pattern. Anything else on a triangle
    with an iris vertex follows the nearest lid vertex instead: the lids still carry it, the gaze doesn't.
    """
    v = np.asarray(verts, dtype=np.float64)
    moved = np.isin(idx, [i for region, _ in IRIS_EYES for i in rig.REGIONS[region]]).any(axis=1)
    for (region, _), eye in zip(IRIS_EYES, EYE_LAYERS.values()):
        vertex, centre, radius = iris_disc(v, region)
        iris = (layer == eye) & (np.linalg.norm(xy - centre, axis=1) <= radius)
        idx[iris], bary[iris] = vertex, (1.0, 0.0, 0.0)
        offset[iris] = xy[iris] - centre
        layer[iris] = eye + IRIS_LAYER
        moved &= ~iris
    lids = np.array(sorted({i for _, contour in IRIS_EYES for i in contour}))
    nearest = lids[np.argmin(np.linalg.norm(xy[moved, None, :] - v[None, lids, :], axis=2), axis=1)]
    idx[moved] = nearest[:, None]
    bary[moved] = (1.0, 0.0, 0.0)
    offset[moved] = xy[moved] - v[nearest]


def bind_rings(xy: np.ndarray, tri: np.ndarray, bary: np.ndarray, rings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tri, bary, offset (N, 2)): every point inside a ring follows the point nearest its centre, at a fixed offset.

    ``rings`` holds ((x, y), (rx, ry)) ellipses. The mesh then only moves each ring, it can't reshape it.
    """
    tri, bary = tri.copy(), bary.copy()
    offset = np.zeros((len(xy), 2), np.float32)
    for centre, radii in rings:
        d = xy - np.asarray(centre, dtype=np.float32)
        inside = np.flatnonzero(((d / np.asarray(radii, dtype=np.float32)) ** 2).sum(axis=1) <= 1.0)
        if not len(inside):
            continue
        anchor = inside[np.argmin((d[inside] ** 2).sum(axis=1))]
        offset[inside] = xy[inside] - xy[anchor]
        tri[inside], bary[inside] = tri[anchor], bary[anchor]
    return tri, bary, offset


def face_buffers(asset, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """(idx (N, 3) int32, attrs (N, 11) float32: w3, colour3, size, phase, layer, rigid2) for face + teeth."""
    rng = np.random.default_rng(seed)
    art = asset.build.get("inputs", {}).get("image_sha256")
    tri, bary, offset = bind_rings(asset.xy, asset.tri, asset.bary, MOUTH_RINGS.get(art, ()))
    idx = asset.triangles[tri].astype(np.int32)
    n = len(idx)
    layer = eye_layers(asset.xy, asset.verts)
    bind_irises(asset.xy, idx, bary, offset, layer, asset.verts)
    attrs = np.empty((n, 11), np.float32)
    attrs[:, 0:3] = bary
    attrs[:, 3:6] = asset.colour
    attrs[:, 6] = asset.size
    attrs[:, 7] = rng.uniform(0, 2 * np.pi, n)
    attrs[:, 8] = layer
    attrs[:, 9:11] = offset
    t_idx, t_w, t_layer = mouth_points(asset.verts)
    m = len(t_idx)
    teeth = np.zeros((m, 11), np.float32)
    teeth[:, 0:3] = t_w
    lip = asset.colour[np.isin(idx, list(rig._LIPS_UPPER_INNER)).any(axis=1)]
    tint = lip.mean(axis=0) if len(lip) else np.array([0.5, 0.5, 0.5])
    teeth[:, 3:6] = 0.35 + 0.4 * tint / max(float(tint.max()), 1e-6)   # pale, in the art's hue
    teeth[:, 6] = 0.6
    teeth[:, 7] = rng.uniform(0, 2 * np.pi, m)
    teeth[:, 8] = t_layer
    return np.vstack([idx, t_idx]), np.vstack([attrs, teeth])


def face_positions(idx: np.ndarray, attrs: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """(N, 2) image px of face_buffers' points on ``verts``: FACE_VERTEX's path without shimmer or head pose."""
    return np.einsum("nk,nkd->nd", attrs[:, 0:3], np.asarray(verts)[idx]) + attrs[:, 9:11]


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

class HairWave:
    """The hair's travelling-wave phase: outward normally, toward the head while listening.

    The shader used ``hair_phase * (1 - 2 * ripple)``. hair_phase is a running
    total (thousands of rad), so as ripple eased in or out at a state change the
    wave jumped thousands of rad in half a second: the fast wiggle Adam saw
    with each colour change (2026-11-04). Integrating the step instead makes
    direction change only the speed.
    """

    def __init__(self) -> None:
        self.wave = 0.0
        self._last: float | None = None

    def advance(self, hair_phase: float, ripple: float) -> float:
        if self._last is not None:
            step = (hair_phase - self._last) % HAIR_PHASE_WRAP   # the modulo crosses the wrap
            self.wave = (self.wave + step * (1.0 - 2.0 * ripple)) % (2.0 * np.pi)
        self._last = hair_phase
        return self.wave


class HairPulse:
    """Recent speech as a ring buffer, so the hair can show it with a delay along each strand.

    Rows are low, high, openness and a smoothed low (swell, for the sway). Samples
    are by time, not by frame, because the window's fps changes with the state; a
    skipped sample holds the latest values. Time going backwards clears the history.
    """

    def __init__(self) -> None:
        self.ring = np.zeros((HAIR_PULSE_SAMPLES, 4), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.ring[:] = 0.0
        self.head = 0
        self._tick: int | None = None
        self._swell = 0.0

    def advance(self, t: float, low: float, high: float, openness: float) -> bool:
        """Record speech up to ``t``; True if the ring changed."""
        tick = int(t * HAIR_PULSE_HZ)
        if self._tick is None or tick < self._tick:
            self.reset()
            self._tick = tick - 1
        steps = min(tick - self._tick, HAIR_PULSE_SAMPLES)
        self._tick = tick
        k = 1.0 - float(np.exp(-HAIR_SWELL_OMEGA / HAIR_PULSE_HZ))
        for _ in range(steps):
            self._swell += (low - self._swell) * k
            self.head = (self.head + 1) % HAIR_PULSE_SAMPLES
            self.ring[self.head] = (low, high, openness, self._swell)
        return steps > 0


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
            (self.face_attrs, "3f 3f 1f 1f 1f 2f", "w", "col", "size", "phase", "layer", "rigid")])
        self.hair_prog = self.hair_vbo = self.hair_vao = self.pulse_tex = None
        if hair:
            self.hair_prog = ctx.program(vertex_shader=HAIR_VERTEX, fragment_shader=DOT_FRAGMENT)
            layout = asset_hair(asset, self.spacing)
            self.hair_vbo = ctx.buffer(layout.tobytes())
            self.hair_vao = ctx.vertex_array(self.hair_prog, [
                (self.hair_vbo, "2f 1f 1f 1f 1f 1f 2f 1f 1f", "root", "side", "len", "s", "phase", "size",
                 "aim", "droop", "bend")])
            self.hair_points = len(layout)
            self.pulse_tex = ctx.texture((HAIR_PULSE_SAMPLES, 1), 4, dtype="f4")
            self.pulse_tex.filter = moderngl.NEAREST, moderngl.NEAREST
        self.windows = eye_windows(asset.verts, asset.xy, attrs[:len(asset.xy), 8])
        self.overlay = Overlay(ctx)
        self.wave = HairWave()
        self.pulse = HairPulse()
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
            if self.pulse.advance(t, frame.low, frame.high, frame.openness):
                self.pulse_tex.write(self.pulse.ring.tobytes())
            self.pulse_tex.use(1)
            for name, value in {**common, "spread": HAIR_SPREAD, "drop": HAIR_DROP, "gravity": HAIR_GRAVITY,
                                "point_px": max(1.5, 1.6 * px_per_img * self.spacing),
                                "hair_phase": p.get("hair_phase", 0.0), "hair_amp": p.get("hair_amp", 1.0),
                                "hair_wave": self.wave.advance(p.get("hair_phase", 0.0), frame.ripple),
                                "swirl": frame.swirl, "ripple": frame.ripple, "ring": frame.ring,
                                "ring_amp": frame.ring_amp, "glow": frame.base_glow, "pulse": 1,
                                "pulse_head": self.pulse.head, "face_h": self.geo["face_h"], "hair_wind_amount": HAIR_WIND,
                                "pulse_per_px": HAIR_PULSE_HZ / (HAIR_PULSE_SPEED * self.geo["face_h"]),
                                "colour": tuple(frame.colour)}.items():
                self._set(self.hair_prog, name, value)
            self.hair_vao.render(self._mgl.POINTS)
        teeth = teeth_visibility(verts, p.get("teeth_show", 0.0), self.geo["face_h"])
        for name, value in {**common, "t": t, "point_px": max(1.5, DOT_SPRITE * px_per_img * self.spacing),
                            "bright": p.get("bright", 1.0), "grey": p.get("grey", 0.0),
                            "glow": p.get("glow", 0.0), "teeth": teeth, "shimmer": SHIMMER_PX,
                            "lids": (p.get("blink_r", 0.0), p.get("blink_l", 0.0)),
                            "eye_window_r": self.windows[0], "eye_window_l": self.windows[1],
                            "eye_tint": (*frame.colour, p.get("eye_tint", 0.0))}.items():
            self._set(self.face_prog, name, value)
        self.verts_tex.use(0)
        self._set(self.face_prog, "verts", 0)
        self.face_vao.render(self._mgl.POINTS)
        if captions is not None:
            self.overlay.draw(captions)

    def release(self) -> None:
        for obj in (self.face_vao, self.face_idx, self.face_attrs, self.face_prog, self.verts_tex,
                    self.hair_vao, self.hair_vbo, self.hair_prog, self.pulse_tex):
            if obj is not None:
                obj.release()
        self.overlay.release()
