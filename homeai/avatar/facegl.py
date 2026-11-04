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
HAIR_STRANDS = 140
HAIR_POINTS = 28            # per strand
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
    if (layer > 0.5) k *= 0.55 * teeth * (layer > 1.5 ? 0.45 : 1.0);
    c = mix(col, g, grey) * k;
    gl_PointSize = point_px * size * (1.0 + 0.25 * low * (1.0 - small));
}
"""
HAIR_VERTEX = "#version 330\n" + _POSE + """
uniform vec2 hc;
uniform vec2 radii;
uniform float hlen;
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
in float theta;
in float s;
in float phase;
in float size;
in float kink;
out vec3 c;
void main() {
    float th = theta + 0.12 * sin(swirl + theta * 2.0);
    vec2 root = hc + vec2(cos(th) * radii.x, -sin(th) * radii.y);
    vec2 n = normalize(vec2(cos(th) / radii.x, -sin(th) / radii.y));
    vec2 perp = vec2(-n.y, n.x);
    float L = hlen * (0.7 + 0.6 * kink);
    float dir = 1.0 - 2.0 * ripple;                     // listening: waves run in toward the head
    float wv = (sin(9.0 * s - hair_phase * dir + phase) * 0.08
              + sin(4.0 * s - 0.6 * hair_phase + phase * 1.7) * 0.05) * hair_amp * L * s;
    vec2 p = root + n * L * s + vec2(0.0, 1.0) * L * 0.5 * s * s + perp * wv;
    vec2 e = (p - hc) / radii;
    float outside = smoothstep(0.98, 1.04, dot(e, e));   // never over the face
    gl_Position = to_clip(pose(p));
    float fade = pow(1.0 - s, 1.3) * smoothstep(0.0, 0.06, s) * outside;
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
    """Pivot, size and the aura's root ellipse, from the face oval (image px)."""
    v = np.asarray(verts, dtype=np.float64)
    oval = v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"])]
    h = rig.face_height(v)
    width = float(np.ptp(oval[:, 0]))
    top = float(v[10, 1])
    return {"pivot": oval.mean(axis=0), "face_h": h, "hc": np.array([oval[:, 0].mean(), top + 0.58 * h]),
            "radii": np.array([0.68 * width, 0.86 * h]), "hlen": 0.36 * h}


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
    attrs[:, 8] = 0.0
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


def hair_attributes(strands: int = HAIR_STRANDS, per: int = HAIR_POINTS, seed: int = 5) -> np.ndarray:
    """(strands * per, 5) float32: root angle, position along the strand, phase, size, kink."""
    if strands < 1 or per < 2:
        raise ValueError("need at least one strand of two points")
    rng = np.random.default_rng(seed)
    theta = np.linspace(-0.3, np.pi + 0.3, strands) + rng.uniform(-0.01, 0.01, strands)
    kink = rng.random(strands)
    sphase = rng.uniform(0, 2 * np.pi, strands)
    s = (np.arange(per) + rng.random((strands, per)) * 0.6) / per
    size = np.where(rng.random((strands, per)) < 0.3, rng.uniform(1.0, 1.5, (strands, per)),
                    rng.uniform(0.6, 0.99, (strands, per)))
    return np.stack([np.repeat(theta, per), s.ravel(), np.repeat(sphase, per), size.ravel(),
                     np.repeat(kink, per)], axis=1).astype(np.float32)


def view(side: float, size: tuple[int, int], zoom: float = 1.0) -> tuple[float, float, float, float]:
    """(sx, sy, cx, cy): image px -> NDC so the image's height fills the window, centred."""
    w, h = size
    sy = 2.0 * zoom / side
    return sy * h / max(w, 1), sy, side / 2, side / 2


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
            self.hair_vbo = ctx.buffer(hair_attributes().tobytes())
            self.hair_vao = ctx.vertex_array(self.hair_prog, [
                (self.hair_vbo, "1f 1f 1f 1f 1f", "theta", "s", "phase", "size", "kink")])
        self.overlay = Overlay(ctx)
        self.points = len(idx) + (HAIR_STRANDS * HAIR_POINTS if hair else 0)

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
            for name, value in {**common, "hc": tuple(self.geo["hc"]), "radii": tuple(self.geo["radii"]),
                                "hlen": self.geo["hlen"], "point_px": max(1.5, 1.6 * px_per_img * self.spacing),
                                "hair_phase": p.get("hair_phase", 0.0), "hair_amp": p.get("hair_amp", 1.0),
                                "swirl": frame.swirl, "ripple": frame.ripple, "ring": frame.ring,
                                "ring_amp": frame.ring_amp, "glow": frame.glow,
                                "colour": tuple(frame.colour)}.items():
                self._set(self.hair_prog, name, value)
            self.hair_vao.render(self._mgl.POINTS)
        teeth = teeth_visibility(verts, p.get("teeth_show", 0.0), self.geo["face_h"])
        for name, value in {**common, "t": t, "point_px": max(1.5, DOT_SPRITE * px_per_img * self.spacing),
                            "bright": p.get("bright", 1.0), "grey": p.get("grey", 0.0),
                            "glow": p.get("glow", 0.0), "teeth": teeth, "shimmer": SHIMMER_PX}.items():
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
