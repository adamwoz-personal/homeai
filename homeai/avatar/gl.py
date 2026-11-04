"""Draw a scene Frame with moderngl: a glowing point sphere plus captions (AVATAR_PLAN §4.1).

Thin on purpose: every number comes from scene.Frame. Works with a window's
context or a headless EGL one (``headless()``), so tools/avatar/sync_check.py
and the tests can render without a display.

Phase 2 point field: each point has a seeded phase and size; low speech band
swells the large dots, the high band brightens the small ones (§4.6). Motion
per state: listening ripples inward, thinking swirls (latitudes rotate at
different rates), followup sends a bright ring outward across the sphere.
"""
from __future__ import annotations

import numpy as np

VERTEX = """
#version 330
in vec3 pos;
in float phase;
in float size;
uniform float t;
uniform float angle;
uniform float radius;
uniform float aspect;
uniform float point_px;
uniform float openness;
uniform float ripple;
uniform float swirl;
uniform float ring;
uniform float ring_amp;
uniform float low;
uniform float high;
out float shade;
out float boost;
void main() {
    float a = angle + swirl * (1.0 - abs(pos.y));
    float c = cos(a), s = sin(a);
    vec3 p = vec3(pos.x * c - pos.z * s, pos.y, pos.x * s + pos.z * c);
    float rho = length(p.xy);
    float wave = ripple * 0.035 * sin(rho * 14.0 + t * 9.0);      // phase travels inward
    float shimmer = 1.0 + 0.012 * sin(t * 3.0 + phase) + 0.05 * openness * sin(t * 23.0 + phase * 3.0)
                    + wave;
    vec2 xy = p.xy * radius * shimmer;
    gl_Position = vec4(xy.x / max(aspect, 1e-3), xy.y, 0.0, 1.0);
    shade = 0.35 + 0.65 * (p.z + 1.0) * 0.5;
    float big = step(1.0, size);
    float ring_hit = ring < 0.0 ? 0.0 : 1.0 - smoothstep(0.0, 0.08, abs(rho - ring));
    boost = 1.0 + 1.5 * ring_amp * ring_hit + 1.2 * high * (1.0 - big) + 0.3 * ripple * max(wave, 0.0) / 0.035;
    gl_PointSize = point_px * size * (1.0 + 0.6 * low * big) * (0.6 + 0.8 * shade);
}
"""
FRAGMENT = """
#version 330
in float shade;
in float boost;
uniform vec3 colour;
uniform float glow;
out vec4 frag;
void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float core = 1.0 - smoothstep(0.0, 0.25, r2);
    frag = vec4(colour * shade * (0.25 + 0.75 * glow) * core * boost, 1.0);
}
"""
OVERLAY_VERTEX = """
#version 330
in vec2 xy;
out vec2 uv;
void main() { uv = xy * 0.5 + 0.5; gl_Position = vec4(xy, 0.0, 1.0); }
"""
OVERLAY_FRAGMENT = """
#version 330
in vec2 uv;
uniform sampler2D tex;
out vec4 frag;
void main() { frag = texture(tex, uv); }
"""


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` evenly spread unit vectors, float32 (n, 3)."""
    if n < 1:
        raise ValueError("need at least one point")
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)],
                    axis=1).astype(np.float32)


def point_attributes(points: int, seed: int = 7) -> np.ndarray:
    """(n, 5) float32: unit position, phase, size factor; deterministic for a seed.

    Size: 70% small dots (0.6-1.0) and 30% large (1.0-1.5); the shader treats
    size >= 1 as "large" for the low band.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, points)
    large = rng.random(points) < 0.3
    size = np.where(large, rng.uniform(1.0, 1.5, points), rng.uniform(0.6, 0.999, points))
    return np.hstack([fibonacci_sphere(points), phase[:, None], size[:, None]]).astype(np.float32)


class Overlay:
    """Draws an RGBA image (captions) over the whole viewport; re-uploads only a new image."""

    def __init__(self, ctx) -> None:
        import moderngl

        self.ctx, self._mgl = ctx, moderngl
        self.prog = ctx.program(vertex_shader=OVERLAY_VERTEX, fragment_shader=OVERLAY_FRAGMENT)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.quad = ctx.buffer(quad.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.quad, "2f", "xy")])
        self.tex = None
        self._tex_image = None

    def draw(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        if self.tex is None or self.tex.size != (w, h):
            if self.tex is not None:
                self.tex.release()
            self.tex = self.ctx.texture((w, h), 4)
            self._tex_image = None
        if image is not self._tex_image:              # Rasteriser returns the cached array
            self.tex.write(image.tobytes())
            self._tex_image = image                   # held, so identity is never reused
        self.ctx.blend_func = self._mgl.SRC_ALPHA, self._mgl.ONE_MINUS_SRC_ALPHA
        self.tex.use(0)
        self.vao.render(self._mgl.TRIANGLE_STRIP)

    def release(self) -> None:
        for obj in (self.vao, self.quad, self.prog, self.tex):
            if obj is not None:
                obj.release()


class SphereRenderer:
    def __init__(self, ctx, points: int = 12000, seed: int = 7) -> None:
        import moderngl

        self.ctx = ctx
        self._mgl = moderngl
        self._points_mode = moderngl.POINTS
        ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.prog = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.vbo = ctx.buffer(point_attributes(points, seed).tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 1f 1f", "pos", "phase", "size")])
        self.overlay = Overlay(ctx)

    def _set(self, name: str, value) -> None:
        uniform = self.prog.get(name, None)   # the GLSL compiler drops unused uniforms
        if uniform is not None:
            uniform.value = value

    def draw(self, frame, t: float, size: tuple[int, int], captions: np.ndarray | None = None) -> None:
        """Draw the sphere, then ``captions`` (RGBA image of the window size) over it."""
        w, h = size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.blend_func = self._mgl.ONE, self._mgl.ONE      # additive: overlaps glow
        for name, value in (("t", t), ("angle", frame.angle), ("radius", frame.radius),
                            ("aspect", w / max(h, 1)), ("point_px", max(1.5, h / 360)),
                            ("openness", frame.openness), ("colour", tuple(frame.colour)),
                            ("glow", frame.glow), ("ripple", frame.ripple), ("swirl", frame.swirl),
                            ("ring", frame.ring), ("ring_amp", frame.ring_amp),
                            ("low", frame.low), ("high", frame.high)):
            self._set(name, value)
        self.vao.render(self._points_mode)
        if captions is not None:
            self.overlay.draw(captions)

    def release(self) -> None:
        for obj in (self.vao, self.vbo, self.prog):
            obj.release()
        self.overlay.release()


def headless(size: tuple[int, int] = (360, 360)):
    """(ctx, fbo) on an offscreen EGL context. Raises if EGL is unavailable."""
    import moderngl

    ctx = moderngl.create_standalone_context(backend="egl")
    fbo = ctx.simple_framebuffer(size)
    fbo.use()
    return ctx, fbo


def lit_radius(pixels: np.ndarray, threshold: int = 20) -> float:
    """Sphere radius in NDC (y) from an RGB frame: half the lit rows' span / height.

    Pure; used by sync_check and the golden test to measure what was drawn.
    """
    rows = np.flatnonzero(pixels.max(axis=(1, 2)) > threshold)
    if rows.size == 0:
        return 0.0
    return (rows[-1] - rows[0] + 1) / pixels.shape[0]
