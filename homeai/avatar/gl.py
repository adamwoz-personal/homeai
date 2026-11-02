"""Draw a scene Frame with moderngl: a glowing point sphere (AVATAR_PLAN §4.1, Phase 1).

Thin on purpose: every number comes from scene.Frame. Works with a window's
context or a headless EGL one (``headless()``), so tools/avatar/sync_check.py
and the tests can render without a display.
"""
from __future__ import annotations

import numpy as np

VERTEX = """
#version 330
in vec3 pos;
in float phase;
uniform float t;
uniform float angle;
uniform float radius;
uniform float aspect;
uniform float point_px;
uniform float openness;
out float shade;
void main() {
    float c = cos(angle), s = sin(angle);
    vec3 p = vec3(pos.x * c - pos.z * s, pos.y, pos.x * s + pos.z * c);
    float shimmer = 1.0 + 0.012 * sin(t * 3.0 + phase) + 0.05 * openness * sin(t * 23.0 + phase * 3.0);
    vec2 xy = p.xy * radius * shimmer;
    gl_Position = vec4(xy.x / max(aspect, 1e-3), xy.y, 0.0, 1.0);
    shade = 0.35 + 0.65 * (p.z + 1.0) * 0.5;
    gl_PointSize = point_px * (0.6 + 0.8 * shade);
}
"""
FRAGMENT = """
#version 330
in float shade;
uniform vec3 colour;
uniform float glow;
out vec4 frag;
void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float core = 1.0 - smoothstep(0.0, 0.25, r2);
    frag = vec4(colour * shade * (0.25 + 0.75 * glow) * core, 1.0);
}
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


class SphereRenderer:
    def __init__(self, ctx, points: int = 12000, seed: int = 7) -> None:
        import moderngl

        self.ctx = ctx
        self._points_mode = moderngl.POINTS
        ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE      # additive: overlaps glow
        self.prog = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, points).astype(np.float32)
        data = np.hstack([fibonacci_sphere(points), phase[:, None]]).astype(np.float32)
        self.vbo = ctx.buffer(data.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 1f", "pos", "phase")])

    def _set(self, name: str, value) -> None:
        uniform = self.prog.get(name, None)   # the GLSL compiler drops unused uniforms
        if uniform is not None:
            uniform.value = value

    def draw(self, frame, t: float, size: tuple[int, int]) -> None:
        w, h = size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0)
        self._set("t", t)
        self._set("angle", frame.angle)
        self._set("radius", frame.radius)
        self._set("aspect", w / max(h, 1))
        self._set("point_px", max(1.5, h / 360))
        self._set("openness", frame.openness)
        self._set("colour", tuple(frame.colour))
        self._set("glow", frame.glow)
        self.vao.render(self._points_mode)

    def release(self) -> None:
        for obj in (self.vao, self.vbo, self.prog):
            obj.release()


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
