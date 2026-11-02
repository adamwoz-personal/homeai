#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Avatar Phase 0 spike: which window setup renders smoothly on this desktop?

Why: AVATAR_PLAN §4.2 / §12. Native Wayland windows can't choose their
position; XWayland (SDL x11 driver) can, at some risk of tearing. Rendering on
the Ryzen iGPU (DRI_PRIME=1) would leave all 7900 XT VRAM for models, if it
still holds 60 fps. This probe measures each combination; Adam judges
tearing by eye (a vertical bar sweeps across; tearing breaks it).

Each run opens a resizable OpenGL window (default 1440x1440), draws --points
animated points on a pulsing sphere plus the sweeping bar, resizes once
halfway, and records: GL renderer, frame-time stats vs the monitor refresh,
dropped frames (> 1.5 refresh intervals), process CPU, and whether a
requested window position was honoured. GNOME may hide the title bar, so
the run number is also drawn as white blocks at the top left.

Usage:
    .venv/bin/python tools/avatar/window_probe.py                    # all 4 combos, 10 s each
    .venv/bin/python tools/avatar/window_probe.py --driver x11 --prime --seconds 20
Needs the avatar extra (pygame-ce, moderngl) and a desktop session.
Writes ~/homeai-bench/window_probe-<stamp>.json. Exit 0 all pass, 1 some fail, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BENCH_DIR = Path.home() / "homeai-bench"
COMBOS = [("wayland", False), ("x11", False), ("wayland", True), ("x11", True)]
REQUESTED_POS = (200, 100)
MIN_FPS_RATIO = 0.97
MAX_DROPPED = 0.01
HITCH_MS = 50.0
EXCUSE_S = 0.5

VERTEX = """
#version 330
in vec3 pos;
uniform float t;
uniform float aspect;
void main() {
    float pulse = 0.55 + 0.08 * sin(t * 6.0) + 0.03 * sin(t * 17.0 + pos.y * 9.0);
    float a = t * 0.4;
    vec3 p = vec3(pos.x * cos(a) - pos.z * sin(a), pos.y, pos.x * sin(a) + pos.z * cos(a));
    gl_Position = vec4(p.x * pulse / aspect, p.y * pulse, 0.0, 1.0);
    gl_PointSize = 2.0 + 2.0 * (p.z + 1.0);
}
"""
FRAGMENT = """
#version 330
out vec4 colour;
void main() {
    vec2 d = gl_PointCoord - 0.5;
    if (dot(d, d) > 0.25) discard;
    colour = vec4(0.2, 1.0, 0.35, 1.0);
}
"""
BAR_VERTEX = """
#version 330
in vec2 corner;
uniform float x;
uniform float y;
uniform float half_w;
uniform float half_h;
void main() { gl_Position = vec4(x + corner.x * half_w, y + corner.y * half_h, 0.0, 1.0); }
"""
BAR_FRAGMENT = """
#version 330
out vec4 colour;
void main() { colour = vec4(1.0, 1.0, 1.0, 1.0); }
"""


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` evenly spread unit vectors, float32 (n, 3)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    pts = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=1)
    return pts.astype(np.float32)


def frame_stats(frame_s: list[float], refresh_hz: float, frame_t: list[float] | None = None,
                excused_t: tuple[float, ...] = ()) -> dict:
    """Pure: fps, p50/p99 frame time and dropped-frame share against the refresh rate.

    A hitch is one frame over HITCH_MS: a visible jerk even when the dropped
    share is tiny. Hitches within EXCUSE_S after an excused event (a resize
    the probe itself caused) are reported but don't fail the run; any other
    hitch does. ``frame_t`` gives each frame's end time (s since start).
    """
    if not frame_s:
        return {"frames": 0, "fps": 0.0, "dropped": 1.0, "hitches": [], "passed": False}
    ms = np.array(frame_s) * 1000
    interval = 1000 / refresh_hz
    dropped = float((ms > 1.5 * interval).mean())
    fps = len(ms) / (ms.sum() / 1000)
    times = frame_t if frame_t is not None else list(np.cumsum(frame_s))
    hitches = [{"t_s": round(float(t), 3), "ms": round(float(m), 1),
                "excused": any(0 <= t - e <= EXCUSE_S for e in excused_t)}
               for t, m in zip(times, ms) if m > HITCH_MS]
    unexcused = [h for h in hitches if not h["excused"]]
    return {"frames": len(ms), "fps": round(fps, 2), "p50_ms": round(float(np.median(ms)), 2),
            "p99_ms": round(float(np.percentile(ms, 99)), 2), "max_ms": round(float(ms.max()), 2),
            "dropped": round(dropped, 4), "hitches": hitches,
            "passed": bool(fps >= MIN_FPS_RATIO * refresh_hz and dropped <= MAX_DROPPED
                           and not unexcused)}


def refresh_rate() -> float:
    """Primary monitor refresh from xrandr (works on GNOME Wayland via XWayland)."""
    try:
        out = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 60.0
    for line in out.splitlines():
        if "*" in line:
            for tok in line.split():
                if "*" in tok:
                    return float(tok.strip("*+"))
    return 60.0


def run_single(args) -> dict:
    """One window run; environment (driver, DRI_PRIME) is already set by the parent."""
    import moderngl
    import pygame

    pygame.init()
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    size = (args.size, args.size)
    pygame.display.set_mode(size, flags, vsync=1)
    pygame.display.set_caption(args.label)
    window = pygame.Window.from_display_module()
    ctx = moderngl.create_context()

    prog = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
    vbo = ctx.buffer(fibonacci_sphere(args.points).tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f", "pos")])
    bar = ctx.program(vertex_shader=BAR_VERTEX, fragment_shader=BAR_FRAGMENT)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    bar_vao = ctx.vertex_array(bar, [(quad, "2f", "corner")])
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    frame_s: list[float] = []
    frame_t: list[float] = []
    resize_t: list[float] = []
    cpu0 = time.process_time()
    start = last = time.monotonic()
    resized = False
    slow: list[dict] = []
    while (now := time.monotonic()) - start < args.seconds:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                args.seconds = 0
        if not resized and now - start > args.seconds / 2:
            window.size = (args.size * 3 // 4, args.size)
            resize_t.append(now - start)
            resized = True
        pumped = time.monotonic()
        w, h = pygame.display.get_window_size()
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.0, 0.0, 0.0)
        t = now - start
        prog["t"].value = t
        prog["aspect"].value = w / max(h, 1)
        vao.render(moderngl.POINTS)
        bar["half_w"].value, bar["half_h"].value, bar["y"].value = 0.02, 1.0, 0.0
        bar["x"].value = ((t * 1.5) % 2.0) - 1.0
        bar_vao.render(moderngl.TRIANGLE_STRIP)
        # GNOME may hide the title bar, so show the run number as white blocks, top left.
        bar["half_w"].value, bar["half_h"].value, bar["y"].value = 0.025, 0.04, 0.9
        for i in range(args.run):
            bar["x"].value = -0.93 + i * 0.08
            bar_vao.render(moderngl.TRIANGLE_STRIP)
        rendered = time.monotonic()
        pygame.display.flip()
        flipped = time.monotonic()
        if flipped - now > HITCH_MS / 1000:
            # Which call blocked: event pump, GL submit or the swap (compositor).
            slow.append({"t_s": round(flipped - start, 3), "pump_ms": round((pumped - now) * 1000, 1),
                         "render_ms": round((rendered - pumped) * 1000, 1),
                         "flip_ms": round((flipped - rendered) * 1000, 1)})
        frame_s.append(now - last)
        frame_t.append(now - start)
        last = now
    wall = time.monotonic() - start
    cpu = (time.process_time() - cpu0) / max(wall, 1e-6)
    position = list(window.position)
    renderer = ctx.info.get("GL_RENDERER", "?")
    pygame.quit()
    return {"renderer": renderer, "cpu_core_share": round(cpu, 3), "start_mono": start,
            "position": position, "requested_position": list(REQUESTED_POS),
            "frames_raw": len(frame_s), "frame_s": frame_s, "frame_t": frame_t,
            "resize_t": resize_t, "slow_phases": slow}


def child_env(driver: str, prime: bool) -> dict:
    env = dict(os.environ, SDL_VIDEODRIVER=driver,
               SDL_VIDEO_WINDOW_POS=f"{REQUESTED_POS[0]},{REQUESTED_POS[1]}")
    if prime:
        env["DRI_PRIME"] = "1"
    else:
        env.pop("DRI_PRIME", None)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--driver", choices=["wayland", "x11"])
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=20000)
    parser.add_argument("--size", type=int, default=1440)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", default="window probe", help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:
        try:
            print(json.dumps(run_single(args)))
            return 0
        except Exception as exc:  # noqa: BLE001 - reported to the parent as data
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
            return 2

    combos = [(args.driver, args.prime)] if args.driver else COMBOS
    hz = refresh_rate()
    results = []
    for n, (driver, prime) in enumerate(combos, 1):
        gpu = "iGPU (DRI_PRIME=1)" if prime else "default GPU"
        label = f"run {n} of {len(combos)}: {driver} / {gpu}"
        print(f"{label} ...", flush=True)
        cmd = [sys.executable, __file__, "--single", "--label", label, "--run", str(n),
               "--seconds", str(args.seconds),
               "--points", str(args.points), "--size", str(args.size)]
        try:
            proc = subprocess.run(cmd, env=child_env(driver, prime), capture_output=True,
                                  text=True, timeout=args.seconds + 60)
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
            data = json.loads(lines[-1]) if lines else {"error": proc.stderr.strip()[-300:]}
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            data = {"error": str(exc)}
        row = {"label": label, "driver": driver, "prime": prime, "refresh_hz": hz}
        if "error" in data:
            row.update(error=data["error"], passed=False)
        else:
            frames = data.pop("frame_s")
            times = data.pop("frame_t")
            # Window mapping and our own resize may hitch; anything else is a real jerk.
            excused = (0.0, *data["resize_t"])
            row.update(data, **frame_stats(frames, hz, times, excused))
            row["position_honoured"] = data["position"] == data["requested_position"]
        results.append(row)

    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"window_probe-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"when": datetime.now().isoformat(timespec="seconds"),
                               "points": args.points, "size": args.size, "results": results}, indent=2))
    for r in results:
        if "error" in r:
            print(f"{r['label']:40} ERROR {r['error'][:120]}")
            continue
        print(f"{r['label']:40} {'PASS' if r['passed'] else 'FAIL'}  {r['fps']:.1f} fps"
              f"  p99 {r['p99_ms']} ms  dropped {r['dropped']:.1%}  cpu {r['cpu_core_share']:.0%}"
              f"  pos {'kept' if r['position_honoured'] else r['position']}  [{r['renderer'][:40]}]")
        for h in r["hitches"]:
            print(f"    hitch {h['ms']:.0f} ms at {h['t_s']:.2f} s"
                  f" {'(start/resize, excused)' if h['excused'] else '<- unexplained'}")
        for p in r.get("slow_phases", []):
            print(f"    slow loop ending {p['t_s']:.2f} s: pump {p['pump_ms']} ms"
                  f"  render {p['render_ms']} ms  flip {p['flip_ms']} ms")
    print(f"-> {out}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
