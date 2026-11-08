"""homeai-avatar: the Jarvis window (AVATAR_PLAN §4, Phase 1 sphere).

Reads the daemon's event bus, plays speech timelines against the monotonic
clock and draws the face (Phase 4) or, without a chosen face, the sphere.
Runs without the daemon (shows "offline") and reconnects when it comes back.
Keys: F11 fullscreen, Esc leaves fullscreen.

Display choice (Phase 0 spike): XWayland (SDL_VIDEODRIVER=x11) so the window
position can be restored; DRI_PRIME=1 renders on the iGPU (set by the service).
Window size/position/fullscreen persist in ~/.config/homeai/avatar.json.

Usage: .venv/bin/python -m homeai.avatar.app [--socket PATH] [--look auto|face|sphere] [--points N]
Needs the avatar extra (pygame-ce, moderngl) and a desktop session.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import time
import warnings
from pathlib import Path

from ..fsutil import atomic_write_json
from .client import BusClient
from .scene import Scene
from .tour import TOUR, staged, tour_index

log = logging.getLogger("homeai.avatar.app")

GEOMETRY_PATH = Path.home() / ".config" / "homeai" / "avatar.json"
DEFAULT_GEOMETRY = {"size": [720, 720], "position": None, "fullscreen": False}
VSYNC_FPS = 60
SETTLE_FRAMES = 15          # frames before trusting the window manager's placement
STATUS = "_status"          # internal queue item: connection up/down
MODE_POLL_S = 1.0
LOOKS = ("auto", "face", "sphere")


def lock_held(path: Path, proc_locks: Path = Path("/proc/locks")) -> bool:
    """True if any process holds a flock on ``path``.

    Read from /proc/locks rather than by trying the lock: a probe could make a
    real ``homeai-mode`` switch fail with "another switch is running".
    """
    try:
        inode = path.stat().st_ino
        table = proc_locks.read_text()
    except OSError:
        return False
    for line in table.splitlines():
        parts = line.split()
        # "1: FLOCK  ADVISORY  WRITE 1234 103:02:5678 0 EOF"
        if len(parts) >= 6 and parts[1] == "FLOCK" and parts[5].rsplit(":", 1)[-1] == str(inode):
            return True
    return False


def local_mode(dropin: Path, lock: Path) -> str | None:
    """"switching" while homeai-mode runs, "coding" in coding mode, else None."""
    if lock_held(lock):
        return "switching"
    return "coding" if dropin.exists() else None


def mode_paths() -> tuple[Path, Path]:
    from ..gpu_mode import DROPIN_NAME, runtime_dropin_dir

    return runtime_dropin_dir() / DROPIN_NAME, Path(f"/run/user/{os.getuid()}/homeai-mode.lock")


def build_scene() -> Scene:
    """Scene with the sleep settings from the environment (.env via the service)."""
    from ..config import AvatarConfig
    from ..doorbell import parse_quiet_hours

    cfg = AvatarConfig()
    return Scene(sleep_s=cfg.sleep_min * 60, quiet=parse_quiet_hours(cfg.quiet_hours))


def prepare_env(env: dict) -> dict:
    """SDL settings that must exist before pygame is imported; never overrides the caller's."""
    env.setdefault("SDL_VIDEODRIVER", "x11")
    env.setdefault("SDL_VIDEO_X11_WMCLASS", "homeai-avatar")
    return env


def load_geometry(path: Path = GEOMETRY_PATH) -> dict:
    """Saved window geometry, or defaults for anything missing or malformed."""
    geo = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_GEOMETRY.items()}
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return geo
    if not isinstance(saved, dict):
        return geo
    size = saved.get("size")
    if (isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) for v in size)
            and all(160 <= v <= 16384 for v in size)):
        geo["size"] = size
    pos = saved.get("position")
    if isinstance(pos, list) and len(pos) == 2 and all(isinstance(v, int) for v in pos):
        geo["position"] = pos
    geo["fullscreen"] = saved.get("fullscreen") is True
    return geo


def save_geometry(geo: dict, path: Path = GEOMETRY_PATH) -> None:
    """Atomic write; a failure is logged, never raised (it's only window placement)."""
    try:
        atomic_write_json(path, geo)
    except OSError as exc:
        log.warning("could not save window geometry: %s", exc)


def position_to_save(current: list[int], requested: list[int] | None,
                     landed: list[int] | None) -> list[int]:
    """Position to save so the next start lands in the same place.

    Under x11 the window lands offset from the requested position by the title
    bar (measured: 37 px down per restart, so it crept). Subtract that offset.
    """
    if requested is None or landed is None:
        return list(current)
    return [c - (l - r) for c, l, r in zip(current, landed, requested)]


def frame_delay(fps: int, frame_start: float, now: float) -> float:
    """Seconds to sleep so frames start every 1/fps; vsync paces 60 fps by itself."""
    if fps >= VSYNC_FPS or fps <= 0:
        return 0.0
    return max(0.0, frame_start + 1.0 / fps - now)


def drain(events: "queue.SimpleQueue[dict]", scene: Scene) -> int:
    """Apply every queued bus event to the scene (main thread only). Returns the count."""
    n = 0
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            return n
        n += 1
        if event.get("type") == STATUS:
            scene.set_connected(bool(event.get("connected")))
        else:
            scene.apply(event)


def choose_look(look: str, load=None) -> tuple[str, object | None, str]:
    """(look, face asset or None, why): ``auto`` uses the face when one is chosen and loads."""
    if look not in LOOKS:
        raise ValueError(f"look must be one of {LOOKS}")
    if look == "sphere":
        return "sphere", None, "asked for the sphere"
    if load is None:
        from .assets import load_current as load
    try:
        asset = load()
    except (OSError, ValueError, KeyError) as exc:
        if look == "face":
            raise
        return "sphere", None, f"face asset unusable ({exc}); showing the sphere"
    if asset is None:
        if look == "face":
            raise ValueError("no face chosen; build one and run: homeai-avatar use vN")
        return "sphere", None, "no face chosen; showing the sphere"
    return "face", asset, f"face {asset.version}"


def run(args) -> int:
    geo = load_geometry(args.geometry)
    env = prepare_env(os.environ)
    if geo["position"] and not geo["fullscreen"]:
        env.setdefault("SDL_VIDEO_WINDOW_POS", f"{geo['position'][0]},{geo['position'][1]}")
    try:
        import moderngl  # noqa: F401 - fail early with a clear message
        import pygame
    except ImportError as exc:
        log.error("avatar extra missing (%s); install with: uv pip install -e '.[avatar]'", exc)
        return 2
    from .captions import Rasteriser
    from .gl import SphereRenderer

    try:
        look, asset, why = choose_look(args.look)
    except (OSError, ValueError, KeyError) as exc:
        log.error("--look face: %s", exc)
        return 2
    log.info("look: %s (%s)", look, why)
    pygame.init()
    for attr, value in ((pygame.GL_CONTEXT_MAJOR_VERSION, 3), (pygame.GL_CONTEXT_MINOR_VERSION, 3),
                        (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)):
        pygame.display.gl_set_attribute(attr, value)
    pygame.display.set_mode(tuple(geo["size"]), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
                            vsync=1)
    pygame.display.set_caption("Jarvis")
    with warnings.catch_warnings():
        # Deprecated for surface rendering only; we render with OpenGL.
        warnings.simplefilter("ignore", DeprecationWarning)
        window = pygame.Window.from_display_module()
    if geo["fullscreen"]:
        window.set_fullscreen(True)

    import moderngl

    ctx = moderngl.create_context()
    if look == "face":
        from .facegl import FaceRenderer

        renderer = FaceRenderer(ctx, asset)
    else:
        renderer = SphereRenderer(ctx, points=args.points)
    log.info("avatar window up: %s, driver %s", ctx.info.get("GL_RENDERER", "?"),
             env.get("SDL_VIDEODRIVER"))

    events: queue.SimpleQueue = queue.SimpleQueue()
    client = BusClient(events.put, lambda up: events.put({"type": STATUS, "connected": up}),
                       args.socket)
    if not args.tour:
        client.start()
    scene = build_scene()
    eyes = eye_follow(look, args.tour)
    captions = Rasteriser()
    dropin, lock = mode_paths()
    next_mode_poll = 0.0
    tour_start, tour_at = time.monotonic(), None
    running = True
    minimised = False
    requested = geo["position"] if geo["position"] and not geo["fullscreen"] else None
    landed: list[int] | None = None
    frames = 0

    def stop(signum, frame):  # noqa: ANN001, ARG001
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while running:
            start = time.monotonic()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.WINDOWMINIMIZED:
                    minimised = True
                elif ev.type in (pygame.WINDOWRESTORED, pygame.WINDOWSHOWN, pygame.WINDOWEXPOSED):
                    minimised = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                    geo["fullscreen"] = not geo["fullscreen"]
                    window.set_fullscreen(True) if geo["fullscreen"] else window.set_windowed()
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE and geo["fullscreen"]:
                    geo["fullscreen"] = False
                    window.set_windowed()
            drain(events, scene)
            if client.version_error:
                log.error("daemon speaks another protocol: %s", client.version_error)
                return 3
            if minimised:
                if eyes is not None:
                    eyes.stop()                 # nobody to look at
                pygame.event.wait(250)          # draw nothing while hidden (§4.3)
                continue
            now = time.monotonic()
            if args.tour:
                i = tour_index(now - tour_start, args.tour)
                if i is None:
                    break
                if i != tour_at:
                    tour_at = i
                    scene, _ = staged(TOUR[i], t0=now, settle=False)
                    log.info("tour %d/%d: %s", i + 1, len(TOUR), TOUR[i])
                    pygame.display.set_caption(f"Jarvis - tour: {TOUR[i]}")
            elif now >= next_mode_poll:
                scene.local_mode = local_mode(dropin, lock)
                next_mode_poll = now + MODE_POLL_S
            frame = scene.frame(now)
            if eyes is not None:
                scene.face.look_at = eyes.update(now, frame.state)
            size = pygame.display.get_window_size()
            renderer.draw(frame, now, size, captions.image(frame.captions, size))
            pygame.display.flip()
            time.sleep(frame_delay(frame.fps, start, time.monotonic()))
            frames += 1
            if frames == SETTLE_FRAMES:
                landed = list(window.position)   # the window manager has placed it by now
            if not geo["fullscreen"] and frames >= SETTLE_FRAMES:
                geo["size"] = list(window.size)
                geo["position"] = position_to_save(list(window.position), requested, landed)
    finally:
        if eyes is not None:
            eyes.stop()
        client.stop()
        if not args.tour:                       # a tour window must not move the real one
            save_geometry(geo, args.geometry)
        renderer.release()
        pygame.quit()
    return 0


def eye_follow(look: str, tour: float):
    """An EyeFollow if it's opted in and there is a face to do it; else None."""
    from ..config import AvatarConfig
    from .eyes import EyeFollow

    return EyeFollow() if look == "face" and not tour and AvatarConfig().eye_follow else None


def check() -> list[str]:
    """Problems that would stop the window from starting (no display needed)."""
    problems = []
    for name in ("pygame", "moderngl"):
        try:
            __import__(name)
        except ImportError as exc:
            problems.append(f"{name} not importable ({exc}); install with: "
                            "uv pip install --python .venv/bin/python -e '.[avatar]'")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jarvis avatar window")
    parser.add_argument("--socket", type=Path, default=None, help="event bus (default: the daemon's)")
    parser.add_argument("--points", type=int, default=12000, help="sphere points")
    parser.add_argument("--look", choices=LOOKS, default="auto",
                        help="auto: the chosen face if there is one, else the sphere")
    parser.add_argument("--geometry", type=Path, default=GEOMETRY_PATH)
    parser.add_argument("--check", action="store_true",
                        help="check the avatar extra imports, print problems, exit 1 if any")
    parser.add_argument("--tour", type=float, default=0.0, metavar="SECONDS",
                        help="show every state for SECONDS each, then exit (ignores the daemon)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if args.check:
        problems = check()
        for p in problems:
            print(f"problem: {p}")
        print("check: OK" if not problems else f"check: {len(problems)} problem(s)")
        return 1 if problems else 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
