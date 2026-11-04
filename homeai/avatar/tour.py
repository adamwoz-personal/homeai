"""Put a Scene into each shown state on demand (AVATAR_PLAN §10 Phase 2).

One definition shared by the golden-image tests, ``tools/avatar/update_golden.py``,
``tools/avatar/state_tour.py`` (Adam's across-the-room check) and
``tools/avatar/avatar_profile.py``, so they all show the same thing.
"""
from __future__ import annotations

import math

from .scene import Scene

TOUR = ("idle", "listening", "thinking", "speaking", "followup", "sleep", "offline", "busy",
        "mode_switching")
SETTLE_S = 2.0              # long enough for colour and motion to finish easing
SPEECH_S = 6.0
SAID = ("The moon probably formed when a Mars-sized body hit the early Earth. "
        "The debris settled into orbit and pulled itself together.")
HEARD = "what made the moon?"


def speech(start: float, seconds: float = SPEECH_S, hz: int = 60) -> list[dict]:
    """Two synthetic utterances: a syllable-rate level, band envelopes, open/close visemes."""
    out = []
    half = seconds / 2
    for k in range(2):
        n = int(half * hz)
        level = [int(200 * abs(math.sin(i / hz * math.pi * 4))) for i in range(n)]
        low = [int(v * (0.5 + 0.5 * math.sin(i / 7))) for i, v in enumerate(level)]
        high = [int(v * (0.5 + 0.5 * math.cos(i / 5))) for i, v in enumerate(level)]
        visemes = [[int(ms), "AA" if (ms // 125) % 2 else "PP", 1.0]
                   for ms in range(0, int(half * 1000), 125)]
        out.append({"type": "utterance", "data": {
            "id": f"tour-{k}", "start_mono": start + k * half, "level_hz": hz, "level": level,
            "duration_ms": half * 1000, "visemes": visemes, "bands": {"low": low, "high": high}}})
    return out


def staged(state: str, t0: float = 100.0, settle: bool = True) -> tuple[Scene, float]:
    """A Scene showing ``state`` and the time to render it at (after easing if ``settle``)."""
    if state not in TOUR:
        raise ValueError(f"unknown tour state {state!r}; pick from {', '.join(TOUR)}")
    scene = Scene(sleep_s=1.0 if state == "sleep" else 3600.0)
    scene.local_mode = {"busy": "coding", "mode_switching": "switching"}.get(state)
    if state not in ("offline", "busy"):
        scene.set_connected(True)
        daemon = "idle" if state in ("sleep", "mode_switching") else state
        scene.apply({"type": "state", "data": {"state": daemon}})
    if state == "speaking":
        scene.apply({"type": "caption", "t_mono": t0, "data": {"who": "user", "text": HEARD}})
        scene.apply({"type": "caption", "t_mono": t0, "data": {"who": "jarvis", "text": SAID}})
        for event in speech(t0):
            scene.apply(event)
    t = t0
    scene.frame(t)
    end = t0 + (SETTLE_S if settle else 0.0)
    while t < end:
        t = min(end, t + 1 / 60)
        scene.frame(t)
    return scene, t


def tour_index(elapsed: float, per_state: float, count: int = len(TOUR)) -> int | None:
    """Which tour state to show ``elapsed`` s in; None once the tour has finished."""
    if per_state <= 0:
        raise ValueError("per_state must be positive")
    i = int(max(0.0, elapsed) // per_state)
    return i if i < count else None


GOLDEN_SIZE = (240, 240)
GOLDEN_POINTS = 6000
MAX_MEAN_DIFF = 2.0         # 0-255 per channel; driver rounding stays well under this
MAX_BAD_SHARE = 0.01        # share of pixels allowed to differ by more than BAD_PIXEL
BAD_PIXEL = 48


def compare(a, b) -> dict:
    """Pure: how far two RGB images are apart, and whether that is within the golden tolerance."""
    import numpy as np

    if a.shape != b.shape:
        return {"ok": False, "why": f"shape {a.shape} != {b.shape}"}
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mean = float(diff.mean())
    bad = float((diff.max(axis=2) > BAD_PIXEL).mean())
    return {"ok": mean <= MAX_MEAN_DIFF and bad <= MAX_BAD_SHARE, "mean": round(mean, 3),
            "bad_share": round(bad, 4)}


def render_states(states=TOUR, size: tuple[int, int] = (360, 360), points: int = 12000):
    """Yield ``(state, rgb uint8 (h, w, 3) top row first)`` drawn headless (EGL), captions included."""
    import numpy as np

    from . import gl
    from .captions import Rasteriser

    ctx, fbo = gl.headless(size)
    renderer = gl.SphereRenderer(ctx, points=points)
    raster = Rasteriser()
    try:
        for name in states:
            scene, t = staged(name)
            frame = scene.frame(t)
            renderer.draw(frame, t, size, captions=raster.image(frame.captions, size))
            rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(size[1], size[0], 3)
            yield name, np.flipud(rgb).copy()
    finally:
        renderer.release()
        ctx.release()
