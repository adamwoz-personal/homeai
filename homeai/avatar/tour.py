"""Put a Scene into each shown state on demand (AVATAR_PLAN §10 Phase 2).

One definition shared by the golden-image tests, ``tools/avatar/update_golden.py``,
``tools/avatar/state_tour.py`` (Adam's across-the-room check) and
``tools/avatar/avatar_profile.py``, so they all show the same thing.
"""
from __future__ import annotations

import math

from .face import BLINK_CLOSE_S
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


FACE_VISEMES = ("rest", "PP", "FF", "TH", "DD", "SH", "RR", "KK", "OU", "AA", "EH", "IH")
FACE_TOUR = FACE_VISEMES + ("blink", "sleep", "listening", "thinking", "offline", "pulse")
PULSE_BURST_S = 0.12        # one loud syllable
PULSE_POSE_DELAY_S = 0.35   # the "pulse" pose: this long after the burst began


def held_viseme(start: float, cls: str, seconds: float = 1.0, hz: int = 60) -> dict:
    """One utterance holding viseme ``cls`` at full loudness: the mouth settles on its shape."""
    n = int(seconds * hz)
    return {"type": "utterance", "data": {
        "id": f"hold-{cls}", "start_mono": start, "level_hz": hz, "level": [255] * n,
        "duration_ms": seconds * 1000, "visemes": [[0, cls, 1.0]], "bands": {"low": [0] * n, "high": [0] * n}}}


def staged_face(name: str, t0: float = 100.0) -> tuple[Scene, float]:
    """A Scene posed for face golden ``name``: a held viseme, a mid-blink, or a settled state."""
    if name not in FACE_TOUR:
        raise ValueError(f"unknown face pose {name!r}; pick from {', '.join(FACE_TOUR)}")
    if name in FACE_VISEMES:
        scene, t = staged("speaking", t0=t0, settle=False)
        scene.utterances.clear()
        scene.apply(held_viseme(t, name))
        end = t + 0.5
    else:
        scene, t = staged("idle" if name in ("blink", "pulse") else name, t0=t0)
        end = t + (BLINK_CLOSE_S + 0.01 if name == "blink" else 0.0)
        if name == "blink":
            scene.face.blink_now(t)
    while t < end:
        t = min(end, t + 1 / 60)
        scene.frame(t)
    return scene, t


def speech_burst(pulse, t: float, delay: float, length: float = PULSE_BURST_S, low: float = 1.0) -> None:
    """Give a HairPulse the history of one loud burst that began ``delay`` s before ``t``.

    ``low=0`` leaves brightness alone: no sway or dot growth to confuse a measurement.
    """
    pulse.reset()
    start, step = t - delay, 1 / 60
    for i in range(int(round(delay / step))):
        u = start + i * step
        loud = 1.0 if u < start + length else 0.0
        pulse.advance(u, low * loud, loud, loud)


def pulse_arc(rgb, silent, rest_px, arc, threshold: int = 24) -> tuple[float, float]:
    """Where a pulse is: median ``arc`` of the hair points nearest the changed pixels, and their share.

    ``rest_px`` is each hair point's resting pixel and ``arc`` its distance along
    its strand. Distance from the image centre won't do: roots run down the whole
    silhouette, so a pulse starts at every radius.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    diff = np.abs(rgb.astype(np.int16) - silent.astype(np.int16)).max(axis=2)
    ys, xs = np.nonzero(diff > threshold)
    if not len(xs):
        return 0.0, 0.0
    near = cKDTree(rest_px).query(np.column_stack([xs, ys]))[1]
    return float(np.median(arc[near])), float(len(xs) / diff.size)


def pulse_light(rgb, silent) -> float:
    """How much light a pulse adds: mean brightness gain over the frame."""
    import numpy as np

    return float((rgb.astype(np.float64) - silent.astype(np.float64)).clip(min=0).mean())


def tour_index(elapsed: float, per_state: float, count: int = len(TOUR)) -> int | None:
    """Which tour state to show ``elapsed`` s in; None once the tour has finished."""
    if per_state <= 0:
        raise ValueError("per_state must be positive")
    i = int(max(0.0, elapsed) // per_state)
    return i if i < count else None


GOLDEN_SIZE = (240, 240)
GOLDEN_POINTS = 6000
FACE_GOLDEN_SIZE = (256, 256)
FACE_GOLDEN_POINTS = 6000
FACE_FIXTURE = "tests/fixtures/avatar/landmarks_3341.json"
# Pairs that must look different (viseme pairs that share a shape, e.g. DD/TH, are allowed to be close).
FACE_DISTINCT = (("rest", "AA"), ("rest", "OU"), ("rest", "IH"), ("AA", "OU"), ("PP", "AA"),
                 ("rest", "blink"), ("rest", "sleep"), ("rest", "listening"), ("rest", "thinking"),
                 ("rest", "offline"))
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


def golden_asset(fixture=None):
    """The synthetic test face (tests/fixtures/avatar landmarks)."""
    import json
    import tempfile
    from pathlib import Path

    import numpy as np

    from . import assets, facebuild

    fixture = Path(fixture) if fixture else Path(__file__).resolve().parents[2] / FACE_FIXTURE
    data = json.loads(fixture.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        facebuild.synthetic_face(np.array(data["landmarks"]), int(data["image_size"][0]), Path(tmp) / "v1",
                                 points=FACE_GOLDEN_POINTS)
        return assets.load(Path(tmp) / "v1")


def render_face_golden(poses=FACE_TOUR, fixture=None):
    """Yield ``(pose, rgb)`` of the synthetic test face."""
    yield from render_face(golden_asset(fixture), poses, FACE_GOLDEN_SIZE)


def render_face(asset, poses=FACE_TOUR, size: tuple[int, int] = (360, 360), hair: bool = True,
                captions: bool = False):
    """Yield ``(pose, rgb)`` of the face look drawn headless (EGL)."""
    import numpy as np

    from . import gl
    from .captions import Rasteriser
    from .facegl import FaceRenderer

    ctx, fbo = gl.headless(size)
    renderer = FaceRenderer(ctx, asset, hair=hair)
    raster = Rasteriser()
    try:
        for name in poses:
            scene, t = staged_face(name)
            frame = scene.frame(t)
            renderer.pulse.reset()                          # poses don't inherit each other's speech
            if name == "pulse":
                speech_burst(renderer.pulse, t, PULSE_POSE_DELAY_S)
            renderer.draw(frame, t, size, captions=raster.image(frame.captions, size) if captions else None)
            rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(size[1], size[0], 3)
            yield name, np.flipud(rgb).copy()
    finally:
        renderer.release()
        ctx.release()


def wind_coherence(rest, face_h: float, centre_x: float, phase_span: float, samples: int = 240,
                   pairs: int = 200, seed: int = 1) -> dict:
    """How the breeze moves the hair over ``phase_span`` rad: {"neighbours", "mirror"}, mean correlations.

    ``neighbours``: sideways motion of a point and one 0.1 face heights away. ``mirror``:
    a point's sideways motion against the reflection of its opposite point's. 1 is a perfect mirror.
    """
    import numpy as np

    from .facegl import hair_wind

    q = np.asarray(rest, dtype=np.float64) / face_h
    rng = np.random.default_rng(seed)
    pick = q[rng.choice(len(q), pairs)]
    cx = centre_x / face_h
    times = np.linspace(0.0, phase_span, samples)

    def motion(points):
        return np.stack([hair_wind(points, t)[:, 0] for t in times], axis=1)

    def corr(a, b):
        a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
        return float(((a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).mean())

    here = motion(pick)
    mirrored = pick * (-1.0, 1.0) + (2.0 * cx, 0.0)
    return {"neighbours": round(corr(here, motion(pick + (0.1, 0.0))), 3),
            "mirror": round(corr(here, -motion(mirrored)), 3)}


def render_wind(asset, seconds, size: tuple[int, int] = (360, 360), speed: float = 0.6):
    """RGB frames of the idle hair ``seconds`` into a breeze blowing at ``speed`` rad/s."""
    import dataclasses

    import numpy as np

    from . import gl
    from .facegl import FaceRenderer

    ctx, fbo = gl.headless(size)
    renderer = FaceRenderer(ctx, asset)
    scene, t = staged("idle")
    frame = scene.frame(t)
    out = []
    try:
        for s in seconds:
            params = {**frame.params, "hair_phase": frame.params["hair_phase"] + speed * s}
            renderer.draw(dataclasses.replace(frame, params=params), t, size)
            rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(size[1], size[0], 3)
            out.append(np.flipud(rgb).copy())
        return out
    finally:
        renderer.release()
        ctx.release()


def render_pulse(asset, delays, size: tuple[int, int] = (360, 360), low: float = 1.0):
    """``(silent rgb, {delay: rgb}, hair rest px, arc in face heights)``: the idle face with no
    speech, then ``delay`` s after a burst."""
    import numpy as np

    from . import gl
    from . import facegl

    ctx, fbo = gl.headless(size)
    renderer = facegl.FaceRenderer(ctx, asset)
    scene, t = staged("idle")
    frame = scene.frame(t)

    def shot():
        renderer.draw(frame, t, size)
        rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(size[1], size[0], 3)
        return np.flipud(rgb).copy()

    try:
        renderer.pulse.reset()
        silent = shot()
        out = {}
        for delay in delays:
            speech_burst(renderer.pulse, t, delay, low=low)
            out[delay] = shot()
        layout = facegl.asset_hair(asset, renderer.spacing)
        rest = facegl.to_pixels(facegl.hair_rest(layout), renderer.side, size, zoom=renderer.zoom)
        return silent, out, rest, layout[:, 3] * layout[:, 4] / renderer.geo["face_h"]
    finally:
        renderer.release()
        ctx.release()
