"""What the avatar looks like at time t: pure numpy, no GL, no window (AVATAR_PLAN §4.1).

The renderer feeds bus events in with ``apply`` and asks ``frame(now)`` for the
numbers to draw. Speech is a schedule: each ``utterance`` event carries its
playback start, a 60 Hz level envelope, low/high band envelopes and visemes,
and the scene plays them against the shared monotonic clock. Mouth openness
follows a critically damped spring, aimed LOOKAHEAD_S ahead to cancel its lag.

Phase 2 (§4.4): the *shown* state can differ from the daemon's. The sphere
sleeps after ``sleep_s`` idle (or 30 s in quiet hours) and wakes on any event;
with the daemon down it shows "busy" in coding mode (``local_mode``, detected
by the window) instead of "offline". Per-state motion (ripple, swirl, pulse
ring) eases in and out, and captions fade after CAPTION_HOLD_S.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Callable

from ..doorbell import in_quiet_hours
from ..speech import split_sentences
from .face import FaceMotion, hold_filter, spring  # noqa: F401 - spring re-exported

# Mouth openness of each viseme class (timeline.VISEME_CLASSES), 0 closed .. 1 wide.
VISEME_OPEN = {"rest": 0.0, "PP": 0.0, "FF": 0.15, "TH": 0.3, "DD": 0.3, "SH": 0.35,
               "RR": 0.4, "KK": 0.35, "OU": 0.55, "AA": 1.0, "EH": 0.7, "IH": 0.5}

# Frames per second by state (§4.3): smooth when something is happening, cheap otherwise.
FPS = {"speaking": 60, "listening": 60, "followup": 60, "thinking": 60, "generating": 30,
       "mode_switching": 30, "idle": 30, "muted": 30, "sleep": 10, "offline": 10, "busy": 10}

# (r, g, b) 0-1 and base glow per state.
LOOK = {"idle": ((0.20, 0.75, 1.00), 0.35), "listening": ((0.35, 0.95, 1.00), 0.85),
        "thinking": ((0.55, 0.55, 1.00), 0.55), "speaking": ((0.25, 0.85, 1.00), 0.70),
        "followup": ((0.35, 0.95, 0.85), 0.60), "sleep": ((0.15, 0.35, 0.60), 0.12),
        "muted": ((1.00, 0.30, 0.25), 0.40), "offline": ((0.45, 0.45, 0.45), 0.10),
        "generating": ((0.90, 0.70, 0.30), 0.55), "mode_switching": ((0.90, 0.70, 0.30), 0.45),
        "busy": ((0.55, 0.45, 0.85), 0.18)}

# Motion amounts (ripple, swirl, ring) per shown state; anything absent is still.
MOTION = {"listening": (1.0, 0.0, 0.0), "thinking": (0.0, 1.0, 0.0),
          "followup": (0.0, 0.0, 1.0), "mode_switching": (0.0, 0.0, 1.0)}
MOTION_OMEGA = 6.0          # 1/s; motion eases in/out over ~0.3 s
SWIRL_RATE = 1.6            # rad/s of extra equatorial rotation at swirl 1
RING_PERIOD_S = 1.6
STATUS_TEXT = {"busy": "coding mode", "mode_switching": "switching modes"}

SLEEP_S = 300.0             # Adam 2026-11-04: five idle minutes
QUIET_SLEEP_S = 30.0        # in quiet hours, sleep again soon after any activity
CAPTION_HOLD_S = 4.0        # after it was said (Jarvis: after he stops talking)
CAPTION_FADE_S = 1.0

BASE_RADIUS = 0.55
OPEN_RADIUS = 0.18          # extra radius at full openness
BREATHE = {"idle": (0.02, 4.0), "sleep": (0.015, 7.0), "offline": (0.0, 1.0)}
JAW_OMEGA = 45.0            # rad/s; a critically damped spring lags ~2/omega
LOOKAHEAD_S = 0.040
COLOUR_OMEGA = 8.0
STALE_S = 1.0               # forget an utterance this long after it ends
LEVEL_FLOOR = 0.15          # level (0-1) below which a viseme is not voiced


@dataclass
class Utterance:
    id: str
    start: float
    end: float
    level: list[int]
    level_hz: float
    vis_t: list[float]          # seconds from start
    vis_open: list[float]       # openness * weight
    low: list[int] = field(default_factory=list)    # band envelopes at level_hz
    high: list[int] = field(default_factory=list)
    mouth_t: list[float] = field(default_factory=list)   # the face's visemes after face.hold_filter
    mouth_cls: list[str] = field(default_factory=list)
    mouth_w: list[float] = field(default_factory=list)

    @classmethod
    def from_event(cls, data: dict) -> "Utterance":
        start = float(data["start_mono"])
        level = [int(v) for v in data.get("level") or []]
        hz = float(data.get("level_hz") or 60.0)
        duration = float(data.get("duration_ms") or len(level) / hz * 1000) / 1000
        vis = data.get("visemes") or []
        bands = data.get("bands") or {}
        held = hold_filter([(float(v[0]) / 1000, str(v[1]), float(v[2])) for v in vis])
        return cls(id=str(data.get("id", "")), start=start, end=start + duration, level=level,
                   level_hz=hz, vis_t=[float(v[0]) / 1000 for v in vis],
                   vis_open=[VISEME_OPEN.get(v[1], 0.5) * float(v[2]) for v in vis],
                   low=[int(v) for v in bands.get("low") or []],
                   high=[int(v) for v in bands.get("high") or []],
                   mouth_t=[k[0] for k in held], mouth_cls=[k[1] for k in held],
                   mouth_w=[k[2] for k in held])

    def _sample(self, values: list[int], t: float) -> float:
        i = int((t - self.start) * self.level_hz)
        return values[i] / 255 if 0 <= i < len(values) else 0.0

    def level_at(self, t: float) -> float:
        return self._sample(self.level, t)

    def bands_at(self, t: float) -> tuple[float, float]:
        return self._sample(self.low, t), self._sample(self.high, t)

    def openness_at(self, t: float) -> float:
        """Viseme shape scaled by loudness; level alone when there are no visemes."""
        level = self.level_at(t)
        if not self.vis_t:
            return level
        i = bisect.bisect_right(self.vis_t, t - self.start) - 1
        shape = self.vis_open[i] if i >= 0 else 0.0
        if level < LEVEL_FLOOR:
            return 0.0
        return shape * (0.5 + 0.5 * level)


@dataclass
class Frame:
    state: str                  # shown state (may be sleep/busy while the daemon says idle/-)
    radius: float
    openness: float
    colour: tuple[float, float, float]
    glow: float
    spin: float                 # rad/s
    angle: float                # integrated rotation, rad
    fps: int
    ripple: float = 0.0         # 0-1 inward ripple (listening)
    swirl: float = 0.0          # integrated equatorial swirl, rad (thinking)
    ring: float = -1.0          # pulse ring position 0-1 across the sphere, -1 none
    ring_amp: float = 0.0
    low: float = 0.0            # speech bands 0-1
    high: float = 0.0
    captions: list[tuple[str, str, float]] = field(default_factory=list)  # (who, text, alpha)
    params: dict[str, float] = field(default_factory=dict)   # face rig values (face.FaceMotion)


@dataclass
class _Said:
    sentences: list[str]
    at: float | None
    starts: list[float] = field(default_factory=list)   # utterance starts since the caption


def _fade(age: float) -> float:
    """Caption alpha: 1 for CAPTION_HOLD_S, then linear to 0 over CAPTION_FADE_S."""
    if age < CAPTION_HOLD_S:
        return 1.0
    return max(0.0, 1.0 - (age - CAPTION_HOLD_S) / CAPTION_FADE_S)


def _local_time() -> dtime:
    return datetime.now().time()


@dataclass
class Scene:
    state: str = "offline"      # the daemon's state
    connected: bool = False
    utterances: list[Utterance] = field(default_factory=list)
    sleep_s: float = SLEEP_S    # <= 0: never sleep from idleness
    quiet: tuple[dtime, dtime] | None = None
    clock: Callable[[], dtime] = _local_time
    local_mode: str | None = None   # None | "coding" | "switching" (window-detected)
    _heard: tuple[str, float | None] | None = None
    _said: _Said | None = None
    _poked: bool = True
    _active_at: float | None = None
    _open: float = 0.0
    _open_v: float = 0.0
    _colour: tuple[float, float, float] = LOOK["offline"][0]
    _glow: float = LOOK["offline"][1]
    _motion: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _swirl: float = 0.0
    _last: float | None = None
    _angle: float = 0.0
    face: FaceMotion = field(default_factory=FaceMotion)

    # ---- events ----
    def apply(self, event: dict) -> None:
        etype, data = event.get("type"), event.get("data") or {}
        self._poked = True                      # any event wakes a sleeping sphere
        t = event.get("t_mono")
        t = float(t) if isinstance(t, (int, float)) and t > 0 else None
        if etype == "snapshot":
            for inner in (data.get("events") or {}).values():
                self.apply(inner)
        elif etype == "state":
            name = data.get("state")
            if name in FPS:
                self.state = name
        elif etype == "utterance":
            try:
                u = Utterance.from_event(data)
            except (KeyError, TypeError, ValueError, IndexError):
                return
            self.utterances.append(u)
            self.utterances.sort(key=lambda x: x.start)
            if self._said is not None:
                self._said.starts.append(u.start)
        elif etype == "speech_stop":
            self.utterances.clear()
        elif etype == "caption":
            text = str(data.get("text", "")).strip()
            if not text:
                return
            if data.get("who") == "jarvis":
                self._said = _Said(split_sentences(text) or [text], t)
            else:
                self._heard, self._said = (text, t), None     # a new turn

    def set_connected(self, connected: bool) -> None:
        self.connected = connected
        self._poked = True
        if not connected:
            self.state = "offline"
            self.utterances.clear()

    # ---- evaluation ----
    def shown_state(self, now: float) -> str:
        if self.local_mode == "switching":
            return "mode_switching"
        if not self.connected:
            return "busy" if self.local_mode == "coding" else "offline"
        if self.state == "idle" and self._active_at is not None:
            quiet = in_quiet_hours(self.quiet, self.clock())
            limit = QUIET_SLEEP_S if quiet else self.sleep_s
            if limit > 0 and now - self._active_at >= limit:
                return "sleep"
        return self.state

    def openness_target(self, t: float) -> float:
        for u in self.utterances:
            if u.start <= t < u.end:
                return u.openness_at(t)
        return 0.0

    def bands_at(self, t: float) -> tuple[float, float]:
        for u in self.utterances:
            if u.start <= t < u.end:
                return u.bands_at(t)
        return 0.0, 0.0

    def fps(self, now: float | None = None) -> int:
        if self.utterances:
            return 60
        state = self.state if now is None else self.shown_state(now)
        return FPS.get(state, 30)

    def captions(self, now: float, shown: str) -> list[tuple[str, str, float]]:
        out = []
        if shown in STATUS_TEXT:
            out.append(("status", STATUS_TEXT[shown], 1.0))
        if self._heard is not None:
            text, at = self._heard
            alpha = _fade(now - at)
            if alpha > 0:
                out.append(("user", text, alpha))
        if self._said is not None:
            said = self._said
            begun = sum(1 for s in said.starts if s <= now)
            sentence = said.sentences[min(max(begun - 1, 0), len(said.sentences) - 1)]
            talking = any(u.start <= now < u.end for u in self.utterances) \
                or self.state == "speaking"
            if talking or said.at is None:
                said.at = now                   # fade counts from when he stops talking
            alpha = _fade(now - said.at)
            if alpha > 0:
                out.append(("jarvis", sentence, alpha))
        return out

    def frame(self, now: float) -> Frame:
        dt = 0.0 if self._last is None else min(max(now - self._last, 0.0), 0.25)
        self._last = now
        if self._poked or self._active_at is None:
            self._active_at, self._poked = now, False
        if self._heard is not None and self._heard[1] is None:
            self._heard = (self._heard[0], now)
        self.utterances = [u for u in self.utterances if now < u.end + STALE_S]
        if self.utterances:
            self._active_at = now
        shown = self.shown_state(now)

        target = self.openness_target(now + LOOKAHEAD_S)
        self._open, self._open_v = spring(self._open, self._open_v, target, JAW_OMEGA, dt)
        openness = min(max(self._open, 0.0), 1.0)
        low, high = self.bands_at(now + LOOKAHEAD_S)

        colour, glow = LOOK.get(shown, LOOK["idle"])
        k = 1 - math.exp(-COLOUR_OMEGA * dt)
        self._colour = tuple(c + (n - c) * k for c, n in zip(self._colour, colour))
        self._glow += (glow - self._glow) * k

        km = 1 - math.exp(-MOTION_OMEGA * dt)
        goal = MOTION.get(shown, (0.0, 0.0, 0.0))
        self._motion = tuple(m + (g - m) * km for m, g in zip(self._motion, goal))
        ripple, swirl, ring_amp = self._motion
        self._swirl = (self._swirl + SWIRL_RATE * swirl * dt) % (2 * math.pi)

        amp, period = BREATHE.get(shown, (0.01, 3.0))
        radius = BASE_RADIUS * (1 + amp * math.sin(2 * math.pi * now / period)) + OPEN_RADIUS * openness
        spin = {"thinking": 1.2, "sleep": 0.05, "offline": 0.0, "busy": 0.05}.get(shown, 0.25)
        self._angle = (self._angle + spin * dt) % (2 * math.pi)
        level = next((u.level_at(now) for u in self.utterances if u.start <= now < u.end), 0.0)
        params = self.face.update(now, dt, shown, self.utterances, level, low)
        return Frame(params=params,state=shown, radius=radius, openness=openness, colour=self._colour,
                     glow=min(1.0, self._glow + 0.3 * openness), spin=spin, angle=self._angle,
                     fps=self.fps(now), ripple=ripple, swirl=self._swirl,
                     ring=(now % RING_PERIOD_S) / RING_PERIOD_S if ring_amp > 0.01 else -1.0,
                     ring_amp=ring_amp, low=low, high=high, captions=self.captions(now, shown))
