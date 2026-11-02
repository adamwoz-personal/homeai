"""What the avatar looks like at time t: pure numpy, no GL, no window (AVATAR_PLAN §4.1).

The renderer feeds bus events in with ``apply`` and asks ``frame(now)`` for the
numbers to draw. Speech is a schedule: each ``utterance`` event carries its
playback start, a 60 Hz level envelope and visemes, and the scene plays them
against the shared monotonic clock. Mouth openness follows a critically damped
spring, aimed LOOKAHEAD_S ahead to cancel the spring's lag.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

# Mouth openness of each viseme class (timeline.VISEME_CLASSES), 0 closed .. 1 wide.
VISEME_OPEN = {"rest": 0.0, "PP": 0.0, "FF": 0.15, "TH": 0.3, "DD": 0.3, "SH": 0.35,
               "RR": 0.4, "KK": 0.35, "OU": 0.55, "AA": 1.0, "EH": 0.7, "IH": 0.5}

# Frames per second by state (§4.3): smooth when something is happening, cheap otherwise.
FPS = {"speaking": 60, "listening": 60, "followup": 60, "thinking": 60, "generating": 30,
       "mode_switching": 30, "idle": 30, "muted": 30, "sleep": 10, "offline": 10}

# (r, g, b) 0-1 and base glow per state.
LOOK = {"idle": ((0.20, 0.75, 1.00), 0.35), "listening": ((0.35, 0.95, 1.00), 0.85),
        "thinking": ((0.55, 0.55, 1.00), 0.55), "speaking": ((0.25, 0.85, 1.00), 0.70),
        "followup": ((0.35, 0.95, 0.85), 0.60), "sleep": ((0.15, 0.35, 0.60), 0.12),
        "muted": ((1.00, 0.30, 0.25), 0.40), "offline": ((0.45, 0.45, 0.45), 0.10),
        "generating": ((0.90, 0.70, 0.30), 0.55), "mode_switching": ((0.90, 0.70, 0.30), 0.45)}

BASE_RADIUS = 0.55
OPEN_RADIUS = 0.18          # extra radius at full openness
BREATHE = {"idle": (0.02, 4.0), "sleep": (0.015, 7.0), "offline": (0.0, 1.0)}
JAW_OMEGA = 45.0            # rad/s; a critically damped spring lags ~2/omega
LOOKAHEAD_S = 0.040
COLOUR_OMEGA = 8.0
STALE_S = 1.0               # forget an utterance this long after it ends
LEVEL_FLOOR = 0.15          # level (0-1) below which a viseme is not voiced


def spring(x: float, v: float, target: float, omega: float, dt: float) -> tuple[float, float]:
    """One exact step of a critically damped spring; stable for any dt >= 0."""
    if dt <= 0:
        return x, v
    d = x - target
    e = math.exp(-omega * dt)
    nx = target + (d + (v + omega * d) * dt) * e
    nv = (v - omega * (v + omega * d) * dt) * e
    return nx, nv


@dataclass
class Utterance:
    id: str
    start: float
    end: float
    level: list[int]
    level_hz: float
    vis_t: list[float]          # seconds from start
    vis_open: list[float]       # openness * weight

    @classmethod
    def from_event(cls, data: dict) -> "Utterance":
        start = float(data["start_mono"])
        level = [int(v) for v in data.get("level") or []]
        hz = float(data.get("level_hz") or 60.0)
        duration = float(data.get("duration_ms") or len(level) / hz * 1000) / 1000
        vis = data.get("visemes") or []
        return cls(id=str(data.get("id", "")), start=start, end=start + duration, level=level,
                   level_hz=hz, vis_t=[float(v[0]) / 1000 for v in vis],
                   vis_open=[VISEME_OPEN.get(v[1], 0.5) * float(v[2]) for v in vis])

    def level_at(self, t: float) -> float:
        i = int((t - self.start) * self.level_hz)
        return self.level[i] / 255 if 0 <= i < len(self.level) else 0.0

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
    state: str
    radius: float
    openness: float
    colour: tuple[float, float, float]
    glow: float
    spin: float                 # rad/s
    angle: float                # integrated rotation, rad
    fps: int
    caption: str = ""


@dataclass
class Scene:
    state: str = "offline"
    connected: bool = False
    utterances: list[Utterance] = field(default_factory=list)
    caption: tuple[str, str, float] = ("", "", 0.0)
    _open: float = 0.0
    _open_v: float = 0.0
    _colour: tuple[float, float, float] = LOOK["offline"][0]
    _glow: float = LOOK["offline"][1]
    _last: float | None = None
    _angle: float = 0.0

    # ---- events ----
    def apply(self, event: dict) -> None:
        etype, data = event.get("type"), event.get("data") or {}
        if etype == "snapshot":
            for inner in (data.get("events") or {}).values():
                self.apply(inner)
        elif etype == "state":
            name = data.get("state")
            if name in FPS:
                self.state = name
        elif etype == "utterance":
            try:
                self.utterances.append(Utterance.from_event(data))
            except (KeyError, TypeError, ValueError, IndexError):
                return
            self.utterances.sort(key=lambda u: u.start)
        elif etype == "speech_stop":
            self.utterances.clear()
        elif etype == "caption":
            self.caption = (str(data.get("who", "")), str(data.get("text", "")),
                            float(event.get("t_mono") or 0.0))

    def set_connected(self, connected: bool) -> None:
        self.connected = connected
        if not connected:
            self.state = "offline"
            self.utterances.clear()

    # ---- evaluation ----
    def openness_target(self, t: float) -> float:
        for u in self.utterances:
            if u.start <= t < u.end:
                return u.openness_at(t)
        return 0.0

    def fps(self) -> int:
        if self.utterances:
            return 60
        return FPS.get(self.state, 30)

    def frame(self, now: float) -> Frame:
        dt = 0.0 if self._last is None else min(max(now - self._last, 0.0), 0.25)
        self._last = now
        self.utterances = [u for u in self.utterances if now < u.end + STALE_S]

        target = self.openness_target(now + LOOKAHEAD_S)
        self._open, self._open_v = spring(self._open, self._open_v, target, JAW_OMEGA, dt)
        openness = min(max(self._open, 0.0), 1.0)

        colour, glow = LOOK.get(self.state, LOOK["idle"])
        k = 1 - math.exp(-COLOUR_OMEGA * dt)
        self._colour = tuple(c + (n - c) * k for c, n in zip(self._colour, colour))
        self._glow += (glow - self._glow) * k

        amp, period = BREATHE.get(self.state, (0.01, 3.0))
        radius = BASE_RADIUS * (1 + amp * math.sin(2 * math.pi * now / period)) + OPEN_RADIUS * openness
        spin = {"thinking": 1.2, "sleep": 0.05, "offline": 0.0}.get(self.state, 0.25)
        self._angle = (self._angle + spin * dt) % (2 * math.pi)
        who, text, _ = self.caption
        return Frame(state=self.state, radius=radius, openness=openness, colour=self._colour,
                     glow=min(1.0, self._glow + 0.3 * openness), spin=spin, angle=self._angle,
                     fps=self.fps(), caption=f"{who}: {text}" if text else "")
