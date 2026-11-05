"""Face motion: rig parameter values over time, from speech and state (AVATAR_PLAN §4.4, §4.5).

Pure Python (no numpy, no GL): the Scene asks ``FaceMotion.update`` once per
frame and gets every parameter the face renderer needs.

- Speech: viseme class -> mouth targets (VISEME_PARAMS). The jaw looks
  JAW_LEAD_S ahead (cancels its spring's lag); the lips look LIP_LEAD_S ahead,
  so they close before a bilabial and round before a rounded vowel
  (coarticulation). A viseme shorter than HOLD_S is dropped (no flicker),
  except closures (p/b/m, f/v), which must be seen.
- Every parameter follows a critically damped spring: no overshoot, so lips
  never pass through each other. The jaw closes within ~100 ms of
  ``speech_stop``.
- Life: blinks every 3-6 s, small saccades, breathing, slow head drift.
- State (§4.4): listening widens the eyes and centres the gaze, thinking
  glances up and aside, followup tilts the head, sleep and offline close the
  eyes. Brightness/grey and the aura hair's speed come from the state too.
"""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field


def spring(x: float, v: float, target: float, omega: float, dt: float) -> tuple[float, float]:
    """One exact step of a critically damped spring; stable for any dt >= 0."""
    if dt <= 0:
        return x, v
    d = x - target
    e = math.exp(-omega * dt)
    nx = target + (d + (v + omega * d) * dt) * e
    nv = (v - omega * (v + omega * d) * dt) * e
    return nx, nv

MOUTH_PARAMS = ("jaw_open", "lips_close", "lips_round", "lips_wide", "lip_bite", "teeth_show")

# Targets per viseme class at weight 1 (timeline.VISEME_CLASSES). The jaw is
# also scaled by loudness; everything else is the shape of the lips.
VISEME_PARAMS: dict[str, dict[str, float]] = {
    "rest": {},
    "PP": {"lips_close": 1.0},
    "FF": {"lip_bite": 1.0, "jaw_open": 0.10, "teeth_show": 0.8},
    "TH": {"jaw_open": 0.25, "teeth_show": 0.7},
    "DD": {"jaw_open": 0.30, "lips_wide": 0.25, "teeth_show": 0.6},
    "SH": {"jaw_open": 0.22, "lips_round": 0.6, "teeth_show": 0.5},
    "RR": {"jaw_open": 0.30, "lips_round": 0.45},
    "KK": {"jaw_open": 0.40},
    "OU": {"jaw_open": 0.35, "lips_round": 1.0},
    "AA": {"jaw_open": 1.00, "teeth_show": 0.3},
    "EH": {"jaw_open": 0.60, "lips_wide": 0.35, "teeth_show": 0.5},
    "IH": {"jaw_open": 0.30, "lips_wide": 0.85, "teeth_show": 0.8},
}
CLOSURES = frozenset({"PP", "FF"})
HOLD_S = 0.040
JAW_LEAD_S = 0.040
LIP_LEAD_S = 0.060
LEVEL_FLOOR = 0.15
LEVEL_ONLY_JAW = 0.7        # no visemes (subprocess TTS): jaw follows loudness alone

OMEGA = {"jaw_open": 50.0, "lips_close": 60.0, "lips_round": 45.0, "lips_wide": 45.0, "lip_bite": 60.0,
         "teeth_show": 30.0, "gaze_x": 45.0, "gaze_y": 45.0, "brow_raise": 14.0, "brow_frown": 14.0,
         "head_yaw": 3.0, "head_pitch": 3.0, "head_roll": 3.0, "bright": 5.0, "grey": 5.0,
         "hair_amp": 4.0, "hair_speed": 3.0, "eye_tint": 5.0}
EYES_CLOSE_OMEGA = 5.0      # falling asleep: slow
EYES_OPEN_OMEGA = 16.0      # waking: open in ~0.3 s (§4.4)
HAIR_PHASE_WRAP = 2 * math.pi * 1000   # hair_phase wraps here (a whole number of turns)

BLINK_EVERY_S = (3.0, 6.0)
BLINK_CLOSE_S = 0.07
BLINK_HOLD_S = 0.03
BLINK_OPEN_S = 0.12
DOUBLE_BLINK = 0.15         # share of blinks followed by a second one

GAZE_LIMIT = 1.0
HEAD_LIMIT = 0.2            # rad
FOLLOW_STATES = frozenset({"idle", "listening", "speaking", "followup"})   # thinking keeps looking away
FOLLOW_JITTER = 0.3         # her own glances, while looking at someone
FOLLOW_HEAD = (0.06, 0.04)  # rad of yaw, pitch at full gaze: the head turns a little toward them

# Per shown state. eyes: lid target (1 closed, <0 wide). gaze: (centre x, centre y,
# jitter x, jitter y, seconds between saccades lo, hi); thinking's centre x
# alternates sides. head: (yaw, pitch, roll) targets; drift: idle sway amplitude.
@dataclass(frozen=True)
class StateFace:
    eyes: float = 0.0
    gaze: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.25, 0.15, 0.7, 2.5)
    brows: tuple[float, float] = (0.0, 0.0)          # raise, frown
    head: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drift: float = 1.0
    breathe: tuple[float, float] = (0.012, 4.0)      # head pitch amplitude (rad), period (s)
    bright: float = 1.0
    grey: float = 0.0
    hair: tuple[float, float] = (0.6, 1.0)           # speed (rad/s), amplitude
    eye_tint: float = 0.25                           # irises glow the state's colour (hair's too)


FACE_STATES: dict[str, StateFace] = {
    "idle": StateFace(),
    "listening": StateFace(eyes=-0.18, gaze=(0.0, 0.0, 0.04, 0.03, 0.8, 2.0), brows=(0.35, 0.0),
                           hair=(1.6, 1.25), eye_tint=0.55),
    "thinking": StateFace(gaze=(0.5, 0.55, 0.08, 0.06, 1.2, 2.8), brows=(0.15, 0.35),
                          head=(0.0, -0.03, -0.03), hair=(1.0, 0.9), eye_tint=0.9),
    "speaking": StateFace(gaze=(0.0, 0.0, 0.12, 0.08, 1.0, 3.0), drift=0.6, hair=(1.2, 1.0), eye_tint=0.35),
    "followup": StateFace(eyes=-0.08, gaze=(0.0, 0.0, 0.05, 0.03, 1.0, 2.5), brows=(0.2, 0.0),
                          head=(0.0, 0.0, 0.07), hair=(1.0, 1.1), eye_tint=0.5),
    "sleep": StateFace(eyes=1.0, drift=0.2, breathe=(0.02, 7.0), bright=0.35, hair=(0.15, 0.5), eye_tint=0.0),
    "offline": StateFace(eyes=1.0, drift=0.0, breathe=(0.0, 1.0), bright=0.35, grey=1.0, hair=(0.0, 0.3),
                         eye_tint=0.0),
    "busy": StateFace(eyes=0.6, drift=0.2, bright=0.35, grey=0.6, hair=(0.1, 0.4)),
    "mode_switching": StateFace(gaze=(0.0, 0.3, 0.1, 0.05, 1.0, 2.0), bright=0.7, hair=(0.8, 1.0), eye_tint=0.7),
    "muted": StateFace(bright=0.8, eye_tint=0.6),
    "generating": StateFace(gaze=(0.4, 0.2, 0.1, 0.1, 1.0, 2.0), hair=(1.0, 1.0), eye_tint=0.8),
}


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def hold_filter(keys: list[tuple[float, str, float]], hold_s: float = HOLD_S) -> list[tuple[float, str, float]]:
    """Drop visemes held shorter than ``hold_s`` (closures excepted), then merge repeats.

    ``keys`` are (t_s, class, weight) in time order. The dropped viseme's time
    goes to the one before it, so the mouth does not twitch for one frame.
    """
    out: list[tuple[float, str, float]] = []
    for i, (t, cls, w) in enumerate(keys):
        nxt = keys[i + 1][0] if i + 1 < len(keys) else math.inf
        if out and nxt - t < hold_s and cls not in CLOSURES:
            continue
        if out and out[-1][1] == cls:
            out[-1] = (out[-1][0], cls, max(out[-1][2], w))
            continue
        out.append((t, cls, w))
    return out


def _viseme_at(utterances, t: float):
    """(class, weight, level, has_visemes) of the utterance playing at ``t``, or None."""
    for u in utterances:
        if u.start <= t < u.end:
            level = u.level_at(t)
            if not u.mouth_t:
                return None, 0.0, level, False
            i = bisect.bisect_right(u.mouth_t, t - u.start) - 1
            if i < 0:
                return "rest", 0.0, level, True
            return u.mouth_cls[i], u.mouth_w[i], level, True
    return None


def mouth_targets(utterances, t: float) -> dict[str, float]:
    """Mouth parameter targets at ``t`` (jaw and lips each looking ahead by their lead)."""
    out = dict.fromkeys(MOUTH_PARAMS, 0.0)
    jaw = _viseme_at(utterances, t + JAW_LEAD_S)
    if jaw is not None:
        cls, w, level, has = jaw
        if not has:
            out["jaw_open"] = LEVEL_ONLY_JAW * level
        elif level >= LEVEL_FLOOR:
            p = VISEME_PARAMS.get(cls, VISEME_PARAMS["EH"])
            out["jaw_open"] = p.get("jaw_open", 0.0) * w * (0.5 + 0.5 * level)
            out["teeth_show"] = p.get("teeth_show", 0.0) * w
    lips = _viseme_at(utterances, t + LIP_LEAD_S)
    if lips is not None and lips[3]:
        cls, w = lips[0], lips[1]
        p = VISEME_PARAMS.get(cls, VISEME_PARAMS["EH"])
        for k in ("lips_close", "lips_round", "lips_wide", "lip_bite"):
            out[k] = p.get(k, 0.0) * (1.0 if cls in CLOSURES else w)
    # Lips can only meet (or bite) with the jaw nearly closed.
    out["jaw_open"] *= (1.0 - out["lips_close"]) * (1.0 - 0.7 * out["lip_bite"])
    return out


def blink_profile(since: float) -> float:
    """Lid closure 0-1, ``since`` seconds after a blink started; 0 outside the blink."""
    if since < 0:
        return 0.0
    if since < BLINK_CLOSE_S:
        x = since / BLINK_CLOSE_S
        return x * x * (3 - 2 * x)
    since -= BLINK_CLOSE_S
    if since < BLINK_HOLD_S:
        return 1.0
    since -= BLINK_HOLD_S
    if since < BLINK_OPEN_S:
        x = 1 - since / BLINK_OPEN_S
        return x * x * (3 - 2 * x)
    return 0.0


BLINK_S = BLINK_CLOSE_S + BLINK_HOLD_S + BLINK_OPEN_S


@dataclass
class FaceMotion:
    """Stateful but deterministic for a seed and the same sequence of calls."""
    seed: int = 7
    _rng: random.Random = field(init=False, repr=False)
    _x: dict[str, float] = field(default_factory=dict)
    _v: dict[str, float] = field(default_factory=dict)
    _eyes: float = 0.0
    _eyes_v: float = 0.0
    _blink_at: float | None = None      # start of the current/next blink
    _next_blink: float | None = None
    _next_saccade: float | None = None
    _saccade: tuple[float, float] = (0.0, 0.0)
    _side: float = 1.0
    _hair_phase: float = 0.0
    _state: str | None = None
    look_at: tuple[float, float] | None = None      # gaze toward someone the camera sees (eyes.EyeFollow)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        for k in OMEGA:
            self._x.setdefault(k, 0.0)
            self._v.setdefault(k, 0.0)
        self._x["bright"] = 1.0

    def _step(self, name: str, target: float, dt: float) -> float:
        self._x[name], self._v[name] = spring(self._x[name], self._v[name], target, OMEGA[name], dt)
        return self._x[name]

    def next_blink_after(self, now: float) -> float:
        gap = self._rng.uniform(*BLINK_EVERY_S)
        return now + gap

    def _blinks(self, now: float, eyes_target: float) -> float:
        if self._next_blink is None:
            self._next_blink = self.next_blink_after(now)
        if eyes_target >= 0.5:                              # eyes shut or heavy: no blinking
            self._blink_at, self._next_blink = None, self.next_blink_after(now)
            return 0.0
        if now >= self._next_blink:
            self._blink_at = self._next_blink
            double = self._rng.random() < DOUBLE_BLINK
            self._next_blink = (self._blink_at + BLINK_S + 0.08 if double
                                else self.next_blink_after(self._blink_at))
        return blink_profile(now - self._blink_at) if self._blink_at is not None else 0.0

    def blink_now(self, now: float) -> None:
        """Start a blink at ``now`` (tests, golden frames)."""
        self._blink_at, self._next_blink = now, self.next_blink_after(now)

    def _gaze(self, now: float, sf: StateFace, state_changed: bool) -> tuple[float, float]:
        cx, cy, jx, jy, lo, hi = sf.gaze
        if state_changed or self._next_saccade is None or now >= self._next_saccade:
            self._side = -self._side if sf.gaze[0] else 1.0
            self._saccade = (self._side * cx + self._rng.uniform(-jx, jx), cy + self._rng.uniform(-jy, jy))
            self._next_saccade = now + self._rng.uniform(lo, hi)
        return self._saccade

    def update(self, now: float, dt: float, state: str, utterances=(), level: float = 0.0,
               low: float = 0.0) -> dict[str, float]:
        sf = FACE_STATES.get(state, FACE_STATES["idle"])
        changed = state != self._state
        self._state = state
        out = {k: self._step(k, v, dt) for k, v in mouth_targets(utterances, now).items()}

        omega = EYES_CLOSE_OMEGA if sf.eyes > self._eyes else EYES_OPEN_OMEGA
        self._eyes, self._eyes_v = spring(self._eyes, self._eyes_v, sf.eyes, omega, dt)
        blink = self._blinks(now, sf.eyes)
        lids = self._eyes + (1.0 - self._eyes) * blink        # a blink shuts from wherever the lids are
        out["blink_l"] = out["blink_r"] = clamp(lids, -0.3, 1.0)

        gx, gy = self._gaze(now, sf, changed)
        look = self.look_at if state in FOLLOW_STATES else None
        if look is not None:
            gx = look[0] + FOLLOW_JITTER * (gx - sf.gaze[0])
            gy = look[1] + FOLLOW_JITTER * (gy - sf.gaze[1])
        out["gaze_x"] = clamp(self._step("gaze_x", gx, dt), -GAZE_LIMIT, GAZE_LIMIT)
        out["gaze_y"] = clamp(self._step("gaze_y", gy, dt), -GAZE_LIMIT, GAZE_LIMIT)

        raise_, frown = sf.brows
        out["brow_raise"] = self._step("brow_raise", clamp(raise_ + 0.6 * low, 0.0, 1.0), dt)
        out["brow_frown"] = self._step("brow_frown", frown, dt)

        yaw, pitch, roll = sf.head
        if look is not None:
            yaw += FOLLOW_HEAD[0] * look[0]
            pitch -= FOLLOW_HEAD[1] * look[1]             # +gaze is up, +pitch is down
        d = sf.drift
        amp, period = sf.breathe
        sway = (0.035 * math.sin(0.23 * now) + 0.015 * math.sin(0.61 * now + 1.0),
                0.020 * math.sin(0.19 * now + 2.0) + amp * math.sin(2 * math.pi * now / period),
                0.012 * math.sin(0.17 * now + 0.5))
        for name, base, s in (("head_yaw", yaw, sway[0] * d), ("head_pitch", pitch, sway[1]),
                              ("head_roll", roll, sway[2] * d)):
            out[name] = clamp(self._step(name, base + s, dt), -HEAD_LIMIT, HEAD_LIMIT)

        out["bright"] = self._step("bright", sf.bright, dt)
        out["grey"] = self._step("grey", sf.grey, dt)
        out["eye_tint"] = self._step("eye_tint", sf.eye_tint, dt)
        speed, hamp = sf.hair
        out["hair_amp"] = self._step("hair_amp", hamp, dt)
        self._hair_phase = (self._hair_phase + self._step("hair_speed", speed, dt) * dt) % HAIR_PHASE_WRAP
        out["hair_phase"] = self._hair_phase
        out["glow"] = clamp(0.4 * level + 0.6 * out["jaw_open"], 0.0, 1.0)
        return out
