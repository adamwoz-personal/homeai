# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Home control: lights, kitchen timers, announcements and music.

One service layer shared by the daemon's fast path (``home_intents``) and the
MCP tools the model can call. Every public method returns a short sentence to
speak, or raises ``HomeError`` / ``HAError`` whose text is speakable.

Design notes (see plans/HOME_ASSISTANT_PLAN.md):

* Lights go to Home Assistant's Hue integration. Hue *rooms* (entities with
  ``is_hue_group``) are the names people use -- "the foyer", "the den fan" --
  so they are matched first. Lights HA doesn't know (bulbs paired straight to
  an Echo) fall back to asking Alexa.
* Timers, music and announcements go to the Echos through ``alexa_devices``.
  Alexa commands are always built from fixed templates around a cleaned-up
  phrase. There is no free-form "say this to Alexa": that would let a
  misheard sentence -- or the model -- tell Alexa to buy, unlock or call.
* HA's timer sensor lags ~90 s and has no label, so Jarvis keeps its own
  ledger of the timers it set, shared between processes via a small file.
* An announcement in Jarvis's name woke Jarvis (2026-10-09). Wake words are
  stripped from anything sent to the Echos, and a quiet window tells the
  daemon to ignore its wake word while the announcement plays.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .ha_client import HAClient

log = logging.getLogger("homeai.home")


class HomeError(RuntimeError):
    """A request Jarvis understood but can't carry out. Speakable."""


# Anything sent to Alexa containing these is refused outright.
BLOCKED_WORDS = frozenset("""
    buy order orders purchase reorder cart checkout pay payment shop shopping
    call calls dial drop unlock lock open opening disarm arm alarm security
    delete erase send message text email routine routines enable disable skill
    password code pin
""".split())

EVERYWHERE = frozenset({"everywhere", "all", "every room", "whole house", "the whole house",
                        "house", "the house", "all rooms", "everyone", "everybody"})
ALL_LIGHTS = frozenset({"all", "everything", "all of them", "every light", "house",
                        "the house", "whole house", "downstairs", "everywhere"})

UNAVAILABLE = ("unavailable", "unknown")


def norm(text: str) -> str:
    """Lowercase, drop apostrophes and punctuation, drop a leading 'the'."""
    text = text.lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9%]+", " ", text).strip()
    text = re.sub(r"^(?:the|my|our)\s+", "", text)
    return text


def norm_light(text: str) -> str:
    """Like ``norm`` but also drops a trailing 'light(s)'."""
    text = norm(text)
    text = re.sub(r"^all (?:the |of the )?", "all ", text)
    stripped = re.sub(r"\s*\blights?$", "", text).strip()
    return stripped or text


def safe_phrase(text: str, max_words: int) -> str:
    """A phrase fit to embed in an Alexa command template, or HomeError."""
    clean = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower())
    words = clean.split()
    if not words:
        raise HomeError("I didn't get what to ask Alexa for.")
    if len(words) > max_words:
        raise HomeError("That's too long to pass on to Alexa.")
    if BLOCKED_WORDS & {w.strip("'") for w in words}:
        raise HomeError("I won't pass that on to Alexa.")
    return " ".join(words)


def strip_wake_words(text: str, wake_words: Iterable[str]) -> str:
    for word in wake_words:
        text = re.sub(rf"\b(?:hey|okay|ok)?\s*{re.escape(word)}\b[,.!]?", "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip(" ,")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    for value, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second")):
        if value:
            parts.append(f"{value} {unit}{'' if value == 1 else 's'}")
    if not parts:
        return "0 seconds"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def format_remaining(seconds: float) -> str:
    """Spoken time left: seconds matter near the end, not an hour out."""
    seconds = max(0, int(round(seconds)))
    if seconds >= 600:
        seconds = int(round(seconds / 60.0)) * 60
    return format_duration(seconds)


def join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


# -- cross-process state ----------------------------------------------------

def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    path = Path(base) / "homeai" if base else Path(tempfile.gettempdir()) / f"homeai-{os.getuid()}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class EchoQuiet:
    """'Ignore the wake word until T' -- written by whoever announces, read
    by the daemon's wake loop (a different process when the model announces)."""

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock

    @property
    def path(self) -> Path:
        return self._path or runtime_dir() / "echo-quiet"

    def mark(self, seconds: float) -> None:
        try:
            _write_atomic(self.path, f"{self._clock() + seconds:.3f}")
        except OSError as exc:
            log.warning("could not write echo quiet window: %s", exc)

    def remaining(self) -> float:
        try:
            until = float(self.path.read_text().strip())
        except (OSError, ValueError):
            return 0.0
        return max(0.0, until - self._clock())


def announcement_seconds(message: str) -> float:
    """How long an Echo takes to chime and read a message out, generously."""
    return min(45.0, 4.0 + len(message.split()) / 2.2)


@dataclass
class TimerEntry:
    label: str
    ends_at: float
    set_at: float
    echo: str

    def remaining(self, now: float) -> float:
        return self.ends_at - now


class TimerLedger:
    """Timers Jarvis set, kept in a JSON file. A missing or corrupt file is
    an empty ledger: losing timer labels must never break a voice turn."""

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock

    @property
    def path(self) -> Path:
        return self._path or runtime_dir() / "timers.json"

    def _load(self) -> list[TimerEntry]:
        try:
            raw = json.loads(self.path.read_text())
            return [TimerEntry(str(t["label"]), float(t["ends_at"]), float(t["set_at"]),
                               str(t["echo"])) for t in raw]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("timer ledger unreadable (%s); starting empty", exc)
            return []

    def _save(self, entries: list[TimerEntry]) -> None:
        try:
            _write_atomic(self.path, json.dumps([e.__dict__ for e in entries]))
        except OSError as exc:
            log.warning("could not save timer ledger: %s", exc)

    def active(self) -> list[TimerEntry]:
        now = self._clock()
        entries = self._load()
        live = [e for e in entries if e.ends_at > now]
        if len(live) != len(entries):
            self._save(live)
        return sorted(live, key=lambda e: e.ends_at)

    def add(self, label: str, seconds: float, echo: str) -> TimerEntry:
        now = self._clock()
        entry = TimerEntry(label, now + seconds, now, echo)
        self._save(self.active() + [entry])
        return entry

    def remove(self, entries: list[TimerEntry]) -> None:
        drop = {(e.label, e.ends_at) for e in entries}
        self._save([e for e in self.active() if (e.label, e.ends_at) not in drop])


# -- the house --------------------------------------------------------------

@dataclass(frozen=True)
class Light:
    entity_id: str
    name: str
    group: bool
    state: str


@dataclass(frozen=True)
class Echo:
    name: str
    device_id: str
    group: bool
    available: bool


# Minimum seconds between setting a timer and cancelling it (see _settle_timers).
TIMER_CANCEL_MIN_GAP_S = 15


class Home:
    def __init__(self, client: HAClient, *, default_echo: str = "kitchen",
                 wake_words: Iterable[str] = ("jarvis",),
                 ledger: TimerLedger | None = None, quiet: EchoQuiet | None = None,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.client = client
        self.default_echo = default_echo
        self.wake_words = tuple(wake_words)
        self.ledger = ledger or TimerLedger(clock=clock)
        self.quiet = quiet or EchoQuiet(clock=clock)
        self._clock = clock
        self._sleep = sleep

    # -- lights -------------------------------------------------------------

    def _lights(self, states: list[dict]) -> list[Light]:
        out = []
        for s in states:
            if s.get("entity_id", "").startswith("light."):
                a = s.get("attributes") or {}
                out.append(Light(s["entity_id"], a.get("friendly_name") or s["entity_id"],
                                 bool(a.get("is_hue_group")), s.get("state", "")))
        return out

    def match_lights(self, target: str, lights: list[Light]) -> tuple[list[Light], bool]:
        """(matches, everything). Hue rooms first; a bare word such as
        'den' means every room with that word in it."""
        t = norm_light(target)
        groups = [lt for lt in lights if lt.group]
        if t in ALL_LIGHTS or t.startswith("all "):
            return (groups or lights), True
        exact = [lt for lt in groups if norm_light(lt.name) == t] or \
                [lt for lt in lights if norm_light(lt.name) == t]
        if exact:
            return exact[:1] if len(exact) > 1 and exact[0].group else exact, False
        words = set(t.split())
        return [lt for lt in groups if words <= set(norm_light(lt.name).split())], False

    def lights(self, target: str | None, action: str, brightness: int | None = None) -> str:
        action = (action or "").lower().strip()
        if action not in ("on", "off", "dim", "brighten", "set"):
            raise HomeError(f"I can turn lights on or off, dim or brighten them, not {action}.")
        if not target or not norm_light(target):
            raise HomeError("Which lights?")
        if action == "set" and brightness is None:
            raise HomeError("What brightness?")
        if brightness is not None:
            try:
                brightness = round(float(str(brightness).strip().rstrip("%")))
            except ValueError:
                raise HomeError(f"I need a brightness from 0 to 100, not {brightness}.") from None
            if brightness <= 0:
                # Models map "dark"/"dim all the way" to 0; HA would clamp to 1%.
                action, brightness = "off", None
            else:
                brightness = min(100, brightness)
        if action == "dim" and brightness is None:
            brightness = 30
        if action == "brighten":
            brightness = 100

        matches, everything = self.match_lights(target, self._lights(self.client.states()))
        if not matches:
            return self._lights_via_alexa(target, action, brightness)
        if all(m.state in UNAVAILABLE for m in matches):
            names = join_names([m.name for m in matches])
            return f"{_capital(names)} isn't responding right now."

        ids = [m.entity_id for m in matches]
        if action == "off":
            self.client.call_service("light", "turn_off", {"entity_id": ids})
            what = "off"
        elif brightness is None:
            self.client.call_service("light", "turn_on", {"entity_id": ids})
            what = "on"
        else:
            self.client.call_service("light", "turn_on",
                                     {"entity_id": ids, "brightness_pct": brightness})
            what = f"at {brightness} percent"
        if everything:
            return f"All the lights {what}."
        return f"{_capital(join_names([m.name for m in matches]))} {what}."

    def _lights_via_alexa(self, target: str, action: str, brightness: int | None) -> str:
        name = safe_phrase(norm_light(target), 5)
        if action in ("on", "off"):
            command = f"turn {action} the {name}"
        elif action == "dim" and brightness == 30:
            command = f"dim the {name}"
        else:
            command = f"set the {name} to {brightness} percent"
        echo = self.echo(None)
        self._alexa(echo, command)
        return f"I don't control the {name} directly, so I've asked Alexa to {command}."

    # -- echos --------------------------------------------------------------

    def echos(self) -> list[Echo]:
        states = {(s.get("attributes") or {}).get("friendly_name", "").lower(): s.get("state")
                  for s in self.client.states() if s.get("entity_id", "").startswith("media_player.")}
        out = []
        for d in self.client.devices():
            if d.get("disabled_by"):
                continue
            if not any(i and i[0] == "alexa_devices" for i in d.get("identifiers") or []):
                continue
            name = d.get("name_by_user") or d.get("name") or ""
            group = (d.get("model") or "").lower() == "speaker group"
            out.append(Echo(name, d["id"], group, states.get(name.lower()) not in (None, *UNAVAILABLE)))
        return out

    def echo(self, where: str | None, echos: list[Echo] | None = None) -> Echo:
        echos = self.echos() if echos is None else echos
        w = norm(where or "") or norm(self.default_echo)
        w = re.sub(r"\s*\b(?:echo|alexa)$", "", w).strip() or w
        if w in EVERYWHERE:
            groups = [e for e in echos if e.group]
            named = [e for e in groups if norm(e.name) == "everywhere"]
            if named or groups:
                return (named or groups)[0]
            raise HomeError("There's no Everywhere speaker group set up in Alexa.")
        exact = [e for e in echos if norm(e.name) == w]
        words = set(w.split())
        loose = exact or [e for e in echos if words <= set(norm(e.name).split())]
        live = [e for e in loose if e.available] or loose
        if len(live) == 1:
            return live[0]
        if not live:
            raise HomeError(f"I don't know an Echo called {where or self.default_echo}.")
        raise HomeError(f"Which one: {join_names([e.name for e in live])}?")

    def _alexa(self, echo: Echo, command: str) -> None:
        self.client.call_service("alexa_devices", "send_text_command",
                                 {"device_id": echo.device_id, "text_command": command})

    # -- announcements -------------------------------------------------------

    def announce(self, message: str, where: str | None = "everywhere") -> str:
        text = strip_wake_words(message or "", self.wake_words)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise HomeError("What should I announce?")
        if len(text.split()) > 40:
            raise HomeError("That's too long for an announcement.")
        echo = self.echo(where or "everywhere")
        entity = self._notify_entity(echo, "announce")
        # Set first: the Echo can start speaking before the service call returns.
        self.quiet.mark(announcement_seconds(text))
        self.client.call_service("notify", "send_message", {"entity_id": entity, "message": text})
        return "Announced." if echo.group else f"Announced on the {echo.name}."

    def _notify_entity(self, echo: Echo, kind: str) -> str:
        want = norm(f"{echo.name} {kind}")
        for s in self.client.states():
            eid = s.get("entity_id", "")
            if eid.startswith("notify.") and norm((s.get("attributes") or {}).get("friendly_name", "")) == want:
                return eid
        raise HomeError(f"The {echo.name} can't take announcements.")

    # -- music --------------------------------------------------------------

    def music(self, action: str, request: str | None = None, where: str | None = None) -> str:
        action = (action or "play").lower().strip()
        verbs = {"stop": "stop", "pause": "pause", "resume": "resume", "next": "next",
                 "skip": "next"}
        echos = self.echos()
        echo = self.echo(where, echos)
        if action in verbs:
            sender = self.echo(None, echos) if echo.group else echo
            self._alexa(sender, verbs[action])
            return {"stop": "Stopped.", "pause": "Paused.", "resume": "Resuming.",
                    "next": "Skipping."}[verbs[action]]
        if action != "play":
            raise HomeError(f"I can play, pause, resume, skip or stop music, not {action}.")
        what = safe_phrase(re.sub(r"^play\s+", "", (request or "").strip(), flags=re.I), 12)
        if echo.group:
            self._alexa(self.echo(None, echos), f"play {what} on {echo.name}")
            return f"Playing {what} {norm(echo.name)}." if norm(echo.name) == "everywhere" \
                else f"Playing {what} on {echo.name}."
        self._alexa(echo, f"play {what}")
        return f"Playing {what} on the {echo.name}."

    # -- timers -------------------------------------------------------------

    def timer_set(self, seconds: float, label: str | None = None, where: str | None = None) -> str:
        try:
            seconds = int(round(float(seconds)))
        except (TypeError, ValueError) as exc:
            raise HomeError("How long should the timer be?") from exc
        if seconds < 1:
            raise HomeError("How long should the timer be?")
        if seconds > 24 * 3600:
            raise HomeError("Timers can't be longer than a day.")
        name = safe_phrase(re.sub(r"\btimer$", "", label).strip(), 3) if label and label.strip() else ""
        echo = self.echo(where)
        duration = format_duration(seconds)
        article = "an" if name[:1] in "aeiou" and name else "a"
        self._alexa(echo, f"set {article} {name} timer for {duration}" if name
                    else f"set a timer for {duration}")
        self.ledger.add(name, seconds, echo.name)
        return f"{_capital(name)} timer set for {duration}." if name else f"Timer set for {duration}."

    def timer_cancel(self, label: str | None = None, where: str | None = None) -> str:
        active = self.ledger.active()
        name = norm(re.sub(r"\btimers?$", "", label or "", flags=re.I))
        if name in ("all", "all the", "every", "all my", "all of the", "all of my"):
            name = "all"
        if name and name != "all":
            hits = [e for e in active if e.label == name]
            echo = self.echo(where or (hits[0].echo if hits else None))
            self._settle_timers(echo, active)
            self._alexa(echo, f"cancel the {safe_phrase(name, 3)} timer")
            self.ledger.remove(hits)
            return f"{_capital(name)} timer cancelled."
        if name != "all" and len(active) > 1:
            labels = [e.label or "unnamed" for e in active]
            raise HomeError(f"There are {len(active)} timers: {join_names(labels)}. Which one?")
        echo = self.echo(where or (active[0].echo if active else None))
        self._settle_timers(echo, active)
        if len(active) == 1 and active[0].label and name != "all":
            self._alexa(echo, f"cancel the {active[0].label} timer")
            self.ledger.remove(active)
            return f"{_capital(active[0].label)} timer cancelled."
        self._alexa(echo, "cancel all timers")
        self.ledger.remove(active)
        return "All timers cancelled." if name == "all" or len(active) != 1 else "Timer cancelled."

    def _settle_timers(self, echo: Echo, active: list[TimerEntry]) -> None:
        """Alexa silently ignores a cancel sent seconds after the set.

        Seen live 2026-10-09: set, then "cancel the test timer" 3 s later;
        HA accepted both and the timer still rang (twice). ~90 s later works.
        """
        recent = [e.set_at for e in active if e.echo == echo.name]
        if recent:
            wait = max(recent) + TIMER_CANCEL_MIN_GAP_S - self._clock()
            if wait > 0:
                log.info("waiting %.0fs before cancelling: Alexa drops early cancels", wait)
                self._sleep(wait)

    def timer_status(self, label: str | None = None) -> str:
        now = self._clock()
        states = self.client.states()
        sensor_ends = self._sensor_timer_ends(states, now)
        entries = self._reconcile(self.ledger.active(), sensor_ends, now)
        timers = [(e.label, e.ends_at) for e in entries]
        for echo_name, ends in sensor_ends:
            if not any(abs(ends - e.ends_at) < 120 for e in entries):
                timers.append(("", ends))
        name = norm(re.sub(r"\btimers?$", "", label or "", flags=re.I))
        if name:
            timers = [t for t in timers if t[0] == name]
            if not timers:
                return f"There's no {name} timer running."
        if not timers:
            return "There are no timers running."
        timers.sort(key=lambda t: t[1])
        if len(timers) == 1:
            lbl, ends = timers[0]
            what = f"The {lbl} timer" if lbl else "The timer"
            return f"{what} has {format_remaining(ends - now)} left."
        parts = [f"{_capital(lbl) if lbl else 'An unnamed one'}, {format_remaining(ends - now)}"
                 for lbl, ends in timers[:4]]
        return f"{len(timers)} timers. " + ". ".join(parts) + "."

    def _sensor_timer_ends(self, states: list[dict], now: float) -> list[tuple[str, float]]:
        out = []
        for s in states:
            eid = s.get("entity_id", "")
            if not (eid.startswith("sensor.") and eid.endswith("_next_timer")):
                continue
            try:
                ends = datetime.fromisoformat(s.get("state", "")).timestamp()
            except (TypeError, ValueError):
                continue
            if ends > now:
                name = (s.get("attributes") or {}).get("friendly_name", "")
                out.append((re.sub(r"\s*next timer$", "", name, flags=re.I), ends))
        return out

    def _reconcile(self, entries: list[TimerEntry], sensor_ends: list[tuple[str, float]],
                   now: float) -> list[TimerEntry]:
        """Drop ledger timers that were cancelled on the Echo itself.

        The sensor shows each Echo's *next* timer, ~90 s late. So only the
        earliest ledger timer per Echo can be checked, and only once it is
        old enough for the sensor to have caught up.
        """
        gone = []
        for echo_name in {e.echo for e in entries}:
            mine = sorted((e for e in entries if e.echo == echo_name), key=lambda e: e.ends_at)
            first = mine[0]
            if now - first.set_at < 150:
                continue
            seen = [ends for n, ends in sensor_ends if norm(n) == norm(echo_name)]
            if not any(abs(ends - first.ends_at) < 120 for ends in seen):
                gone.append(first)
        if gone:
            log.info("timers no longer on the Echo, dropping: %s", [g.label for g in gone])
            self.ledger.remove(gone)
        return [e for e in entries if e not in gone]


# -- wiring -----------------------------------------------------------------

def build_home(env_path: Path | None = None, *, default_echo: str = "kitchen",
               wake_words: Iterable[str] = ("jarvis",)) -> Home | None:
    """A Home if HA credentials are set up, else None. Never touches the
    network: HA may be restarting when Jarvis starts."""
    from .ha_client import HAClient, HAError, load_settings  # noqa: PLC0415

    try:
        settings = load_settings(env_path)
    except HAError as exc:
        log.info("home control off: %s", exc)
        return None
    return Home(HAClient(settings), default_echo=default_echo, wake_words=wake_words)


def run_intent(home: Home, kind: str, args: dict) -> str:
    """Carry out a recognised home intent; returns the sentence to speak."""
    if kind == "lights":
        return home.lights(args.get("target"), args.get("action", ""), args.get("brightness"))
    if kind == "timer_set":
        return home.timer_set(args.get("seconds"), args.get("label"), args.get("where"))
    if kind == "timer_cancel":
        return home.timer_cancel(args.get("label"), args.get("where"))
    if kind == "timer_status":
        return home.timer_status(args.get("label"))
    if kind == "announce":
        return home.announce(args.get("message", ""), args.get("where") or "everywhere")
    if kind == "music":
        return home.music(args.get("action", "play"), args.get("request"), args.get("where"))
    raise HomeError(f"I don't know how to {kind}.")
