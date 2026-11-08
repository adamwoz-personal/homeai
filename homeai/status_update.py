"""'Hey Jarvis, status update': what's new since you last asked (asked for by Adam 2026-11-07).

For an admin (HOMEAI_STATUS_ADMINS, default Adam):
- voice profiles created since their last update (first time: the last week),
- doorbell rings and motion since their last update, at most the last 24 hours,
- a short weather summary.
Anyone else gets the weather. Nothing goes to the agent: the sentences are
built here, so the reply is short and can't wander off into a lookup.

"Since your last update" is a small JSON ledger per person, moved on only
after a complete update, so a doorbell lookup that failed is reported again.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from homeai.fsutil import atomic_write_json

log = logging.getLogger("homeai.status_update")

LEDGER = Path.home() / ".local/share/homeai/status_update.json"
DAY_S = 24 * 3600
FIRST_LOOK_S = 7 * DAY_S
TRIGGER = re.compile(r"^(?:(?:give|get) me |can i (?:get|have) |let'?s (?:have|get) |i'?d like |what'?s )?"
                     r"(?:a |an |the |my )?status(?: update| report)?$|^what did i miss$|^catch me up$")

MSG_NO_NEWS = "No new voices and no doorbell alerts since your last update."
MSG_DOOR_FAILED = "I couldn't check the doorbell."
MSG_WEATHER_FAILED = "I couldn't get the weather."


def is_request(text: str) -> bool:
    said = re.sub(r"^(?:hey jarvis|jarvis)[, ]+", "", text.lower().strip())
    said = re.sub(r"^(?:please|can you|could you)[, ]+|[, ]+please$", "", said)
    return bool(TRIGGER.match(re.sub(r"[.!?]+$", "", said).strip()))


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _clock_time(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%I:%M %p").lstrip("0").replace("AM", "am").replace("PM", "pm")


def _count(n: int, one: str, many: str) -> str:
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    return f"{words.get(n, n)} {one if n == 1 else many}"


def voices_sentence(names: list[str]) -> str:
    return f"New voice{'s' if len(names) > 1 else ''} since your last update: {_and(names)}."


def doors_sentence(alerts: dict[tuple[str, str], list[float]]) -> str:
    """'At the front door: two rings and five motion alerts, the latest at 3:40 pm.'"""
    parts = []
    for door in sorted({door for door, _ in alerts}):
        rings, motion = alerts.get((door, "ring"), []), alerts.get((door, "motion"), [])
        what = [x for x in (rings and _count(len(rings), "ring", "rings"),
                            motion and _count(len(motion), "motion alert", "motion alerts")) if x]
        latest = max(rings + motion)
        parts.append(f"At the {door}: {_and(what)}, the latest at {_clock_time(latest)}.")
    return " ".join(parts)


class Ledger:
    """When each person last had a status update. A broken file starts fresh, never stops Jarvis."""

    def __init__(self, path: Path = LEDGER) -> None:
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def last(self, person: str) -> float | None:
        value = self._read().get(person.lower())
        return float(value) if isinstance(value, (int, float)) else None

    def mark(self, person: str, when: float) -> None:
        data = self._read()
        data[person.lower()] = when
        try:
            atomic_write_json(self.path, data)
        except OSError as exc:
            log.warning("could not save the status ledger: %s", exc)


class StatusUpdate:
    def __init__(self, profiles: Callable[[], list], door_alerts: Callable[[float, float], dict] | None,
                 weather: Callable[[], str], admins: tuple[str, ...] = ("Adam",), ledger: Ledger | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._profiles, self._doors, self._weather = profiles, door_alerts, weather
        self.admins = tuple(a.lower() for a in admins)
        self.ledger, self._clock = ledger or Ledger(), clock

    def handle(self, text: str, speaker: str | None) -> str | None:
        """The update to speak, or None if ``text`` isn't asking for one."""
        if not is_request(text):
            return None
        return self.compose(speaker)

    def compose(self, speaker: str | None) -> str:
        now = self._clock()
        parts: list[str] = []
        complete = True
        admin = bool(speaker) and speaker.lower() in self.admins
        if admin:
            last = self.ledger.last(speaker)
            since_voices = last if last is not None else now - FIRST_LOOK_S
            since_doors = max(last if last is not None else 0.0, now - DAY_S)
            names = [p.name for p in self._profiles()
                     if p.created_at > since_voices and p.name.lower() != speaker.lower()]
            news = [voices_sentence(sorted(names))] if names else []
            if self._doors is not None:
                try:
                    alerts = self._doors(since_doors, now)
                    if alerts:
                        news.append(doors_sentence(alerts))
                except Exception as exc:  # noqa: BLE001 - HA down: say so, report it again next time
                    log.warning("status update: doorbell history failed: %s", exc)
                    news.append(MSG_DOOR_FAILED)
                    complete = False
            parts += news or [MSG_NO_NEWS]
        try:
            parts.append(self._weather())
        except Exception as exc:  # noqa: BLE001 - weather service down
            log.warning("status update: weather failed: %s", exc)
            parts.append(MSG_WEATHER_FAILED)
        if admin and complete:
            self.ledger.mark(speaker, now)
        return " ".join(p for p in parts if p)


def build(home, admins: tuple[str, ...], profiles: Callable[[], list], location: str) -> StatusUpdate:
    """The real one: Home Assistant history for the doorbell, Open-Meteo for the weather."""
    from .doorbell import door_alerts
    from .weather import get_weather

    doors = (lambda since, until: door_alerts(home, since, until)) if home is not None else None
    return StatusUpdate(profiles, doors, lambda: get_weather(location).spoken(include_place=False), admins=admins)
