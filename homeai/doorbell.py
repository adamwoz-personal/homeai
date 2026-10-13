# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Ring doorbells through Home Assistant: announce rings, answer "who's at the
door" and "how's the doorbell battery".

Home Assistant's Ring integration exposes per-device entities:
  event.<name>_ding / event.<name>_motion   state = timestamp of the last event
  sensor.<name>_last_activity               persists across restarts (category)
  sensor.<name>_battery
The watcher polls the two event entities every couple of seconds; the first
poll only records a baseline, so a restart never replays an old ring. Polling
(rather than a websocket subscription) keeps it trivially recoverable: any
failure is logged once and retried on the next tick.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import Callable

from .ha_client import HAError
from .home import Home, HomeError, format_duration, norm

log = logging.getLogger("homeai.doorbell")

DAY_S = 24 * 3600


def parse_quiet_hours(spec: str | None) -> tuple[dtime, dtime] | None:
    """"22:00-07:00" -> (22:00, 07:00); empty or malformed -> None (no quiet hours)."""
    try:
        start, end = (dtime.fromisoformat(part.strip()) for part in str(spec).split("-"))
    except (TypeError, ValueError):
        return None
    return None if start == end else (start, end)


def in_quiet_hours(window: tuple[dtime, dtime] | None, now: dtime) -> bool:
    if window is None:
        return False
    start, end = window
    return start <= now < end if start < end else now >= start or now < end


@dataclass(frozen=True)
class Door:
    name: str                 # "front door"
    ding: str | None = None
    motion: str | None = None
    activity: str | None = None
    battery: str | None = None


@dataclass(frozen=True)
class DoorEvent:
    door: str
    kind: str                 # ding | motion


def parse_time(value: str | None) -> float | None:
    """Epoch seconds from an HA timestamp state, or None (unknown/unavailable)."""
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def discover(states: list[dict]) -> list[Door]:
    """Group Ring entities into doors. Ring events are told apart by their
    device_class (doorbell / motion), not by guessing from entity names."""
    by_name: dict[str, dict[str, str]] = {}
    friendly = {s.get("entity_id", ""): ((s.get("attributes") or {}).get("friendly_name") or "")
                for s in states}
    for s in states:
        eid = s.get("entity_id", "")
        attrs = s.get("attributes") or {}
        if "Ring.com" not in str(attrs.get("attribution", "")):
            continue
        if eid.startswith("event."):
            kind = {"doorbell": "ding", "motion": "motion"}.get(attrs.get("device_class"))
            suffix = {"ding": " Ding", "motion": " Motion"}.get(kind or "")
        elif eid.startswith("sensor.") and attrs.get("device_class") == "battery":
            kind, suffix = "battery", " Battery"
        elif eid.startswith("sensor.") and eid.endswith("_last_activity"):
            kind, suffix = "activity", " Last activity"
        else:
            continue
        if not kind:
            continue
        label = friendly[eid]
        name = label[: -len(suffix)] if suffix and label.endswith(suffix) else label
        by_name.setdefault(norm(name), {})[kind] = eid
    return [Door(n, **parts) for n, parts in sorted(by_name.items())
            if parts.get("ding") or parts.get("motion")]


def ago(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    if seconds >= DAY_S:
        return "more than a day ago"
    return f"{format_duration(seconds // 60 * 60)} ago"


def _state(states: dict[str, dict], eid: str | None) -> str | None:
    return (states.get(eid) or {}).get("state") if eid else None


def _pick(doors: list[Door], where: str | None) -> list[Door]:
    if not where:
        return doors
    want = norm(where)
    return [d for d in doors if want in d.name or d.name in want]


def door_status(home: Home, where: str | None = None) -> str:
    """The latest ring or motion, in a sentence."""
    states = {s["entity_id"]: s for s in home.client.states()}
    doors = _pick(discover(list(states.values())), where)
    if not doors:
        raise HomeError("I don't see a doorbell." if not where else f"I don't see a {where}.")
    now = home._clock()
    parts = []
    for door in doors:
        latest: tuple[float, str] | None = None
        for kind, eid in (("ring", door.ding), ("motion", door.motion)):
            t = parse_time(_state(states, eid))
            if t is not None and (latest is None or t > latest[0]):
                latest = (t, kind)
        # The activity sensor outlives restarts of Home Assistant.
        act = states.get(door.activity or "") or {}
        t = parse_time(act.get("state"))
        if t is not None and (latest is None or t > latest[0] + 5):
            category = (act.get("attributes") or {}).get("category")
            latest = (t, "ring" if category in ("ding", "ring") else "motion")
        if latest is None:
            parts.append(f"No activity at the {door.name} that I know of.")
            continue
        age = max(0.0, now - latest[0])
        what = "Someone rang" if latest[1] == "ring" else "Motion at"
        parts.append(f"{what} the {door.name}, {ago(age)}." if age >= 60
                     else f"{what} the {door.name} just now.")
    return " ".join(parts)


def door_battery(home: Home, where: str | None = None) -> str:
    states = {s["entity_id"]: s for s in home.client.states()}
    doors = [d for d in _pick(discover(list(states.values())), where) if d.battery]
    if not doors:
        raise HomeError("I can't see a doorbell battery.")
    out = []
    for door in doors:
        value = _state(states, door.battery)
        try:
            out.append(f"The {door.name} battery is at {int(float(value))} percent.")
        except (TypeError, ValueError):
            out.append(f"The {door.name} battery level isn't available.")
    return " ".join(out)


class DoorbellWatcher:
    """Announces rings (and optionally motion) on the Echos."""

    def __init__(self, home: Home, *, announce_motion: bool = False, where: str = "everywhere",
                 poll_s: float = 2.0, phone: bool = True, quiet_hours: tuple[dtime, dtime] | None = None,
                 local_time: Callable[[], dtime] = lambda: datetime.now().time()) -> None:
        self.home = home
        self.quiet_hours = quiet_hours
        self.phone = phone
        self._local_time = local_time
        self.announce_motion = announce_motion
        self.where = where
        self.poll_s = poll_s
        self._seen: dict[str, str | None] | None = None
        self._failing = False

    def poll(self) -> list[DoorEvent]:
        """Events since the last poll. The first call only takes a baseline."""
        states = {s["entity_id"]: s for s in self.home.client.states()}
        current: dict[str, str | None] = {}
        watch: dict[str, DoorEvent] = {}
        for door in discover(list(states.values())):
            for kind, eid in (("ding", door.ding), ("motion", door.motion)):
                if eid:
                    current[eid] = _state(states, eid)
                    watch[eid] = DoorEvent(door.name, kind)
        previous, self._seen = self._seen, current
        if previous is None:
            return []
        return [watch[eid] for eid, state in current.items()
                if eid in previous and state != previous[eid]
                and parse_time(state) is not None]

    def announce(self, event: DoorEvent) -> None:
        if event.kind == "motion" and not self.announce_motion:
            return
        if self.phone:  # unaffected by quiet hours: the phone is how you hear at night
            sent = self.home.notify_phones(
                f"Someone is at the {event.door}." if event.kind == "ding"
                else f"Motion at the {event.door}.", "Front door" if event.kind == "ding" else "Motion")
            log.info("doorbell: notified %d phone(s)", sent)
        if in_quiet_hours(self.quiet_hours, self._local_time()):
            log.info("doorbell: %s at the %s during quiet hours; not announcing",
                     event.kind, event.door)
            return
        text = (f"Someone is at the {event.door}." if event.kind == "ding"
                else f"Motion at the {event.door}.")
        try:
            self.home.announce(text, self.where)
        except (HomeError, HAError) as exc:
            log.warning("could not announce %s at the %s: %s", event.kind, event.door, exc)

    def tick(self) -> None:
        try:
            events = self.poll()
        except HAError as exc:
            if not self._failing:
                log.warning("doorbell watch: %s (will keep trying)", exc)
            self._failing = True
            return
        except Exception:  # noqa: BLE001 - the watcher must never die
            log.exception("doorbell watch crashed; continuing")
            return
        if self._failing:
            log.info("doorbell watch: house controller is back")
            self._failing = False
        for event in events:
            log.info("doorbell: %s at the %s", event.kind, event.door)
            self.announce(event)

    def run(self, stop: threading.Event, wait: Callable[[float], bool] | None = None) -> None:
        wait = wait or stop.wait
        log.info("doorbell watch: on (motion announcements %s)",
                 "on" if self.announce_motion else "off")
        while not stop.is_set():
            self.tick()
            wait(self.poll_s)

