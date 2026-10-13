# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Ring doorbell: discovery, "who's at the door", battery, ring announcements."""

from __future__ import annotations

import threading
from datetime import datetime, time as dtime, timezone

import pytest

from homeai.ha_client import HAUnavailable
from homeai.home import Home, HomeError, run_intent
from homeai.home_intents import parse
from homeai import doorbell
from homeai.doorbell import DoorbellWatcher, discover, door_battery, door_status
from tests.fake_ha import FakeHA, default_states, ring_entities

NOW = datetime(2026, 10, 13, 20, 0, 0, tzinfo=timezone.utc).timestamp()


def stamp(seconds_ago: float) -> str:
    return datetime.fromtimestamp(NOW - seconds_ago, timezone.utc).isoformat()


@pytest.fixture
def ha():
    return FakeHA(states=default_states() + ring_entities())


@pytest.fixture
def home(ha, tmp_path):
    from homeai.home import EchoQuiet, TimerLedger
    return Home(ha, default_echo="kitchen", wake_words=("jarvis",),
                ledger=TimerLedger(tmp_path / "t.json", clock=lambda: NOW),
                quiet=EchoQuiet(tmp_path / "q", clock=lambda: NOW), clock=lambda: NOW,
                sleep=lambda s: None)


def announcements(ha):
    return [d["message"] for dom, svc, d in ha.calls if (dom, svc) == ("notify", "send_message")]


# -- discovery ----------------------------------------------------------

def test_discover_finds_the_front_door():
    (door,) = discover(default_states() + ring_entities(activity=stamp(5)))
    assert door.name == "front door"
    assert door.ding == "event.front_door_ding" and door.motion == "event.front_door_motion"
    assert door.battery == "sensor.front_door_battery"
    assert door.activity == "sensor.front_door_last_activity"


def test_discover_ignores_non_ring_and_handles_two_doors():
    doors = discover(default_states() + ring_entities() + ring_entities("Back Door"))
    assert [d.name for d in doors] == ["back door", "front door"]
    assert discover(default_states()) == []


# -- status answers -----------------------------------------------------

def test_status_recent_ring(ha, home):
    ha.set_state("event.front_door_ding", stamp(300))
    assert door_status(home) == "Someone rang the front door, 5 minutes ago."


def test_status_latest_of_ring_and_motion(ha, home):
    ha.set_state("event.front_door_ding", stamp(3600))
    ha.set_state("event.front_door_motion", stamp(120))
    assert door_status(home) == "Motion at the front door, 2 minutes ago."


def test_status_uses_last_activity_after_restart():
    ha = FakeHA(states=default_states() + ring_entities(activity=stamp(600), category="ding"))
    home = Home(ha, clock=lambda: NOW)
    assert door_status(home) == "Someone rang the front door, 10 minutes ago."


def test_status_just_now_and_old(ha, home):
    ha.set_state("event.front_door_ding", stamp(10))
    assert door_status(home) == "Someone rang the front door just now."
    ha.set_state("event.front_door_ding", stamp(3 * 86400))
    assert "more than a day ago" in door_status(home)


def test_status_nothing_known(home):
    assert "No activity at the front door" in door_status(home)


def test_no_doorbell_installed():
    home = Home(FakeHA(), clock=lambda: NOW)
    with pytest.raises(HomeError, match="don't see a doorbell"):
        door_status(home)


def test_battery(home):
    assert door_battery(home) == "The front door battery is at 39 percent."


def test_battery_unavailable(ha, home):
    ha.set_state("sensor.front_door_battery", "unavailable")
    assert "isn't available" in door_battery(home)


def test_run_intent_routes(ha, home):
    ha.set_state("event.front_door_ding", stamp(300))
    assert run_intent(home, "door", {}).startswith("Someone rang")
    assert "39 percent" in run_intent(home, "door_battery", {})


@pytest.mark.parametrize("text,kind", [
    ("who's at the door", "door"), ("Who is at the front door?", "door"),
    ("did anyone ring the doorbell", "door"), ("did the doorbell ring", "door"),
    ("has someone come by", "door"), ("is there someone at the door", "door"),
    ("how's the doorbell battery", "door_battery"),
    ("what's the doorbell battery level", "door_battery")])
def test_parse_door_questions(text, kind):
    assert parse(text).kind == kind


@pytest.mark.parametrize("text", ["turn off the door light", "open the door", "who is the president",
                                  "ring the bell on my phone"])
def test_parse_leaves_other_things_alone(text):
    assert parse(text) is None or parse(text).kind not in ("door", "door_battery")


# -- watcher ------------------------------------------------------------

def test_first_poll_is_a_baseline_not_a_ring(ha, home):
    ha.set_state("event.front_door_ding", stamp(30))
    watcher = DoorbellWatcher(home)
    assert watcher.poll() == []
    watcher.tick()
    assert announcements(ha) == []


def test_ring_is_announced_once(ha, home):
    watcher = DoorbellWatcher(home)
    watcher.tick()
    ha.set_state("event.front_door_ding", stamp(1))
    watcher.tick()
    watcher.tick()
    assert announcements(ha) == ["Someone is at the front door."]


def test_motion_is_off_by_default_and_optional(ha, home):
    watcher = DoorbellWatcher(home)
    watcher.tick()
    ha.set_state("event.front_door_motion", stamp(1))
    watcher.tick()
    assert announcements(ha) == []
    loud = DoorbellWatcher(home, announce_motion=True)
    ha.set_state("event.front_door_motion", stamp(100))
    loud.tick()
    ha.set_state("event.front_door_motion", stamp(1))
    loud.tick()
    assert announcements(ha) == ["Motion at the front door."]


def test_unavailable_transitions_are_not_rings(ha, home):
    watcher = DoorbellWatcher(home)
    watcher.tick()
    ha.set_state("event.front_door_ding", "unavailable")
    watcher.tick()
    assert announcements(ha) == []


def test_where_is_configurable(ha, home):
    watcher = DoorbellWatcher(home, where="kitchen")
    watcher.tick()
    ha.set_state("event.front_door_ding", stamp(1))
    watcher.tick()
    assert [d["entity_id"] for _, svc, d in ha.calls if svc == "send_message"] == \
        ["notify.kitchen_alexa_announce"]


def test_announcement_marks_echo_quiet_so_jarvis_does_not_wake(ha, home):
    watcher = DoorbellWatcher(home)
    watcher.tick()
    ha.set_state("event.front_door_ding", stamp(1))
    watcher.tick()
    assert home.quiet.remaining() > 0


@pytest.mark.parametrize("spec,start,end", [("22:00-07:00", dtime(22), dtime(7)),
                                            (" 09:30 - 10:00 ", dtime(9, 30), dtime(10))])
def test_quiet_hours_parse(spec, start, end):
    assert doorbell.parse_quiet_hours(spec) == (start, end)


@pytest.mark.parametrize("spec", ["", None, "late", "10:00-10:00", "25:00-26:00"])
def test_quiet_hours_garbage_means_none(spec):
    assert doorbell.parse_quiet_hours(spec) is None


@pytest.mark.parametrize("now,quiet", [(dtime(23), True), (dtime(3), True), (dtime(7), False),
                                       (dtime(12), False), (dtime(22), True)])
def test_overnight_window(now, quiet):
    assert doorbell.in_quiet_hours((dtime(22), dtime(7)), now) is quiet


def test_same_day_window():
    window = (dtime(13), dtime(14))
    assert doorbell.in_quiet_hours(window, dtime(13, 30))
    assert not doorbell.in_quiet_hours(window, dtime(14))
    assert not doorbell.in_quiet_hours(None, dtime(3))


def test_no_announcement_in_quiet_hours(ha, home):
    watcher = DoorbellWatcher(home, quiet_hours=(dtime(22), dtime(7)), local_time=lambda: dtime(23))
    watcher.tick()
    ha.set_state("event.front_door_ding", stamp(1))
    watcher.tick()
    assert announcements(ha) == []


def test_outage_is_survivable_and_recovers(ha, home):
    watcher = DoorbellWatcher(home)
    watcher.tick()
    ha.fail = HAUnavailable("down")
    watcher.tick()
    watcher.tick()
    ha.fail = None
    ha.set_state("event.front_door_ding", stamp(1))
    watcher.tick()
    assert announcements(ha) == ["Someone is at the front door."]


def test_announce_failure_does_not_kill_the_watcher(ha, home, monkeypatch):
    watcher = DoorbellWatcher(home)
    watcher.tick()
    monkeypatch.setattr(home, "announce", lambda *a: (_ for _ in ()).throw(HomeError("no echo")))
    ha.set_state("event.front_door_ding", stamp(1))
    watcher.tick()


def test_run_loop_stops(ha, home):
    stop = threading.Event()
    waits = []
    watcher = DoorbellWatcher(home, poll_s=0.5)

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            stop.set()
        return stop.is_set()

    watcher.run(stop, wait)
    assert waits == [0.5, 0.5, 0.5]
