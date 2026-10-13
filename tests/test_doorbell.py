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


# -- phone notifications ------------------------------------------------

def phone_calls(ha):
    return [(svc, d) for dom, svc, d in ha.calls if dom == "notify" and svc.startswith("mobile_app_")]


def ring(watcher, ha):
    watcher.tick()
    ha.set_state("event.front_door_ding", stamp(1))
    watcher.tick()


def test_ring_also_notifies_phones(home):
    ha = home.client
    ha.phones = ["adams_pixel", "ani_phone"]
    ring(DoorbellWatcher(home), ha)
    calls = phone_calls(ha)
    assert [c[0] for c in calls] == ["mobile_app_adams_pixel", "mobile_app_ani_phone"]
    assert calls[0][1]["message"] == "Someone is at the front door."
    # only phones got the push; the Echo announcement is the single send_message
    assert [svc for dom, svc, _ in ha.calls if dom == "notify"].count("send_message") == 1


def test_no_phones_is_fine(ha, home):
    ring(DoorbellWatcher(home), ha)
    assert phone_calls(ha) == [] and announcements(ha) == ["Someone is at the front door."]


def test_phone_can_be_turned_off(home):
    home.client.phones = ["pixel"]
    ring(DoorbellWatcher(home, phone=False), home.client)
    assert phone_calls(home.client) == []


def test_phone_still_buzzes_in_quiet_hours(home):
    home.client.phones = ["pixel"]
    watcher = DoorbellWatcher(home, quiet_hours=(dtime(22), dtime(7)), local_time=lambda: dtime(23))
    ring(watcher, home.client)
    assert len(phone_calls(home.client)) == 1 and announcements(home.client) == []


def test_phone_failure_does_not_block_the_echo(home, monkeypatch):
    home.client.phones = ["pixel"]
    real = home.client.call_service

    def flaky(domain, service, data=None):
        if service.startswith("mobile_app_"):
            raise HAUnavailable("phone offline")
        return real(domain, service, data)

    monkeypatch.setattr(home.client, "call_service", flaky)
    ring(DoorbellWatcher(home), home.client)
    assert announcements(home.client) == ["Someone is at the front door."]


def test_notify_phones_survives_service_listing_failure(home, monkeypatch):
    monkeypatch.setattr(home.client, "services", lambda: (_ for _ in ()).throw(HAUnavailable("x")))
    assert home.notify_phones("hi") == 0


# -- goodnight ----------------------------------------------------------

def test_goodnight_turns_everything_off(ha, home):
    reply = home.goodnight()
    assert reply == "Goodnight. All the lights are off."
    assert any(svc == "turn_off" for _, svc, _ in ha.calls)


def test_goodnight_when_house_unreachable_is_still_polite(home):
    home.client.fail = None
    home.client._states = [s for s in home.client._states if not s["entity_id"].startswith("light.")]
    assert home.goodnight().startswith("Goodnight.")


@pytest.mark.parametrize("text", ["goodnight routine", "run the bedtime routine", "I'm going to bed",
                                  "time for bed", "heading to bed"])
def test_goodnight_phrases(text):
    assert parse(text).kind == "goodnight"


@pytest.mark.parametrize("text", ["good night", "goodnight", "what time is bedtime", "bed"])
def test_bare_goodnight_stays_a_farewell(text):
    assert parse(text) is None
