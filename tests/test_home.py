# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Home control against a fake HA shaped like the real house."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from homeai.ha_client import HAUnavailable
from homeai.home import (
    TIMER_CANCEL_MIN_GAP_S,
    EchoQuiet,
    Home,
    HomeError,
    TimerLedger,
    announcement_seconds,
    format_duration,
    format_remaining,
    norm_light,
    safe_phrase,
    strip_wake_words,
)
from tests.fake_ha import FakeHA


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds

    @property
    def slept(self):
        return self.__dict__.setdefault("_slept", [])


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ha():
    return FakeHA()


@pytest.fixture
def home(ha, tmp_path, clock):
    return Home(ha, default_echo="kitchen", wake_words=("jarvis",),
                ledger=TimerLedger(tmp_path / "timers.json", clock=clock),
                quiet=EchoQuiet(tmp_path / "quiet", clock=clock), clock=clock,
                sleep=clock.sleep)


# -- helpers --------------------------------------------------------------

def test_format_duration():
    assert format_duration(600) == "10 minutes"
    assert format_duration(60) == "1 minute"
    assert format_duration(5400) == "1 hour and 30 minutes"
    assert format_duration(3725) == "1 hour, 2 minutes and 5 seconds"
    assert format_duration(0) == "0 seconds"


def test_format_remaining_rounds_long_times():
    assert format_remaining(250) == "4 minutes and 10 seconds"
    assert format_remaining(1850) == "31 minutes"


def test_norm_light():
    assert norm_light("the Foyer lights") == "foyer"
    assert norm_light("Den lamp 2") == "den lamp 2"
    assert norm_light("lights") == "lights"


@pytest.mark.parametrize("bad", ["buy more paper towels", "unlock the front door",
                                 "call mom", "open the garage", "drop in on the kitchen"])
def test_safe_phrase_blocks_dangerous_alexa_commands(bad):
    with pytest.raises(HomeError, match="won't"):
        safe_phrase(bad, 12)


def test_safe_phrase_limits():
    assert safe_phrase("Some JAZZ!", 5) == "some jazz"
    with pytest.raises(HomeError):
        safe_phrase("", 5)
    with pytest.raises(HomeError, match="too long"):
        safe_phrase("one two three four", 3)


def test_strip_wake_words():
    assert strip_wake_words("This is Jarvis. Dinner is ready", ["jarvis"]) == "This is Dinner is ready"
    assert strip_wake_words("Hey Jarvis, dinner", ["jarvis"]) == "dinner"


# -- lights ----------------------------------------------------------------

def test_light_room_by_name(home, ha):
    assert home.lights("the foyer", "off") == "Foyer off."
    assert ha.calls == [("light", "turn_off", {"entity_id": ["light.foyer_foyer"]})]


def test_light_exact_name_beats_word_match(home, ha):
    home.lights("den lamp", "on")
    assert ha.calls[-1][2]["entity_id"] == ["light.den_lamp_den_lamp"]


def test_light_word_matches_every_room_with_it(home, ha):
    reply = home.lights("den", "off")
    assert set(ha.calls[-1][2]["entity_id"]) == {
        "light.den_fan_den_fan", "light.den_lamp_den_lamp", "light.den_lamp_2_den_lamp_2"}
    assert reply == "Den fan, den lamp and Den lamp 2 off."


def test_light_all(home, ha):
    assert home.lights("all", "off") == "All the lights off."
    ids = ha.calls[-1][2]["entity_id"]
    assert "light.foyer_foyer" in ids and "light.foyer_hue_color_lamp_13" not in ids


def test_light_brightness(home, ha):
    assert home.lights("foyer", "set", 40) == "Foyer at 40 percent."
    assert ha.calls[-1] == ("light", "turn_on",
                            {"entity_id": ["light.foyer_foyer"], "brightness_pct": 40})
    home.lights("foyer", "dim")
    assert ha.calls[-1][2]["brightness_pct"] == 30
    home.lights("foyer", "brighten")
    assert ha.calls[-1][2]["brightness_pct"] == 100
    home.lights("foyer", "set", 400)
    assert ha.calls[-1][2]["brightness_pct"] == 100


@pytest.mark.parametrize("value,pct", [("40", 40), ("55%", 55), (" 7 ", 7), (1, 1), (0.6, 1)])
def test_light_brightness_from_model_strings(home, ha, value, pct):
    """The model sends JSON strings ("0", "55%") as often as numbers."""
    home.lights("foyer", "set", value)
    assert ha.calls[-1][2]["brightness_pct"] == pct


@pytest.mark.parametrize("value", [0, "0", -5, "0%"])
def test_light_brightness_zero_means_off(home, ha, value):
    """Seen live: "make the den lamp dark" -> brightness "0", which HA
    clamps to 1% and the model then reports as dark."""
    assert home.lights("foyer", "set", value) == "Foyer off."
    assert ha.calls[-1] == ("light", "turn_off", {"entity_id": ["light.foyer_foyer"]})


def test_light_unavailable(home, ha):
    assert "isn't responding" in home.lights("utility room", "on")
    assert ha.calls == []


def test_light_unknown_falls_back_to_alexa(home, ha):
    reply = home.lights("porch", "on")
    assert ha.alexa_commands() == [("dev-kitchen_alexa", "turn on the porch")]
    assert "asked Alexa" in reply


def test_light_alexa_fallback_is_sanitised(home, ha):
    with pytest.raises(HomeError):
        home.lights("front door lock", "off")
    assert ha.calls == []


@pytest.mark.parametrize("target,action,brightness,msg", [
    (None, "off", None, "Which lights"),
    ("", "on", None, "Which lights"),
    ("foyer", "explode", None, "not explode"),
    ("foyer", "set", None, "What brightness"),
    ("foyer", "set", "bright", "brightness from 0 to 100"),
])
def test_light_bad_requests(home, target, action, brightness, msg):
    with pytest.raises(HomeError, match=msg):
        home.lights(target, action, brightness)


def test_ha_down_propagates(tmp_path, clock):
    home = Home(FakeHA(fail=HAUnavailable("I can't reach the house controller")),
                ledger=TimerLedger(tmp_path / "t.json", clock=clock),
                quiet=EchoQuiet(tmp_path / "q", clock=clock), clock=clock)
    with pytest.raises(HAUnavailable):
        home.lights("foyer", "off")


# -- echos -----------------------------------------------------------------

def test_echo_default_is_kitchen(home):
    assert home.echo(None).name == "Kitchen alexa"


def test_echo_prefers_available(home):
    # "bedroom" matches Adam's bedroom and the (unavailable) farm bedroom.
    assert home.echo("bedroom").name == "Adam's bedroom"


def test_echo_everywhere_is_the_group(home):
    assert home.echo("the whole house").name == "Everywhere"


def test_echo_unknown(home):
    with pytest.raises(HomeError, match="don't know an Echo"):
        home.echo("garage")


def test_echo_ambiguous(ha, tmp_path, clock):
    ha.set_state("media_player.farm_bedroom_alexa", "idle")
    home = Home(ha, ledger=TimerLedger(tmp_path / "t", clock=clock),
                quiet=EchoQuiet(tmp_path / "q", clock=clock), clock=clock)
    with pytest.raises(HomeError, match="Which one"):
        home.echo("bedroom")


def test_echo_ignores_non_alexa_devices(home):
    assert all(e.name != "Hue Bridge" for e in home.echos())


# -- announcements ------------------------------------------------------------

def test_announce_everywhere_and_quiet_window(home, ha, clock):
    assert home.announce("Dinner is ready") == "Announced."
    assert ha.calls == [("notify", "send_message",
                         {"entity_id": "notify.everywhere_announce", "message": "Dinner is ready"})]
    assert home.quiet.remaining() == pytest.approx(announcement_seconds("Dinner is ready"), abs=0.01)


def test_announce_never_says_the_wake_word(home, ha):
    home.announce("This is Jarvis. Testing whole house announcements.")
    assert "jarvis" not in ha.calls[-1][2]["message"].lower()


def test_announce_one_room(home, ha):
    assert home.announce("come up please", where="basement") == "Announced on the basement echo."
    assert ha.calls[-1][2]["entity_id"] == "notify.basement_echo_announce"


def test_announce_empty_or_long(home):
    with pytest.raises(HomeError):
        home.announce("Hey Jarvis")
    with pytest.raises(HomeError, match="too long"):
        home.announce("word " * 41)


def test_quiet_window_expires(home, clock):
    home.quiet.mark(5)
    clock.t += 6
    assert home.quiet.remaining() == 0


def test_quiet_window_missing_or_garbage(tmp_path, clock):
    q = EchoQuiet(tmp_path / "q", clock=clock)
    assert q.remaining() == 0
    (tmp_path / "q").write_text("not a number")
    assert q.remaining() == 0


# -- music -----------------------------------------------------------------

def test_music_play_default_echo(home, ha):
    assert home.music("play", "some jazz") == "Playing some jazz on the Kitchen alexa."
    assert ha.alexa_commands() == [("dev-kitchen_alexa", "play some jazz")]


def test_music_everywhere_goes_through_default_echo(home, ha):
    home.music("play", "jazz", where="everywhere")
    assert ha.alexa_commands() == [("dev-kitchen_alexa", "play jazz on Everywhere")]


def test_music_controls(home, ha):
    assert home.music("stop") == "Stopped."
    assert home.music("skip") == "Skipping."
    assert [c for _, c in ha.alexa_commands()] == ["stop", "next"]


def test_music_refuses_purchases(home, ha):
    with pytest.raises(HomeError):
        home.music("play", "buy the new album")
    assert ha.calls == []


def test_music_unknown_action(home):
    with pytest.raises(HomeError):
        home.music("rewind")


# -- timers ----------------------------------------------------------------

def test_timer_set_uses_adams_phrasing(home, ha):
    assert home.timer_set(600, "pasta") == "Pasta timer set for 10 minutes."
    assert ha.alexa_commands() == [("dev-kitchen_alexa", "set a pasta timer for 10 minutes")]
    assert [e.label for e in home.ledger.active()] == ["pasta"]


def test_timer_set_unnamed(home, ha):
    assert home.timer_set(90) == "Timer set for 1 minute and 30 seconds."
    assert ha.alexa_commands()[-1][1] == "set a timer for 1 minute and 30 seconds"


@pytest.mark.parametrize("seconds", [0, -5, "soon", None, 90000])
def test_timer_set_bad_durations(home, ha, seconds):
    with pytest.raises(HomeError):
        home.timer_set(seconds)
    assert ha.calls == []


def test_timer_label_drops_trailing_word_timer(home, ha):
    home.timer_set(60, "pasta timer")
    assert ha.alexa_commands()[-1][1] == "set a pasta timer for 1 minute"


def _sensor_shows(ha, clock, seconds_from_now):
    ends = datetime.fromtimestamp(clock.t + seconds_from_now, tz=timezone.utc).isoformat()
    ha.set_state("sensor.kitchen_alexa_next_timer", ends)


def test_timer_status_from_ledger(home, ha, clock):
    home.timer_set(600, "pasta")
    _sensor_shows(ha, clock, 600)
    clock.t += 350
    assert home.timer_status() == "The pasta timer has 4 minutes and 10 seconds left."
    assert home.timer_status("pasta") == "The pasta timer has 4 minutes and 10 seconds left."
    assert home.timer_status("soup") == "There's no soup timer running."


def test_timer_status_none(home):
    assert home.timer_status() == "There are no timers running."


def test_timer_expires_from_ledger(home, clock):
    home.timer_set(60, "eggs")
    clock.t += 61
    assert home.timer_status() == "There are no timers running."


def test_timer_status_includes_alexa_only_timers(home, ha, clock):
    ends = datetime.fromtimestamp(clock.t + 300, tz=timezone.utc).isoformat()
    ha.set_state("sensor.kitchen_alexa_next_timer", ends)
    assert home.timer_status() == "The timer has 5 minutes left."


def test_timer_status_several(home, clock):
    home.timer_set(600, "pasta")
    home.timer_set(1200, "bread")
    assert home.timer_status() == "2 timers. Pasta, 10 minutes. Bread, 20 minutes."


def test_timer_cancelled_on_the_echo_is_dropped(home, ha, clock):
    home.timer_set(600, "pasta")
    clock.t += 200  # the sensor has had time to show it; it shows nothing
    assert home.timer_status() == "There are no timers running."
    assert home.ledger.active() == []


def test_timer_seen_on_the_echo_is_kept(home, ha, clock):
    home.timer_set(600, "pasta")
    ends = datetime.fromtimestamp(clock.t + 600, tz=timezone.utc).isoformat()
    ha.set_state("sensor.kitchen_alexa_next_timer", ends)
    clock.t += 200
    assert home.timer_status() == "The pasta timer has 6 minutes and 40 seconds left."


def test_timer_cancel_by_label(home, ha):
    home.timer_set(600, "pasta")
    assert home.timer_cancel("pasta") == "Pasta timer cancelled."
    assert ha.alexa_commands()[-1][1] == "cancel the pasta timer"
    assert home.ledger.active() == []


def test_timer_cancel_right_after_set_waits(home, ha, clock):
    """Alexa ignored "cancel the test timer" sent 3 s after the set (live,
    twice); the timer rang. Jarvis holds the cancel until the gap has passed."""
    home.timer_set(60, "test")
    clock.t += 3
    assert home.timer_cancel("test") == "Test timer cancelled."
    assert clock.slept == [pytest.approx(TIMER_CANCEL_MIN_GAP_S - 3)]
    assert ha.alexa_commands()[-1][1] == "cancel the test timer"


def test_timer_cancel_later_does_not_wait(home, ha, clock):
    home.timer_set(600, "pasta")
    clock.t += TIMER_CANCEL_MIN_GAP_S
    home.timer_cancel(None)
    home.timer_set(600, "bread", "basement")
    clock.t += 1
    home.timer_cancel("all", "kitchen")  # the recent timer is on another Echo
    assert clock.slept == []


def test_timer_cancel_the_only_one(home, ha):
    home.timer_set(600, "pasta")
    assert home.timer_cancel(None) == "Pasta timer cancelled."
    assert ha.alexa_commands()[-1][1] == "cancel the pasta timer"


def test_timer_cancel_ambiguous_asks(home, ha):
    home.timer_set(600, "pasta")
    home.timer_set(900, "bread")
    with pytest.raises(HomeError, match="pasta and bread. Which one"):
        home.timer_cancel(None)


def test_timer_cancel_all(home, ha):
    home.timer_set(600, "pasta")
    home.timer_set(900, "bread")
    assert home.timer_cancel("all") == "All timers cancelled."
    assert ha.alexa_commands()[-1][1] == "cancel all timers"
    assert home.ledger.active() == []


def test_timer_cancel_unknown_label_still_asks_alexa(home, ha):
    # Set on the Echo directly: Jarvis has no record, Alexa does.
    assert home.timer_cancel("soup") == "Soup timer cancelled."
    assert ha.alexa_commands()[-1] == ("dev-kitchen_alexa", "cancel the soup timer")


def test_ledger_survives_corruption(tmp_path, clock):
    path = tmp_path / "timers.json"
    path.write_text("{not json")
    ledger = TimerLedger(path, clock=clock)
    assert ledger.active() == []
    ledger.add("pasta", 60, "Kitchen alexa")
    assert json.loads(path.read_text())[0]["label"] == "pasta"


def test_ledger_shared_between_instances(tmp_path, clock):
    TimerLedger(tmp_path / "t.json", clock=clock).add("pasta", 60, "Kitchen alexa")
    assert [e.label for e in TimerLedger(tmp_path / "t.json", clock=clock).active()] == ["pasta"]


def test_timer_set_uses_an_before_vowels(home, ha):
    home.timer_set(60, "egg")
    assert ha.alexa_commands()[-1][1] == "set an egg timer for 1 minute"
