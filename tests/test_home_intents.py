# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Fast-path home command recognition. A false match does the wrong thing
to the house, so the negative cases matter as much as the positive ones."""

from __future__ import annotations

import pytest

from homeai.home_intents import HomeIntent, parse, parse_duration, parse_number


@pytest.mark.parametrize("text,seconds", [
    ("10 minutes", 600),
    ("ten minutes", 600),
    ("a minute", 60),
    ("an hour", 3600),
    ("90 seconds", 90),
    ("twenty five minutes", 1500),
    ("twenty-five minutes", 1500),
    ("1 hour 20 minutes", 4800),
    ("1 hour and 20 minutes", 4800),
    ("an hour and a half", 5400),
    ("two and a half minutes", 150),
    ("half an hour", 1800),
    ("a quarter of an hour", 900),
    ("1.5 hours", 5400),
    ("5 mins", 300),
    ("10-minute", 600),
    ("a couple of minutes", 120),
])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "a while", "ten", "minutes", "10 minutes ish", "soon"])
def test_parse_duration_rejects(text):
    assert parse_duration(text) is None


def test_parse_number():
    assert parse_number("forty two") == 42
    assert parse_number("7") == 7
    assert parse_number("lots") is None


@pytest.mark.parametrize("text,label,seconds", [
    ("Set a pasta timer for 10 minutes.", "pasta", 600),
    ("set a timer for ten minutes", None, 600),
    ("Set a timer for 5 minutes called soup", "soup", 300),
    ("set a timer for 5 minutes for the eggs", "eggs", 300),
    ("Can you set a 10 minute timer please", None, 600),
    ("set a 10 minute pasta timer", "pasta", 600),
    ("start a pizza timer for twenty minutes", "pizza", 1200),
    ("Set a chocolate chip timer for 12 minutes", "chocolate chip", 720),
    ("set a timer for an hour and a half", None, 5400),
])
def test_timer_set(text, label, seconds):
    assert parse(text) == HomeIntent("timer_set", {"seconds": seconds, "label": label})


@pytest.mark.parametrize("text,label", [
    ("Stop the pasta timer.", "pasta"),
    ("cancel the pasta timer", "pasta"),
    ("turn off the pasta timer", "pasta"),
    ("cancel the timer", None),
    ("cancel my timer", None),
    ("cancel all timers", "all"),
    ("cancel all the timers", "all"),
    ("delete the soup timer please", "soup"),
])
def test_timer_cancel(text, label):
    assert parse(text) == HomeIntent("timer_cancel", {"label": label})


@pytest.mark.parametrize("text,label", [
    ("How much time is left on the pasta timer?", "pasta"),
    ("how long is left on the timer", None),
    ("how much time is left", None),
    ("how's the pasta timer doing", "pasta"),
    ("what timers are running", None),
    ("are there any timers", None),
    ("check my timers", None),
])
def test_timer_status(text, label):
    assert parse(text) == HomeIntent("timer_status", {"label": label})


@pytest.mark.parametrize("text,action,target", [
    ("Turn off the foyer.", "off", "foyer"),
    ("turn on the foyer lights", "on", "foyer"),
    ("Turn the den lamp off", "off", "den lamp"),
    ("switch off the kitchen cabinets", "off", "kitchen cabinets"),
    ("turn off all the lights", "off", "all"),
    ("turn off everything", "off", "all"),
    ("lights off in the den", "off", "den"),
    ("den lights on", "on", "den"),
    ("could you turn on the breakfast nook please", "on", "breakfast nook"),
    ("Turn off the lights.", "off", None),
])
def test_lights_on_off(text, action, target):
    assert parse(text) == HomeIntent("lights", {"action": action, "target": target})


def test_dim_and_levels():
    assert parse("dim the den lamp") == HomeIntent(
        "lights", {"action": "dim", "target": "den lamp", "brightness": None})
    assert parse("dim the foyer to 20%") == HomeIntent(
        "lights", {"action": "set", "target": "foyer", "brightness": 20})
    assert parse("set the foyer to 40 percent") == HomeIntent(
        "lights", {"action": "set", "target": "foyer", "brightness": 40})
    assert parse("brighten the kitchen cabinets") == HomeIntent(
        "lights", {"action": "brighten", "target": "kitchen cabinets"})


@pytest.mark.parametrize("text", [
    "turn off the music",          # music control, matched separately below? no: "turn off" + music
    "turn off the tv",
    "turn on the garage",
    "turn it off",
    "open the garage door",
])
def test_not_lights(text):
    intent = parse(text)
    assert intent is None or intent.kind != "lights"


def test_turn_off_the_music_is_music():
    assert parse("turn off the music") == HomeIntent("music", {"action": "stop"})


@pytest.mark.parametrize("text,msg", [
    ("Announce that dinner is ready", "dinner is ready"),
    ("announce dinner is ready", "dinner is ready"),
    ("tell everyone that we're leaving in five minutes", "we're leaving in five minutes"),
    ("make an announcement: the car is here", "the car is here"),
])
def test_announce(text, msg):
    assert parse(text) == HomeIntent("announce", {"message": msg})


def test_music():
    assert parse("play some jazz") == HomeIntent(
        "music", {"action": "play", "request": "some jazz", "where": None})
    assert parse("Play jazz on Amazon Music.") == HomeIntent(
        "music", {"action": "play", "request": "jazz", "where": None})
    assert parse("play some jazz in the kitchen") == HomeIntent(
        "music", {"action": "play", "request": "some jazz", "where": "kitchen"})
    assert parse("pause the music") == HomeIntent("music", {"action": "pause"})
    assert parse("resume the music") == HomeIntent("music", {"action": "resume"})
    assert parse("skip this song") == HomeIntent("music", {"action": "next"})
    assert parse("next song") == HomeIntent("music", {"action": "next"})


@pytest.mark.parametrize("text", [
    "play a game with me",         # no music cue: the model decides
    "play twenty questions",
    "what do you think about free will",
    "what's the weather",
    "stop",                        # a dismissal, not a command
    "thank you",
    "who am I",
    "remember my voice as Adam",
    "turn",
    "",
    "how long is a piece of string",
    "set the table",
])
def test_not_home(text):
    assert parse(text) is None


@pytest.mark.parametrize("text,label", [
    ("set aside the timer for five minutes", "aside"),   # heard live for "a pasta"
    ("set a pasta the timer for 2 minutes", "pasta"),
    ("set the my timer for 2 minutes", None),
])
def test_timer_label_drops_stray_articles(text, label):
    intent = parse(text)
    assert intent.kind == "timer_set" and intent.args["label"] == label
