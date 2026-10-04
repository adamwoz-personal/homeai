# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Tests for the second-stage wake check (homeai/wake_verify.py)."""

from __future__ import annotations

import pytest

from homeai.wake_verify import is_pleasantry_only, mentions_wake_word, strip_wake_phrase

# What Whisper heard after real false wakes, 2026-10-02/03.
FALSE_WAKES = [
    "Amazing. Love you, Valor.",
    "This is 94 Belches, slow you down, flat feet and throw. I'm coming at you buddy!",
    "David.",
    "I don't know what conversation",
    "Cody will be back in a moment. He will help them. Please sign up yourself.",
    "Thank you. You're welcome.",
    "Hey Travis, come here.",
    "Marvin, dinner!",
    "Harvey, put the jar away.",
]


@pytest.mark.parametrize("text", [
    "Hey Jarvis, what is 3 plus 3?",
    "Hey Jarvis.",
    "hey jervis what time is it",
    "Javis, is the universe alive?",
    "So anyway. Hey, Jarvis's voice is nice. What's the weather?",
    "Hey Garvis, tell me a joke.",
    "Got so close though. Hey Jarvan, what is 2 level 2?",  # live, Piper hfc_female
])
def test_real_wakes_are_verified(text):
    assert mentions_wake_word(text)


@pytest.mark.parametrize("text", FALSE_WAKES)
def test_recorded_false_wakes_are_rejected(text):
    assert not mentions_wake_word(text)


def test_other_wake_models_use_their_own_name():
    assert mentions_wake_word("Alexa, stop", "alexa")
    assert not mentions_wake_word("Hey Jarvis", "alexa")


@pytest.mark.parametrize("text,expected", [
    ("Hey Jarvis, what is 3 plus 3?", "what is 3 plus 3?"),
    ("so anyway, hey Jarvis. What's the time?", "What's the time?"),
    ("Hey Jarvis.", ""),
    ("Jervis - is the universe alive?", "is the universe alive?"),
    ("no wake word here", "no wake word here"),
])
def test_strip_wake_phrase(text, expected):
    assert strip_wake_phrase(text) == expected


@pytest.mark.parametrize("text", [
    "Thank you. You're welcome.",
    "Thank you!",
    "thanks for watching",
    "Thank you so much. Bye.",
    "You\u2019re welcome.",
    "you",
])
def test_pleasantries(text):
    assert is_pleasantry_only(text)


@pytest.mark.parametrize("text", [
    "Thank you, what's the weather?",
    "okay tell me more",
    "Thanks. Is the universe alive?",
    "Hello, how are you?",
    "",
    "Youth is wasted on the young.",
])
def test_requests_are_not_pleasantries(text):
    assert not is_pleasantry_only(text)


@pytest.mark.parametrize("text,expected", [
    ("Thanks for watching!", True),
    ("Please subscribe.", True),
    ("you", True),
    ("Thank you.", False),
    ("Thank you. You're welcome.", False),
    ("Thanks for watching. What's the weather?", False),
])
def test_hallucinations_are_a_subset_of_pleasantries(text, expected):
    from homeai.wake_verify import is_hallucination_only
    assert is_hallucination_only(text) is expected
    if expected:
        assert is_pleasantry_only(text)


@pytest.mark.parametrize("text,reply", [
    ("Thank you.", "You're welcome."),
    ("Thanks a lot!", "You're welcome."),
    ("Thank you. You're welcome.", "You're welcome."),
    ("Bye bye.", "Bye for now."),
    ("Okay.", ""),
    ("Great.", ""),
])
def test_closing_reply(text, reply):
    from homeai.wake_verify import closing_reply
    assert closing_reply(text) == reply
