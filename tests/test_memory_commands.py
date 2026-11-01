# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Spoken memory command parsing (memory_store/commands.py)."""

from __future__ import annotations

import pytest

from homeai.memory_store.commands import (confirmation, correction, first_person_to_third,
                                          parse)


@pytest.mark.parametrize("text,kind,arg", [
    ("remember that my sister is called Kate", "remember", "my sister is called Kate"),
    ("Jarvis, remember I like my coffee black.", "remember", "I like my coffee black"),
    ("please make a note that the boiler was serviced", "remember", "the boiler was serviced"),
    ("always remember that I'm allergic to penicillin", "remember_always",
     "I'm allergic to penicillin"),
    ("never forget my anniversary is June 3rd", "remember_always", "my anniversary is June 3rd"),
    ("what do you remember about me?", "recall_me", ""),
    ("What do you know about me", "recall_me", ""),
    ("what do you remember about the snake", "recall_about", "the snake"),
    ("forget that", "forget_last", ""),
    ("okay forget what I just said", "forget_last", ""),
    ("forget the last thing I said", "forget_last", ""),
    ("forget what I said about the party", "forget_about", "the party"),
    ("forget everything about work", "forget_about", "work"),
    ("forget my conversations about the car", "forget_about", "the car"),
    ("forget that whole idea", "forget_thread", ""),
    ("forget the whole plan about the shed", "forget_thread", "the shed"),
    ("forget everything about me", "forget_all", ""),
    ("forget everything you know about me", "forget_all", ""),
    ("stop remembering things about me", "stop", ""),
    ("don't remember anything about me", "stop", ""),
    ("start remembering things about me again", "start", ""),
    ("you can remember stuff about me again", "start", ""),
])
def test_commands(text, kind, arg):
    cmd = parse(text)
    assert cmd is not None, text
    assert (cmd.kind, cmd.text) == (kind, arg)


@pytest.mark.parametrize("text", [
    "do you remember when we went to Rome",
    "remember when I asked about the snake",
    "remember what I said yesterday",
    "remember my voice",
    "remember me",
    "remember to buy milk",
    "I can't remember the name of that film",
    "what's the weather",
    "can you forget about the lights and play music",   # mid-sentence: not anchored
    "remember it",                                       # too short to keep
    "what do you remember about it",
    "",
    None,
])
def test_not_commands(text):
    assert parse(text) is None


@pytest.mark.parametrize("text,want", [
    ("yes", True), ("Yeah.", True), ("okay yes please", True), ("go ahead", True),
    ("forget them", True), ("no", False), ("nope", False), ("keep them", False),
    ("never mind", False), ("no thanks", False), ("what time is it", None),
    ("yes and turn on the lights", None), ("", None),
])
def test_confirmation(text, want):
    assert confirmation(text) is want


@pytest.mark.parametrize("text,want", [
    ("that's wrong, it's Pickle now", "Pickle now"),
    ("No, that's not right. My snake is called Pickle", "My snake is called Pickle"),
    ("that is out of date, I moved to Leeds", "I moved to Leeds"),
    ("that's wrong", None),
    ("I think you're wrong", None),
    ("that's wrong, ok", None),
])
def test_correction(text, want):
    assert correction(text) == want


@pytest.mark.parametrize("text,want", [
    ("my sister is called Kate", "Adam's sister is called Kate"),
    ("I'm allergic to penicillin", "Adam is allergic to penicillin"),
    ("I am a nurse", "Adam is a nurse"),
    ("I've got two cats", "Adam has got two cats"),
    ("I like it when you call me sir", 'Adam said: "I like it when you call me sir"'),
    ("i think my bike is broken", 'Adam said: "I think my bike is broken"'),
    ("the bike is mine", "The bike is Adam's"),
    ("the boiler was serviced", "The boiler was serviced"),
])
def test_first_person_to_third(text, want):
    assert first_person_to_third(text, "Adam") == want
