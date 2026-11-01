# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""What long-term memory may record, redact and recall (memory_store/policy.py)."""

from __future__ import annotations

import pytest

from homeai.memory_store import policy
from homeai.memory_store.policy import REDACTED, redact, should_recall, should_record, username_for

OK = dict(heard="hey jarvis what's a good snake for a beginner",
          reply="A corn snake is a classic first snake.", verdict="allow", speaker="Adam")


@pytest.mark.parametrize("name,expected", [
    ("Adam", "adam"), ("  Mary Jane ", "mary_jane"), ("O'Neil", "o_neil"),
    ("x" * 40, "x" * 32), ("", None), (None, None), ("!!!", None),
    ("Household", None), ("guest", None),
])
def test_username_for(name, expected):
    assert username_for(name) == expected


def test_records_an_answered_turn_from_a_recognised_speaker():
    assert should_record(**OK) == policy.Decision(True, "ok")


@pytest.mark.parametrize("change,reason", [
    ({"speaker": None}, "speaker not recognised"),
    ({"verdict": "unverified"}, "verdict unverified"),
    ({"verdict": "other-speaker"}, "verdict other-speaker"),
    ({"verdict": "home-lights"}, "verdict home-lights"),
    ({"verdict": "closing"}, "verdict closing"),
    ({"verdict": ""}, "verdict none"),
    ({"ok": False}, "failed turn"),
    ({"reply": "  "}, "empty"),
    ({"heard": ""}, "empty"),
    ({"reply": '<tool_call>{"name": "shell"}</tool_call>'}, "leaked markup"),
])
def test_does_not_record(change, reason):
    d = should_record(**{**OK, **change})
    assert d == policy.Decision(False, reason)


def test_continued_is_recorded():
    assert should_record(**{**OK, "verdict": "continued"}).record


def test_test_voices_are_not_people():
    d = should_record(**{**OK, "speaker": "Ryan"}, ignore_speakers=("ryan", "amy"))
    assert d == policy.Decision(False, "test voice")
    assert should_record(**{**OK, "speaker": "Ryan"}, ignore_speakers=()).record


def test_clean_heard_strips_the_wake_phrase():
    assert policy.clean_heard("hey jarvis what time is it") == "what time is it"
    assert policy.clean_heard("what time is it") == "what time is it"
    assert policy.clean_heard("hey jarvis") == "hey jarvis"  # nothing else left: keep it


@pytest.mark.parametrize("text,secret", [
    ("my wifi password is hunter2", "hunter2"),
    ("the gate code is 4 4 9 1", "4 4 9 1"),
    ("the alarm code: 8812", "8812"),
    ("pin number is 1234", "1234"),
    ("use sk-abcdefghijklmnop1234 for it", "sk-abcdefghijklmnop1234"),
    ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "ghp_abcdefghijklmnopqrstuvwxyz0123"),
    ("AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP"),
    ("ssn 123-45-6789", "123-45-6789"),
    ("card 4111 1111 1111 1111 thanks", "4111 1111 1111 1111"),
    ("x " + "A" * 48, "A" * 48),
])
def test_redacts_secrets(text, secret):
    out = redact(text)
    assert secret not in out
    assert REDACTED in out


@pytest.mark.parametrize("text", [
    "call me at 555 1234 tomorrow",
    "the order number is not a card 1234 5678 9012 3456",  # fails Luhn
    "my favourite snake is a king cobra",
    "",
])
def test_leaves_ordinary_text_alone(text):
    assert redact(text) == text


def test_luhn():
    assert policy._luhn("4111111111111111")
    assert not policy._luhn("4111111111111112")


@pytest.mark.parametrize("utterance,speaker,expected", [
    ("what's my favourite snake", "Adam", True),
    ("what's my favourite snake", None, False),
    ("what's my favourite snake", "Household", False),
    ("thank you", "Adam", False),
    ("hello", "Adam", False),          # one word: nothing to search on
    ("", "Adam", False),
    ('<tool_call>{"x": 1}</tool_call> please', "Adam", False),
])
def test_should_recall(utterance, speaker, expected):
    assert should_recall(utterance, speaker) is expected
