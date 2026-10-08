"""Tests for homeai.voice_id: commands, enrolment state, the follow-up gate."""
from __future__ import annotations

import numpy as np
import pytest

from homeai import voice_id as vid
from homeai.config import SpeakerConfig
from homeai.speaker import Identification
from homeai.voice_id import Command, Enrolment, VoiceId, parse_command

SR = 16000


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("text,expected", [
    ("remember my voice as Adam.", Command("enrol", "Adam")),
    ("Jarvis, learn my voice. My name is beth", Command("enrol", "Beth")),
    ("please remember my voice, I'm Charlie", Command("enrol", "Charlie")),
    ("remember my voice", Command("enrol", None)),
    ("remember my voice as the boss", Command("enrol", None)),
    ("who am I?", Command("who")),
    ("Do you recognize my voice?", Command("who")),
    ("forget my voice", Command("forget")),
    ("what is the weather", None),
    ("remember to buy milk", None),
    ("I am Adam", None),
])
def test_parse_command(text, expected):
    assert parse_command(text) == expected


def test_enrolment_needs_enough_speech(monkeypatch):
    monkeypatch.setattr(vid, "ENROL_TARGET_SPEECH_S", 3.0)
    e = Enrolment("Adam")
    e.add(tone(2.0))
    assert not e.done and not e.gave_up
    e.add(np.concatenate([tone(1.5), np.zeros(SR, dtype=np.float32)]))
    assert e.done and e.rounds == 2 and e.speech_s == pytest.approx(3.5, abs=0.1)
    assert len(e.audio()) == int(4.5 * SR)


def test_enrolment_gives_up_after_too_many_quiet_answers():
    e = Enrolment("Adam")
    for _ in range(vid.ENROL_MAX_ROUNDS):
        e.add(np.zeros(SR, dtype=np.float32))
    assert e.gave_up and not e.done


def test_enrolment_expires():
    e = Enrolment("Adam")
    assert not e.expired(e.started + 1) and e.expired(e.started + vid.ENROL_TIMEOUT_S + 1)


@pytest.fixture
def voice(tmp_path):
    v = VoiceId(SpeakerConfig(registry_path=tmp_path / "speakers.json"))
    v.registry.add("Adam", unit(1, 0, 0))
    v.registry.add("Beth", unit(0, 1, 0))
    return v


def ident(name, score=0.8):
    return Identification(name, score, name or "Adam", 0.5, "" if name else "below threshold")


def test_followup_from_the_same_person_is_let_through(voice):
    assert voice.is_someone_else("Adam", ident("Adam"), unit(1, 0.1, 0)) == (False, pytest.approx(0.995, abs=0.01))


def test_followup_recognised_as_someone_else_is_rejected(voice):
    other, _ = voice.is_someone_else("Adam", ident("Beth"), unit(0, 1, 0))
    assert other


def test_followup_from_an_unknown_but_clearly_different_voice_is_rejected(voice):
    assert voice.is_someone_else("Adam", ident(None), unit(0, 0, 1))[0]


def test_followup_that_cannot_be_judged_is_let_through(voice):
    assert voice.is_someone_else("Adam", ident(None), None) == (False, None)
    assert voice.is_someone_else(None, ident("Beth"), unit(0, 1, 0)) == (False, None)
    # Unsure but not clearly different: let through.
    assert not voice.is_someone_else("Adam", ident(None, 0.4), unit(0.4, 0, 0.9))[0]


def test_followup_gate_can_be_turned_off(tmp_path):
    v = VoiceId(SpeakerConfig(registry_path=tmp_path / "s.json", followup_same_speaker=False))
    v.registry.add("Adam", unit(1, 0))
    assert not v.is_someone_else("Adam", ident("Beth"), unit(0, 1))[0]


def test_finish_enrolment_replaces_and_saves(voice, monkeypatch):
    monkeypatch.setattr(voice.embedder, "embed", lambda audio, trim=True: unit(0, 0, 1))
    e = Enrolment("adam")
    e.add(tone(1.0))
    profile = voice.finish_enrolment(e)
    assert profile.name == "Adam" and profile.embedding == pytest.approx([0, 0, 1])
    again = VoiceId(SpeakerConfig(registry_path=voice.registry.path))
    assert again.registry.get("Adam").embedding == pytest.approx([0, 0, 1])


def test_finish_enrolment_with_no_usable_audio(voice, monkeypatch):
    monkeypatch.setattr(voice.embedder, "embed", lambda audio, trim=True: None)
    assert voice.finish_enrolment(Enrolment("Adam")) is None


def test_forget(voice):
    assert voice.forget("adam") and voice.registry.names() == ["Beth"]
    assert not voice.forget("Adam")


def test_describe():
    assert vid.describe(Identification("Adam", 0.62, "Adam", 0.4)) == "Adam (0.62, margin 0.40)"
    assert "best Adam 0.31: below threshold" in vid.describe(
        Identification(None, 0.31, "Adam", 0.31, "below threshold"))
    assert vid.describe(Identification(None, reason="nobody enrolled")) == "unknown (nobody enrolled)"
