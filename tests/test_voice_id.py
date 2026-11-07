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
    # Live 2026-10-08: Whisper heard "as Adam" as "is Adam".
    ("remember my voice is Adam.", Command("enrol", "Adam")),
    ("Remember my voice for Adam", Command("enrol", "Adam")),
    ("remember my voice, Adam", Command("enrol", "Adam")),
    ("remember my voice, my name's Adam", Command("enrol", "Adam")),
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


@pytest.mark.parametrize("answer,name", [
    ("Adam.", "Adam"), ("It's Adam.", "Adam"), ("My name is adam", "Adam"),
    ("Um, Adam", "Adam"), ("I don't know.", None), ("", None),
])
def test_extract_name(answer, name):
    assert vid.extract_name(answer) == name


# -- other ways to ask, talk-or-read, offered profiles ----------------------------

@pytest.mark.parametrize("said, name", [("register me", None), ("register my voice", None),
                                        ("create a profile for me", None), ("set up my voice profile", None),
                                        ("register me as Sam", "Sam"), ("make a profile, I'm Lily", "Lily")])
def test_more_ways_to_ask_for_enrolment(said, name):
    assert parse_command(said) == Command("enrol", name)


@pytest.mark.parametrize("said", ["register the car", "create a shopping list", "what's my profile picture"])
def test_not_enrolment(said):
    assert parse_command(said) is None


@pytest.mark.parametrize("said, how", [("put something on the screen", "read"), ("I'd rather read", "read"),
                                       ("give me a script", "read"), ("I'll just talk", "talk"),
                                       ("talk", "talk"), ("um, I don't know", "talk"),
                                       ("I'll read, no, just talk", "talk")])
def test_talk_or_read(said, how):
    assert vid.enrol_method(said) == how


def test_script_is_about_twelve_seconds_of_reading():
    words = len(vid.ENROL_SCRIPT.split())
    assert 30 <= words <= 45                  # ~2.7 words a second read aloud


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def speech(seconds):
    return (0.3 * np.sin(np.arange(int(seconds * 16000)) / 16000 * 2 * np.pi * 220)).astype(np.float32)


def test_pool_is_ready_after_enough_turns_and_speech():
    pool = vid.StrangerPool(need_s=6.0, min_turns=3)
    kid = unit(0, 0, 1)
    assert pool.add(kid, speech(4)) is None
    assert pool.add(unit(0, 0.1, 1), speech(4)) is None            # 8 s, but only two turns
    voice = pool.add(kid, speech(1))
    assert voice is not None and len(voice.clips) == 3


def test_pool_keeps_voices_apart():
    pool = vid.StrangerPool(need_s=2.0, min_turns=2)
    pool.add(unit(0, 0, 1), speech(2))
    assert pool.add(unit(0, 1, 0), speech(2)) is None              # someone else: a new voice
    assert len(pool.voices) == 2


def test_pool_forgets_old_voices_and_never_saves():
    clock = FakeClock()
    pool = vid.StrangerPool(max_age_s=60, clock=clock)
    pool.add(unit(0, 0, 1), speech(2))
    clock.now = 61
    pool.add(unit(0, 1, 0), speech(2))
    assert len(pool.voices) == 1


def test_pool_asks_each_voice_once():
    pool = vid.StrangerPool(need_s=1.0, min_turns=1)
    voice = pool.add(unit(0, 0, 1), speech(2))
    enrolment = pool.take(voice)
    assert enrolment.offered and enrolment.stage == "name" and enrolment.speech_s >= 1.0
    assert enrolment.rounds == 0
    assert pool.add(unit(0, 0, 1), speech(2)) is None


def test_pool_matches_only_its_own_voice():
    pool = vid.StrangerPool(need_s=1.0, min_turns=1)
    voice = pool.add(unit(0, 0, 1), speech(2))
    assert pool.matches(voice, unit(0, 0.2, 1)) and not pool.matches(voice, unit(1, 0, 0))
    assert not pool.matches(voice, None)
    assert pool.add(None, speech(2)) is None
