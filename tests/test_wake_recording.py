# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Recording "hey jeeves" from the family inside Jarvis (homeai/wake_recording.py, plans/WAKE_WORD_PLAN.md)."""
from __future__ import annotations

import json
import wave
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from homeai import wake_recording as W
from homeai.config import Config

VOICE = (0.1 * np.sin(np.arange(16000) / 5)).astype(np.float32)     # 1 s, clearly louder than the room


@pytest.mark.parametrize("text,name", [
    ("Record my wake word.", None), ("record the wake word for Ani", "Ani"), ("Record Ani's wake word", "Ani"),
    ("please record my wake phrase", None), ("Record the wake phrase for izzy.", "Izzy"),
])
def test_starting_a_session_takes_the_name_if_said(text, name):
    assert W.parse_start(text) == W.Start(name)


@pytest.mark.parametrize("text", ["record a song", "what's my wake word", "Hey Jeeves", "remember my voice"])
def test_other_requests_do_not_start_one(text):
    assert W.parse_start(text) is None


def test_stop_words_end_it_but_the_phrase_does_not():
    assert W.is_stop("Stop.") and W.is_stop("that's enough") and W.is_stop("done")
    assert not W.is_stop("Hey Jeeves.") and not W.is_stop("hey jeeves, stop the music")


def test_only_a_voice_is_kept():
    assert W.usable(VOICE) == ""
    assert W.usable(VOICE[:2000]) == "too short"
    assert W.usable(np.tile(VOICE, 9)) == "too long"
    assert W.usable(VOICE * 0.01) == "too quiet"


def test_a_session_asks_in_rotating_styles_then_sound_alikes_and_never_says_hey_jarvis():
    items = W.plan(0)
    pos = [i for i in items if i[0] == "positive"]
    assert len(pos) == W.POSITIVES and all(t == "hey jeeves" for _, t, _ in pos)
    assert {p for _, _, p in pos} == {p for _, p in W.STYLES}
    neg = [i for i in items if i[0] == "negative"]
    assert [t for _, t, _ in neg] == list(W.SOUND_ALIKES)
    assert not any("hey jarvis" in p.lower() for _, _, p in items)          # the barge-in listener would hear it
    assert W.plan(1)[0][2] == W.STYLES[1][1]                                # carries on where the last one left off


def test_a_person_stops_at_the_maximum():
    assert len([i for i in W.plan(W.MAX_POSITIVES - 5) if i[0] == "positive"]) == 5
    assert W.plan(W.MAX_POSITIVES) == []


def test_answers_are_saved_with_what_was_asked_and_heard(tmp_path):
    s = W.Session("Ani", root=tmp_path)
    assert "Ani" in s.opening() and str(W.POSITIVES) in s.opening()
    reply = s.take(VOICE, "Hey Jeeves.", now=0)
    assert reply.startswith("1 of 30.") and s.kept == 1
    files = list((tmp_path / "ani" / "positive").glob("*.wav"))
    assert len(files) == 1
    with wave.open(str(files[0])) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 16000)
    row = json.loads((tmp_path / "ani" / "index.jsonl").read_text())
    assert row["asked"] == "hey jeeves" and row["heard"] == "Hey Jeeves." and row["kind"] == "positive"
    assert W.saved("Ani", tmp_path) == 1 and W.Session("ani", root=tmp_path).items == W.plan(1)


def test_a_bad_answer_is_asked_again_then_skipped(tmp_path):
    s = W.Session("Adam", root=tmp_path)
    for _ in range(W.RETRIES):
        assert s.take(VOICE * 0.01, "").startswith("That was too quiet. Again:")
        assert s.at == 0
    assert s.take(VOICE * 0.01, "").startswith("Skipping that one.") and s.at == 1 and s.kept == 0
    assert not (tmp_path / "adam" / "positive").exists()


def test_sound_alikes_go_in_their_own_folder_and_the_session_ends(tmp_path):
    s = W.Session("Adam", root=tmp_path)
    reply = ""
    while not s.done:
        reply = s.take(VOICE, "x")
    assert reply.startswith("Done") and s.kept == len(s.items)
    assert len(list((tmp_path / "adam" / "negative").glob("*.wav"))) == len(W.SOUND_ALIKES)


# --- inside Jarvis ---------------------------------------------------------------------------------------------

@pytest.fixture
def assistant(tmp_path, monkeypatch):
    monkeypatch.setattr("homeai.daemon.FOLLOWUP_SETTLE_S", 0.0)
    monkeypatch.setattr(W, "DIR", tmp_path)
    with patch("homeai.daemon.Microphone"), patch("homeai.daemon.Transcriber"), patch("homeai.daemon.Speaker"), \
            patch("homeai.daemon.build_agent_client"), patch("homeai.daemon.TranscriptLog"):
        from homeai.daemon import VoiceAssistant
        va = VoiceAssistant(Config())
    va.cfg = replace(va.cfg, wake=replace(va.cfg.wake, barge_in=False, verify=False))
    return va


def _say(va, text, source="wake", audio=VOICE):
    result = va.stt.transcribe_audio.return_value
    result.ok, result.text, result.error = True, text, ""
    va._followup_armed.clear()
    va._handle(audio, source)
    return va.transcript.write.call_args.args[0]


def test_jarvis_records_a_named_person_turn_by_turn_and_stops_on_stop(assistant, tmp_path):
    turn = _say(assistant, "Record Ani's wake word.")
    assert turn.verdict == "wake-rec" and "Recording Ani" in turn.reply
    assert assistant._followup_armed.is_set()                    # listens for the answer without the wake word
    turn = _say(assistant, "Hey Jeeves.", source="followup")
    assert turn.reply.startswith("1 of 30.") and assistant._followup_armed.is_set()
    assert len(list((tmp_path / "ani" / "positive").glob("*.wav"))) == 1
    assistant.agent.ask.assert_not_called()
    turn = _say(assistant, "Stop.", source="followup")
    assert turn.reply == "Stopped. I kept 1 recordings of Ani." and assistant._wake_rec is None
    assert not assistant._followup_armed.is_set()


def test_an_unknown_voice_is_asked_whose_it_is(assistant, tmp_path):
    turn = _say(assistant, "Record my wake word.")
    assert turn.reply.startswith("Whose voice") and assistant._wake_rec is None
    assert not (tmp_path / "ani").exists()


def test_a_new_wake_word_turn_leaves_the_session(assistant, tmp_path):
    _say(assistant, "Record the wake word for Ani.")
    reply = assistant.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "Hello.", 1, ""
    turn = _say(assistant, "Hey Jeeves.")                         # a wake-word turn, not a follow-up answer
    assert assistant._wake_rec is None and turn.verdict != "wake-rec"
    assert not (tmp_path / "ani" / "positive").exists()
