# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Daemon wiring for home control: the fast path, 'which lights?', and the
quiet window that stops an Echo announcement waking Jarvis."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from homeai.config import Config
from homeai.ha_client import HAUnavailable
from homeai.home import EchoQuiet, Home, TimerLedger
from homeai.wake import State
from tests.fake_ha import FakeHA


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    monkeypatch.setattr("homeai.daemon.FOLLOWUP_SETTLE_S", 0.0)
    with patch("homeai.daemon.Microphone"), \
         patch("homeai.daemon.Transcriber"), \
         patch("homeai.daemon.Speaker"), \
         patch("homeai.daemon.build_agent_client"), \
         patch("homeai.daemon.TranscriptLog"):
        from homeai.daemon import VoiceAssistant

        va = VoiceAssistant(Config())
    va.cfg = replace(va.cfg, wake=replace(va.cfg.wake, barge_in=False))
    va.ha = FakeHA()
    va.home = Home(va.ha, ledger=TimerLedger(tmp_path / "t.json"),
                   quiet=EchoQuiet(tmp_path / "quiet"), sleep=va.ha.slept.append)
    va._echo_quiet = va.home.quiet
    return va


def _say(va, text, source="wake"):
    result = va.stt.transcribe_audio.return_value
    result.ok, result.text, result.error = True, text, ""
    va._handle(np.zeros(1600, dtype=np.float32), source)
    return va.transcript.write.call_args.args[0]


def test_light_command_skips_the_model(assistant):
    turn = _say(assistant, "Turn off the foyer.")
    assert assistant.ha.calls == [("light", "turn_off", {"entity_id": ["light.foyer_foyer"]})]
    assistant.agent.ask.assert_not_called()
    assert turn.verdict == "home-lights"
    assert turn.reply == "Foyer off."


def test_home_exchange_is_remembered(assistant):
    _say(assistant, "Turn off the foyer.")
    assert [(e.user, e.assistant) for e in assistant.memory.recent()] == [
        ("Turn off the foyer.", "Foyer off.")]


def test_timer_round_trip(assistant):
    assert _say(assistant, "Set a pasta timer for 10 minutes").reply == \
        "Pasta timer set for 10 minutes."
    assert _say(assistant, "Stop the pasta timer.").reply == "Pasta timer cancelled."
    assert [c for _, c in assistant.ha.alexa_commands()] == [
        "set a pasta timer for 10 minutes", "cancel the pasta timer"]


def test_which_lights_then_answer(assistant):
    turn = _say(assistant, "Turn off the lights.")
    assert turn.reply == "Which lights?"
    assert assistant._followup_armed.is_set()
    assistant._followup_armed.clear()
    turn = _say(assistant, "The foyer.", source="followup")
    assert turn.reply == "Foyer off."
    assert assistant.ha.calls[-1][2]["entity_id"] == ["light.foyer_foyer"]


def test_which_lights_answer_needs_the_followup_window(assistant):
    _say(assistant, "Turn off the lights.")
    reply = assistant.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "Hello.", 1, ""
    _say(assistant, "the foyer")  # a new wake-word turn, not an answer
    assert assistant.ha.calls == []


def test_which_lights_expires(assistant, monkeypatch):
    _say(assistant, "Turn off the lights.")
    intent, asked = assistant._pending_home
    assistant._pending_home = (intent, asked - 31)
    reply = assistant.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "Hello.", 1, ""
    _say(assistant, "the foyer", source="followup")
    assert assistant.ha.calls == []


def test_house_controller_down_is_spoken(assistant):
    assistant.ha.fail = HAUnavailable("I can't reach the house controller")
    turn = _say(assistant, "Turn off the foyer.")
    assert turn.reply == "Sorry, I can't reach the house controller."
    assistant.agent.ask.assert_not_called()


def test_unexpected_crash_is_contained(assistant):
    assistant.home.lights = MagicMock(side_effect=ZeroDivisionError)
    turn = _say(assistant, "Turn off the foyer.")
    assert turn.reply == "Sorry, something went wrong."


def test_questions_still_go_to_the_model(assistant):
    reply = assistant.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "Probably.", 1, ""
    spoken = assistant.tts.say_chunked.return_value
    spoken.first_audio_ms, spoken.interrupted, spoken.chunks_spoken = 100.0, False, 1
    _say(assistant, "Is free will real?")
    assistant.agent.ask.assert_called_once()
    assert assistant.ha.calls == []


def test_no_house_means_old_behaviour(assistant):
    assistant.home = None
    turn = _say(assistant, "Turn off the foyer.")
    assert turn.verdict == "confirm"  # safety.py: device state change, refused
    assistant.agent.ask.assert_not_called()


def test_fast_path_can_be_turned_off(assistant):
    assistant.cfg = replace(assistant.cfg, home=replace(assistant.cfg.home, fast_path=False))
    reply = assistant.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "Okay.", 1, ""
    spoken = assistant.tts.say_chunked.return_value
    spoken.first_audio_ms, spoken.interrupted, spoken.chunks_spoken = 100.0, False, 1
    _say(assistant, "Set a pasta timer for 10 minutes")
    assert assistant.ha.calls == []


def test_announcement_never_carries_the_wake_word(assistant):
    _say(assistant, "Announce that Jarvis says dinner is ready")
    assert "jarvis" not in assistant.ha.calls[-1][2]["message"].lower()
    assert assistant._echo_quiet.remaining() > 0


# -- the quiet window in the wake loop ----------------------------------------

class OneShotBuffer:
    def __init__(self, frames):
        self.total_written = 0
        self._frames = frames

    def read_new(self, cursor, max_samples=None):
        data, self._frames = self._frames, np.zeros(0, dtype=np.float32)
        self.total_written += len(data)
        return data, cursor + len(data)


def _drive(va, seconds=0.3):
    va.mic.paused = False
    va.tts.speaking = False
    va.mic.buffer = OneShotBuffer(np.zeros(va.cfg.audio.block_size * 4, dtype=np.float32))
    va._detector = MagicMock()
    va._detector.detect.return_value = True
    va._detector.last_score = 0.77
    t = threading.Thread(target=va._wake_loop, daemon=True)
    t.start()
    time.sleep(seconds)
    va._stop.set()
    t.join(timeout=2)
    va._stop.clear()


def test_wake_ignored_while_an_echo_announces(assistant):
    assistant._echo_quiet.mark(10)
    _drive(assistant)
    assert assistant.machine.state is State.IDLE


def test_wake_works_after_the_announcement(assistant):
    _drive(assistant)
    assert assistant.machine.state is State.LISTENING
