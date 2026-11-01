# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Daemon wiring for spoken memory commands: handled without the model,
never recorded as an episode, and a yes/no question listens for the answer."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from homeai.config import Config


@pytest.fixture
def assistant(monkeypatch):
    monkeypatch.setattr("homeai.daemon.FOLLOWUP_SETTLE_S", 0.0)
    with patch("homeai.daemon.Microphone"), \
         patch("homeai.daemon.Transcriber"), \
         patch("homeai.daemon.Speaker"), \
         patch("homeai.daemon.build_agent_client"), \
         patch("homeai.daemon.TranscriptLog"):
        from homeai.daemon import VoiceAssistant

        va = VoiceAssistant(Config())
    va.cfg = replace(va.cfg, wake=replace(va.cfg.wake, barge_in=False))
    va.ltm = MagicMock()
    return va


def _say(va, text, source="wake"):
    result = va.stt.transcribe_audio.return_value
    result.ok, result.text, result.error = True, text, ""
    va._handle(np.zeros(1600, dtype=np.float32), source)
    return va.transcript.write.call_args.args[0]


def test_memory_command_skips_the_model(assistant):
    assistant.ltm.handle_command.return_value = ("Okay, I'll remember that.", False)
    turn = _say(assistant, "Remember that my sister is Kate.")
    assistant.agent.ask.assert_not_called()
    assert turn.verdict == "memory-command" and turn.reply == "Okay, I'll remember that."
    assert assistant.ltm.handle_command.call_args.args[1] == "Remember that my sister is Kate."
    assert not assistant._followup_armed.is_set()


def test_memory_question_listens_for_the_answer(assistant):
    assistant.ltm.handle_command.return_value = ("Shall I forget them?", True)
    _say(assistant, "Forget what I said about the party.")
    assert assistant._followup_armed.is_set()


def test_not_a_memory_command_goes_on_to_the_model(assistant):
    assistant.ltm.handle_command.return_value = None
    assistant.agent.ask.return_value = MagicMock(ok=True, text="It's sunny.", attempts=1,
                                                 error="")
    _say(assistant, "What's the weather like?")
    assistant.ltm.handle_command.assert_called_once()
    assert assistant.agent.ask.called


def test_memory_command_verdict_is_never_recorded():
    from homeai.memory_store.policy import should_record

    assert not should_record(heard="remember that x", reply="ok", verdict="memory-command",
                             speaker="Adam").record


def test_no_long_term_memory_means_no_command_handling(assistant):
    assistant.ltm = None
    assistant.agent.ask.return_value = MagicMock(ok=True, text="Sure.", attempts=1, error="")
    _say(assistant, "Remember that my sister is Kate.")
    assert assistant.agent.ask.called
