# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for daemon orchestration that needs no audio hardware.

The daemon is the least-covered module despite being the orchestrator, and two
field bugs in a row lived here rather than in the components it wires together.
These tests construct a VoiceAssistant with its heavyweight collaborators
replaced, so no microphone, model, or subprocess is touched.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from homeai.config import Config
from homeai.wake import State


@pytest.fixture
def assistant():
    """A VoiceAssistant whose expensive collaborators are inert fakes."""
    with patch("homeai.daemon.Microphone"), \
         patch("homeai.daemon.Transcriber"), \
         patch("homeai.daemon.Speaker"), \
         patch("homeai.daemon.build_agent_client"), \
         patch("homeai.daemon.TranscriptLog"):
        from homeai.daemon import VoiceAssistant

        va = VoiceAssistant(Config())
    return va


class FakeBuffer:
    def __init__(self) -> None:
        self.total_written = 0
        self._pending = np.zeros(0, dtype=np.float32)

    def read_new(self, cursor, max_samples=None):
        data, self._pending = self._pending, np.zeros(0, dtype=np.float32)
        return data, cursor + len(data)

    def provide(self, frames: np.ndarray) -> None:
        self._pending = frames
        self.total_written += len(frames)


def _drive_wake_loop(assistant, iterations: float = 0.25) -> None:
    """Run the real wake loop briefly in a thread, then stop it."""
    t = threading.Thread(target=assistant._wake_loop, daemon=True)
    t.start()
    time.sleep(iterations)
    assistant._stop.set()
    t.join(timeout=2)
    assistant._stop.clear()


# ---------------------------------------------------------------------------
# Barge-in handoff
#
# Field failure: the interrupt worked (10ms cut) but the follow-up question was
# never captured, and the log showed nothing at all afterwards. The wake loop
# skips every frame while tts.speaking is true, so it never observed the wake
# word that caused the interrupt -- only InterruptListener did, and that
# listener captures no audio.
# ---------------------------------------------------------------------------


def test_interrupt_arms_the_wake_loop(assistant) -> None:
    assistant.mic.buffer = FakeBuffer()
    assistant.mic.paused = False
    assistant.tts.speaking = False
    assistant._bargein_armed.set()

    assert assistant.machine.state is State.IDLE
    _drive_wake_loop(assistant)

    assert assistant.machine.state is State.LISTENING, (
        "after an interrupt the daemon must capture the follow-up without "
        "requiring a second wake word"
    )
    assert not assistant._bargein_armed.is_set(), "flag must be consumed once"


def test_no_arming_means_no_capture(assistant) -> None:
    """Without the flag the loop stays idle, so this cannot fire on its own."""
    assistant.mic.buffer = FakeBuffer()
    assistant.mic.paused = False
    assistant.tts.speaking = False

    _drive_wake_loop(assistant)
    assert assistant.machine.state is State.IDLE


def test_arming_bypasses_the_refractory_window(assistant) -> None:
    """_speak resets the machine, opening a refractory window.

    The handoff must not be blocked by it: the person has already spoken the
    wake word deliberately, so refusing to listen would drop their question.
    """
    assistant.mic.buffer = FakeBuffer()
    assistant.mic.paused = False
    assistant.tts.speaking = False

    assistant.machine.reset(time.monotonic())
    assert assistant.machine.accepts_wake(time.monotonic()) is False

    assistant._bargein_armed.set()
    _drive_wake_loop(assistant)
    assert assistant.machine.state is State.LISTENING


def test_loop_ignores_arming_while_still_speaking(assistant) -> None:
    """Arming must not take effect until our own audio has actually stopped."""
    assistant.mic.buffer = FakeBuffer()
    assistant.mic.paused = False
    assistant.tts.speaking = True
    assistant._bargein_armed.set()

    _drive_wake_loop(assistant)

    assert assistant.machine.state is State.IDLE
    assert assistant._bargein_armed.is_set(), "flag must survive until audible"
