# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for barge-in: the wake-word-during-playback interrupt path.

No audio hardware is used. The microphone and detector are both fakes, which
is the point of keeping this logic out of the daemon.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from homeai.bargein import InterruptListener, contains_wake_fragments


class FakeBuffer:
    """Minimal stand-in for RingBuffer.read_new."""

    def __init__(self) -> None:
        self._data = np.zeros(0, dtype=np.float32)
        self._total = 0
        self._lock = threading.Lock()

    def push(self, frames: np.ndarray) -> None:
        with self._lock:
            self._data = np.concatenate([self._data, frames])
            self._total += len(frames)

    def read_new(self, cursor: int, max_samples: int | None = None):
        with self._lock:
            available = self._total - cursor
            if available <= 0:
                return np.zeros(0, dtype=np.float32), cursor
            n = available if max_samples is None else min(available, max_samples)
            start = len(self._data) - available
            return self._data[start : start + n].copy(), cursor + n


class FakeMic:
    def __init__(self) -> None:
        self.buffer = FakeBuffer()


class FakeDetector:
    """Fires once after ``fire_after`` blocks have been inspected."""

    def __init__(self, fire_after: int | None = 2) -> None:
        self.fire_after = fire_after
        self.blocks = 0
        self.reset_calls = 0

    def detect(self, frame: np.ndarray) -> bool:
        self.blocks += 1
        return self.fire_after is not None and self.blocks >= self.fire_after

    def reset(self) -> None:
        self.reset_calls += 1


BLOCK = 128


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# contains_wake_fragments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("General relativity extended that to gravity.", False),
        ("Hey, that reminds me. Jarvis is a name in fiction.", True),
        ("The answer is jarvis.", True),
        ("Hey there, how are you?", True),
        ("", False),
        ("HEY JARVIS", True),
    ],
)
def test_fragment_detection(text: str, expected: bool) -> None:
    assert contains_wake_fragments(text, "hey_jarvis") is expected


def test_fragments_ignore_short_words() -> None:
    """Two-letter tokens are not distinctive enough to gate on."""
    assert contains_wake_fragments("an ox in a box", "ox_jarvis") is False


def test_empty_wake_word_never_blocks() -> None:
    assert contains_wake_fragments("anything at all", "") is False


# ---------------------------------------------------------------------------
# InterruptListener
# ---------------------------------------------------------------------------


def test_triggers_when_detector_fires() -> None:
    mic = FakeMic()
    detector = FakeDetector(fire_after=2)
    listener = InterruptListener(mic, lambda: (detector, "openWakeWord"), BLOCK)

    assert listener.start() is True
    try:
        mic.buffer.push(np.ones(BLOCK * 4, dtype=np.float32))
        assert _wait_for(lambda: listener.triggered)
        assert listener.should_stop() is True
    finally:
        listener.stop()


def test_silent_when_detector_never_fires() -> None:
    mic = FakeMic()
    detector = FakeDetector(fire_after=None)
    listener = InterruptListener(mic, lambda: (detector, "openWakeWord"), BLOCK)

    assert listener.start() is True
    try:
        mic.buffer.push(np.ones(BLOCK * 8, dtype=np.float32))
        assert _wait_for(lambda: detector.blocks >= 8)
        assert listener.triggered is False
        assert listener.should_stop() is False
    finally:
        listener.stop()


def test_refuses_energy_fallback_detector() -> None:
    """An energy detector would fire on Jarvis's own voice immediately."""
    mic = FakeMic()
    listener = InterruptListener(
        mic, lambda: (FakeDetector(), "energy fallback (no model)"), BLOCK
    )
    assert listener.start() is False
    assert listener.triggered is False


def test_detector_factory_failure_is_non_fatal() -> None:
    def boom():
        raise RuntimeError("no model")

    listener = InterruptListener(FakeMic(), boom, BLOCK)
    assert listener.start() is False
    assert listener.should_stop() is False


def test_read_failure_does_not_raise() -> None:
    class ExplodingBuffer(FakeBuffer):
        def read_new(self, cursor, max_samples=None):
            raise OSError("device lost")

    mic = FakeMic()
    mic.buffer = ExplodingBuffer()
    listener = InterruptListener(mic, lambda: (FakeDetector(), "openWakeWord"), BLOCK)
    assert listener.start() is True
    listener.stop()
    assert listener.triggered is False


def test_stop_resets_detector_and_joins() -> None:
    mic = FakeMic()
    detector = FakeDetector(fire_after=None)
    listener = InterruptListener(mic, lambda: (detector, "openWakeWord"), BLOCK)
    listener.start()
    listener.stop()
    assert detector.reset_calls == 1
    assert listener._thread is None


def test_stop_is_idempotent() -> None:
    listener = InterruptListener(
        FakeMic(), lambda: (FakeDetector(), "openWakeWord"), BLOCK
    )
    listener.start()
    listener.stop()
    listener.stop()


def test_ignores_audio_buffered_before_start() -> None:
    """Only speech arriving *during* playback may interrupt.

    Otherwise the user's original request, still sitting in the ring buffer,
    would immediately interrupt the reply it just asked for.
    """
    mic = FakeMic()
    mic.buffer.push(np.ones(BLOCK * 10, dtype=np.float32))
    detector = FakeDetector(fire_after=1)
    listener = InterruptListener(mic, lambda: (detector, "openWakeWord"), BLOCK)

    assert listener.start() is True
    try:
        time.sleep(0.15)
        assert listener.triggered is False, "stale audio must not interrupt"
        mic.buffer.push(np.ones(BLOCK * 2, dtype=np.float32))
        assert _wait_for(lambda: listener.triggered)
    finally:
        listener.stop()


def test_partial_block_is_not_scored() -> None:
    mic = FakeMic()
    detector = FakeDetector(fire_after=1)
    listener = InterruptListener(mic, lambda: (detector, "openWakeWord"), BLOCK)
    assert listener.start() is True
    try:
        mic.buffer.push(np.ones(BLOCK // 2, dtype=np.float32))
        time.sleep(0.15)
        assert detector.blocks == 0
        assert listener.triggered is False
    finally:
        listener.stop()


def test_consecutive_requirement() -> None:
    """With consecutive=3 a single isolated hit must not interrupt."""

    class AlternatingDetector(FakeDetector):
        def detect(self, frame):
            self.blocks += 1
            return self.blocks % 2 == 1

    mic = FakeMic()
    listener = InterruptListener(
        mic, lambda: (AlternatingDetector(), "openWakeWord"), BLOCK, consecutive=3
    )
    assert listener.start() is True
    try:
        mic.buffer.push(np.ones(BLOCK * 12, dtype=np.float32))
        time.sleep(0.2)
        assert listener.triggered is False
    finally:
        listener.stop()
