# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for the audio ring buffer and helpers.

The ring buffer is pure logic and is tested exhaustively here, including the
wrap-around cases that are easy to get subtly wrong and hard to debug live.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from homeai.audio import Microphone, MicWatchdog, RingBuffer, rms
from homeai.config import AudioConfig


class TestRingBufferBasics:
    def test_rejects_invalid_capacity(self) -> None:
        with pytest.raises(ValueError):
            RingBuffer(0)

    def test_starts_empty(self) -> None:
        assert len(RingBuffer(10)) == 0

    def test_read_from_empty_returns_empty(self) -> None:
        assert RingBuffer(10).read_last(5).size == 0

    def test_write_then_read(self) -> None:
        buf = RingBuffer(10)
        buf.write(np.array([1, 2, 3], dtype=np.float32))
        assert len(buf) == 3
        np.testing.assert_allclose(buf.read_last(3), [1, 2, 3])

    def test_write_empty_is_noop(self) -> None:
        buf = RingBuffer(10)
        buf.write(np.array([], dtype=np.float32))
        assert len(buf) == 0

    def test_read_more_than_available_clamps(self) -> None:
        buf = RingBuffer(10)
        buf.write(np.array([1, 2], dtype=np.float32))
        assert buf.read_last(100).size == 2


class TestRingBufferWrapAround:
    def test_wraps_and_keeps_newest(self) -> None:
        buf = RingBuffer(5)
        buf.write(np.arange(1, 8, dtype=np.float32))  # 1..7 into a 5-slot buffer
        assert len(buf) == 5
        np.testing.assert_allclose(buf.read_last(5), [3, 4, 5, 6, 7])

    def test_multiple_writes_across_boundary(self) -> None:
        buf = RingBuffer(5)
        buf.write(np.array([1, 2, 3], dtype=np.float32))
        buf.write(np.array([4, 5, 6], dtype=np.float32))
        np.testing.assert_allclose(buf.read_last(5), [2, 3, 4, 5, 6])

    def test_write_larger_than_capacity_keeps_tail(self) -> None:
        buf = RingBuffer(3)
        buf.write(np.arange(1, 11, dtype=np.float32))
        np.testing.assert_allclose(buf.read_last(3), [8, 9, 10])

    def test_exact_capacity_write(self) -> None:
        buf = RingBuffer(4)
        buf.write(np.array([1, 2, 3, 4], dtype=np.float32))
        np.testing.assert_allclose(buf.read_last(4), [1, 2, 3, 4])

    def test_partial_read_after_wrap(self) -> None:
        buf = RingBuffer(5)
        buf.write(np.arange(1, 9, dtype=np.float32))  # keeps 4..8
        np.testing.assert_allclose(buf.read_last(2), [7, 8])

    def test_chronological_order_preserved(self) -> None:
        """Regression guard: wrap-around must not reverse or rotate samples."""
        buf = RingBuffer(100)
        for i in range(0, 250, 10):
            buf.write(np.arange(i, i + 10, dtype=np.float32))
        out = buf.read_last(100)
        assert np.all(np.diff(out) == 1), "samples must remain in order"
        assert out[-1] == 249


class TestRingBufferClear:
    def test_clear_empties(self) -> None:
        buf = RingBuffer(10)
        buf.write(np.array([1, 2, 3], dtype=np.float32))
        buf.clear()
        assert len(buf) == 0
        assert buf.read_last(3).size == 0

    def test_usable_after_clear(self) -> None:
        buf = RingBuffer(5)
        buf.write(np.array([1, 2, 3], dtype=np.float32))
        buf.clear()
        buf.write(np.array([9, 8], dtype=np.float32))
        np.testing.assert_allclose(buf.read_last(2), [9, 8])


class TestRingBufferTypes:
    def test_accepts_2d_column(self) -> None:
        buf = RingBuffer(10)
        buf.write(np.array([[1.0], [2.0]], dtype=np.float32).reshape(-1))
        assert len(buf) == 2

    def test_casts_to_float32(self) -> None:
        buf = RingBuffer(10)
        buf.write(np.array([1, 2, 3], dtype=np.int16))
        assert buf.read_last(3).dtype == np.float32


class TestRms:
    def test_silence_is_zero(self) -> None:
        assert rms(np.zeros(100, dtype=np.float32)) == pytest.approx(0.0)

    def test_constant_signal(self) -> None:
        assert rms(np.full(100, 0.5, dtype=np.float32)) == pytest.approx(0.5, abs=1e-6)

    def test_empty_is_zero(self) -> None:
        assert rms(np.array([], dtype=np.float32)) == 0.0

    def test_none_is_zero(self) -> None:
        assert rms(None) == 0.0

    def test_louder_signal_scores_higher(self) -> None:
        quiet = rms(np.full(50, 0.1, dtype=np.float32))
        loud = rms(np.full(50, 0.8, dtype=np.float32))
        assert loud > quiet


class TestReadNewCursor:
    """The cursor API guarantees each sample is consumed exactly once.

    Regression: using read_last() for wake detection re-examined the same
    audio every poll, so one spoken wake word fired the detector repeatedly.
    """

    def test_returns_only_new_samples(self) -> None:
        buf = RingBuffer(100)
        cursor = 0
        buf.write(np.arange(1, 6, dtype=np.float32))
        data, cursor = buf.read_new(cursor)
        np.testing.assert_allclose(data, [1, 2, 3, 4, 5])

        buf.write(np.arange(6, 9, dtype=np.float32))
        data, cursor = buf.read_new(cursor)
        np.testing.assert_allclose(data, [6, 7, 8])

    def test_no_new_data_returns_empty(self) -> None:
        buf = RingBuffer(100)
        buf.write(np.arange(1, 4, dtype=np.float32))
        _, cursor = buf.read_new(0)
        data, cursor2 = buf.read_new(cursor)
        assert data.size == 0
        assert cursor2 == cursor

    def test_never_replays_same_samples(self) -> None:
        """The core guarantee: repeated reads must not duplicate audio."""
        buf = RingBuffer(1000)
        cursor = 0
        seen: list[float] = []
        for i in range(0, 50, 5):
            buf.write(np.arange(i, i + 5, dtype=np.float32))
            data, cursor = buf.read_new(cursor)
            seen.extend(data.tolist())
        assert seen == list(range(50))
        assert len(seen) == len(set(seen)), "no sample may be delivered twice"

    def test_max_samples_limits_batch(self) -> None:
        buf = RingBuffer(100)
        buf.write(np.arange(1, 11, dtype=np.float32))
        data, cursor = buf.read_new(0, max_samples=4)
        assert data.size == 4
        assert cursor == 4

    def test_cursor_jumps_forward_when_overrun(self) -> None:
        """A slow consumer must not receive misaligned stale audio."""
        buf = RingBuffer(10)
        buf.write(np.arange(1, 31, dtype=np.float32))  # 20 samples overwritten
        data, cursor = buf.read_new(0)
        assert data.size == 10
        assert cursor == 30
        np.testing.assert_allclose(data, np.arange(21, 31))

    def test_clear_returns_adoptable_cursor(self) -> None:
        buf = RingBuffer(50)
        buf.write(np.arange(1, 11, dtype=np.float32))
        cursor = buf.clear()
        data, cursor2 = buf.read_new(cursor)
        assert data.size == 0, "after clear there is nothing new to read"

        buf.write(np.array([99.0], dtype=np.float32))
        data, _ = buf.read_new(cursor2)
        np.testing.assert_allclose(data, [99.0])

    def test_total_written_tracks_all_writes(self) -> None:
        buf = RingBuffer(5)
        buf.write(np.ones(3, dtype=np.float32))
        buf.write(np.ones(4, dtype=np.float32))
        assert buf.total_written == 7

    def test_total_written_counts_oversized_write(self) -> None:
        buf = RingBuffer(5)
        buf.write(np.ones(20, dtype=np.float32))
        assert buf.total_written == 20


class FakeStream:
    """Stands in for sd.InputStream: delivers audio only when told to."""

    opened: list["FakeStream"] = []

    def __init__(self, callback, **_kwargs) -> None:
        self.callback = callback
        self.closed = False
        FakeStream.opened.append(self)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def deliver(self) -> None:
        self.callback(np.ones((4, 1), dtype=np.float32), 4, None, None)


@pytest.fixture
def mic(monkeypatch):
    FakeStream.opened = []
    fake = types.SimpleNamespace(InputStream=FakeStream, query_devices=lambda: [
        {"name": "HD WEBCAM", "max_input_channels": 1, "default_samplerate": 16000}])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    m = Microphone(AudioConfig(input_match="WEBCAM"))
    assert m.start() == (True, "")
    return m


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def watched(mic, **kwargs):
    clock, gave_up = Clock(), []
    return MicWatchdog(mic, lambda: gave_up.append(True), clock=clock, **kwargs), clock, gave_up


class TestMicWatchdog:
    def test_a_silent_stream_is_reopened_and_audio_carries_on(self, mic) -> None:
        dog, clock, gave_up = watched(mic)
        first = FakeStream.opened[0]
        first.deliver()
        dog.check()
        cursor = mic.buffer.total_written
        clock.now = 5.0
        dog.check()
        assert first.closed and len(FakeStream.opened) == 2 and not gave_up
        FakeStream.opened[1].deliver()
        assert mic.buffer.total_written > cursor        # same buffer: readers' cursors stay valid

    def test_a_short_gap_is_left_alone(self, mic) -> None:
        dog, clock, _ = watched(mic)
        clock.now = 4.9
        dog.check()
        assert len(FakeStream.opened) == 1

    def test_flowing_audio_is_left_alone(self, mic) -> None:
        dog, clock, _ = watched(mic)
        for second in range(1, 20):
            FakeStream.opened[0].deliver()
            clock.now = float(second)
            dog.check()
        assert len(FakeStream.opened) == 1

    def test_pausing_for_speech_is_not_a_stall(self, mic) -> None:
        """While Jarvis talks nothing reaches the buffer, but the device is fine."""
        dog, clock, _ = watched(mic)
        mic.pause()
        for second in range(1, 20):
            FakeStream.opened[0].deliver()
            clock.now = float(second)
            dog.check()
        assert mic.buffer.total_written == 0 and len(FakeStream.opened) == 1

    def test_gives_up_once_after_the_reopens_run_out(self, mic) -> None:
        dog, clock, gave_up = watched(mic, max_reopens=3)
        for _ in range(3):
            clock.now += 5.0
            dog.check()
        assert len(FakeStream.opened) == 4 and not gave_up
        clock.now += 5.0
        dog.check()
        clock.now += 5.0
        dog.check()
        assert gave_up == [True] and len(FakeStream.opened) == 4

    def test_each_new_stream_gets_a_full_stall_period(self, mic) -> None:
        dog, clock, _ = watched(mic)
        clock.now = 5.0
        dog.check()
        clock.now = 9.9
        dog.check()
        assert len(FakeStream.opened) == 2

    def test_recovery_earns_back_the_reopens(self, mic) -> None:
        dog, clock, gave_up = watched(mic, max_reopens=1)
        for _ in range(3):
            clock.now += 5.0
            dog.check()                                 # silent: one reopen
            FakeStream.opened[-1].deliver()
            clock.now += 1.0
            dog.check()                                 # audio came back
        assert not gave_up and len(FakeStream.opened) == 4

    def test_a_failed_open_is_retried(self, mic, monkeypatch) -> None:
        dog, clock, gave_up = watched(mic)
        monkeypatch.setattr(mic, "reopen", lambda: (False, "no device"))
        clock.now = 5.0
        dog.check()
        assert not gave_up
        calls = []
        monkeypatch.setattr(mic, "reopen", lambda: (calls.append(1), (True, ""))[1])
        clock.now = 10.0
        dog.check()
        assert calls == [1]

    def test_a_reopen_that_raises_does_not_stop_the_watch(self, mic, monkeypatch) -> None:
        def boom():
            raise OSError("PortAudio")

        dog, clock, gave_up = watched(mic)
        monkeypatch.setattr(mic, "reopen", boom)
        clock.now = 5.0
        dog.check()
        assert not gave_up

    def test_a_reopen_stuck_on_a_wedged_device_gives_up_without_hanging(self, mic, monkeypatch) -> None:
        release = threading.Event()
        monkeypatch.setattr(mic, "reopen", lambda: release.wait(5) and (True, ""))
        dog, clock, gave_up = watched(mic, reopen_timeout_s=0.05)
        clock.now = 5.0
        started = time.monotonic()
        dog.check()
        release.set()
        assert gave_up == [True] and time.monotonic() - started < 1.0

    def test_runs_on_its_own_thread_until_stopped(self, mic) -> None:
        dog = MicWatchdog(mic, lambda: None, stall_s=0.02, check_s=0.005, max_reopens=10_000, reopen_timeout_s=0.1)
        dog.start()
        try:
            deadline = time.monotonic() + 2.0
            while len(FakeStream.opened) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            dog.stop()
            settled = len(FakeStream.opened)
            assert settled >= 2
            time.sleep(0.05)
            assert len(FakeStream.opened) == settled
        finally:
            # If stop() is broken (mutant mic-dog-thread-outlives-stop) the thread
            # would still be reopening at interpreter exit, which segfaulted
            # pytest and raised desktop crash pop-ups (2026-11-07). SystemExit
            # gets past the loop's `except Exception` and ends the thread.
            def halt():
                raise SystemExit
            dog.check = halt
            if dog._thread is not None:
                dog._thread.join(timeout=2)
