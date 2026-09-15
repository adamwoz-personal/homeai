# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for the audio ring buffer and helpers.

The ring buffer is pure logic and is tested exhaustively here, including the
wrap-around cases that are easy to get subtly wrong and hard to debug live.
"""

from __future__ import annotations

import numpy as np
import pytest

from homeai.audio import RingBuffer, rms


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
