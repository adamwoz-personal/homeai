# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for the transcript logging module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from homeai.transcript import TranscriptLog, Turn, Stopwatch


class TestTranscriptLog:
    def test_write_single_turn(self, tmp_path: Path) -> None:
        """Test that writing a single Turn appends exactly one JSON line to the file."""
        path = tmp_path / "test.log"
        log = TranscriptLog(str(path))

        turn = Turn(
            heard="hello",
            reply="hi",
            verdict="confirmed",
            ok=True,
            error="",
            attempts=1,
            stt_ms=100.0,
            agent_ms=200.0,
            tts_ms=300.0,
            total_ms=600.0,
        )
        log.write(turn)

        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert "ts" in record
        assert record["heard"] == "hello"
        assert record["reply"] == "hi"
        assert record["verdict"] == "confirmed"
        assert record["ok"] is True
        assert record["attempts"] == 1
        assert "timings_ms" in record
        assert record["timings_ms"]["stt"] == 100
        assert record["timings_ms"]["agent"] == 200
        assert record["timings_ms"]["tts"] == 300
        assert record["timings_ms"]["total"] == 600

    def test_write_multiple_turns(self, tmp_path: Path) -> None:
        """Test that writing two Turns produces exactly two lines."""
        path = tmp_path / "test.log"
        log = TranscriptLog(str(path))

        turn1 = Turn(heard="first", reply="reply1", verdict="ok", ok=True, attempts=1)
        turn2 = Turn(heard="second", reply="reply2", verdict="ok", ok=True, attempts=1)
        log.write(turn1)
        log.write(turn2)

        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2

        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])
        assert record1["heard"] == "first"
        assert record2["heard"] == "second"

    def test_written_lines_are_valid_json_with_required_keys(self, tmp_path: Path) -> None:
        """Test that each written line is valid JSON and contains the required keys."""
        path = tmp_path / "test.log"
        log = TranscriptLog(str(path))

        turn = Turn(
            heard="test",
            reply="reply",
            verdict="test",
            ok=True,
            error="",
            attempts=1,
            stt_ms=10.0,
            agent_ms=20.0,
            tts_ms=30.0,
            total_ms=60.0,
        )
        log.write(turn)

        content = path.read_text()
        record = json.loads(content.strip())

        required_keys = ["ts", "heard", "reply", "verdict", "ok", "attempts", "timings_ms"]
        for key in required_keys:
            assert key in record

        assert "error" not in record  # error key should not be present when empty

    def test_error_key_present_when_turn_has_error(self, tmp_path: Path) -> None:
        """Test that the 'error' key is present only when Turn.error is non-empty."""
        path = tmp_path / "test.log"
        log = TranscriptLog(str(path))

        turn_with_error = Turn(
            heard="test",
            reply="reply",
            verdict="test",
            ok=False,
            error="something went wrong",
            attempts=1,
        )
        log.write(turn_with_error)

        content = path.read_text()
        record = json.loads(content.strip())
        assert "error" in record
        assert record["error"] == "something went wrong"

    def test_enabled_false_no_file_creation(self) -> None:
        """Test that when enabled=False, no file is created and write() does nothing."""
        # Use a path that would fail if the file was actually created
        path = "/root/protected_file.log"
        log = TranscriptLog(path, enabled=False)

        turn = Turn(heard="test", reply="reply", verdict="test", ok=True, attempts=1)
        log.write(turn)

        # The file should not exist
        assert not os.path.exists(path)

    def test_stopwatch_ms_returns_float_ge_zero(self) -> None:
        """Test that Stopwatch.ms() returns a float >= 0."""
        stop = Stopwatch()
        ms = stop.ms()
        assert isinstance(ms, float)
        assert ms >= 0.0

    def test_timings_ms_contains_all_subkeys(self, tmp_path: Path) -> None:
        """Test that timings_ms contains the sub-keys stt, agent, tts, total."""
        path = tmp_path / "test.log"
        log = TranscriptLog(str(path))

        turn = Turn(
            heard="test",
            reply="reply",
            verdict="test",
            ok=True,
            error="",
            attempts=1,
            stt_ms=10.0,
            agent_ms=20.0,
            tts_ms=30.0,
            total_ms=60.0,
        )
        log.write(turn)

        content = path.read_text()
        record = json.loads(content.strip())
        timings = record["timings_ms"]

        expected_subkeys = ["stt", "agent", "tts", "total"]
        for key in expected_subkeys:
            assert key in timings