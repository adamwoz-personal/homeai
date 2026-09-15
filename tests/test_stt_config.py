# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for the whisper.cpp wrapper and configuration validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from homeai.config import AudioConfig, Config, SttConfig
from homeai.stt import Transcriber, clean_transcript, write_wav


class TestCleanTranscript:
    def test_strips_bracketed_non_speech(self) -> None:
        assert clean_transcript("[BLANK_AUDIO] hello there") == "hello there"
        assert clean_transcript("(door closes) come in") == "come in"

    def test_collapses_whitespace(self) -> None:
        assert clean_transcript("  hello   world \n") == "hello world"

    def test_pure_annotation_becomes_empty(self) -> None:
        assert clean_transcript("[BLANK_AUDIO]") == ""
        assert clean_transcript("(silence)") == ""

    def test_punctuation_only_becomes_empty(self) -> None:
        """Whisper sometimes emits bare punctuation for noise."""
        assert clean_transcript("...") == ""
        assert clean_transcript(" . , ! ") == ""

    def test_empty_input(self) -> None:
        assert clean_transcript("") == ""
        assert clean_transcript(None) == ""

    def test_preserves_real_speech(self) -> None:
        assert clean_transcript("What is the weather today?") == "What is the weather today?"


class TestWriteWav:
    def test_writes_readable_wav_from_float(self, tmp_path: Path) -> None:
        import wave

        audio = np.zeros(1600, dtype=np.float32)
        path = tmp_path / "t.wav"
        write_wav(path, audio, 16000)

        assert path.exists()
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
            assert handle.getsampwidth() == 2
            assert handle.getnframes() == 1600

    def test_accepts_int16_directly(self, tmp_path: Path) -> None:
        audio = np.zeros(800, dtype=np.int16)
        path = tmp_path / "t.wav"
        write_wav(path, audio, 16000)
        assert path.exists()

    def test_clips_out_of_range_floats(self, tmp_path: Path) -> None:
        """Values beyond [-1, 1] must not wrap around into loud noise."""
        import wave

        audio = np.array([5.0, -5.0], dtype=np.float32)
        path = tmp_path / "t.wav"
        write_wav(path, audio, 16000)

        with wave.open(str(path), "rb") as handle:
            data = np.frombuffer(handle.readframes(2), dtype=np.int16)
        assert data[0] == 32767
        assert data[1] == -32767


class TestTranscriberAvailability:
    def test_reports_missing_binary(self, tmp_path: Path) -> None:
        cfg = replace(SttConfig(), binary=tmp_path / "nope", model=tmp_path / "also-nope")
        ok, problem = Transcriber(cfg).available()
        assert ok is False
        assert "binary not found" in problem

    def test_reports_missing_model(self, tmp_path: Path) -> None:
        binary = tmp_path / "whisper-cli"
        binary.touch()
        cfg = replace(SttConfig(), binary=binary, model=tmp_path / "absent.bin")
        ok, problem = Transcriber(cfg).available()
        assert ok is False
        assert "model not found" in problem

    def test_transcribe_returns_error_not_raise(self, tmp_path: Path) -> None:
        cfg = replace(SttConfig(), binary=tmp_path / "nope", model=tmp_path / "nope")
        result = Transcriber(cfg).transcribe_file(tmp_path / "x.wav")
        assert result.ok is False
        assert result.error

    def test_empty_audio_rejected(self) -> None:
        result = Transcriber(SttConfig()).transcribe_audio(np.array([], dtype=np.float32), 16000)
        assert result.ok is False
        assert "empty" in result.error


class TestConfigValidation:
    def test_flags_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HOMEAI_AGENT_TOKEN", raising=False)
        problems = Config().validation_errors()
        assert any("HOMEAI_AGENT_TOKEN" in p for p in problems)

    def test_flags_wrong_sample_rate(self) -> None:
        cfg = Config(audio=replace(AudioConfig(), sample_rate=44100))
        assert any("16000" in p for p in cfg.validation_errors())

    def test_flags_stereo(self) -> None:
        cfg = Config(audio=replace(AudioConfig(), channels=2))
        assert any("mono" in p for p in cfg.validation_errors())

    def test_reports_all_problems_not_just_first(self) -> None:
        cfg = Config(audio=replace(AudioConfig(), sample_rate=8000, channels=2))
        problems = cfg.validation_errors()
        assert len(problems) >= 2, "user should see every problem at once"

    def test_env_override_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOMEAI_WAKE_THRESHOLD", "0.9")
        from homeai.config import WakeConfig

        assert WakeConfig().threshold == pytest.approx(0.9)

    def test_malformed_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo in an env var must not crash the service at boot."""
        monkeypatch.setenv("HOMEAI_SAMPLE_RATE", "not-a-number")
        assert AudioConfig().sample_rate == 16000

    def test_ring_frames_derived(self) -> None:
        cfg = replace(AudioConfig(), sample_rate=16000, ring_seconds=30)
        assert cfg.ring_frames == 480000
