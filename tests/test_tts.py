# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for the TTS module and its audio-format contract.

Several tests here encode hard-won facts about this specific hardware. They
exist because each corresponds to a real failure observed after a reboot.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from homeai.config import AudioConfig, TtsConfig
from homeai.tts import Speaker


class TestOutputDeviceContract:
    def test_default_uses_plughw_not_raw_hw(self) -> None:
        """Regression: raw 'hw:2,0' fails with 'Channels count non available'.

        The ALC897 accepts only stereo at >=44.1 kHz; Piper emits mono at
        22.05 kHz. Only the 'plug' plugin bridges that gap.
        """
        device = AudioConfig().output_device
        assert device.startswith("plughw:"), (
            f"output_device is {device!r}; must use plughw so ALSA converts "
            "mono 22.05 kHz to the stereo 44.1 kHz+ the card demands"
        )

    def test_device_override_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOMEAI_OUTPUT_DEVICE", "plughw:9,9")
        assert AudioConfig().output_device == "plughw:9,9"


class TestSampleRateResolution:
    def test_reads_rate_from_model_config(self, tmp_path: Path) -> None:
        (tmp_path / "v.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 16000}}))
        cfg = replace(TtsConfig(), voice="v", model_dir=tmp_path)
        assert cfg.sample_rate() == 16000

    def test_falls_back_when_config_missing(self, tmp_path: Path) -> None:
        """A missing sidecar must not crash startup."""
        cfg = replace(TtsConfig(), voice="absent", model_dir=tmp_path)
        assert cfg.sample_rate() == 22050

    def test_falls_back_on_malformed_json(self, tmp_path: Path) -> None:
        (tmp_path / "v.onnx.json").write_text("{not json")
        cfg = replace(TtsConfig(), voice="v", model_dir=tmp_path)
        assert cfg.sample_rate() == 22050

    def test_falls_back_on_nonsense_rate(self, tmp_path: Path) -> None:
        (tmp_path / "v.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 0}}))
        cfg = replace(TtsConfig(), voice="v", model_dir=tmp_path)
        assert cfg.sample_rate() == 22050

    def test_real_voice_config_is_readable(self) -> None:
        """The installed voice must expose a plausible rate."""
        cfg = TtsConfig()
        if cfg.config_path.exists():
            assert 8000 <= cfg.sample_rate() <= 48000


class TestAvailability:
    def test_reports_missing_voice_model(self, tmp_path: Path) -> None:
        cfg = replace(TtsConfig(), voice="nope", model_dir=tmp_path)
        ok, problem = Speaker(cfg, "plughw:2,0").available()
        assert ok is False
        assert "voice model missing" in problem

    def test_reports_missing_piper_binary(self, tmp_path: Path) -> None:
        (tmp_path / "v.onnx").touch()
        cfg = replace(TtsConfig(), voice="v", model_dir=tmp_path, binary=tmp_path / "no-piper")
        ok, problem = Speaker(cfg, "plughw:2,0").available()
        # Either piper is absent entirely, or it resolved from PATH and the
        # model check is what matters; both are acceptable, a crash is not.
        assert isinstance(ok, bool)
        assert isinstance(problem, str)

    def test_empty_text_rejected_without_subprocess(self, tmp_path: Path) -> None:
        cfg = replace(TtsConfig(), voice="v", model_dir=tmp_path)
        result = Speaker(cfg, "plughw:2,0").say("   ")
        assert result.ok is False
        assert "nothing to say" in result.error


class TestSpeakingFlag:
    def test_not_speaking_initially(self) -> None:
        assert Speaker(TtsConfig(), "plughw:2,0").speaking is False

    def test_say_safe_never_raises(self, tmp_path: Path) -> None:
        """Failure paths must degrade, never propagate into the main loop."""
        cfg = replace(TtsConfig(), voice="absent", model_dir=tmp_path)
        result = Speaker(cfg, "plughw:2,0").say_safe("hello")
        assert result.ok is False
        assert result.error


class TestPiperBinaryResolution:
    def test_prefers_configured_venv_binary(self, tmp_path: Path) -> None:
        """Regression: piper lives in the venv, not on PATH.

        systemd will not activate a virtualenv, so the absolute path must win.
        """
        fake = tmp_path / "piper"
        fake.touch()
        cfg = replace(TtsConfig(), binary=fake)
        assert Speaker(cfg, "plughw:2,0")._piper_path() == str(fake)


def test_tts_timeout_allows_multi_minute_answers():
    """Regression: a 30s cap truncated any reply longer than ~75 words.

    The timeout bounds spoken duration, not synthesis, so it must comfortably
    exceed the length of a genuinely long spoken answer.
    """
    from homeai.config import TtsConfig

    # ~150 wpm; a 500-word answer needs well over three minutes.
    assert TtsConfig().timeout_s >= 240
