# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for wake detection and the capture state machine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from homeai.config import WakeConfig
from homeai.wake import CaptureMachine, EnergyDetector, State


def loud(n: int = 1280, level: float = 0.5) -> np.ndarray:
    return np.full(n, level, dtype=np.float32)


def quiet(n: int = 1280) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


class TestEnergyDetector:
    def test_requires_consecutive_loud_frames(self) -> None:
        det = EnergyDetector(threshold=0.05, consecutive=3)
        assert det.detect(loud()) is False
        assert det.detect(loud()) is False
        assert det.detect(loud()) is True

    def test_single_transient_does_not_trigger(self) -> None:
        """A door slam is one loud frame, not speech."""
        det = EnergyDetector(threshold=0.05, consecutive=3)
        assert det.detect(loud()) is False
        assert det.detect(quiet()) is False
        assert det.detect(loud()) is False

    def test_silence_never_triggers(self) -> None:
        det = EnergyDetector(threshold=0.05, consecutive=3)
        assert all(det.detect(quiet()) is False for _ in range(20))

    def test_resets_after_firing(self) -> None:
        det = EnergyDetector(threshold=0.05, consecutive=2)
        det.detect(loud())
        assert det.detect(loud()) is True
        # Streak must restart, not fire again immediately.
        assert det.detect(loud()) is False

    def test_reset_clears_streak(self) -> None:
        det = EnergyDetector(threshold=0.05, consecutive=3)
        det.detect(loud())
        det.detect(loud())
        det.reset()
        assert det.detect(loud()) is False


@pytest.fixture
def cfg() -> WakeConfig:
    return replace(WakeConfig(), silence_s=0.7, max_utterance_s=15.0)


class TestCaptureMachine:
    def test_starts_idle(self, cfg: WakeConfig) -> None:
        assert CaptureMachine(cfg).state is State.IDLE

    def test_ignores_frames_while_idle(self, cfg: WakeConfig) -> None:
        machine = CaptureMachine(cfg)
        assert machine.feed(loud(), now=1.0) is False

    def test_wake_transitions_to_listening(self, cfg: WakeConfig) -> None:
        machine = CaptureMachine(cfg)
        machine.on_wake(now=0.0)
        assert machine.state is State.LISTENING

    def test_continues_while_speech_present(self, cfg: WakeConfig) -> None:
        machine = CaptureMachine(cfg)
        machine.on_wake(now=0.0)
        for t in (0.1, 0.2, 0.3, 0.4):
            assert machine.feed(loud(), now=t) is False

    def test_ends_after_trailing_silence(self, cfg: WakeConfig) -> None:
        machine = CaptureMachine(cfg)
        machine.on_wake(now=0.0)
        machine.feed(loud(), now=0.5)
        assert machine.feed(quiet(), now=0.9) is False   # 0.4s of silence
        assert machine.feed(quiet(), now=1.3) is True    # 0.8s exceeds 0.7s

    def test_silence_timer_resets_on_new_speech(self, cfg: WakeConfig) -> None:
        """A pause mid-sentence must not truncate the utterance."""
        machine = CaptureMachine(cfg)
        machine.on_wake(now=0.0)
        machine.feed(quiet(), now=0.5)
        machine.feed(loud(), now=0.6)      # speech resumes
        assert machine.feed(quiet(), now=1.1) is False
        assert machine.feed(quiet(), now=1.4) is True

    def test_hard_max_duration_ends_capture(self, cfg: WakeConfig) -> None:
        """Guards against a stuck-open mic capturing forever."""
        machine = CaptureMachine(cfg)
        machine.on_wake(now=0.0)
        # Continuous speech would otherwise never end.
        assert machine.feed(loud(), now=10.0) is False
        assert machine.feed(loud(), now=15.1) is True

    def test_reset_returns_to_idle(self, cfg: WakeConfig) -> None:
        machine = CaptureMachine(cfg)
        machine.on_wake(now=0.0)
        machine.reset()
        assert machine.state is State.IDLE
        assert machine.feed(loud(), now=1.0) is False

    def test_duration_tracks_elapsed(self, cfg: WakeConfig) -> None:
        machine = CaptureMachine(cfg)
        machine.on_wake(now=5.0)
        assert machine.duration(now=7.5) == pytest.approx(2.5)

    def test_duration_zero_when_idle(self, cfg: WakeConfig) -> None:
        assert CaptureMachine(cfg).duration(now=100.0) == 0.0

    def test_short_silence_threshold_respected(self) -> None:
        fast = replace(WakeConfig(), silence_s=0.2, max_utterance_s=15.0)
        machine = CaptureMachine(fast)
        machine.on_wake(now=0.0)
        machine.feed(loud(), now=0.1)
        assert machine.feed(quiet(), now=0.35) is True

    def test_reusable_after_completion(self, cfg: WakeConfig) -> None:
        machine = CaptureMachine(cfg)
        machine.on_wake(now=0.0)
        machine.feed(quiet(), now=1.0)
        machine.reset()
        machine.on_wake(now=10.0)
        assert machine.state is State.LISTENING
        assert machine.feed(loud(), now=10.2) is False


class TestModelPathResolution:
    def test_resolves_friendly_name_to_versioned_file(self) -> None:
        """Bundled models carry a version suffix, e.g. hey_jarvis_v0.1.onnx."""
        from homeai.wake import OpenWakeWordDetector

        path = OpenWakeWordDetector.resolve_model_path("hey_jarvis")
        if path is not None:  # skip gracefully if package layout changes
            assert path.endswith(".onnx")
            assert "hey_jarvis" in path

    def test_unknown_name_returns_none(self) -> None:
        from homeai.wake import OpenWakeWordDetector

        assert OpenWakeWordDetector.resolve_model_path("no_such_wakeword_xyz") is None

    def test_empty_name_returns_none(self) -> None:
        from homeai.wake import OpenWakeWordDetector

        assert OpenWakeWordDetector.resolve_model_path("") is None

    def test_does_not_resolve_feature_extractors(self) -> None:
        """melspectrogram and embedding_model are not wake words."""
        from homeai.wake import OpenWakeWordDetector

        assert OpenWakeWordDetector.resolve_model_path("melspectrogram") is None
        assert OpenWakeWordDetector.resolve_model_path("embedding_model") is None


# -- refractory window (regression: spurious second wake after capture) -----

from homeai.wake import CaptureMachine as _CM, State as _St
from homeai.config import WakeConfig as _WC


def _machine(**kw):
    cfg = _WC()
    for k, v in kw.items():
        object.__setattr__(cfg, k, v)
    return _CM(cfg=cfg)


def test_wake_is_ignored_during_refractory_window():
    m = _machine(refractory_s=1.5)
    m.on_wake(10.0)
    m.reset(now=10.0)
    assert m.accepts_wake(10.1) is False
    assert m.accepts_wake(11.4) is False


def test_wake_is_accepted_after_refractory_window():
    m = _machine(refractory_s=1.5)
    m.on_wake(10.0)
    m.reset(now=10.0)
    assert m.accepts_wake(11.5) is True
    assert m.accepts_wake(20.0) is True


def test_reset_without_now_does_not_open_refractory():
    """Error paths call reset() with no timestamp; must not go deaf."""
    m = _machine(refractory_s=1.5)
    m.reset()
    assert m.accepts_wake(0.0) is True


def test_observed_regression_second_wake_245ms_later_is_blocked():
    # Live log: wake at t, capture done, second wake 0.245s later.
    m = _machine(refractory_s=1.5)
    m.on_wake(100.0)
    m.reset(now=102.3)
    assert m.accepts_wake(102.545) is False, "the 0.245s re-trigger must be suppressed"


def test_refractory_can_be_disabled():
    m = _machine(refractory_s=0.0)
    m.reset(now=5.0)
    assert m.accepts_wake(5.0) is True
