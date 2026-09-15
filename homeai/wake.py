# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Wake-word detection and utterance capture state machine.

Two detector implementations share one interface:

* ``OpenWakeWordDetector`` - the real thing, ONNX on CPU.
* ``EnergyDetector``       - a level-triggered fallback requiring no model.

The fallback exists so the pipeline is testable and demonstrable before the
wake-word models are downloaded, and so a missing model degrades the service
rather than preventing it from starting. It is **not** suitable for always-on
use: it triggers on any loud sound.

The state machine is kept separate from both detectors and from audio I/O, so
it can be tested exhaustively with synthetic timestamps and no hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import rms
from .config import WakeConfig

log = logging.getLogger(__name__)


class Detector(Protocol):
    """Anything that can say 'the wake word just occurred'."""

    def detect(self, frame: np.ndarray) -> bool: ...
    def reset(self) -> None: ...


class EnergyDetector:
    """Level-triggered fallback. Fires when audio exceeds a threshold.

    Requires ``consecutive`` loud frames in a row so that a single click or
    door slam does not trigger it.
    """

    def __init__(self, threshold: float = 0.05, consecutive: int = 3) -> None:
        self._threshold = threshold
        self._consecutive = consecutive
        self._streak = 0

    def detect(self, frame: np.ndarray) -> bool:
        if rms(frame) >= self._threshold:
            self._streak += 1
            if self._streak >= self._consecutive:
                self._streak = 0
                return True
        else:
            self._streak = 0
        return False

    def reset(self) -> None:
        self._streak = 0


class OpenWakeWordDetector:
    """openWakeWord via onnxruntime. Loaded lazily so import never fails."""

    def __init__(self, model: str, threshold: float) -> None:
        self._model_name = model
        self._threshold = threshold
        self._model = None
        self._error = ""

    @staticmethod
    def resolve_model_path(name: str) -> str | None:
        """Map a friendly name such as 'hey_jarvis' to its bundled .onnx path.

        The API takes filesystem paths, and bundled models carry a version
        suffix (``hey_jarvis_v0.1.onnx``), so an exact filename match fails.
        """
        if not name:
            return None
        if name.endswith(".onnx") and Path(name).exists():
            return name

        try:
            import openwakeword
        except ImportError:
            return None

        models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        if not models_dir.is_dir():
            return None

        stem = Path(name).stem.lower()
        candidates = sorted(
            p for p in models_dir.glob("*.onnx") if p.stem.lower().startswith(stem)
        )
        # Exclude the shared feature extractors, which are not wake models.
        candidates = [
            p for p in candidates
            if p.stem not in {"melspectrogram", "embedding_model", "silero_vad"}
        ]
        return str(candidates[0]) if candidates else None

    def load(self) -> tuple[bool, str]:
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            self._error = f"openwakeword not installed: {exc}"
            return False, self._error

        path = self.resolve_model_path(self._model_name)
        if path is None:
            self._error = f"no bundled wake model matching {self._model_name!r}"
            return False, self._error

        try:
            self._model = Model(wakeword_model_paths=[path])
        except Exception as exc:  # noqa: BLE001 - model loading raises broadly
            self._error = f"could not load wake model {self._model_name!r}: {exc}"
            return False, self._error

        return True, ""

    def detect(self, frame: np.ndarray) -> bool:
        if self._model is None:
            return False
        try:
            # openWakeWord expects int16 PCM.
            pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
            scores = self._model.predict(pcm)
        except Exception:  # noqa: BLE001 - must not kill the audio thread
            log.exception("wake word prediction failed")
            return False
        return any(score >= self._threshold for score in scores.values())

    def reset(self) -> None:
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:  # noqa: BLE001
                log.debug("wake model reset failed", exc_info=True)


def build_detector(cfg: WakeConfig) -> tuple[Detector, str]:
    """Return the best available detector plus a note about which was chosen."""
    oww = OpenWakeWordDetector(cfg.model, cfg.threshold)
    ok, problem = oww.load()
    if ok:
        return oww, f"openWakeWord ({cfg.model})"
    log.warning("falling back to energy detector: %s", problem)
    return EnergyDetector(), f"energy fallback ({problem})"


# ---------------------------------------------------------------------------
# Capture state machine
# ---------------------------------------------------------------------------


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"


@dataclass
class CaptureMachine:
    """Decides when an utterance starts and stops.

    Pure logic: it is fed frames plus a monotonic timestamp and returns whether
    an utterance is complete. No audio I/O, no threads, fully testable.
    """

    cfg: WakeConfig
    speech_threshold: float = 0.02
    state: State = State.IDLE
    _speech_last_seen: float = field(default=0.0, repr=False)
    _heard_speech: bool = field(default=False, repr=False)
    _started_at: float = field(default=0.0, repr=False)
    _refractory_until: float = field(default=0.0, repr=False)

    def reset(self, now: float | None = None) -> None:
        """Return to IDLE.

        ``now`` opens a refractory window. Calling ``reset()`` on the detector
        was **not** sufficient in practice: openWakeWord keeps roughly a second
        of internal audio context, so the tail of the just-captured utterance
        could immediately re-trigger it. Observed live as a second wake 0.245s
        after a capture completed, which then recorded silence for 25s and
        blocked the pipeline. A short deaf window is the reliable fix.
        """
        self.state = State.IDLE
        self._heard_speech = False
        self._speech_last_seen = 0.0
        self._started_at = 0.0
        if now is not None:
            self._refractory_until = now + self.cfg.refractory_s

    def accepts_wake(self, now: float) -> bool:
        """False while still inside the post-capture refractory window."""
        return now >= self._refractory_until

    def on_wake(self, now: float) -> None:
        self.state = State.LISTENING
        self._started_at = now
        self._speech_last_seen = now
        self._heard_speech = False

    def feed(self, frame: np.ndarray, now: float) -> bool:
        """Feed one frame while LISTENING. Returns True when the utterance ends.

        Ends on either trailing silence or the hard maximum duration. The
        maximum matters: without it, a stuck-open or noisy microphone would
        capture forever and never dispatch.
        """
        if self.state is not State.LISTENING:
            return False

        if rms(frame) >= self.speech_threshold:
            self._speech_last_seen = now
            self._heard_speech = True

        if now - self._started_at >= self.cfg.max_utterance_s:
            log.info("utterance hit max duration")
            return True

        # Before any speech has been heard, wait `lead_in_s` rather than
        # `silence_s`. A person who says the wake word and then pauses -- to
        # think, or to check that an interruption actually worked -- would
        # otherwise have the capture close on the wake word alone, which is
        # then thrown away as too short to be a real utterance. Observed in
        # the field: an interrupt succeeded but the follow-up question was
        # never captured.
        limit = self.cfg.silence_s if self._heard_speech else self.cfg.lead_in_s
        if now - self._speech_last_seen >= limit:
            if not self._heard_speech:
                log.info("no speech within lead-in; closing capture")
            return True

        return False

    def duration(self, now: float) -> float:
        if self.state is not State.LISTENING:
            return 0.0
        return max(0.0, now - self._started_at)
