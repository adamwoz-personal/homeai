# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""In-process Piper: load the voice once, synthesise PCM chunks with phoneme timing.

The piper CLI reloads the voice for every utterance (~0.5 s before the first
sample, measured by tools/avatar/tts_latency_bench.py); keeping it loaded makes
first audio ~20 ms. Piper can also report per-phoneme sample counts, which the
avatar turns into a lip-sync timeline (homeai/avatar/timeline.py).

Alignments need the ``onnx`` package to patch the model in memory at load. If
that fails the voice loads without them and the avatar falls back to
loudness-driven lip sync.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterator

from .config import TtsConfig

log = logging.getLogger(__name__)

ENGINES = ("subprocess", "inprocess")


@dataclass(frozen=True)
class PcmChunk:
    pcm: bytes                                   # int16 mono
    sample_rate: int
    alignments: list[tuple[str, int]] | None     # (phoneme, samples) or None

    @property
    def num_samples(self) -> int:
        return len(self.pcm) // 2


def normalise_engine(name: str) -> str:
    """Validated engine name; unknown values fall back to subprocess with a warning."""
    value = (name or "").strip().lower()
    if value in ENGINES:
        return value
    log.warning("unknown HOMEAI_TTS_ENGINE %r; using subprocess", name)
    return "subprocess"


def _load_piper(model: str, alignments: bool):
    from piper import PiperVoice
    return PiperVoice.load(model, include_alignments=alignments) if alignments \
        else PiperVoice.load(model)


class PiperEngine:
    """Thread-safe lazy holder of one loaded PiperVoice."""

    def __init__(self, cfg: TtsConfig, loader: Callable[[str, bool], object] = _load_piper) -> None:
        self._cfg = cfg
        self._loader = loader
        self._voice = None
        self._aligned = False
        self._lock = threading.Lock()

    @property
    def aligned(self) -> bool:
        return self._aligned

    def available(self) -> tuple[bool, str]:
        if not self._cfg.model_path.exists():
            return False, f"piper voice model missing: {self._cfg.model_path}"
        if self._voice is None:
            try:
                import piper  # noqa: F401
            except ImportError as exc:
                return False, f"piper python package not importable: {exc}"
        return True, ""

    def load(self):
        """Load once; try with alignments, fall back to plain. Raises if neither loads."""
        with self._lock:
            if self._voice is not None:
                return self._voice
            model = str(self._cfg.model_path)
            try:
                self._voice = self._loader(model, True)
                self._aligned = True
            except Exception as exc:  # noqa: BLE001 - piper raises plain Exceptions
                log.warning("piper alignments unavailable (%s); avatar uses loudness lip sync", exc)
                self._voice = self._loader(model, False)
                self._aligned = False
            log.info("piper voice loaded in-process (alignments %s)", "on" if self._aligned else "off")
            return self._voice

    def synthesize(self, text: str) -> Iterator[PcmChunk]:
        """Yield one PcmChunk per Piper sentence."""
        voice = self.load()
        kwargs = {"include_alignments": True} if self._aligned else {}
        for chunk in voice.synthesize(text, **kwargs):
            aligns = None
            if self._aligned and chunk.phoneme_alignments:
                aligns = [(a.phoneme, int(a.num_samples)) for a in chunk.phoneme_alignments]
            yield PcmChunk(chunk.audio_int16_bytes, int(chunk.sample_rate), aligns)
