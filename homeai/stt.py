# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Speech-to-text via whisper.cpp.

Runs the ``whisper-cli`` binary as a subprocess. A subprocess boundary is used
deliberately rather than Python bindings: it keeps the heavy C++ dependency out
of the Python process, and a hung or crashed transcription cannot take the
voice service down with it.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SttConfig

log = logging.getLogger(__name__)

# whisper.cpp emits bracketed annotations for non-speech audio. These are not
# words and must not be forwarded to the agent.
_NON_SPEECH = re.compile(r"[\(\[][^)\]]*[\)\]]")


@dataclass(frozen=True)
class Transcript:
    ok: bool
    text: str = ""
    error: str = ""


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono int16 PCM. Accepts float32 in [-1, 1] or int16."""
    if audio.dtype != np.int16:
        clipped = np.clip(audio, -1.0, 1.0)
        audio = (clipped * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


def clean_transcript(raw: str) -> str:
    """Strip whisper artefacts and normalise whitespace."""
    if not raw:
        return ""
    without_annotations = _NON_SPEECH.sub(" ", raw)
    collapsed = re.sub(r"\s+", " ", without_annotations).strip()
    # A transcript of only punctuation carries no instruction.
    if not re.search(r"[A-Za-z0-9]", collapsed):
        return ""
    return collapsed


class Transcriber:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg

    def available(self) -> tuple[bool, str]:
        if not self._cfg.binary.exists():
            return False, f"whisper binary not found: {self._cfg.binary}"
        if not self._cfg.model.exists():
            return False, f"whisper model not found: {self._cfg.model}"
        return True, ""

    def transcribe_file(self, wav_path: Path) -> Transcript:
        ok, problem = self.available()
        if not ok:
            return Transcript(ok=False, error=problem)

        cmd = [
            str(self._cfg.binary),
            "-m", str(self._cfg.model),
            "-f", str(wav_path),
            "-t", str(self._cfg.threads),
            "-nt",          # no timestamps
            "--no-prints",  # keep stdout to the transcript alone
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._cfg.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return Transcript(ok=False, error=f"whisper timed out after {self._cfg.timeout_s}s")
        except OSError as exc:
            return Transcript(ok=False, error=f"failed to launch whisper: {exc}")

        if proc.returncode != 0:
            detail = (proc.stderr or "").strip().splitlines()
            tail = detail[-1] if detail else "no stderr"
            return Transcript(ok=False, error=f"whisper exited {proc.returncode}: {tail}")

        text = clean_transcript(proc.stdout)
        if len(text) < self._cfg.min_chars:
            return Transcript(ok=False, error="transcript below minimum length (likely noise)")

        return Transcript(ok=True, text=text)

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        """Transcribe an in-memory buffer by staging it to a temporary WAV."""
        if audio is None or audio.size == 0:
            return Transcript(ok=False, error="empty audio buffer")

        with tempfile.TemporaryDirectory(prefix="homeai-stt-") as tmp:
            wav_path = Path(tmp) / "utterance.wav"
            try:
                write_wav(wav_path, audio, sample_rate)
            except (OSError, ValueError) as exc:
                return Transcript(ok=False, error=f"failed to write wav: {exc}")
            return self.transcribe_file(wav_path)
