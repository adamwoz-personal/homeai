# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Text-to-speech via Piper, played through ALSA.

Piper writes WAV to stdout and ``aplay`` consumes it from stdin, so speech
starts before synthesis finishes and there is no temporary file.

The ``speaking`` flag exists so the capture thread can mute the microphone
while audio is playing. Without that, the assistant hears its own voice,
retriggers the wake word, and can loop indefinitely. This is the interim
measure until PipeWire echo cancellation is configured.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import threading
from dataclasses import dataclass
from pathlib import Path

from .config import TtsConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechResult:
    ok: bool
    error: str = ""
    # Time from call to the first PCM sample reaching the sound card. This,
    # not total elapsed time, is the latency a listener actually perceives:
    # everything after it is Jarvis talking, which is not a delay.
    first_audio_ms: float = 0.0
    # Total wall clock, including the full spoken duration.
    total_ms: float = 0.0
    # True when playback was cut short by the stop callback. Distinguished
    # from failure: an interruption is the user getting what they wanted.
    interrupted: bool = False
    chunks_spoken: int = 0


class Speaker:
    def __init__(self, cfg: TtsConfig, output_device: str) -> None:
        self._cfg = cfg
        self._output_device = output_device
        self._lock = threading.Lock()
        self._speaking = threading.Event()

    @property
    def speaking(self) -> bool:
        """True while audio is being played. Read by the capture thread."""
        return self._speaking.is_set()

    def _piper_path(self) -> str | None:
        """Prefer the configured (venv) binary; fall back to PATH."""
        if self._cfg.binary.exists():
            return str(self._cfg.binary)
        return shutil.which("piper")

    def available(self) -> tuple[bool, str]:
        if not self._piper_path():
            return False, f"piper executable not found: {self._cfg.binary}"
        if shutil.which("aplay") is None:
            return False, "aplay not found (install alsa-utils)"
        if not self._cfg.model_path.exists():
            return False, f"piper voice model missing: {self._cfg.model_path}"
        return True, ""

    def say(self, text: str) -> SpeechResult:
        """Synthesise and play ``text``. Blocks until playback completes."""
        if not text or not text.strip():
            return SpeechResult(ok=False, error="nothing to say")

        ok, problem = self.available()
        if not ok:
            return SpeechResult(ok=False, error=problem)

        # One utterance at a time; overlapping speech is unintelligible.
        with self._lock:
            self._speaking.set()
            try:
                return self._run_pipeline(text)
            finally:
                self._speaking.clear()

    def _run_pipeline(self, text: str) -> SpeechResult:
        started = time.monotonic()
        # '--output-raw' streams headerless PCM. The alternative, writing a WAV
        # to stdout, fails because a WAV header needs a length that is not
        # known until synthesis finishes and stdout cannot be seeked.
        piper_cmd = [
            self._piper_path(),
            "--model", str(self._cfg.model_path),
            "--output-raw",
        ]
        aplay_cmd = [
            "aplay", "-q",
            "-D", self._output_device,
            "-f", "S16_LE",
            "-r", str(self._cfg.sample_rate()),
            "-c", "1",
        ]

        piper = None
        aplay = None
        try:
            piper = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            aplay = subprocess.Popen(
                aplay_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()

            # Relay piper -> aplay in Python so the arrival of the first PCM
            # chunk can be timed. Chunks are forwarded immediately, so this
            # stays a streaming pipeline and does not buffer whole utterances.
            first_audio: list[float] = []

            def relay() -> None:
                try:
                    while True:
                        chunk = piper.stdout.read(4096)
                        if not chunk:
                            break
                        if not first_audio:
                            first_audio.append(time.monotonic())
                        aplay.stdin.write(chunk)
                        aplay.stdin.flush()
                except (OSError, BrokenPipeError, ValueError):
                    pass
                finally:
                    try:
                        aplay.stdin.close()
                    except (OSError, ValueError):
                        pass

            pump = threading.Thread(target=relay, name="tts-relay", daemon=True)
            pump.start()

            aplay.wait(timeout=self._cfg.timeout_s)
            pump.join(timeout=5)
            piper.wait(timeout=5)

            if first_audio:
                first_audio_ms = (first_audio[0] - started) * 1000.0
            else:
                first_audio_ms = 0.0

        except subprocess.TimeoutExpired:
            for proc in (aplay, piper):
                if proc and proc.poll() is None:
                    proc.kill()
            return SpeechResult(ok=False, error="tts pipeline timed out")
        except (OSError, BrokenPipeError) as exc:
            for proc in (aplay, piper):
                if proc and proc.poll() is None:
                    proc.kill()
            return SpeechResult(ok=False, error=f"tts pipeline failed: {exc}")

        if aplay.returncode != 0:
            detail = (aplay.stderr.read().decode(errors="replace").strip() if aplay.stderr else "")
            return SpeechResult(ok=False, error=f"aplay exited {aplay.returncode}: {detail}")

        return SpeechResult(
            ok=True,
            first_audio_ms=first_audio_ms,
            total_ms=(time.monotonic() - started) * 1000.0,
        )

    def say_safe(self, text: str) -> SpeechResult:
        """Speak, logging rather than raising. For use on failure paths where
        an exception would be worse than silence."""
        try:
            result = self.say(text)
        except Exception as exc:  # noqa: BLE001 - must never kill the daemon
            log.exception("unexpected TTS failure")
            return SpeechResult(ok=False, error=str(exc))
        if not result.ok:
            log.error("TTS failed: %s", result.error)
        return result


    def say_chunked(self, text: str, should_stop=None) -> SpeechResult:
        """Speak sentence by sentence, checking for an interrupt between chunks.

        Chunking is what makes a long answer interruptible at all: previously
        a 68-second reply was a single `aplay` call with no seam to stop at.

        ``should_stop`` is polled *between* chunks, never during. That keeps
        the check trivially correct -- no audio is playing at that moment, so
        there is nothing to race with and no echo for a microphone to hear.
        The cost is granularity: an interruption takes effect at the end of
        the current sentence rather than instantly.
        """
        from .speech import chunk_for_speech

        started = time.perf_counter()
        chunks = chunk_for_speech(text)
        if not chunks:
            return SpeechResult(ok=True, total_ms=0.0)

        first_audio_ms = 0.0
        spoken = 0

        for index, chunk in enumerate(chunks):
            # Checked before each chunk except the first: the first has not
            # produced any sound yet, so stopping there would be silence.
            if index and should_stop is not None:
                try:
                    if should_stop():
                        log.info("speech interrupted after %d of %d chunks",
                                 spoken, len(chunks))
                        return SpeechResult(
                            ok=True,
                            first_audio_ms=first_audio_ms,
                            total_ms=(time.perf_counter() - started) * 1000.0,
                            interrupted=True,
                            chunks_spoken=spoken,
                        )
                except Exception:  # noqa: BLE001 - a bad callback must not mute us
                    log.warning("stop callback raised; continuing", exc_info=True)

            result = self.say_safe(chunk)
            if not result.ok:
                # Partial speech beats silence, so report what was said.
                return SpeechResult(
                    ok=False,
                    error=result.error,
                    first_audio_ms=first_audio_ms,
                    total_ms=(time.perf_counter() - started) * 1000.0,
                    chunks_spoken=spoken,
                )

            spoken += 1
            if index == 0:
                first_audio_ms = result.first_audio_ms

        return SpeechResult(
            ok=True,
            first_audio_ms=first_audio_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
            chunks_spoken=spoken,
        )
