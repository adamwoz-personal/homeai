# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Central configuration for the home AI voice stack.

All hardware identifiers here were verified on the target host. Override any
value via environment variables so that tests and alternate machines do not
require editing code.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .hardware import detect_output_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR = PROJECT_ROOT / "vendor"


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AudioConfig:
    """Capture and playback settings.

    16 kHz mono is not arbitrary: it is the only format the USB webcam mic
    offers, and it is exactly what Whisper and openWakeWord expect. Keeping it
    means no resampling anywhere in the pipeline.
    """

    sample_rate: int = field(default_factory=lambda: _env_int("HOMEAI_SAMPLE_RATE", 16000))
    channels: int = field(default_factory=lambda: _env_int("HOMEAI_CHANNELS", 1))
    block_size: int = field(default_factory=lambda: _env_int("HOMEAI_BLOCK_SIZE", 1280))
    ring_seconds: int = field(default_factory=lambda: _env_int("HOMEAI_RING_SECONDS", 30))

    # Substring matched against PortAudio device names. Safer than a numeric
    # index, which shifts when USB devices are re-enumerated.
    #
    # PORTABILITY: "HD WEBCAM" is the development box's microphone. On another
    # machine set HOMEAI_INPUT_MATCH to any distinctive substring of the
    # desired mic's name; run `python -m homeai.hardware` to list candidates.
    input_match: str = field(default_factory=lambda: _env_str("HOMEAI_INPUT_MATCH", "HD WEBCAM"))

    # Detected, not hard-coded: ALSA card numbers differ per machine, and a
    # wrong guess here plays audio into a powered-off HDMI monitor while every
    # layer reports success. See homeai/hardware.py for the ranking rationale.
    # HOMEAI_OUTPUT_DEVICE always overrides detection.
    output_device: str = field(
        default_factory=lambda: _env_str("HOMEAI_OUTPUT_DEVICE", "")
        or detect_output_device()
    )

    @property
    def ring_frames(self) -> int:
        return self.sample_rate * self.ring_seconds


@dataclass(frozen=True)
class SttConfig:
    binary: Path = field(
        default_factory=lambda: Path(
            _env_str("HOMEAI_WHISPER_BIN", str(VENDOR / "whisper.cpp/build-blas/bin/whisper-cli"))
        )
    )
    model: Path = field(
        default_factory=lambda: Path(
            _env_str("HOMEAI_WHISPER_MODEL", str(VENDOR / "whisper.cpp/models/ggml-small.en.bin"))
        )
    )
    threads: int = field(default_factory=lambda: _env_int("HOMEAI_STT_THREADS", 8))
    timeout_s: float = field(default_factory=lambda: _env_float("HOMEAI_STT_TIMEOUT", 30.0))
    # Transcripts shorter than this are treated as noise and discarded.
    min_chars: int = field(default_factory=lambda: _env_int("HOMEAI_STT_MIN_CHARS", 2))


@dataclass(frozen=True)
class TtsConfig:
    voice: str = field(default_factory=lambda: _env_str("HOMEAI_PIPER_VOICE", "en_US-lessac-medium"))
    model_dir: Path = field(
        default_factory=lambda: Path(_env_str("HOMEAI_PIPER_DIR", str(VENDOR / "piper")))
    )
    # Default to the interpreter's own bin directory so the venv copy is found
    # without requiring the venv to be activated (systemd will not activate it).
    binary: Path = field(
        default_factory=lambda: Path(
            _env_str("HOMEAI_PIPER_BIN", str(Path(sys.executable).parent / "piper"))
        )
    )
    # Generous, because this bounds the *spoken duration*, not synthesis. At
    # roughly 150 words per minute a 30s cap would truncate any answer longer
    # than ~75 words mid-sentence. This is a runaway guard, not a style limit.
    timeout_s: float = field(default_factory=lambda: _env_float("HOMEAI_TTS_TIMEOUT", 300.0))

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.voice}.onnx"

    @property
    def config_path(self) -> Path:
        return self.model_dir / f"{self.voice}.onnx.json"

    def sample_rate(self, default: int = 22050) -> int:
        """Read the voice's native sample rate from its sidecar config.

        Piper streams headerless PCM with ``--output-raw``, so aplay must be
        told the rate explicitly. Reading it from the model avoids a hardcoded
        value silently becoming wrong when the voice is changed.
        """
        try:
            with open(self.config_path, encoding="utf-8") as handle:
                data = json.load(handle)
            rate = int(data.get("audio", {}).get("sample_rate", default))
            return rate if rate > 0 else default
        except (OSError, ValueError, TypeError):
            return default


@dataclass(frozen=True)
class AgentConfig:
    """ZeroClaw gateway connection.

    The token is read from the environment only. It must never be written into
    source control or into this file.
    """

    url: str = field(
        default_factory=lambda: _env_str("HOMEAI_AGENT_URL", "http://127.0.0.1:42617/webhook")
    )
    health_url: str = field(
        default_factory=lambda: _env_str("HOMEAI_AGENT_HEALTH", "http://127.0.0.1:42617/health")
    )
    token: str = field(default_factory=lambda: _env_str("HOMEAI_AGENT_TOKEN", ""))
    timeout_s: float = field(default_factory=lambda: _env_float("HOMEAI_AGENT_TIMEOUT", 45.0))
    max_retries: int = field(default_factory=lambda: _env_int("HOMEAI_AGENT_RETRIES", 2))

    # Transport: "cli" (default) or "http".
    #
    # "cli" is the default because the gateway webhook was measured NOT to
    # enforce the target agent's risk profile, while the CLI does. The voice
    # path is reachable by anyone in earshot, so it must run restricted.
    # The CLI is also ~10x faster (~0.5s vs ~5.2s) because the restricted
    # profile loads ~3 tools instead of 62.
    transport: str = field(default_factory=lambda: _env_str("HOMEAI_AGENT_TRANSPORT", "cli"))
    cli_binary: str = field(
        default_factory=lambda: _env_str(
            "HOMEAI_ZEROCLAW_BIN", os.path.expanduser("~/.cargo/bin/zeroclaw")
        )
    )
    # The ZeroClaw agent to address. Must be bound to a restricted risk
    # profile in ~/.zeroclaw/config.toml.
    agent_name: str = field(default_factory=lambda: _env_str("HOMEAI_AGENT_NAME", "local"))


@dataclass(frozen=True)
class TranscriptConfig:
    """Append-only audit log of every interaction (plan item 5A#5)."""

    enabled: bool = field(
        default_factory=lambda: _env_str("HOMEAI_TRANSCRIPT_ENABLED", "1") not in ("0", "false", "no")
    )
    path: str = field(
        default_factory=lambda: _env_str(
            "HOMEAI_TRANSCRIPT_PATH", os.path.expanduser("~/.local/share/homeai/transcript.jsonl")
        )
    )


@dataclass(frozen=True)
class WakeConfig:
    model: str = field(default_factory=lambda: _env_str("HOMEAI_WAKE_MODEL", "hey_jarvis"))
    threshold: float = field(default_factory=lambda: _env_float("HOMEAI_WAKE_THRESHOLD", 0.5))
    # Stop capturing after this much trailing silence.
    silence_s: float = field(default_factory=lambda: _env_float("HOMEAI_SILENCE_S", 0.7))
    # Grace period after the wake word, before any speech has been heard. Longer
    # than silence_s because a person often pauses between "Hey Jarvis" and
    # their actual question -- especially when interrupting a reply.
    lead_in_s: float = field(default_factory=lambda: _env_float("HOMEAI_LEAD_IN_S", 2.5))
    # Hard ceiling on one utterance, so a stuck-open mic cannot capture forever.
    max_utterance_s: float = field(default_factory=lambda: _env_float("HOMEAI_MAX_UTTERANCE_S", 15.0))
    # Deaf window after a capture ends. Stops the just-captured wake word from
    # re-triggering the detector out of its internal audio context.
    refractory_s: float = field(default_factory=lambda: _env_float("HOMEAI_REFRACTORY_S", 1.5))
    # Keep the mic open during replies so the wake word can interrupt them.
    # Safe because measurement showed ordinary speech peaks at 0.10 against a
    # 0.5 threshold; see homeai/bargein.py and tools/probe_bargein.py.
    barge_in: bool = field(default_factory=lambda: _env_bool("HOMEAI_BARGE_IN", True))


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    transcript: TranscriptConfig = field(default_factory=TranscriptConfig)

    def validation_errors(self) -> list[str]:
        """Return human-readable problems. Empty list means good to start.

        Deliberately returns all problems rather than raising on the first, so
        a first-run user sees everything they need to fix at once.
        """
        problems: list[str] = []
        if not self.stt.binary.exists():
            problems.append(f"whisper binary missing: {self.stt.binary}")
        if not self.stt.model.exists():
            problems.append(f"whisper model missing: {self.stt.model}")
        if not self.agent.token:
            problems.append(
                "HOMEAI_AGENT_TOKEN is unset - obtain one via 'zeroclaw gateway get-paircode'"
            )
        if self.audio.sample_rate != 16000:
            problems.append(
                f"sample_rate is {self.audio.sample_rate}; whisper and openWakeWord expect 16000"
            )
        if self.audio.channels != 1:
            problems.append(f"channels is {self.audio.channels}; expected mono (1)")
        return problems


def load() -> Config:
    return Config()
