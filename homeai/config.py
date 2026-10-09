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
    # Style limit, unlike timeout_s. Replies longer than this many words are
    # cut at a sentence boundary and Jarvis offers to keep going; 110 words
    # is under forty seconds at Piper's ~3 words/second. 0 disables. See
    # homeai/dialogue.py split_for_budget.
    spoken_budget_words: int = field(
        default_factory=lambda: _env_int("HOMEAI_SPOKEN_BUDGET_WORDS", 110)
    )

    @property
    def model_path(self) -> Path:
        # Extra voices are downloaded into model_dir/voices/; the original
        # voice sits directly in model_dir. Prefer model_dir when both exist.
        direct = self.model_dir / f"{self.voice}.onnx"
        extra = self.model_dir / "voices" / f"{self.voice}.onnx"
        return extra if not direct.exists() and extra.exists() else direct

    @property
    def config_path(self) -> Path:
        return self.model_path.with_suffix(".onnx.json")

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


DEFAULT_STYLE_HINT = (
    "(Spoken reply. If I ask what you think, give your own view in the first "
    "sentence, then your reason; never say you have no opinion or that it is "
    "a complex topic. If you already gave your view, do not restate it: add "
    "something new, such as another reason, an objection, or what follows "
    "from it. Keep it under a hundred words.)"
)


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
    # One line of guidance attached to every spoken request. SOUL.md says the
    # same things, but it sits under ~8000 characters of ZeroClaw preamble
    # and an 8B model follows instructions near the question far better than
    # ones buried in a system prompt. Set HOMEAI_AGENT_STYLE_HINT=off to disable.
    style_hint: str = field(
        default_factory=lambda: _env_str("HOMEAI_AGENT_STYLE_HINT", DEFAULT_STYLE_HINT)
    )


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
    # Second-stage check: Whisper must hear the wake word in the audio around
    # the trigger, or the wake is ignored. See homeai/wake_verify.py.
    verify: bool = field(default_factory=lambda: _env_bool("HOMEAI_WAKE_VERIFY", True))
    # Audio kept from before the trigger for that check. "Hey Jarvis" takes
    # about 0.8 s; the detector fires near its end.
    preroll_s: float = field(default_factory=lambda: _env_float("HOMEAI_WAKE_PREROLL_S", 2.0))
    # Keep the mic open during replies so the wake word can interrupt them.
    # Safe because measurement showed ordinary speech peaks at 0.10 against a
    # 0.5 threshold; see homeai/bargein.py and tools/probe_bargein.py.
    barge_in: bool = field(default_factory=lambda: _env_bool("HOMEAI_BARGE_IN", True))
    # When a reply ends with a question, listen for the answer without
    # requiring the wake word. See homeai/dialogue.py.
    followup: bool = field(default_factory=lambda: _env_bool("HOMEAI_FOLLOWUP", True))
    # How long to wait for the answer to begin. Longer than lead_in_s: the
    # person was just asked something and may need a moment to think.
    followup_lead_in_s: float = field(
        default_factory=lambda: _env_float("HOMEAI_FOLLOWUP_LEAD_IN_S", 6.0)
    )
    # Maximum follow-up windows in a row before the wake word is required
    # again. Bounds a loop where background audio keeps "answering" Jarvis.
    followup_max_chain: int = field(
        default_factory=lambda: _env_int("HOMEAI_FOLLOWUP_MAX_CHAIN", 3)
    )


@dataclass(frozen=True)
class SpeakerConfig:
    """Who is speaking. See homeai/speaker.py and plans/VOICE_ID_PLAN.md."""

    enabled: bool = field(default_factory=lambda: _env_bool("HOMEAI_SPEAKER_ID", False))
    model_path: Path = field(
        default_factory=lambda: Path(_env_str(
            "HOMEAI_SPEAKER_MODEL", str(VENDOR / "speaker" / "wespeaker_en_voxceleb_resnet34_LM.onnx")
        ))
    )
    registry_path: Path = field(
        default_factory=lambda: Path(_env_str(
            "HOMEAI_SPEAKER_REGISTRY", os.path.expanduser("~/.config/homeai/speakers.json")
        ))
    )
    # Cosine similarity needed to name someone, and how far ahead of the
    # next-best profile the match must be. Tuned with tools/bench_speaker_separation.py.
    threshold: float = field(default_factory=lambda: _env_float("HOMEAI_SPEAKER_THRESHOLD", 0.45))
    margin: float = field(default_factory=lambda: _env_float("HOMEAI_SPEAKER_MARGIN", 0.1))
    # Follow-ups (no wake word) from a voice that is confidently NOT the
    # person in the conversation are ignored: family chatter is not an answer.
    followup_same_speaker: bool = field(
        default_factory=lambda: _env_bool("HOMEAI_SPEAKER_FOLLOWUP_GATE", True)
    )
    # Similarity below which a follow-up voice counts as someone else.
    different_below: float = field(
        default_factory=lambda: _env_float("HOMEAI_SPEAKER_DIFFERENT_BELOW", 0.25)
    )


@dataclass(frozen=True)
class HomeConfig:
    """Home control through Home Assistant (plans/HOME_ASSISTANT_PLAN.md).

    On whenever the HA credentials file exists; HOMEAI_HOME=0 turns it off.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("HOMEAI_HOME", True))
    env_path: Path = field(
        default_factory=lambda: Path(_env_str(
            "HOMEAI_HA_ENV", os.path.expanduser("~/.config/homeai/ha.env")
        ))
    )
    # Echo used for timers, music and unknown lights when no room is named.
    default_echo: str = field(default_factory=lambda: _env_str("HOMEAI_HA_DEFAULT_ECHO", "kitchen"))
    # Recognise everyday commands without the model (homeai/home_intents.py).
    fast_path: bool = field(default_factory=lambda: _env_bool("HOMEAI_HOME_FASTPATH", True))


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    transcript: TranscriptConfig = field(default_factory=TranscriptConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    home: HomeConfig = field(default_factory=HomeConfig)

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
        # Only the HTTP transport authenticates with a token; the default CLI
        # transport runs the agent directly. Requiring it unconditionally made
        # a fresh install (no gateway pairing) refuse to start.
        if self.agent.transport == "http" and not self.agent.token:
            problems.append(
                "HOMEAI_AGENT_TOKEN is unset (needed by the http transport) - obtain one "
                "via 'zeroclaw gateway get-paircode', or use HOMEAI_AGENT_TRANSPORT=cli"
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
