# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Speaker recognition as the daemon uses it: identify, enrol, voice commands.

homeai/speaker.py does the maths; this holds the state and wording, so the
daemon only has to call it. Spoken commands:

- "remember my voice as Adam"  -> Jarvis asks for ~10 s of speech, then saves
- "who am I?"                   -> says who it thinks is speaking
- "forget my voice"             -> removes the speaker it recognises
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .config import SpeakerConfig
from .speaker import (
    Identification,
    SpeakerEmbedder,
    SpeakerIdentifier,
    SpeakerRegistry,
    cosine,
    speech_seconds,
)

log = logging.getLogger(__name__)

# Total speech needed for a voice profile, across however many answers.
ENROL_TARGET_SPEECH_S = 8.0
# Answers to ask for before giving up.
ENROL_MAX_ROUNDS = 4
# An enrolment nobody finished is abandoned after this long.
ENROL_TIMEOUT_S = 120.0

MSG_ENROL_START = ("Okay, {name}. To learn your voice, talk to me for about ten seconds, "
                   "about anything at all. Go ahead.")
MSG_ENROL_MORE = "Thanks. A little more, please."
MSG_ENROL_DONE = "Got it. I'll know your voice now, {name}."
MSG_ENROL_FAILED = "Sorry, I didn't hear enough to learn your voice. We can try again later."
MSG_ENROL_NO_NAME = "Say: remember my voice as, and then your name."
MSG_OFF = "Voice recognition is turned off."
MSG_WHO_KNOWN = "You sound like {name}."
MSG_WHO_UNSURE = "I'm not sure who you are."
MSG_WHO_NOBODY = "I don't know anyone's voice yet. Say: remember my voice as, and then your name."
MSG_FORGOTTEN = "Okay, {name}. I've forgotten your voice."
MSG_FORGET_UNKNOWN = "I don't recognise your voice, so there's nothing to forget."

_ENROL = re.compile(r"\b(?:remember|learn|memori[sz]e|save)\b.{0,20}\bmy voice\b", re.I)
_NAME = re.compile(r"\b(?:as|i am|i'm|my name is|this is|call me|it's)\s+([a-z][a-z'\-]{1,30})", re.I)
_WHO = re.compile(r"\b(?:who am i|do you know who i am|do you recogni[sz]e (?:me|my voice)|"
                  r"who(?:'s| is) (?:this|speaking|talking))\b", re.I)
_FORGET = re.compile(r"\b(?:forget|delete|erase|remove)\b.{0,10}\bmy voice\b", re.I)
_NOT_NAMES = {"a", "an", "the", "my", "your", "me", "him", "her", "it", "well", "please", "now"}


@dataclass(frozen=True)
class Command:
    kind: str            # "enrol", "who", "forget"
    name: str | None = None


def parse_command(text: str) -> Command | None:
    """A voice-ID command in ``text``, or None if it is an ordinary request."""
    if _FORGET.search(text):
        return Command("forget")
    if _WHO.search(text):
        return Command("who")
    if _ENROL.search(text):
        m = _NAME.search(text.split("voice", 1)[-1]) or _NAME.search(text)
        name = m.group(1) if m and m.group(1).lower() not in _NOT_NAMES else None
        return Command("enrol", name.strip("'-").capitalize() if name else None)
    return None


@dataclass
class Enrolment:
    """Speech collected so far for one person's new voice profile."""

    name: str
    clips: list[np.ndarray] = field(default_factory=list)
    speech_s: float = 0.0
    rounds: int = 0
    started: float = field(default_factory=time.monotonic)

    def add(self, audio: np.ndarray) -> None:
        self.clips.append(np.asarray(audio, dtype=np.float32))
        self.speech_s += speech_seconds(audio)
        self.rounds += 1

    @property
    def done(self) -> bool:
        return self.speech_s >= ENROL_TARGET_SPEECH_S

    @property
    def gave_up(self) -> bool:
        return not self.done and self.rounds >= ENROL_MAX_ROUNDS

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.monotonic()) - self.started > ENROL_TIMEOUT_S

    def audio(self) -> np.ndarray:
        return np.concatenate(self.clips) if self.clips else np.zeros(0, dtype=np.float32)


class VoiceId:
    """Model, registry and identifier, loaded together."""

    def __init__(self, cfg: SpeakerConfig) -> None:
        self.cfg = cfg
        self.embedder = SpeakerEmbedder(cfg.model_path)
        self.registry = SpeakerRegistry(cfg.registry_path, self.embedder.model_name)
        self.identifier = SpeakerIdentifier(self.registry, cfg.threshold, cfg.margin)

    def load(self) -> tuple[bool, str]:
        return self.embedder.load()

    def identify(self, audio: np.ndarray) -> tuple[Identification, np.ndarray | None]:
        emb = self.embedder.embed(audio)
        return self.identifier.identify(emb), emb

    def similarity_to(self, name: str, emb: np.ndarray | None) -> float | None:
        profile = self.registry.get(name)
        if profile is None or emb is None:
            return None
        return cosine(emb, profile.embedding)

    def is_someone_else(self, owner: str | None, ident: Identification,
                        emb: np.ndarray | None) -> tuple[bool, float | None]:
        """Whether a wake-word-free turn confidently comes from another voice.

        Only says yes when it is sure: too little speech, or nobody known to
        compare with, is "can't tell", and that lets the turn through.
        """
        if not owner or not self.cfg.followup_same_speaker:
            return False, None
        sim = self.similarity_to(owner, emb)
        if ident.name and ident.name != owner:
            return True, sim
        return (sim is not None and sim < self.cfg.different_below), sim

    def finish_enrolment(self, enrolment: Enrolment):
        """Save the profile; returns it, or None if the audio was unusable."""
        emb = self.embedder.embed(enrolment.audio())
        if emb is None:
            return None
        profile = self.registry.add(enrolment.name, emb, count=len(enrolment.clips), replace=True)
        self.registry.save()
        return profile

    def forget(self, name: str) -> bool:
        removed = self.registry.remove(name)
        if removed:
            self.registry.save()
        return removed


def describe(ident: Identification) -> str:
    """For the log: "Adam (0.62, margin 0.40)" or "unknown (best Adam 0.31: below threshold)"."""
    if ident.name:
        return f"{ident.name} ({ident.score:.2f}, margin {ident.margin:.2f})"
    if ident.best:
        return f"unknown (best {ident.best} {ident.score:.2f}: {ident.reason})"
    return f"unknown ({ident.reason})"
