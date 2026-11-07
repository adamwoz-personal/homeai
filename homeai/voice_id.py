# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Speaker recognition as the daemon uses it: identify, enrol, voice commands.

homeai/speaker.py does the maths; this holds the state and wording, so the
daemon only has to call it. Spoken commands:

- "remember my voice as Adam", "register me", "create a profile for me"
                                -> asks for a name if none was given, then whether
                                   they'd rather talk or read a script shown on the
                                   avatar screen, then saves after ~8 s of speech
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
ENROL_MAX_ROUNDS = 6     # command, name, talk-or-read, then up to three answers
# An enrolment nobody finished is abandoned after this long.
ENROL_TIMEOUT_S = 120.0

MSG_ENROL_START = ("Okay, {name}. To learn your voice, talk to me for about ten seconds, "
                   "about anything at all. Go ahead.")
MSG_ENROL_MORE = "Thanks. A little more, please."
MSG_ENROL_DONE = "Got it. I'll know your voice now, {name}."
MSG_ENROL_FAILED = "Sorry, I didn't hear enough to learn your voice. We can try again later."
MSG_ENROL_NO_NAME = "Sure. What name should I remember you by?"
MSG_ENROL_HOW = ("Okay, {name}. Would you like to just talk for a bit, "
                 "or shall I put something on the screen for you to read?")
MSG_ENROL_TALK = "Great. Talk to me about anything at all, for about ten seconds. Go ahead."
MSG_ENROL_READ = "It's on the screen. Read it out loud when you're ready."
MSG_ENROL_READ_MORE = "Thanks. Keep reading, please."
MSG_ENROL_TAKEN = "I already know someone called {name}. Ask Adam to help with that one."
MSG_OFFER = "By the way, I don't know your voice yet. What's your name?"
MSG_OFFER_DONE = "Nice to meet you, {name}. I'll know your voice now."
MSG_OFFER_DECLINED = "No problem."
# Read aloud during enrolment: about 12 s, varied sounds, and the words people
# say to Jarvis anyway. Shown on the avatar screen (the "script" event).
ENROL_SCRIPT = ("Hey Jarvis, what's the weather like this afternoon? "
                "Please turn on the kitchen lights and play some quiet music. "
                "My favourite colours are blue, green and orange, "
                "and I like pizza, puzzles and long walks with the dog.")
MSG_ENROL_NAME_AGAIN = "Sorry, I didn't catch a name. Say hey Jarvis, remember my voice, to try again."
MSG_OFF = "Voice recognition is turned off."
MSG_WHO_KNOWN = "You sound like {name}."
MSG_WHO_UNSURE = "I'm not sure who you are."
MSG_WHO_NOBODY = "I don't know anyone's voice yet. Say: remember my voice as, and then your name."
MSG_FORGOTTEN = "Okay, {name}. I've forgotten your voice."
MSG_FORGET_UNKNOWN = "I don't recognise your voice, so there's nothing to forget."

_ENROL = re.compile(r"\b(?:remember|learn|memori[sz]e|save)\b.{0,20}\bmy voice\b", re.I)
_WORD = re.compile(r"[a-z][a-z'\-]*", re.I)
# Words that can sit between "my voice" and the name. Whisper hears "as
# Adam" as "is Adam" (live 2026-10-08), so this is generous on purpose.
_FILLER = {"as", "is", "it's", "its", "for", "under", "with", "to", "be", "i", "am", "i'm",
           "im", "my", "name", "name's", "this", "call", "me", "please", "jarvis", "and",
           "so", "that", "the", "of", "it", "s", "okay", "ok", "um", "uh", "oh", "yes",
           "yeah", "sure", "hi", "hey", "hello", "you", "can", "should", "remember"}
_WHO = re.compile(r"\b(?:who am i|do you know who i am|do you recogni[sz]e (?:me|my voice)|"
                  r"who(?:'s| is) (?:this|speaking|talking))\b", re.I)
_ENROL_MORE = re.compile(r"\b(?:register\b.{0,12}\b(?:me|my voice|myself)|"
                         r"(?:create|make|set up|setup|add)\b.{0,12}\b(?:voice )?profile|"
                         r"(?:learn|know) (?:who i am|my voice))\b", re.I)
_NAME_AFTER = re.compile(r"\b(?:as|called|named|i'?m|i am)\s+([a-z][a-z'\-]*)", re.I)
_READ = re.compile(r"\b(?:read|script|screen|something to read|show me|write|words)\b", re.I)
_TALK = re.compile(r"\b(?:talk|speak|chat|say|myself|my own|just talk)\b", re.I)
_FORGET = re.compile(r"\b(?:forget|delete|erase|remove)\b.{0,10}\bmy voice\b", re.I)
_NOT_NAMES = {"a", "an", "your", "him", "her", "well", "now", "voice", "boss", "here",
              "there", "what", "who", "not", "no", "don't", "know", "sorry"}


def extract_name(text: str) -> str | None:
    """The first name-like word in ``text``, skipping filler.

    "is Adam", "as Adam", "it's Adam", "my name is Adam", "Adam." -> "Adam".
    """
    for word in _WORD.findall(text):
        low = word.lower().strip("'-")
        if low in _FILLER:
            continue
        if low in _NOT_NAMES or len(low) < 2 or len(low) > 30:
            return None
        return low.capitalize()
    return None


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
        after = re.split(r"\bmy voice\b", text, maxsplit=1, flags=re.I)[-1]
        return Command("enrol", extract_name(after))
    if _ENROL_MORE.search(text):
        named = _NAME_AFTER.search(text)
        return Command("enrol", extract_name(named.group(1)) if named else None)
    return None


def enrol_method(text: str) -> str:
    """The answer to "talk, or read something?": "read" or "talk" (also when unclear: no reading needed)."""
    if _READ.search(text) and not _TALK.search(text):
        return "read"
    return "talk"


@dataclass
class Enrolment:
    """Speech collected so far for one person's new voice profile."""

    name: str | None     # None until asked for
    stage: str = "speech"  # "name" (asked for it), "how" (talk or read?), "speech" (collecting)
    method: str = "talk"   # "talk" or "read" (the script is on the avatar screen)
    offered: bool = False  # Jarvis offered: the speech was gathered in passing, then the name asked
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
        if self.registry.reload_if_changed():
            log.info("speaker registry reloaded: %s", ", ".join(self.registry.names()) or "nobody")
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
        profile = self.registry.add(enrolment.name, emb, count=len(enrolment.clips), replace=True,
                                    source="offered" if enrolment.offered else "asked")
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


@dataclass
class _Voice:
    """One unknown voice heard in passing: its clips and their mean embedding."""

    embeddings: list[np.ndarray] = field(default_factory=list)
    clips: list[np.ndarray] = field(default_factory=list)
    speech_s: float = 0.0
    last: float = 0.0
    asked: bool = False

    @property
    def centroid(self) -> np.ndarray:
        mean = np.mean(self.embeddings, axis=0)
        return mean / (np.linalg.norm(mean) or 1.0)


class StrangerPool:
    """Unknown voices that keep talking to Jarvis, so it can offer to learn them.

    Held in memory only and forgotten after ``max_age_s`` of silence: nothing is
    saved unless the person gives a name. A clip joins a voice when it is at
    least ``same`` similar to that voice's average. Only clearly unknown clips
    are pooled (the daemon checks the best known match is weak), so a known
    person on a bad day doesn't become a second profile. Each voice is asked
    once; ``ready`` says when one has given enough speech over enough turns.
    """

    def __init__(self, same: float = 0.5, need_s: float = ENROL_TARGET_SPEECH_S, min_turns: int = 3,
                 max_age_s: float = 1800.0, max_voices: int = 8, clock=time.monotonic) -> None:
        self.same, self.need_s, self.min_turns = same, need_s, min_turns
        self.max_age_s, self.max_voices, self._clock = max_age_s, max_voices, clock
        self.voices: list[_Voice] = []

    def add(self, emb: np.ndarray | None, audio: np.ndarray) -> _Voice | None:
        """Pool one clip; returns its voice when that voice is ready to be asked."""
        now = self._clock()
        self.voices = [v for v in self.voices if now - v.last < self.max_age_s]
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32)
        best = max(self.voices, key=lambda v: cosine(emb, v.centroid), default=None)
        if best is None or cosine(emb, best.centroid) < self.same:
            best = _Voice()
            self.voices.append(best)
            self.voices = self.voices[-self.max_voices:]
        best.embeddings.append(emb)
        best.clips.append(np.asarray(audio, dtype=np.float32))
        best.speech_s += speech_seconds(audio)
        best.last = now
        return best if self.ready(best) else None

    def ready(self, voice: _Voice) -> bool:
        return not voice.asked and len(voice.clips) >= self.min_turns and voice.speech_s >= self.need_s

    def matches(self, voice: _Voice, emb: np.ndarray | None) -> bool:
        return emb is not None and cosine(np.asarray(emb, dtype=np.float32), voice.centroid) >= self.same

    def take(self, voice: _Voice) -> Enrolment:
        """An offered enrolment holding this voice's speech, waiting for a name."""
        voice.asked = True
        enrolment = Enrolment(None, stage="name", offered=True)
        for clip in voice.clips:
            enrolment.add(clip)
        enrolment.rounds = 0
        return enrolment

