# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Recording family members saying a new wake phrase ("hey jeeves"; plans/WAKE_WORD_PLAN.md).

Why: five rounds of training on synthetic voices reached 57-83% recall on held-out voices (2026-11-24). Real
recordings of the people who will say it measure true recall and can join training. Jarvis holds the webcam
mic exclusively (raw ALSA), so no separate program can record: this runs inside Jarvis, like voice enrolment.
"Record my wake word" (or "record Ani's wake word") starts a session; Jarvis asks for the phrase in a few styles
(normal, quiet, from across the room, quickly, leading a question), then a few sound-alikes that must NOT wake
it. Each answer is one follow-up turn; its audio is saved as heard. "Stop" or no answer ends it; saying the
start phrase again carries on from the saved count.

Files: ~/.local/share/homeai/wake-recordings/<person>/{positive,negative}/<time>.wav (16 kHz mono 16-bit) plus
index.jsonl (asked, style, heard). Local only, never sent anywhere. Pure: no audio device, no TTS.
"""
from __future__ import annotations

import json
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DIR = Path.home() / ".local/share/homeai/wake-recordings"
PHRASE = "hey jeeves"
POSITIVES = 30                 # per session
MAX_POSITIVES = 100            # per person, over all sessions
STYLES = (("normal", "Say it normally."), ("quiet", "A bit quieter."), ("far", "From across the room."),
          ("fast", "Quickly, like you're busy."), ("request", "Say it, then ask me something, like the time."),
          ("normal", "Once more, normally."))
SOUND_ALIKES = ("hey jarvis", "hey jesus", "hey genie", "hey steve", "cheese please", "he leaves")
MIN_S, MAX_S = 0.3, 8.0
MIN_RMS = 0.003                # quieter than this is the room, not a voice
RETRIES = 2

_START = re.compile(r"\brecord\s+(?:my|the|(?P<name>[a-z]+)'?s)\s+wake\s*(?:word|phrase|name)"
                    r"(?:\s+for\s+(?P<for>[a-z]+))?"
                    r"|\brecord\s+(?:the\s+)?wake\s*(?:word|phrase)\s+for\s+(?P<for2>[a-z]+)", re.IGNORECASE)
_STOP = re.compile(r"^\W*(stop|done|enough|that's enough|finish|finished|cancel|quit|no more)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Start:
    name: str | None           # a name said in the request ("Ani's"), else None (use voice ID)


def parse_start(text: str) -> Start | None:
    m = _START.search(text or "")
    if not m:
        return None
    name = m.group("name") or m.group("for") or m.group("for2")
    return Start(name.capitalize() if name else None)


def is_stop(text: str) -> bool:
    return bool(_STOP.search(text or ""))


def usable(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """'' if the clip is a voice worth keeping, else why not."""
    seconds = len(audio) / sample_rate
    if seconds < MIN_S:
        return "too short"
    if seconds > MAX_S:
        return "too long"
    if float(np.sqrt(np.mean(np.square(audio.astype(np.float32))))) < MIN_RMS:
        return "too quiet"
    return ""


def saved(person: str, root: Path | None = None) -> int:
    return len(list(((root or DIR) / person.lower() / "positive").glob("*.wav")))


def say_safely(text: str) -> str:
    """The prompt as Jarvis speaks it. He never says his own wake phrase: the barge-in listener runs while he
    talks and could hear it."""
    return "hey, and then my name" if text == "hey jarvis" else text


def plan(already: int, phrase: str = PHRASE) -> list[tuple[str, str, str]]:
    """(kind, text to say, prompt) for one session: positives in rotating styles, then the sound-alikes."""
    n = max(0, min(POSITIVES, MAX_POSITIVES - already))
    items = [("positive", phrase, STYLES[(already + i) % len(STYLES)][1]) for i in range(n)]
    if n:
        items += [("negative", s, f"Now one that isn't the name. Say: {say_safely(s)}.") for s in SOUND_ALIKES]
    return items


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


@dataclass
class Session:
    person: str
    phrase: str = PHRASE
    root: Path | None = None   # None: DIR, read at start (tests point it elsewhere)
    items: list = field(default_factory=list)
    at: int = 0
    tries: int = 0
    kept: int = 0

    def __post_init__(self) -> None:
        self.root = self.root or DIR
        self.items = plan(saved(self.person, self.root), self.phrase)

    @property
    def done(self) -> bool:
        return self.at >= len(self.items)

    def opening(self) -> str:
        if not self.items:
            return f"I already have {MAX_POSITIVES} recordings of {self.person}. That's plenty."
        pos = sum(1 for k, _, _ in self.items if k == "positive")
        return (f"Recording {self.person} saying {self.phrase}. {pos} short ones, then a few that aren't the "
                f"name. After each prompt, say just what I ask. Say stop to finish. {self.items[0][2]}")

    def take(self, audio: np.ndarray, heard: str, sample_rate: int = 16000, now: float | None = None) -> str:
        """Keep this answer (if usable) and return what Jarvis says next; ``done`` is set after the last."""
        kind, text, prompt = self.items[self.at]
        why = usable(audio, sample_rate)
        if why:
            self.tries += 1
            if self.tries <= RETRIES:
                return f"That was {why}. Again: {prompt}"
            why_skip = "Skipping that one. "
        else:
            why_skip = ""
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{self.at:03d}"
            folder = self.root / self.person.lower()
            write_wav(folder / kind / f"{stamp}.wav", audio, sample_rate)
            with (folder / "index.jsonl").open("a") as f:
                f.write(json.dumps({"file": f"{kind}/{stamp}.wav", "kind": kind, "asked": text,
                                    "style": prompt, "heard": heard}) + "\n")
            self.kept += 1
        self.at += 1
        self.tries = 0
        if self.done:
            return f"{why_skip}Done, thank you. I kept {self.kept} recordings of {self.person}."
        nxt = self.items[self.at][2]
        pos = sum(1 for k, _, _ in self.items if k == "positive")
        count = f"{self.at} of {pos}. " if self.at < pos else ""
        return f"{why_skip}{count}{nxt}"

    def stopped(self) -> str:
        return f"Stopped. I kept {self.kept} recordings of {self.person}."
