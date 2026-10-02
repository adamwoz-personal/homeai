# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Barge-in: let a person interrupt Jarvis mid-sentence by saying the wake word.

Why this is a separate module
-----------------------------
The daemon normally *pauses* the microphone while speaking, so Jarvis can never
transcribe its own voice. Barge-in requires the exact opposite -- the mic stays
open while audio is playing -- so the risk of self-triggering is real and has to
be handled deliberately rather than as a side effect of daemon control flow.

Measured basis for this design (``tools/probe_bargein.py``, real speakers and
real microphone, not a simulation):

* Ordinary spoken prose:               peak wake score 0.1026, 0 false triggers
* A reply containing "Hey ... Jarvis":  peak wake score 0.9953, 4 false triggers

So the open microphone is safe *except* when Jarvis itself utters the wake
word, which it does only when the reply text happens to contain those words.
Because we always know the text we are about to speak, that case is detectable
in advance and needs no acoustic echo cancellation -- see
``contains_wake_fragments``.

This matters: the alternative was switching the sound card to its ``pro-audio``
profile and re-plumbing desktop audio to get PipeWire echo cancellation, which
is invasive and risks breaking all audio on the machine.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger(__name__)


class _Detector(Protocol):
    def detect(self, frame: np.ndarray) -> bool: ...
    def reset(self) -> None: ...


def contains_wake_fragments(text: str, wake_word: str = "hey_jarvis") -> bool:
    """True if spoken ``text`` could plausibly trigger our own wake word.

    Deliberately conservative: it fires on any individual word of the wake
    phrase, not just the full phrase in order. The probe showed that "Hey,
    that reminds me..." followed a sentence later by "Jarvis is a name..."
    was enough to trigger detection four times, because the detector hears a
    rolling window and does not care about our punctuation.

    A false positive here costs one lost opportunity to interrupt. A false
    negative causes Jarvis to interrupt itself, which looks like a crash.
    The asymmetry justifies being blunt.
    """
    if not text:
        return False
    words = {w for w in re.split(r"[^a-z]+", wake_word.lower()) if len(w) > 2}
    if not words:
        return False
    spoken = set(re.split(r"[^a-z]+", text.lower()))
    return bool(words & spoken)


_DISMISSAL_FILLERS = frozenset({
    "please", "okay", "ok", "just", "now", "um", "uh", "er", "oh",
    "well", "actually", "really", "right", "yeah", "yep", "and", "a",
})

# Whole utterances that mean "be quiet", not "answer this". Matched exactly
# against the normalised utterance rather than by substring: "stop" must not
# swallow "stop and tell me about the moon", and "nothing" must not swallow
# "nothing rhymes with orange, does it". Failing to match costs one needless
# lookup, which is the behaviour we have today; matching too eagerly would
# silently discard a real question, which is much worse.
_DISMISSALS = frozenset({
    "stop", "stop it", "stop talking", "stop that", "stop speaking",
    "quiet", "be quiet", "quiet down", "hush", "shush", "silence",
    "shut up", "shut it", "shut up already",
    "never mind", "nevermind", "never mind then",
    "forget it", "forget about it", "forget i asked", "forget that",
    "enough", "thats enough", "enough already", "no more",
    "cancel", "cancel that", "abort", "stop the answer",
    "go away", "leave me alone",
    "im not talking to you", "i am not talking to you",
    "im not talking to you jarvis", "i wasnt talking to you",
    "i was not talking to you", "not talking to you",
    "wasnt talking to you", "im talking to someone else",
    "im not asking you", "i didnt ask you", "i wasnt asking you",
    "thats all", "that was not for you", "that wasnt for you",
    "not you", "sorry not you", "ignore that", "ignore me",
})


def is_dismissal(text: str, wake_word: str = "hey_jarvis") -> bool:
    """True if ``text`` is a request to be quiet rather than a question.

    After a barge-in the microphone stays open to capture a follow-up, which
    is usually a new question. But sometimes the interruption *is* the whole
    point -- the user wants the reply to stop, or Jarvis woke on a
    conversation that was never addressed to it. Sending "stop" to the agent
    answers a question nobody asked, costs a lookup, and produces yet more
    speech to interrupt.

    The wake word is stripped first, because a dismissal usually arrives
    attached to it ("Jarvis, stop"). Filler words are stripped too, so
    "okay, just stop please" reduces to "stop".
    """
    if not text:
        return False
    cleaned = text.lower().replace("'", "").replace("\u2019", "")
    words = [w for w in re.split(r"[^a-z]+", cleaned) if w]
    wake_tokens = {w for w in re.split(r"[^a-z]+", wake_word.lower()) if len(w) > 2}
    kept = [w for w in words if w not in wake_tokens and w not in _DISMISSAL_FILLERS]
    if not kept:
        # Only the wake word and filler. That is not a dismissal -- the user
        # may simply have been cut off mid-sentence -- so let it through.
        return False
    return " ".join(kept) in _DISMISSALS


class InterruptListener:
    """Watches the live microphone for the wake word while Jarvis is speaking.

    Runs its own detector instance, separate from the daemon's wake detector,
    so that interrupting never disturbs the state of normal wake detection.

    Usage::

        listener = InterruptListener(mic, build_detector_fn, block_size)
        if listener.start():
            try:
                tts.say_chunked(text, should_stop=listener.should_stop)
            finally:
                listener.stop()
    """

    def __init__(
        self,
        mic,
        detector_factory: Callable[[], tuple[_Detector, str]],
        block_size: int,
        *,
        consecutive: int = 1,
    ) -> None:
        self._mic = mic
        self._factory = detector_factory
        self._block = block_size
        self._consecutive = consecutive
        self._triggered = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._detector: _Detector | None = None

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()

    def should_stop(self) -> bool:
        """Callback handed to the speaker; polled between speech chunks."""
        return self._triggered.is_set()

    def start(self) -> bool:
        """Begin listening. Returns False if it could not start.

        A failure here must be non-fatal: the caller simply speaks without the
        ability to be interrupted, which is exactly today's behaviour.
        """
        try:
            detector, description = self._factory()
        except Exception:  # noqa: BLE001 - never break a reply over this
            log.warning("interrupt listener: detector unavailable", exc_info=True)
            return False

        # An energy-based fallback would fire on Jarvis's own speech instantly,
        # turning every reply into a self-interruption.
        if not hasattr(detector, "detect") or "fallback" in description.lower():
            log.info("interrupt listener disabled: %s is not usable here", description)
            return False

        self._detector = detector
        self._stop.clear()
        self._triggered.clear()
        self._thread = threading.Thread(
            target=self._run, name="interrupt-listener", daemon=True
        )
        self._thread.start()
        return True

    def _run(self) -> None:
        cursor = getattr(self._mic.buffer, "_total", 0)
        streak = 0
        while not self._stop.is_set():
            try:
                frames, cursor = self._mic.buffer.read_new(cursor, self._block * 8)
            except Exception:  # noqa: BLE001 - audio thread must not die
                log.warning("interrupt listener: read failed", exc_info=True)
                return

            if frames is None or len(frames) < self._block:
                self._stop.wait(0.02)
                continue

            for start in range(0, len(frames) - self._block + 1, self._block):
                if self._stop.is_set():
                    return
                block = frames[start : start + self._block]
                try:
                    hit = self._detector.detect(block)  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    log.warning("interrupt listener: detect failed", exc_info=True)
                    return
                if hit:
                    streak += 1
                    if streak >= self._consecutive:
                        log.info("barge-in: wake word heard during playback")
                        self._triggered.set()
                        return
                else:
                    streak = 0

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        if self._detector is not None:
            try:
                self._detector.reset()
            except Exception:  # noqa: BLE001
                log.debug("interrupt detector reset failed", exc_info=True)
            self._detector = None
