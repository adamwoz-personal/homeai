"""Interactive mode step 0 (plans/INTERACTIVE_MODE_PLAN.md): keep every stretch of speech the mic hears.

Can Jarvis transcribe a room continuously, beside voice turns and art jobs, and how well does it hear people
across the room? Jarvis holds the webcam mic exclusively (raw ALSA hw), so no second process can listen: this
thread taps Jarvis's own ring buffer with its own cursor, cuts it into speech segments by loudness, and saves
each one (16-bit wav + json) for ``tools/bench_interactive_listen.py`` to transcribe and time. Audio the mic
drops while Jarvis talks (the mic is paused then) is never seen here.

On only until ``until`` (a date, set by Adam); every clip is deleted ``KEEP_DAYS`` after it was recorded
(Adam 2026-11-13: audio 30 days). Never blocks the mic: it only reads the buffer. Pure file handling.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from homeai import art_clips
from homeai.audio import rms

log = logging.getLogger("homeai.listen_probe")

DIR = Path.home() / ".local/share/homeai/listen-probe"
KEEP_DAYS = 30
PRUNE_EVERY_S = 6 * 3600
POLL_S = 0.25


@dataclass
class Segmenter:
    """Cuts a stream of frames into speech segments by loudness (the wake capture's threshold)."""

    sample_rate: int = 16000
    threshold: float = 0.02         # homeai/wake.py speech_threshold
    end_silence_s: float = 1.0      # a pause this long ends the segment
    min_speech_s: float = 0.4       # shorter loud bits (a cough, a door) are dropped
    max_s: float = 30.0             # long talk is cut into pieces
    pre_roll_s: float = 0.3         # keep the start of the first word
    _frames: list = field(default_factory=list)
    _pre: list = field(default_factory=list)
    _start: float = 0.0
    _last_loud: float = 0.0
    _loud_s: float = 0.0

    def feed(self, frame: np.ndarray, now: float) -> tuple[np.ndarray, float, float] | None:
        """``now`` is the time at the frame's end. Returns (audio, start, end) when a segment closes."""
        dur = len(frame) / self.sample_rate
        loud = rms(frame) >= self.threshold
        if not self._frames:
            if not loud:
                self._pre.append(frame)
                while sum(len(f) for f in self._pre) / self.sample_rate > self.pre_roll_s:
                    self._pre.pop(0)
                return None
            pre = sum(len(f) for f in self._pre) / self.sample_rate
            self._frames, self._pre = [*self._pre, frame], []
            self._start, self._last_loud, self._loud_s = now - dur - pre, now, dur
            return None
        self._frames.append(frame)
        if loud:
            self._last_loud, self._loud_s = now, self._loud_s + dur
        if now - self._last_loud >= self.end_silence_s or now - self._start >= self.max_s:
            audio, start, enough = np.concatenate(self._frames), self._start, self._loud_s >= self.min_speech_s
            self._frames = []
            return (audio, start, now) if enough else None
        return None


class ListenProbe:
    """Reads the mic ring buffer on its own thread and saves speech segments until ``until``."""

    def __init__(self, buffer, sample_rate: int, until: str, folder: Path | None = None,
                 state: Callable[[], dict] = dict, clock: Callable[[], float] = time.time,
                 frame: int = 1280) -> None:
        self.buffer, self.sample_rate, self.until = buffer, sample_rate, until
        self.folder = folder or DIR
        self.state, self.clock, self.frame = state, clock, frame
        self.segmenter = Segmenter(sample_rate)
        self.saved = 0
        self._cursor = buffer.total_written
        self._pending = np.zeros(0, dtype=np.float32)
        self._next_prune = 0.0

    def on(self) -> bool:
        return art_clips.recording_on(self.until)

    def step(self) -> int:
        """Process whatever is new in the buffer; returns segments saved."""
        new, self._cursor = self.buffer.read_new(self._cursor)
        audio = np.concatenate((self._pending, new))
        now = self.clock()
        saved = 0
        whole = len(audio) // self.frame * self.frame
        for i in range(0, whole, self.frame):
            end = now - (len(audio) - i - self.frame) / self.sample_rate
            seg = self.segmenter.feed(audio[i:i + self.frame], end)
            if seg is not None:
                self._save(*seg)
                saved += 1
        self._pending = audio[whole:]
        if now >= self._next_prune:
            self._next_prune = now + PRUNE_EVERY_S
            gone = art_clips.prune(self.folder, now, KEEP_DAYS)
            if gone:
                log.info("listen probe: deleted %d old files", gone)
        return saved

    def _save(self, audio: np.ndarray, start: float, end: float) -> None:
        try:
            meta = {"start": start, "end": end, "duration": round(end - start, 2), **self.state()}
            art_clips.save(self.folder, audio, self.sample_rate, meta, now=end)
            self.saved += 1
        except Exception:  # noqa: BLE001 - a full disk must not touch the voice path
            log.warning("listen probe: saving a segment failed", exc_info=True)

    def run(self, stop: threading.Event, shown: Callable[[bool], None] = lambda on: None) -> None:
        """``shown(on)`` tells the avatar (white hair while recording)."""
        log.info("listen probe on until %s: speech segments to %s", self.until, self.folder)
        shown(True)
        while not stop.wait(POLL_S):
            if not self.on():
                log.info("listen probe: past %s, stopping (%d segments saved)", self.until, self.saved)
                shown(False)
                return
            try:
                self.step()
            except Exception:  # noqa: BLE001 - never take Jarvis down
                log.warning("listen probe step failed", exc_info=True)
