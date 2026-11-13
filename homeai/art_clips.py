"""Keep art-mode recordings for a week to find why the other-voice split eats words (ART_FEATURES_PLAN S1).

On 2026-11-12 the split (homeai/speaker_split.py) ran on 11 of ~24 of Adam's art sentences and mostly
dropped his own last word ("fighting a [dragon]", "cancel art [mode]"). Recordings weren't kept, so the
cause couldn't be measured. Adam chose (2026-11-13): split off, keep his art-mode recordings for a week,
deleted after, then fix and turn it back on after his look.

Only sessions owned by a registered adult are kept (not Ani's), only until ``until`` (a date), and every
clip is deleted ``KEEP_DAYS`` after it was recorded: on each save, at Jarvis start, and every few hours.
A clip may contain another household voice: that is the point of it. Pure file handling, no network.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("homeai.art_clips")

DIR = Path.home() / ".local/share/homeai/art-clips"
KEEP_DAYS = 7
PRUNE_EVERY_S = 6 * 3600


def recording_on(until: str, today: dt.date | None = None) -> bool:
    """True while today is on or before ``until`` (YYYY-MM-DD); empty or bad: off."""
    try:
        last = dt.date.fromisoformat(until.strip())
    except (AttributeError, ValueError):
        return False
    return (today or dt.date.today()) <= last


def save(folder: Path, audio: np.ndarray, sample_rate: int, meta: dict, now: float | None = None) -> Path:
    """<stamp>.wav (16-bit mono) and <stamp>.json beside it; returns the wav path."""
    now = time.time() if now is None else now
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
    path = folder / f"{stamp}.wav"
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
    path.with_suffix(".json").write_text(json.dumps({**meta, "recorded": now, "sample_rate": sample_rate}, indent=1))
    return path


def prune(folder: Path, now: float | None = None, days: float = KEEP_DAYS) -> int:
    """Delete clips (and their JSON) recorded more than ``days`` ago; returns how many files went."""
    now = time.time() if now is None else now
    if not folder.is_dir():
        return 0
    gone = 0
    for path in folder.iterdir():
        if path.suffix not in (".wav", ".json"):
            continue
        try:
            if now - path.stat().st_mtime > days * 86400:
                path.unlink()
                gone += 1
        except OSError as exc:
            log.warning("could not delete %s: %s", path, exc)
    return gone
