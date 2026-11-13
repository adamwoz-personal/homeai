"""Art-mode recordings kept for a week to measure the other-voice split (homeai/art_clips.py; Adam 2026-11-13)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import wave

import numpy as np
import pytest

from homeai import art_clips as AC
from tests.test_daemon import assistant, quick, voiced  # noqa: F401 - fixtures
from tests.test_speaker_split import SR, _art_turn, mixed  # noqa: F401 - fixture


def test_recording_window():
    assert AC.recording_on("2026-11-20", dt.date(2026, 11, 20))
    assert not AC.recording_on("2026-11-20", dt.date(2026, 11, 21))
    assert not AC.recording_on("", dt.date(2026, 11, 13)) and not AC.recording_on("soon", dt.date(2026, 11, 13))


def test_save_writes_a_wav_and_its_json(tmp_path):
    audio = np.linspace(-1, 1, SR, dtype=np.float32)
    path = AC.save(tmp_path / "adam", audio, SR, {"heard": "cancel art mode"}, now=1_794_000_000.0)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, SR, SR)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["heard"] == "cancel art mode" and meta["sample_rate"] == SR and meta["recorded"] == 1_794_000_000.0


def test_prune_deletes_only_week_old_clips(tmp_path):
    old = AC.save(tmp_path, np.zeros(10, np.float32), SR, {}, now=1.0)
    new = AC.save(tmp_path, np.zeros(10, np.float32), SR, {}, now=2.0)
    now = 1_794_000_000.0
    for p in (old, old.with_suffix(".json")):
        os.utime(p, (now - 8 * 86400, now - 8 * 86400))
    for p in (new, new.with_suffix(".json")):
        os.utime(p, (now - 6 * 86400, now - 6 * 86400))
    (tmp_path / "notes.txt").write_text("x")
    os.utime(tmp_path / "notes.txt", (0, 0))
    assert AC.prune(tmp_path, now=now) == 2
    assert not old.exists() and new.exists() and (tmp_path / "notes.txt").exists()
    assert AC.prune(tmp_path / "missing", now=now) == 0


def _clips(q):
    return sorted((AC.DIR / "adam").glob("*.json")) if (AC.DIR / "adam").is_dir() else []


def _with_art(q, **changes):
    q.cfg = dataclasses.replace(q.cfg, art=dataclasses.replace(q.cfg.art, **changes))


def test_an_adult_s_art_turn_is_kept(mixed):
    q, audio = mixed
    _with_art(q, keep_clips_until="2999-01-01", split=False)
    _art_turn(q, audio, "cancel art mode")
    clips = _clips(q)
    meta = json.loads(clips[0].read_text())
    assert len(clips) == 1 and meta["heard"] == "cancel art mode" and meta["owner"] == "Adam"
    assert meta["split_on"] is False and clips[0].with_suffix(".wav").exists()


def test_no_clips_after_the_date(mixed):
    q, audio = mixed
    _with_art(q, keep_clips_until="2000-01-01")
    _art_turn(q, audio, "again")
    assert _clips(q) == []


def test_never_a_child_s_session(mixed):
    q, audio = mixed
    _with_art(q, keep_clips_until="2999-01-01")
    q.art.owner = "Ani"
    _art_turn(q, audio, "again")
    assert not AC.DIR.exists() or not any(AC.DIR.rglob("*.wav"))


def test_a_clip_failure_does_not_stop_the_turn(mixed, monkeypatch):
    q, audio = mixed
    _with_art(q, keep_clips_until="2999-01-01", split=False)
    monkeypatch.setattr(AC, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert _art_turn(q, audio, "again")
    assert q.art.seen == [("again", False)]


def test_the_prune_thread_runs_until_stop(mixed, monkeypatch):
    q, _ = mixed
    calls = []
    folder = AC.DIR / "adam"
    folder.mkdir(parents=True)
    monkeypatch.setattr(AC, "prune", lambda f, *a, **k: calls.append(f) or 0)
    waits = iter([False, True])
    monkeypatch.setattr(q._stop, "wait", lambda s: next(waits))
    q._prune_art_clips()
    assert calls == [folder, folder]
