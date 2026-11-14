"""tools/inspect_listen_timeline.py: buckets clips by start time; counts full-length (TV-like) clips and noise."""
from __future__ import annotations

from tools import inspect_listen_timeline as T


def test_buckets_count_full_clips_noise_and_words():
    rows = [{"start": 0, "duration": 30.0, "stt": {"ok": True, "text": "news at ten"}},
            {"start": 100, "duration": 2.0, "stt": {"ok": False}},
            {"start": 700, "duration": 3.0, "stt": {"ok": True, "text": "hi"}},
            {"duration": 1.0}]
    first, second = T.timeline(rows, minutes=10)
    assert (first["clips"], first["full"], first["noise"], first["sound_s"]) == (2, 1, 1, 32)
    assert first["words"] == "news at ten" and second["at"] == 600 and second["full"] == 0


def test_label_appends_a_tv_window(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "LABELS", tmp_path / "tv.jsonl")
    T.main(["--label", "tv", "2026-11-13 18:45", "2026-11-13 20:00", "a", "show"])
    import json
    row = json.loads((tmp_path / "tv.jsonl").read_text())
    assert row["kind"] == "tv" and row["end"] - row["start"] == 75 * 60 and row["note"] == "a show"
