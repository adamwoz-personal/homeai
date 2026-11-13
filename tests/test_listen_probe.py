"""Interactive mode step 0: the listen probe (homeai/listen_probe.py), its worker and report (tools/)."""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pytest

from homeai import listen_probe as LP
from homeai.audio import RingBuffer
from homeai.stt import Transcript
from tools import bench_interactive_listen as W
from tools import report_interactive_listen as R

SR, FRAME = 16000, 1280            # 80 ms frames, as Jarvis's mic delivers them


def tone(seconds, level=0.1):
    return (np.sin(np.arange(int(SR * seconds)) / 5) * level * 1.41).astype(np.float32)


def quiet(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def feed_all(seg, audio, t0=100.0):
    out = []
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        got = seg.feed(audio[i:i + FRAME], t0 + (i + FRAME) / SR)
        if got:
            out.append(got)
    return out


def test_one_sentence_is_one_segment_with_its_start():
    got = feed_all(LP.Segmenter(SR), np.concatenate([quiet(1.0), tone(1.6), quiet(1.5)]))
    assert len(got) == 1
    audio, start, end = got[0]
    assert start == pytest.approx(100.0 + 1.0 - 0.3, abs=0.09)     # pre-roll keeps the first word's start
    assert end - start == pytest.approx(len(audio) / SR, abs=0.01)
    assert end == pytest.approx(100.0 + 1.0 + 1.6 + 1.0, abs=0.09)  # closed by a 1 s pause


def test_a_short_pause_keeps_one_segment():
    got = feed_all(LP.Segmenter(SR), np.concatenate([tone(1.0), quiet(0.6), tone(1.0), quiet(1.2)]))
    assert len(got) == 1


def test_silence_and_a_cough_are_not_speech():
    assert feed_all(LP.Segmenter(SR), quiet(5)) == []
    assert feed_all(LP.Segmenter(SR), np.concatenate([quiet(1), tone(0.16), quiet(2)])) == []


def test_long_talk_is_cut_at_the_maximum():
    got = feed_all(LP.Segmenter(SR, max_s=5.0), np.concatenate([tone(12), quiet(1.2)]))
    assert len(got) == 3 and all(e - s <= 5.1 for _, s, e in got[:2])


def probe(tmp_path, clock, until="2999-01-01"):
    buf = RingBuffer(SR * 30)
    return buf, LP.ListenProbe(buf, SR, until, folder=tmp_path / "lp", clock=clock,
                               state=lambda: {"jarvis_speaking": False, "art_active": True})


def test_probe_saves_speech_from_jarvis_s_buffer_with_what_jarvis_was_doing(tmp_path):
    now = [1_000_000.0]
    buf, p = probe(tmp_path, lambda: now[0])
    audio = np.concatenate([tone(1.0), quiet(1.5)])
    t0 = now[0]
    buf.write(audio[:SR // 2 + 77])     # arrives in pieces, not frame-aligned
    now[0] += (SR // 2 + 77) / SR
    p.step()
    buf.write(audio[SR // 2 + 77:])
    now[0] = t0 + len(audio) / SR
    assert p.step() == 1
    meta = json.loads(next((tmp_path / "lp").glob("*.json")).read_text())
    assert meta["art_active"] is True and meta["duration"] == pytest.approx(2.0, abs=0.09)  # speech + 1 s pause
    assert meta["start"] == pytest.approx(t0, abs=0.09) and len(list((tmp_path / "lp").glob("*.wav"))) == 1


def test_probe_starts_at_the_buffer_s_present_not_its_past(tmp_path):
    buf = RingBuffer(SR * 30)
    buf.write(np.concatenate([tone(1.0), quiet(1.5)]))           # before the probe existed
    p = LP.ListenProbe(buf, SR, "2999-01-01", folder=tmp_path / "lp", clock=lambda: 5.0)
    assert p.step() == 0


def test_probe_deletes_clips_older_than_30_days(tmp_path):
    folder = tmp_path / "lp"
    folder.mkdir()
    old, new = folder / "old.wav", folder / "new.wav"
    old.write_bytes(b"x"), new.write_bytes(b"x")
    now = time.time()
    os.utime(old, (now - 31 * 86400,) * 2)
    os.utime(new, (now - 29 * 86400,) * 2)
    buf, p = probe(tmp_path, lambda: now)
    p.step()
    assert not old.exists() and new.exists()


def test_probe_stops_after_its_date(tmp_path):
    buf, p = probe(tmp_path, time.time, until="2000-01-01")
    stop = threading.Event()
    t = threading.Thread(target=p.run, args=(stop,))
    t.start()
    t.join(2)
    alive = t.is_alive()
    stop.set()
    assert not alive


def test_a_failed_save_never_raises(tmp_path, monkeypatch):
    buf, p = probe(tmp_path, lambda: 1e6)
    monkeypatch.setattr(LP.art_clips, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    p._save(tone(1), 1.0, 2.0)
    assert p.saved == 0


# -- the worker ----------------------------------------------------------------

class FakeSTT:
    def transcribe_file(self, wav):
        return Transcript(ok=True, text="hello there")


def test_worker_transcribes_waiting_segments_once_and_times_them(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "comfy_busy", lambda: True)
    folder = tmp_path / "lp"
    LP.art_clips.save(folder, tone(1), SR, {"end": 1000.0, "duration": 1.0, "art_active": True}, now=1000.0)
    [path] = W.waiting(folder)
    times = iter([1000.5, 1001.25])
    row = W.transcribe(path, FakeSTT(), clock=lambda: next(times))
    assert row["text"] == "hello there" and row["lag_s"] == 1.25 and row["wall_s"] == 0.75
    assert row["rtf"] == 0.75 and row["comfy_busy"] is True and row["art_active"] is True
    assert json.loads(path.read_text())["stt"]["lag_s"] == 1.25
    assert W.waiting(folder) == []


def test_worker_skips_json_without_audio(tmp_path):
    (tmp_path / "x.json").write_text("{}")
    assert W.waiting(tmp_path) == []


# -- the report ----------------------------------------------------------------

def rows(lags, comfy):
    return [{"ok": True, "lag_s": lag, "rtf": 0.3, "duration": 3600 / len(lags), "comfy_busy": comfy} for lag in lags]


def test_report_counts_noise_apart_from_failures_and_not_as_speech():
    noise = {"ok": False, "error": "transcript below minimum length (likely noise)", "duration": 3600}
    broken = {"ok": False, "error": "whisper timed out after 30s", "duration": 3600}
    s = R.summary(rows([1, 2], False) + [noise, broken])
    assert (s["noise"], s["failed"], s["speech_h"]) == (1, 1, 1.0)


def test_worker_never_retries_a_clip_that_raised(tmp_path, monkeypatch, capsys):
    class Broken:
        def transcribe_file(self, wav):
            raise UnicodeDecodeError("utf-8", b"\xe2", 0, 1, "unexpected end of data")
    monkeypatch.setattr(W, "Transcriber", lambda cfg: Broken())
    LP.art_clips.save(tmp_path, tone(1), SR, {"end": 1000.0, "duration": 1.0}, now=1000.0)
    assert W.main(["--once", "--folder", str(tmp_path)]) == 0
    assert W.waiting(tmp_path) == []
    [path] = tmp_path.glob("*.json")
    assert "unexpected end" in json.loads(path.read_text())["stt"]["error"]


def test_report_says_go_only_when_lag_keeps_up_with_an_art_job():
    assert "GO" in R.report(rows([0.5] * 19 + [1.9], True)).splitlines()[-1]
    assert "NO-GO" in R.report(rows([0.5] * 18 + [3.0, 3.0], True))
    assert "no art-job data yet" in R.report(rows([0.5] * 5, False))


def test_report_counts_speech_hours_and_percentiles():
    s = R.summary(rows([1, 2, 3, 4], False))
    assert s["speech_h"] == 1.0 and s["lag_p50"] in (2, 3) and s["lag_max"] == 4


def test_probe_shows_recording_while_on_and_clears_it_at_its_date(tmp_path, monkeypatch):
    shown = []
    buf, p = probe(tmp_path, time.time)
    dates = iter([True, False])
    monkeypatch.setattr(p, "on", lambda: next(dates))
    monkeypatch.setattr(LP, "POLL_S", 0.01)
    p.run(threading.Event(), shown.append)
    assert shown == [True, False]


def test_recording_is_kept_for_clients_that_connect_later():
    from homeai.avatar.bus import COALESCED
    assert "recording" in COALESCED


def test_worker_exits_after_its_date_once_nothing_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "Transcriber", lambda cfg: FakeSTT())
    assert W.main(["--until", "2000-01-01", "--folder", str(tmp_path)]) == 0
