"""In-process Piper engine and the Speaker's in-process path (fake voice, fake aplay)."""

import os
import stat
import time
from types import SimpleNamespace

import pytest

from homeai.config import TtsConfig
from homeai.tts import SpeechResult, Speaker
from homeai.tts_engine import PcmChunk, PiperEngine, normalise_engine

RATE = 22050


@pytest.fixture
def voice_dir(tmp_path):
    (tmp_path / "v.onnx").write_bytes(b"onnx")
    (tmp_path / "v.onnx.json").write_text(f'{{"audio": {{"sample_rate": {RATE}}}}}')
    return tmp_path


def cfg_for(voice_dir, engine="inprocess", **kw):
    return TtsConfig(voice="v", model_dir=voice_dir, engine=engine, **kw)


class FakeEngine:
    def __init__(self, chunks=None, error=None, delay=0.0):
        self.chunks = chunks or []
        self.error = error
        self.delay = delay

    def available(self):
        return True, ""

    def load(self):
        return self

    def synthesize(self, text):
        for c in self.chunks:
            time.sleep(self.delay)
            yield c
        if self.error:
            raise self.error


def tone(n):
    return (b"\x00\x40" * n)


def make_speaker(voice_dir, engine, aplay_cmd, **kw):
    events = []
    sp = Speaker(cfg_for(voice_dir, **kw), "null", engine=engine,
                 publish=lambda t, d: events.append((t, d)))
    sp._aplay_cmd = lambda rate: aplay_cmd
    return sp, events


# ---- engine ---------------------------------------------------------------------

def test_normalise_engine():
    assert normalise_engine(" InProcess ") == "inprocess"
    assert normalise_engine("subprocess") == "subprocess"
    assert normalise_engine("bogus") == "subprocess"
    assert normalise_engine("") == "subprocess"


def test_engine_falls_back_without_alignments_and_loads_once(voice_dir):
    calls = []

    def loader(model, aligned):
        calls.append(aligned)
        if aligned:
            raise RuntimeError("no onnx")
        return "voice"

    eng = PiperEngine(cfg_for(voice_dir), loader=loader)
    assert eng.load() == "voice" and eng.load() == "voice"
    assert calls == [True, False]
    assert eng.aligned is False


def test_engine_load_failure_propagates(voice_dir):
    def loader(model, aligned):
        raise RuntimeError("broken model")

    with pytest.raises(RuntimeError, match="broken model"):
        PiperEngine(cfg_for(voice_dir), loader=loader).load()


def test_engine_synthesize_maps_chunks(voice_dir):
    raw = SimpleNamespace(audio_int16_bytes=tone(10), sample_rate=RATE,
                          phoneme_alignments=[SimpleNamespace(phoneme="a", num_samples=10)])
    voice = SimpleNamespace(synthesize=lambda text, **kw: iter([raw]))
    eng = PiperEngine(cfg_for(voice_dir), loader=lambda m, a: voice)
    [chunk] = list(eng.synthesize("hi"))
    assert chunk == PcmChunk(tone(10), RATE, [("a", 10)])
    assert chunk.num_samples == 10


def test_engine_available_reports_missing_model(tmp_path):
    ok, why = PiperEngine(TtsConfig(voice="nope", model_dir=tmp_path)).available()
    assert not ok and "missing" in why


# ---- speaker, in-process --------------------------------------------------------

def test_inprocess_plays_all_pcm_and_publishes_chained_timelines(voice_dir, tmp_path):
    out = tmp_path / "played.raw"
    chunks = [PcmChunk(tone(RATE // 2), RATE, [("^", 100), ("a", RATE // 2 - 100)]),
              PcmChunk(tone(RATE // 4), RATE, None)]
    sp, events = make_speaker(voice_dir, FakeEngine(chunks), ["sh", "-c", f"cat > {out}"],
                              avatar_lag_ms=70.0)
    before = time.monotonic()
    res = sp.say("Hello there. Bye.")
    assert res.ok and not res.interrupted and res.first_audio_ms > 0
    assert out.read_bytes() == chunks[0].pcm + chunks[1].pcm
    utts = [d for t, d in events if t == "utterance"]
    assert len(utts) == 2
    assert utts[0]["start_mono"] - before >= 0.070
    assert utts[1]["start_mono"] - utts[0]["start_mono"] == pytest.approx(0.5)
    assert utts[0]["visemes"][1][1] == "AA"
    assert "visemes" not in utts[1]                     # no alignments: loudness fallback
    assert len(utts[1]["level"]) == 15                  # 0.25 s at 60 Hz
    assert utts[0]["id"] != utts[1]["id"]


def test_inprocess_barge_in_kills_playback_and_emits_stop(voice_dir):
    chunks = [PcmChunk(tone(RATE), RATE, None)]
    sp, events = make_speaker(voice_dir, FakeEngine(chunks), ["sleep", "10"])
    start = time.monotonic()
    res = sp.say("long answer", should_stop=lambda: True)
    assert res.ok and res.interrupted
    assert time.monotonic() - start < 2
    assert ("speech_stop", {"reason": "interrupted"}) in events
    assert sp.speaking is False


def test_inprocess_synthesis_error_reported(voice_dir):
    sp, _ = make_speaker(voice_dir, FakeEngine(error=RuntimeError("onnx blew up")),
                         ["sh", "-c", "cat > /dev/null"])
    res = sp.say("hi")
    assert not res.ok and "onnx blew up" in res.error


def test_inprocess_rate_mismatch_is_an_error(voice_dir):
    sp, _ = make_speaker(voice_dir, FakeEngine([PcmChunk(tone(10), 16000, None)]),
                         ["sh", "-c", "cat > /dev/null"])
    res = sp.say("hi")
    assert not res.ok and "rate" in res.error


def test_publish_failure_does_not_break_speech(voice_dir):
    def boom(t, d):
        raise RuntimeError("bus down")

    sp = Speaker(cfg_for(voice_dir), "null", engine=FakeEngine([PcmChunk(tone(100), RATE, None)]),
                 publish=boom)
    sp._aplay_cmd = lambda rate: ["sh", "-c", "cat > /dev/null"]
    assert sp.say("hi").ok


def test_aplay_failure_reported(voice_dir):
    sp, _ = make_speaker(voice_dir, FakeEngine([PcmChunk(tone(100), RATE, None)]),
                         ["sh", "-c", "cat > /dev/null; echo bad device >&2; exit 3"])
    res = sp.say("hi")
    assert not res.ok and "aplay exited 3" in res.error and "bad device" in res.error


def test_both_engines_share_the_result_contract(voice_dir, tmp_path):
    fake_piper = tmp_path / "piper"
    fake_piper.write_text("#!/bin/sh\ncat > /dev/null\nhead -c 4410 /dev/zero\n")
    fake_piper.chmod(fake_piper.stat().st_mode | stat.S_IEXEC)
    sub = Speaker(cfg_for(voice_dir, engine="subprocess", binary=fake_piper), "null")
    sub._aplay_cmd = lambda rate: ["sh", "-c", "cat > /dev/null"]
    inp, _ = make_speaker(voice_dir, FakeEngine([PcmChunk(tone(2205), RATE, None)]),
                          ["sh", "-c", "cat > /dev/null"])
    if os.system("which aplay > /dev/null 2>&1") != 0:
        pytest.skip("aplay not installed; available() would refuse")
    results = [sub.say("hi"), inp.say("hi")]
    for r in results:
        assert isinstance(r, SpeechResult) and r.ok and not r.interrupted and r.error == ""
        assert r.first_audio_ms > 0 and r.total_ms >= r.first_audio_ms


def test_unknown_engine_uses_subprocess(voice_dir):
    sp = Speaker(cfg_for(voice_dir, engine="weird"), "null")
    assert sp.engine_name == "subprocess" and sp._engine is None
    sp.warm()                                           # no-op, must not raise


def test_subprocess_barge_in_emits_stop(voice_dir, tmp_path):
    fake_piper = tmp_path / "piper"
    fake_piper.write_text("#!/bin/sh\ncat > /dev/null\nhead -c 4410 /dev/zero\n")
    fake_piper.chmod(fake_piper.stat().st_mode | stat.S_IEXEC)
    events = []
    sp = Speaker(cfg_for(voice_dir, engine="subprocess", binary=fake_piper), "null",
                 publish=lambda t, d: events.append((t, d)))
    sp._aplay_cmd = lambda rate: ["sleep", "10"]
    if not sp.available()[0]:
        pytest.skip("aplay not installed")
    assert sp.say("hi", should_stop=lambda: True).interrupted
    assert ("speech_stop", {"reason": "interrupted"}) in events
