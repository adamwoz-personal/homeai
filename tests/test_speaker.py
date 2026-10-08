"""Tests for homeai.speaker: features, embedder, registry, identification."""
from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from homeai import speaker
from homeai.config import SpeakerConfig
from homeai.speaker import (
    Identification,
    SpeakerEmbedder,
    SpeakerIdentifier,
    SpeakerRegistry,
    cosine,
    fbank,
    speech_mask,
    speech_seconds,
)

SR = 16000
MODEL = "test-model"


def tone(seconds: float, hz: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def unit(*xs) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# -- features ---------------------------------------------------------------

def test_fbank_shape_follows_kaldi_snip_edges():
    feats = fbank(tone(1.0))
    assert feats.shape == (1 + (SR - 400) // 160, 80)
    assert np.isfinite(feats).all()


def test_fbank_of_too_short_audio_is_empty():
    assert fbank(np.zeros(399, dtype=np.float32)).shape == (0, 80)


def test_fbank_puts_a_tone_in_the_right_mel_band():
    feats = fbank(tone(0.5, hz=1000))
    low = fbank(tone(0.5, hz=200))
    assert feats.mean(axis=0).argmax() > low.mean(axis=0).argmax()


def test_silence_is_not_speech():
    assert not speech_mask(np.zeros(SR, dtype=np.float32)).any()
    assert speech_seconds(np.zeros(SR, dtype=np.float32)) == 0.0


def test_speech_mask_keeps_the_loud_part_only():
    audio = np.concatenate([tone(1.0), np.zeros(SR, dtype=np.float32)])
    assert speech_seconds(audio) == pytest.approx(1.0, abs=0.05)


# -- embedder ---------------------------------------------------------------

class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, _outputs, feeds):
        feats = next(iter(feeds.values()))
        self.calls.append(feats)
        return [np.array([[3.0, 4.0]], dtype=np.float32)]


def fake_embedder() -> tuple[SpeakerEmbedder, FakeSession]:
    e = SpeakerEmbedder("unused.onnx")
    e._session, e._input = FakeSession(), "feats"
    return e, e._session


def test_load_reports_a_missing_model(tmp_path):
    ok, problem = SpeakerEmbedder(tmp_path / "nope.onnx").load()
    assert not ok and "missing" in problem


def test_load_reports_a_broken_model(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    ok, problem = SpeakerEmbedder(bad).load()
    assert not ok and "could not load" in problem


def test_embed_before_load_raises_rather_than_returning_nothing():
    with pytest.raises(RuntimeError):
        SpeakerEmbedder("x.onnx").embed(tone(2.0))


def test_embed_returns_unit_vector_of_mean_normalised_speech_frames():
    e, session = fake_embedder()
    audio = np.concatenate([tone(1.0), np.zeros(SR, dtype=np.float32)])
    v = e.embed(audio)
    assert v == pytest.approx([0.6, 0.8])
    feats = session.calls[0]
    assert feats.shape[0] == 1 and feats.shape[2] == 80
    assert feats.shape[1] < 120  # the silent second was trimmed
    assert np.abs(feats[0].mean(axis=0)).max() < 1e-3


def test_embed_of_silence_or_a_blip_is_none():
    e, session = fake_embedder()
    assert e.embed(np.zeros(2 * SR, dtype=np.float32)) is None
    assert e.embed(tone(0.2)) is None
    assert session.calls == []


# -- real model (only where it has been downloaded) -------------------------

REAL = SpeakerConfig().model_path
SAMPLES = REAL.parent / "samples"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


@pytest.mark.skipif(not REAL.is_file() or not SAMPLES.is_dir(), reason="speaker model not downloaded")
def test_real_model_separates_the_sample_speakers():
    e = SpeakerEmbedder(REAL)
    assert e.load() == (True, "")
    a1, a2 = (e.embed(read_wav(SAMPLES / f"fangjun-sr-{i}.wav")) for i in (1, 2))
    b1 = e.embed(read_wav(SAMPLES / "leijun-sr-1.wav"))
    assert a1.shape == (256,) and np.linalg.norm(a1) == pytest.approx(1.0, abs=1e-5)
    assert cosine(a1, a2) > SpeakerConfig().threshold > cosine(a1, b1)


# -- registry ---------------------------------------------------------------

def test_missing_registry_is_empty_without_complaint(tmp_path):
    reg = SpeakerRegistry(tmp_path / "speakers.json", MODEL)
    assert reg.names() == [] and reg.problem == ""


def test_registry_round_trip_is_private(tmp_path):
    path = tmp_path / "sub" / "speakers.json"
    reg = SpeakerRegistry(path, MODEL)
    reg.add("Adam", unit(1, 0, 0), count=3)
    reg.save()
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    again = SpeakerRegistry(path, MODEL)
    assert again.names() == ["Adam"]
    p = again.get("adam")
    assert p.sample_count == 3 and p.embedding == pytest.approx([1, 0, 0])


def test_corrupt_registry_is_reported_and_moved_aside_on_save(tmp_path):
    path = tmp_path / "speakers.json"
    path.write_text("{not json")
    reg = SpeakerRegistry(path, MODEL)
    assert reg.names() == [] and "unreadable" in reg.problem
    reg.add("Adam", unit(1, 0))
    reg.save()
    assert list(tmp_path.glob("speakers.corrupt-*"))
    assert json.loads(path.read_text())["speakers"]["Adam"]


def test_profiles_from_another_model_are_ignored(tmp_path):
    path = tmp_path / "speakers.json"
    reg = SpeakerRegistry(path, "old-model")
    reg.add("Adam", unit(1, 0))
    reg.save()
    reg = SpeakerRegistry(path, MODEL)
    assert reg.names() == [] and "re-enrol" in reg.problem


def test_adding_again_averages_case_insensitively(tmp_path):
    reg = SpeakerRegistry(tmp_path / "s.json", MODEL)
    reg.add("Adam", unit(1, 0))
    p = reg.add("adam", unit(0, 1))
    assert reg.names() == ["Adam"] and p.sample_count == 2
    assert p.embedding == pytest.approx(unit(1, 1))
    assert reg.add("ADAM", unit(0, 1), replace=True).sample_count == 1


def test_remove_and_empty_name(tmp_path):
    reg = SpeakerRegistry(tmp_path / "s.json", MODEL)
    reg.add("Adam", unit(1, 0))
    assert reg.remove("ADAM") and not reg.remove("Adam")
    with pytest.raises(ValueError):
        reg.add("  ", unit(1, 0))


# -- identification ---------------------------------------------------------

def identifier(tmp_path, **profiles) -> SpeakerIdentifier:
    reg = SpeakerRegistry(tmp_path / "s.json", MODEL)
    for name, emb in profiles.items():
        reg.add(name, emb)
    return SpeakerIdentifier(reg, threshold=0.5, margin=0.1)


def test_nobody_enrolled_or_no_speech_is_unknown(tmp_path):
    assert identifier(tmp_path).identify(unit(1, 0)).reason == "nobody enrolled"
    assert identifier(tmp_path, Adam=unit(1, 0)).identify(None).reason == "too little speech"


def test_clear_match_is_named(tmp_path):
    ident = identifier(tmp_path, Adam=unit(1, 0, 0), Beth=unit(0, 1, 0))
    r = ident.identify(unit(0.9, 0.1, 0.1))
    assert r.name == "Adam" and r.score > 0.9 and r.margin > 0.5


def test_weak_match_is_unknown_but_reports_the_best_guess(tmp_path):
    r = identifier(tmp_path, Adam=unit(1, 0)).identify(unit(0.4, 1))
    assert r == Identification(None, pytest.approx(r.score), "Adam", pytest.approx(r.margin),
                               "below threshold")
    assert r.score < 0.5


def test_two_similar_voices_are_not_guessed_between(tmp_path):
    ident = identifier(tmp_path, Adam=unit(1, 0.05), Ben=unit(1, -0.05))
    r = ident.identify(unit(1, 0.01))
    assert r.name is None and r.reason == "ambiguous" and r.score > 0.9


def test_threshold_and_margin_are_inclusive(tmp_path, monkeypatch):
    ident = identifier(tmp_path, Adam=unit(1, 0))
    monkeypatch.setattr(speaker, "cosine", lambda a, b: 0.5)
    assert ident.identify(unit(1, 0)).name == "Adam"


def test_registry_picks_up_changes_made_by_another_process(tmp_path):
    path = tmp_path / "s.json"
    daemon = SpeakerRegistry(path, MODEL)
    daemon.add("Adam", unit(1, 0))
    daemon.add("Ryan", unit(0, 1))
    daemon.save()
    assert not daemon.reload_if_changed()  # its own save is not "a change"
    cli = SpeakerRegistry(path, MODEL)
    cli.remove("Ryan")
    cli.save()
    assert daemon.reload_if_changed() and daemon.names() == ["Adam"]
    path.unlink()
    assert daemon.reload_if_changed() and daemon.names() == []
