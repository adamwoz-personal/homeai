"""The wake-word training and false-wake bench (tools/wake; plans/WAKE_WORD_PLAN.md, "Hey Jeeves")."""
from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from tools.wake import bench_wake_word as B
from tools.wake import train_wake_word as T


def test_a_wake_is_a_score_over_the_threshold_then_quiet_for_the_refractory_time():
    s = [0.1, 0.6, 0.9, 0.7, 0.2] + [0.0] * 20 + [0.55]
    got = B.fires(s, 0.5, refractory_s=1.5, frame_s=0.08)
    assert got == [(0.08, 0.6), (2.0, 0.55)]
    assert B.fires(s, 0.95) == []
    assert B.fires([0.5], 0.5) == [(0.0, 0.5)]                 # at the threshold counts, as in homeai/wake.py
    # inside the refractory time a second peak is the same wake
    assert len(B.fires([0.9] + [0.0] * 5 + [0.9], 0.5, refractory_s=1.5, frame_s=0.08)) == 1
    assert len(B.fires([0.9] + [0.0] * 20 + [0.9], 0.5, refractory_s=1.5, frame_s=0.08)) == 2


@pytest.mark.parametrize("model, words", [
    ("hey_jarvis", ["jarvis"]), ("/x/hey_jeeves.onnx", ["jeeves"]), ("hey_jarvis_v0.1", ["jarvis"]),
    ("alexa", ["alexa"])])
def test_the_name_comes_from_the_model(model, words):
    assert B.name_words(model) == words


def test_a_fire_on_a_clip_that_said_the_name_is_not_false():
    texts = {"a.wav": "Hey Jarvis, what time is it?", "b.wav": "pass the cheese please", "c.wav": ""}
    rows = [{"clip": "a.wav", "seconds": 1800, "max": 0.9, "fires": [(0.5, 0.9)]},
            {"clip": "b.wav", "seconds": 1800, "max": 0.7, "fires": [(0.2, 0.7), (3.0, 0.6)]},
            {"clip": "c.wav", "seconds": 0, "max": 0.3, "fires": []}, {"clip": "d.wav", "error": "bad"}]
    s = B.summary("hey_jarvis", rows, texts)
    assert (s["hours"], s["fires_on_name"], s["false_fires"], s["false_per_hour"]) == (1.0, 1, 2, 2.0)
    assert [f["clip"] for f in s["false"]] == ["b.wav"] and s["errors"] == 1 and s["clips_saying_name"] == 1
    assert [n["clip"] for n in s["near_misses"]] == ["c.wav"]
    assert B.says("JARVIS!", ["jarvis"]) and not B.says("javascript", ["jarvis"])


def test_clips_are_read_only_as_jarvis_hears_them(tmp_path):
    good, bad = tmp_path / "g.wav", tmp_path / "b.wav"
    for path, rate in ((good, 16000), (bad, 44100)):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
            w.writeframes(np.zeros(160, dtype=np.int16).tobytes())
    assert len(B.read_wav(good)) == 160
    with pytest.raises(ValueError):
        B.read_wav(bad)
    (tmp_path / "g.json").write_text(json.dumps({"stt": {"text": "hello"}}))
    assert B.transcript(good) == "hello" and B.transcript(bad) == ""


def test_the_training_config_says_the_phrase_and_never_the_sound_alikes(tmp_path):
    c = T.config_for("hey_jeeves", "hey jeeves", 20000, 50000, tmp_path)
    assert c["target_phrase"] == ["hey jeeves"] and c["model_name"] == "hey_jeeves"
    assert "hey jarvis" in c["custom_negative_phrases"] and "hey jeeves" not in c["custom_negative_phrases"]
    # about a quarter of the negatives, every sound-alike equally often
    assert abs(len(c["custom_negative_phrases"]) - 5000) <= len(T.SOUND_ALIKES)
    assert c["custom_negative_phrases"].count("hey jesus") == c["custom_negative_phrases"].count("sleeves") > 100
    assert T.sound_alike_list(10) == T.SOUND_ALIKES                         # never fewer than once each
    assert {"jesus", "sleeves", "he leaves", "cheese"} <= set(c["custom_negative_phrases"])
    assert c["n_samples_val"] == 2000 and T.config_for("x", "x", 500, 1, tmp_path)["n_samples_val"] == 1000
    assert c["target_false_positives_per_hour"] <= 0.2
    assert c["output_dir"] == str(tmp_path)
    assert (c["max_negative_weight"], c["layer_size"]) == (1500, 32)
    tuned = T.config_for("x", "x", 500, 1, tmp_path, negative_weight=300, layer=64)
    assert (tuned["max_negative_weight"], tuned["layer_size"]) == (300, 64)
    assert (c["augmentation_rounds"], c["batch_n_per_class"]["adversarial_negative"]) == (1, 50)
    more = T.config_for("x", "x", 500, 1, tmp_path, rounds=2, adversarial=200)
    assert (more["augmentation_rounds"], more["batch_n_per_class"]["adversarial_negative"]) == (2, 200)


def test_training_refuses_until_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "PY", tmp_path / "none")
    monkeypatch.setattr(T, "DATA", tmp_path / "data")      # never the real runs folder, even with a mutant
    assert str(tmp_path / "none") in T.missing()
    assert T.main(["--step", "generate"]) == 1


def test_generating_clips_needs_no_negative_data(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "DATA", tmp_path / "nothing")
    for path in ("PY", "TRAIN"):
        f = tmp_path / path
        f.write_text("")
        monkeypatch.setattr(T, path, f)
    piper = tmp_path / "piper"
    (piper / "models").mkdir(parents=True)
    (piper / "generate_samples.py").write_text("")
    (piper / "models/en_US-libritts_r-medium.pt").write_text("")
    monkeypatch.setattr(T, "PIPER", piper)
    assert T.missing("generate") == []
    assert any("ACAV100M" in m for m in T.missing("train")) and any("mit_rirs" in m for m in T.missing("augment"))
    assert not any("ACAV100M" in m for m in T.missing("augment"))       # augmenting can run during the download


# -- held-out voices (make_test_phrases) and the breakdown (report_wake_recall) --------------------------------------

from tools.wake import make_test_phrases as M  # noqa: E402
from tools.wake import report_wake_recall as R  # noqa: E402


def test_the_phrase_is_tested_alone_and_leading_a_request():
    assert M.requests("hey jeeves") == ["hey jeeves", "hey jeeves, what time is it?", "hey jeeves. turn on the lights."]
    assert "hey jeeves" not in M.SOUND_ALIKES and "hey jarvis" in M.SOUND_ALIKES


def test_noise_goes_under_the_speech_by_the_asked_amount():
    speech = np.full(16000, 0.1, dtype=np.float32)
    noise = np.random.default_rng(1).normal(0, 1, 4000).astype(np.float32)        # shorter: it repeats
    out = M.mix(speech, noise, 10.0)
    assert len(out) == 16000 + 2 * 8000                                              # 0.5 s pad each side
    under = out[:8000]                                                               # only noise there
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))                            # noqa: E731
    assert abs(20 * np.log10(0.1 / rms(under)) - 10.0) < 0.5
    assert np.max(np.abs(M.mix(np.ones(100, np.float32), np.ones(10, np.float32), 0.0))) <= 0.9 + 1e-6  # no clipping


def test_resampling_to_16k_keeps_the_length_in_time():
    assert len(M.to_16k(np.zeros(22050, np.float32), 22050)) == 16000
    assert len(M.to_16k(np.zeros(160, np.float32), 16000)) == 160


def test_the_breakdown_counts_hits_and_sound_alike_wakes():
    index = [{"kind": "positive", "file": "a.wav", "voice": "v1", "text": "hey jeeves", "speed": 1.0, "noise_db_below": None},
             {"kind": "positive", "file": "b.wav", "voice": "v2", "text": "hey jeeves", "speed": 1.0, "noise_db_below": 5.0},
             {"kind": "negative", "file": "c.wav", "voice": "v1", "text": "hey jesus", "speed": 1.0, "noise_db_below": None},
             {"kind": "negative", "file": "d.wav", "voice": "v1", "text": "nice sleeves", "speed": 1.0, "noise_db_below": None}]
    result = {"threshold": 0.5, "models": [{"model": "m", "false": [{"clip": "c.wav"}],
                                            "recall": {"rate": 0.5, "missed": ["b.wav"]}}]}
    m = R.report(result, index)["models"][0]
    assert m["recall_by_voice"] == {"v1": "1/1", "v2": "0/1"} and m["recall_by_noise"] == {"5.0": "0/1", "None": "1/1"}
    assert m["sound_alike_wakes"] == {"hey jesus": "1/1", "nice sleeves": "0/1"}
