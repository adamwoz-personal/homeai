"""Speech timeline: phoneme alignments -> visemes, level envelope, chained starts."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from homeai.avatar import timeline as tl

FIXTURE = json.loads((Path(__file__).parent / "fixtures/avatar/alignment_fixture.json")
                     .read_text())


def test_every_voice_phoneme_has_a_rule():
    """Completeness: no symbol in the voice's phoneme_id_map falls through to the default."""
    known = set(tl.VISEME_OF) | tl.PAUSE_SYMBOLS | tl.STRESS_MARKS | tl.EXTENDERS | {tl.WORD_GAP}
    missing = [k for k in FIXTURE["phoneme_id_map_keys"] if k not in known]
    assert missing == []


def test_symbol_classes_do_not_overlap():
    total = sum(len(chars) for chars in tl.VISEME_CLASSES.values())
    assert total == len(tl.VISEME_OF)
    groups = [set(tl.VISEME_OF), tl.PAUSE_SYMBOLS, tl.STRESS_MARKS, tl.EXTENDERS, {tl.WORD_GAP}]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not (a & b)


def test_synthetic_sequence_exact():
    # ^ p ˈa: <short gap> m .   at 1 kHz so samples == ms
    align = [("^", 100), ("p", 100), ("ˈ", 50), ("a", 200), ("ː", 100), (" ", 50),
             ("m", 100), (".", 100)]
    assert tl.visemes_from_alignments(align, 1000) == [
        [0, "rest", 0.0], [100, "PP", 1.0], [200, "AA", 1.0], [600, "PP", 1.0], [700, "rest", 0.0]]


def test_long_gap_rests_and_unstressed_vowel_lighter():
    align = [("a", 100), (" ", 120), ("ə", 100)]
    assert tl.visemes_from_alignments(align, 1000) == [
        [0, "AA", 0.8], [100, "rest", 0.0], [220, "EH", 0.8], [320, "rest", 0.0]]


def test_stress_carries_to_next_vowel_across_consonant():
    # Piper puts the stress mark before the onset: d ˈ e -> e is stressed, starts at the mark.
    align = [("t", 50), ("ˈ", 30), ("d", 40), ("e", 100)]
    assert tl.visemes_from_alignments(align, 1000) == [
        [0, "DD", 1.0], [120, "EH", 1.0], [220, "rest", 0.0]]
    # ...and is used up: the next vowel is unstressed
    assert tl.visemes_from_alignments(align + [("p", 10), ("e", 10)], 1000)[-2] == [230, "EH", 0.8]


def test_unknown_phoneme_falls_back_to_mid():
    assert tl.classify("☃") == "EH"
    with pytest.raises(ValueError):
        tl.visemes_from_alignments([("a", 1)], 0)


def test_real_voice_fixture_timeline():
    for chunk in FIXTURE["chunks"]:
        sr = chunk["sample_rate"]
        frames = tl.visemes_from_alignments([tuple(p) for p in chunk["phonemes"]], sr)
        times = [f[0] for f in frames]
        assert times == sorted(times) and len(set(times)) == len(times)
        assert frames[-1][1] == "rest"
        assert sum(n for _, n in chunk["phonemes"]) == chunk["num_samples"]
        assert frames[-1][0] <= round(chunk["num_samples"] * 1000 / sr)
        assert all(v in tl.VISEMES for _, v, _ in frames)
        assert all(a[1] != b[1] for a, b in zip(frames, frames[1:]))   # no repeated keyframes
        assert any(w == tl.STRESSED_WEIGHT and v in tl.VOWEL_VISEMES for _, v, w in frames)
    # "weather": w ˈ ɛ -> the stressed ɛ starts where the stress mark starts
    first = FIXTURE["chunks"][0]
    sr = first["sample_rate"]
    cursor = 0
    for i, (ph, n) in enumerate(first["phonemes"]):
        if ph == "ˈ":
            mark_ms = round(cursor * 1000 / sr)
            break
        cursor += n
    frames = tl.visemes_from_alignments([tuple(p) for p in first["phonemes"]], sr)
    assert [mark_ms, "EH", 1.0] in frames


def test_level_envelope_length_and_scale():
    sr = 22050
    n = sr // 2 + 7
    t = np.arange(n) / sr
    loud = (np.sin(2 * math.pi * 220 * t) * 32767).astype(np.int16)
    env = tl.level_envelope(loud, sr)
    assert len(env) == math.ceil(n / (sr / 60))
    assert 235 <= min(env[:-1]) <= max(env) <= 255            # -3 dB RMS
    assert tl.level_envelope(np.zeros(n, np.int16), sr) == [0] * len(env)
    assert tl.level_envelope(np.ones(n, np.int16), sr)[0] == 0  # -90 dB clamps to 0
    assert tl.level_envelope(loud.tobytes(), sr) == env
    assert tl.level_envelope(b"", sr) == []
    with pytest.raises(ValueError):
        tl.level_envelope(loud, 0)


def test_chain_starts_by_sample_count():
    counts = [c["num_samples"] for c in FIXTURE["chunks"]]
    sr = FIXTURE["chunks"][0]["sample_rate"]
    starts = tl.chain_starts(100.0, counts, sr)
    assert starts[0] == 100.0
    for i in range(1, len(counts)):
        assert starts[i] == pytest.approx(100.0 + sum(counts[:i]) / sr)
    with pytest.raises(ValueError):
        tl.chain_starts(0, [1], 0)


def test_utterance_data_fallback_omits_visemes():
    d = tl.utterance_data("u1", 5.0, 22050, 22050, [1, 2], None, "hi")
    assert "visemes" not in d and d["duration_ms"] == 1000 and d["level_hz"] == 60
    assert tl.utterance_data("u1", 5.0, 22050, 22050, [], [[0, "rest", 0.0]])["visemes"]


def test_chunk_utterance_with_and_without_alignments():
    pcm = (np.full(2205, 8000, dtype=np.int16)).tobytes()          # 0.1 s
    d = tl.chunk_utterance("c-0", 2.0, pcm, 22050, [("h", 1000), ("a", 1205)])
    assert d["duration_ms"] == 100 and len(d["level"]) == 6 and d["visemes"][-1][1] == "rest"
    assert "visemes" not in tl.chunk_utterance("c-1", 2.0, pcm, 22050, None)


def tone(hz, seconds=0.5, sr=22050, amp=12000):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def test_band_envelopes_separate_low_and_high():
    lo = tl.band_envelopes(tone(150), 22050)
    hi = tl.band_envelopes(tone(5000), 22050)
    assert len(lo["low"]) == len(tl.level_envelope(tone(150), 22050))
    assert np.mean(lo["low"]) > 150 and np.mean(lo["high"]) < 20
    assert np.mean(hi["high"]) > 150 and np.mean(hi["low"]) < 20
    mid = tl.band_envelopes(tone(1200), 22050)       # between the bands: neither lights
    assert np.mean(mid["low"]) < 40 and np.mean(mid["high"]) < 40


def test_band_envelopes_silence_bytes_and_bad_rates():
    quiet = tl.band_envelopes(np.zeros(22050, dtype=np.int16).tobytes(), 22050)
    assert set(quiet["low"]) == {0} and set(quiet["high"]) == {0}
    for sr, hz in ((0, 60), (22050, 0)):
        with pytest.raises(ValueError):
            tl.band_envelopes(tone(150), sr, hz)


def test_bands_never_brighter_than_level():
    pcm = tone(150)
    level = tl.level_envelope(pcm, 22050)
    bands = tl.band_envelopes(pcm, 22050, level=level)
    assert all(b <= lv for b, lv in zip(bands["low"], level))


def test_chunk_utterance_carries_bands():
    d = tl.chunk_utterance("c-0", 0.0, tone(150).tobytes(), 22050, None)
    assert len(d["bands"]["low"]) == len(d["level"])
    assert "bands" not in tl.utterance_data("x", 0.0, 22050, 10, [0])
