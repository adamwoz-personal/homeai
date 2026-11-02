"""Verdict logic of the avatar Phase 0 TTS latency spike (tools/avatar/tts_latency_bench.py)."""

import importlib.util
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "avatar"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("tts_latency_bench")
probe = _load("landmarks_probe")
lag = _load("measure_output_latency")
window = _load("window_probe")
stalls = _load("probe_stalls")


def row(sub_first, inproc_first, sub_samples=1000, inproc_samples=1000):
    def engine(first, samples):
        return {"first_ms": {"p50": first}, "samples_p50": samples}
    return {"text": "hello", "subprocess": engine(sub_first, sub_samples),
            "inprocess": engine(inproc_first, inproc_samples)}


def test_passes_when_gain_and_length_are_within_limits():
    rows = [row(520, 20, 1000, 1050)]
    assert bench.evaluate(rows, 0.10) == (True, [])
    assert rows[0]["gain_ms"] == 500 and rows[0]["length_diff"] == 0.05


@pytest.mark.parametrize("sub,inproc,ok", [(320, 20, True), (319, 20, False)])
def test_gain_boundary_is_inclusive_at_300_ms(sub, inproc, ok):
    assert bench.evaluate([row(sub, inproc)], 0.10)[0] is ok


def test_fails_when_audio_length_differs_too_much():
    passed, reasons = bench.evaluate([row(520, 20, 1000, 1200)], 0.10)
    assert not passed and "length differs" in reasons[0]


def test_summarise_percentiles():
    s = bench.summarise([float(v) for v in range(1, 21)])
    assert s == {"p50": 10.5, "p95": 19.0, "min": 1.0, "max": 20.0}


def face(**moves):
    """478 points at the centre, with eyes, lips and chin placed like a face."""
    pts = [(0.5, 0.5)] * 478
    layout = {"LEFT_EYE": (0.4, 0.4), "RIGHT_EYE": (0.6, 0.4), "UPPER_LIP": (0.5, 0.6),
              "LOWER_LIP": (0.5, 0.62), "CHIN": (0.5, 0.8)}
    layout.update(moves)
    for name, xy in layout.items():
        pts[getattr(probe, name)] = xy
    return pts


def test_plausible_face_layout():
    assert probe.plausible(face())


@pytest.mark.parametrize("moves", [
    {"LEFT_EYE": (0.4, 0.7)},          # eye below the lips
    {"UPPER_LIP": (0.5, 0.63)},        # lips swapped
    {"CHIN": (0.5, 0.61)},             # chin above the lower lip
    {"RIGHT_EYE": (0.41, 0.4)},        # eyes on top of each other
])
def test_implausible_layouts_rejected(moves):
    assert not probe.plausible(face(**moves))


def test_plausible_needs_enough_points():
    assert not probe.plausible([(0.5, 0.5)] * 10)


def test_bbox_is_normalised_extent():
    assert probe.bbox([(0.2, 0.3), (0.6, 0.1), (0.4, 0.9)]) == [0.2, 0.1, 0.6, 0.9]


def recorded(n=5, delay_s=0.080, rate=16000, t0=100.0, noise=0.002, drop=()):
    """A fake mic recording of the click train, heard ``delay_s`` late."""
    import numpy as np
    pcm, offsets = lag.click_train(n, 0.3, rate)
    clean = pcm.astype(np.float32) / 32767
    for k in drop:  # silence a burst, as if it were never heard
        a = int(offsets[k] * rate)
        clean[a:a + rate // 10] = 0
    shift = int(delay_s * rate)
    signal = np.concatenate([np.zeros(shift, np.float32), clean])
    signal += np.random.default_rng(1).normal(0, noise, len(signal)).astype(np.float32)
    return signal, [t0 + o for o in offsets], t0


def test_click_train_offsets_and_bursts():
    pcm, offsets = lag.click_train(3, 0.3, 22050)
    assert offsets == pytest.approx([0.5, 1.1, 1.7], abs=1e-4)
    for o in offsets:
        i = int(o * 22050)
        assert abs(pcm[i:i + 132]).max() > 0.2 * 32767
        assert abs(pcm[i - 200:i]).max() == 0


def test_onsets_recover_the_delay():
    signal, expected, t0 = recorded(delay_s=0.080)
    found = lag.onsets(signal, 16000, t0, expected)
    lags = [(f - e) * 1000 for f, e in zip(found, expected)]
    assert all(78 <= x <= 84 for x in lags), lags


def test_onsets_report_missing_bursts_as_none():
    signal, expected, t0 = recorded(drop=(2,))
    found = lag.onsets(signal, 16000, t0, expected)
    assert found[2] is None and all(f is not None for i, f in enumerate(found) if i != 2)


def test_onsets_ignore_windows_outside_the_recording():
    signal, expected, t0 = recorded()
    assert lag.onsets(signal, 16000, t0 + 5.0, expected)[0] is None


@pytest.mark.parametrize("lags,passed", [
    ([80.0, 85.0, 75.0, 90.0], True),     # median 82.5, worst 7.5 ms off
    ([80.0, 80.0, 80.0, 91.0], False),    # 11 ms off the median
    ([80.0, None, 80.0, 80.0], False),    # one burst missing
    ([None, None, None, None], False),
])
def test_verdict(lags, passed):
    assert lag.verdict(lags, 4)["passed"] is passed


def test_verdict_spread_boundary_inclusive():
    v = lag.verdict([80.0, 80.0, 80.0, 90.0], 4)
    assert v["spread_ms"] == 10.0 and v["passed"]


def test_fibonacci_sphere_unit_vectors_evenly_spread():
    import numpy as np
    pts = window.fibonacci_sphere(2000)
    assert pts.shape == (2000, 3) and pts.dtype == np.float32
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-5)
    assert abs(pts.mean(axis=0)).max() < 0.01


def test_frame_stats_smooth_run_passes():
    s = window.frame_stats([1 / 60] * 600, 60.0)
    assert s["passed"] and s["fps"] == 60.0 and s["dropped"] == 0.0


@pytest.mark.parametrize("frames,passed", [
    ([1 / 60] * 594 + [2 / 60] * 6, True),    # 1.0% dropped: limit, still >= 97% fps
    ([1 / 60] * 593 + [2 / 60] * 7, False),   # 1.17% dropped
    ([1 / 55] * 600, False),                  # steady but too slow
    ([], False),
])
def test_frame_stats_limits(frames, passed):
    assert window.frame_stats(frames, 60.0)["passed"] is passed


def test_hitch_fails_a_run_unless_right_after_an_excused_event():
    frames = [1 / 60] * 300
    frames[150] = 0.117                      # one 117 ms jerk, ends at ~2.6 s
    times = list(__import__("itertools").accumulate(frames))
    s = window.frame_stats(frames, 60.0, times)
    assert not s["passed"] and len(s["hitches"]) == 1 and not s["hitches"][0]["excused"]
    excused = window.frame_stats(frames, 60.0, times, (times[150] - 0.4,))
    assert excused["passed"] and excused["hitches"][0]["excused"]
    late = window.frame_stats(frames, 60.0, times, (times[150] - 0.6,))
    assert not late["passed"]


def test_hitch_threshold():
    frames = [1 / 60] * 300
    frames[100] = 0.050                      # exactly 50 ms: not a hitch
    assert window.frame_stats(frames, 60.0)["hitches"] == []


def test_find_gaps_reports_only_long_intervals():
    stamps = [0.0, 0.002, 0.004, 0.050, 0.052]
    gaps = stalls.find_gaps(stamps, 0.002, 0.020)
    assert len(gaps) == 1
    assert gaps[0][0] == pytest.approx(0.004)
    assert gaps[0][1] == pytest.approx(46.0)
    assert stalls.find_gaps([0.0, 0.021], 0.002, 0.020) == []   # within tolerance
    assert stalls.find_gaps([0.0], 0.002, 0.020) == []


def test_find_bursts_flags_busy_windows():
    stamps = [i * 0.01 for i in range(21)]        # 0..0.2 s, 4 windows of 50 ms
    cpu = [0.0] * 21
    for i in range(5, 21):                          # 2nd window: 40 ms CPU, then idle
        cpu[i] = min(0.04, (i - 5) * 0.008)
    bursts = stalls.find_bursts(stamps, cpu, 0.05, 0.5)
    assert [round(t, 2) for t, _ in bursts] == [0.05]
    assert bursts[0][1] == pytest.approx(40.0)
    assert stalls.find_bursts(stamps, cpu, 0.05, 0.9) == []    # 80% of a core < 90%
    assert stalls.find_bursts(stamps, cpu[:-1], 0.05, 0.5) == []  # length mismatch


def test_stat_path_targets():
    assert stalls.stat_path("12") == "/proc/12/stat"
    assert stalls.stat_path("12/34") == "/proc/12/task/34/stat"
    for bad in ("x", "12/y", ""):
        with pytest.raises(ValueError):
            stalls.stat_path(bad)


demo = _load("avatar_demo")


def test_demo_summary_chains_only_within_a_call():
    def utt(uid, start, dur):
        return ("utterance", {"id": uid, "start_mono": start, "duration_ms": dur, "visemes": [1]})

    events = [utt("aa-0", 10.0, 500), utt("aa-1", 10.5, 300),     # abut
              utt("bb-0", 20.0, 100),                              # new call: not compared
              ("speech_stop", {})]
    s = demo.summarise(events)
    assert s["chain_gaps_ms"] == [0.0] and s["chained"] and s["stops"] == 1
    events.append(utt("bb-1", 20.2, 100))                          # 100 ms hole in one call
    assert not demo.summarise(events)["chained"]
