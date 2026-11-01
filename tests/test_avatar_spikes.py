"""Verdict logic of the avatar Phase 0 TTS latency spike (tools/avatar/tts_latency_bench.py)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "avatar" / "tts_latency_bench.py"
_spec = importlib.util.spec_from_file_location("tts_latency_bench", _PATH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


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
