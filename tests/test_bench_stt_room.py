"""tools/bench_stt_room.py: the word-agreement hint between models, and which clips it samples."""
from __future__ import annotations

import json

from tools import bench_stt_room as B


def test_agreement_ignores_case_and_punctuation_and_counts_repeats():
    assert B.agreement("Hey, what's up?", "hey what's up") == 1.0
    assert B.agreement("yeah yeah", "yeah yeah yeah") == 0.8
    assert B.agreement("", "") == 1.0
    assert B.agreement("thank you", "") == 0.0


def test_sample_takes_only_transcribed_clips_with_audio(tmp_path):
    def clip(name, stt, wav=True):
        (tmp_path / f"{name}.json").write_text(json.dumps({"duration": 1.0, **({"stt": stt} if stt else {})}))
        if wav:
            (tmp_path / f"{name}.wav").write_bytes(b"")
    clip("heard", {"ok": True})
    clip("noise", {"ok": False})
    clip("waiting", None)
    clip("gone", {"ok": True}, wav=False)
    names = [p.stem for p, _ in B.pick(tmp_path, clips=5, noise=5)]
    assert names == ["heard", "noise"]
