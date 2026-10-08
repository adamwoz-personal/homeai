#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Check homeai.speaker's numpy features against sherpa-onnx's reference.

homeai computes Kaldi fbank features itself to avoid a dependency. A
subtle mismatch would not crash; it would quietly cost accuracy. This
embeds the same clips both ways and prints the cosine between them
(1.000 = identical), then how well each separates the speakers -- the
measure that actually matters if they disagree. sherpa-onnx is not a homeai dependency, so run this
from a throwaway venv:

    uv venv /tmp/sherpa-venv && uv pip install -p /tmp/sherpa-venv sherpa-onnx numpy onnxruntime
    PYTHONPATH=. /tmp/sherpa-venv/bin/python tools/probe_speaker_vs_sherpa.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import sherpa_onnx  # noqa: E402

from homeai.config import SpeakerConfig  # noqa: E402
from homeai.speaker import SpeakerEmbedder, cosine  # noqa: E402
from bench_speaker_separation import read_wav  # noqa: E402


def main() -> int:
    model = SpeakerConfig().model_path
    ref = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=1)
    )
    ours = SpeakerEmbedder(model)
    ok, problem = ours.load()
    if not ok:
        print(problem)
        return 2
    worst = 1.0
    both = {"homeai": {}, "sherpa": {}}
    for clip in sorted((ROOT / "vendor/speaker/samples").glob("*.wav")):
        audio = read_wav(clip)
        stream = ref.create_stream()
        stream.accept_waveform(16000, audio)
        stream.input_finished()
        r = ref.compute(stream)
        # Untrimmed, to compare like with like: sherpa does not trim silence.
        sim = cosine(ours.embed(audio, trim=False), r)
        worst = min(worst, sim)
        both["homeai"][clip.stem] = ours.embed(audio)
        both["sherpa"][clip.stem] = r
        print(f"  {clip.stem:24} {sim:.4f}")
    print(f"lowest agreement {worst:.4f}")
    for label, embs in both.items():
        same, diff = [], []
        for (a, ea), (b, eb) in itertools.combinations(embs.items(), 2):
            (same if a.split("-")[0] == b.split("-")[0] else diff).append(cosine(ea, eb))
        print(f"{label:7} same min {min(same):.3f} median {sorted(same)[len(same)//2]:.3f} | "
              f"different max {max(diff):.3f} median {sorted(diff)[len(diff)//2]:.3f} | "
              f"gap {min(same) - max(diff):+.3f}")
    return 0 if worst > 0.98 else 1


if __name__ == "__main__":
    sys.exit(main())
