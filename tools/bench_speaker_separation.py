#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Go/no-go control for speaker recognition (VOICE_ID_PLAN.md step 3).

Embeds labelled clips, prints same-speaker vs different-speaker cosine
similarity, and the time each embedding takes. If the two distributions
overlap, nothing built on top of speaker-id can work; stop.

Clips are WAV files named ``<speaker>-<anything>.wav``. By default this
uses the sherpa-onnx sample clips in vendor/speaker/samples (three
speakers). Point ``--clips`` at a directory of household recordings (from
``python -m homeai.speaker record``) to measure real voices and pick a
threshold.

    .venv/bin/python tools/bench_speaker_separation.py [--clips DIR] [--model PATH]
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homeai.config import SpeakerConfig  # noqa: E402
from homeai.speaker import SpeakerEmbedder, cosine, speech_seconds  # noqa: E402


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz 16-bit WAV")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            raw = raw.reshape(-1, w.getnchannels())[:, 0]
    return raw.astype(np.float32) / 32768.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=Path, default=ROOT / "vendor/speaker/samples")
    ap.add_argument("--model", type=Path, default=None)
    args = ap.parse_args()

    embedder = SpeakerEmbedder(args.model or SpeakerConfig().model_path)
    ok, problem = embedder.load()
    if not ok:
        print(problem)
        return 2

    clips = sorted(args.clips.glob("*.wav"))
    if len(clips) < 2:
        print(f"need at least two clips in {args.clips}")
        return 2
    embs, times = {}, []
    for clip in clips:
        audio = read_wav(clip)
        t0 = time.perf_counter()
        e = embedder.embed(audio)
        times.append((time.perf_counter() - t0) * 1000)
        if e is None:
            print(f"  skip {clip.name}: too little speech")
            continue
        embs[clip.stem] = (clip.stem.split("-")[0], e, len(audio) / 16000, speech_seconds(audio))

    same, diff = [], []
    for (a, (sa, ea, *_)), (b, (sb, eb, *_)) in itertools.combinations(embs.items(), 2):
        (same if sa == sb else diff).append((cosine(ea, eb), a, b))

    print(f"model {embedder.model_name}, {len(embs)} clips, "
          f"{len({v[0] for v in embs.values()})} speakers")
    for name, (_, _, dur, sp) in embs.items():
        print(f"  {name:28} {dur:5.1f} s audio, {sp:5.1f} s speech")
    print(f"embed time: median {statistics.median(times):.0f} ms, max {max(times):.0f} ms "
          f"(includes first-call warm-up)")
    if not same or not diff:
        print("need at least two clips of one speaker and two speakers")
        return 2
    for label, rows in (("same speaker", same), ("different", diff)):
        vals = [r[0] for r in rows]
        print(f"{label:13}: n={len(vals):3}  min {min(vals):.3f}  "
              f"median {statistics.median(vals):.3f}  max {max(vals):.3f}")
    worst_same, worst_diff = min(same), max(diff)
    print(f"lowest same-speaker pair : {worst_same[0]:.3f}  {worst_same[1]} / {worst_same[2]}")
    print(f"highest different pair   : {worst_diff[0]:.3f}  {worst_diff[1]} / {worst_diff[2]}")
    gap = worst_same[0] - worst_diff[0]
    if gap > 0:
        print(f"SEPARATED: gap {gap:.3f}; a threshold near "
              f"{(worst_same[0] + worst_diff[0]) / 2:.2f} splits them")
        return 0
    print(f"OVERLAP by {-gap:.3f}: do not build on this until it separates")
    return 1


if __name__ == "__main__":
    sys.exit(main())
