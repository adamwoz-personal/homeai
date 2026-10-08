#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Compare homeai.speaker.fbank with kaldi-native-fbank, option for option.

Isolates the feature code from the model: if these differ, the bug is in
homeai's fbank. Run from the throwaway venv described in
tools/probe_speaker_vs_sherpa.py, after ``uv pip install kaldi-native-fbank``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import kaldi_native_fbank as knf  # noqa: E402

from homeai import speaker  # noqa: E402
from bench_speaker_separation import read_wav  # noqa: E402


def kaldi(audio: np.ndarray) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.snip_edges = True
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = speaker.N_MELS
    opts.mel_opts.low_freq = speaker.LOW_HZ
    opts.mel_opts.high_freq = 0
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, (audio * 32768.0).tolist())
    fb.input_finished()
    return np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)])


def main() -> int:
    worst = 0.0
    for clip in sorted((ROOT / "vendor/speaker/samples").glob("*.wav"))[:4]:
        audio = read_wav(clip)
        ours, ref = speaker.fbank(audio), kaldi(audio)
        n = min(len(ours), len(ref))
        err = float(np.abs(ours[:n] - ref[:n]).max())
        worst = max(worst, err)
        print(f"  {clip.stem:24} frames ours {len(ours)} kaldi {len(ref)}  max abs diff {err:.4f}")
    print(f"worst {worst:.4f}")
    return 0 if worst < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
