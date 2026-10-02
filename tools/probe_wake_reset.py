#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Does the wake detector re-fire on a wake word it already reported?

    .venv/bin/python tools/probe_wake_reset.py [--voice en_US-joe-medium]

This reproduces the daemon's sequence with the real openWakeWord model:
1. Feed a synthesized "Hey Jarvis" until the detector fires.
2. Stop feeding while the utterance is captured, then call reset().
3. Feed silence and record the highest score.

A score at or above the threshold on pure silence means the wake word is
still in the detector's feature buffers. That is the phantom second wake
seen live about 1-2 s after a question (2026-10-02, 18:27:23 and 19:17:32).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from homeai.config import TtsConfig, WakeConfig  # noqa: E402
from homeai.wake import OpenWakeWordDetector  # noqa: E402

FRAME = 1280  # 80 ms at 16 kHz, what the daemon feeds


def synth_16k(text: str, voice: str) -> np.ndarray:
    tts = TtsConfig(voice=voice)
    rate = json.loads(tts.config_path.read_text())["audio"]["sample_rate"]
    raw = subprocess.run([str(tts.binary), "--model", str(tts.model_path), "--output-raw"],
                         input=text.encode(), capture_output=True, check=True).stdout
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    n = int(len(pcm) * 16000 / rate)
    return np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voice", default="en_US-joe-medium")
    ap.add_argument("--silence-s", type=float, default=3.0)
    args = ap.parse_args()

    cfg = WakeConfig()
    det = OpenWakeWordDetector(cfg.model, cfg.threshold)
    ok, problem = det.load()
    if not ok:
        print(problem, file=sys.stderr)
        return 2
    audio = np.concatenate([np.zeros(16000, np.float32),
                            synth_16k("Hey Jarvis.", args.voice),
                            np.zeros(4000, np.float32)])
    fired_at = None
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        if det.detect(audio[i:i + FRAME]):
            fired_at = i / 16000
            break
    if fired_at is None:
        print("detector never fired on the synthesized wake word; try another --voice")
        return 2
    print(f"wake fired at {fired_at:.2f}s (score {det.last_score:.3f})")

    import time
    t0 = time.perf_counter()
    det.reset()
    print(f"reset() took {(time.perf_counter() - t0) * 1000:.0f} ms")
    silence = np.zeros(FRAME, np.float32)
    scores = []
    for _ in range(int(args.silence_s * 16000 / FRAME)):
        det.detect(silence)
        scores.append(det.last_score)
    peak = max(scores)
    first_hit = next((k for k, s in enumerate(scores) if s >= cfg.threshold), None)
    print(f"after reset, on silence: peak score {peak:.3f}"
          + (f", re-fired after {first_hit * 0.08:.2f}s" if first_hit is not None else ""))
    stale = peak >= cfg.threshold
    print("FAIL: reset() leaves the wake word in the detector" if stale
          else "PASS: reset() fully clears the detector")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
