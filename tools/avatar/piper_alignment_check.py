#!/usr/bin/env python3
"""Spike: can our Piper voice report per-phoneme timing for lip sync?

Synthesises a sentence with ``include_alignments=True`` (Piper patches the
ONNX model in memory; nothing on disk changes) and prints each phoneme with
its start time and the alignment's total length versus the real audio.

Needs the ``onnx`` package (not in the project venv). Without touching the
venv:

    pip install --target /tmp/homeai-onnx onnx
    PYTHONPATH=/tmp/homeai-onnx .venv/bin/python tools/avatar/piper_alignment_check.py ["text"]

Exit codes: 0 alignments available, 1 not available, 2 setup error.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_TEXT = "Good night. I've turned off the lights, and the front door is locked."


def main() -> int:
    text = " ".join(sys.argv[1:]) or DEFAULT_TEXT
    try:
        from piper import PiperVoice
        from homeai.config import TtsConfig
    except ImportError as exc:
        print(f"setup error: {exc}")
        return 2

    model = TtsConfig().model_path
    if not model.exists():
        print(f"setup error: voice model not found at {model}")
        return 2

    started = time.monotonic()
    try:
        voice = PiperVoice.load(str(model), include_alignments=True)
    except Exception as exc:  # noqa: BLE001 - report any load failure plainly
        print(f"alignments unavailable: {exc}")
        return 1
    load_ms = (time.monotonic() - started) * 1000

    audio_samples = 0
    aligned_samples = 0
    rate = 0
    rows: list[tuple[float, str]] = []
    synth_start = time.monotonic()
    for chunk in voice.synthesize(text, include_alignments=True):
        rate = chunk.sample_rate
        if not chunk.phoneme_alignments:
            print("alignments unavailable: voice returned none")
            return 1
        cursor = audio_samples
        for alignment in chunk.phoneme_alignments:
            rows.append((cursor / rate, alignment.phoneme))
            cursor += alignment.num_samples
        aligned_samples += cursor - audio_samples
        audio_samples += len(chunk.audio_float_array)
    synth_ms = (time.monotonic() - synth_start) * 1000

    audio_s = audio_samples / rate
    drift_ms = (aligned_samples - audio_samples) / rate * 1000
    print(f"voice: {model.name}  rate {rate} Hz  load {load_ms:.0f} ms  synth {synth_ms:.0f} ms"
          f"  audio {audio_s:.2f} s  ({audio_s * 1000 / max(synth_ms, 1):.1f}x real time)")
    print(f"phonemes: {len(rows)}  alignment vs audio length: {drift_ms:+.1f} ms")
    print("  ".join(f"{t:5.2f}:{p}" for t, p in rows[:48]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
