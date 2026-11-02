#!/usr/bin/env python3
"""Capture Piper phoneme alignments and the voice's phoneme set as a test fixture.

Why: timeline tests must use real alignments from our voice (stress marks,
pauses, diphthongs as Piper actually emits them) without loading the model in
CI. Re-run after changing the voice.

Usage:
    .venv/bin/python tools/avatar/capture_timeline_fixture.py \
        [--out tests/fixtures/avatar/alignment_fixture.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SENTENCES = [
    "The weather today is cloudy, with a high of twelve degrees.",
    "Should I turn on the lights? Okay, the movie starts in five minutes.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--out", type=Path, default=root / "tests/fixtures/avatar/alignment_fixture.json")
    args = parser.parse_args()
    try:
        from piper import PiperVoice
        from homeai.config import TtsConfig
    except ImportError as exc:
        print(f"setup error: {exc}")
        return 2
    model = TtsConfig().model_path
    if not model.exists():
        print(f"setup error: voice model missing: {model}")
        return 2
    try:
        voice = PiperVoice.load(str(model), include_alignments=True)
    except Exception as exc:                # Piper raises plain Exceptions on patch failure
        print(f"alignments unavailable: {exc}")
        return 1
    phoneme_map = json.loads(Path(f"{model}.json").read_text())["phoneme_id_map"]
    chunks = []
    for text in SENTENCES:
        for chunk in voice.synthesize(text, include_alignments=True):
            if not chunk.phoneme_alignments:
                print("alignments unavailable: voice returned none")
                return 1
            chunks.append({"text": text, "sample_rate": chunk.sample_rate,
                           "num_samples": len(chunk.audio_int16_bytes) // 2,
                           "phonemes": [[a.phoneme, int(a.num_samples)] for a in chunk.phoneme_alignments]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"voice": model.name, "chunks": chunks,
                                    "phoneme_id_map_keys": sorted(phoneme_map)},
                                   ensure_ascii=False, indent=1))
    print(f"{len(chunks)} chunks, {sum(len(c['phonemes']) for c in chunks)} phonemes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
