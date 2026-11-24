#!/usr/bin/env python3
"""Clips of new voices saying the wake phrase and its sound-alikes, for wake-model recall (plans/WAKE_WORD_PLAN.md).

Why: the room recordings (bench_wake_word.py) show false wakes, but nobody in the house has said "hey jeeves" yet,
so they can't show misses. The model trained on Piper's LibriTTS-R generator; the six Piper voices Jarvis has on
disk (vendor/piper) are voices it never heard. Each says the phrase (alone and leading a request) at three speeds,
clean and over household background sound (AudioSet clips, 10 dB and 5 dB below), plus the sound-alikes
("hey jarvis", "hey jesus", "he leaves"...) the same way. Synthetic: a lower bound for misses on clean speech,
not a substitute for family recordings.

Usage:
    .venv/bin/python tools/wake/make_test_phrases.py [--phrase "hey jeeves"] [--noise ~/.cache/homeai/wake-train/audioset_16k]
Output: ~/homeai-bench/wake-phrases-<time>/{positive,negative}/*.wav (16 kHz mono 16-bit), index.json
Then:  bench_wake_word.py --model X --positives DIR/positive --clips DIR/negative
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.benchio import bench_path  # noqa: E402

VOICES = REPO / "vendor/piper"
SPEEDS = (0.8, 1.0, 1.25)            # Piper length_scale: under 1 is faster
NOISE_DB = (None, 10.0, 5.0)         # clean, then background this many dB below the speech
SOUND_ALIKES = ["hey jarvis", "hey jesus", "he leaves", "she believes", "hey steve", "cheese please", "nice sleeves",
                "hey genie", "the thieves", "hey keanu reeves", "jeez, that was close", "hey these jeans fit"]


def voices(folder: Path = VOICES) -> list[Path]:
    return sorted(p for p in folder.rglob("*.onnx") if p.with_suffix(".onnx.json").exists())


def requests(phrase: str) -> list[str]:
    """The phrase alone and leading a request, as people say it."""
    return [phrase, f"{phrase}, what time is it?", f"{phrase}. turn on the lights."]


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly
    if rate == 16000:
        return audio.astype(np.float32)
    g = np.gcd(16000, rate)
    return resample_poly(audio.astype(np.float32), 16000 // g, rate // g).astype(np.float32)


def mix(speech: np.ndarray, noise: np.ndarray, below_db: float) -> np.ndarray:
    """Speech with noise under it, the noise ``below_db`` dB quieter (by RMS); padded 0.5 s each side."""
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) or 1e-9   # noqa: E731
    level = rms(speech)                          # the speech itself, not counting the silent padding
    pad = np.zeros(8000, dtype=np.float32)
    speech = np.concatenate([pad, speech, pad])
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / max(len(noise), 1))))
    noise = noise[: len(speech)].astype(np.float32)
    noise = noise * (level / rms(noise)) * 10 ** (-below_db / 20)
    out = speech + noise
    peak = float(np.max(np.abs(out))) or 1.0
    return out / peak * 0.9 if peak > 1.0 else out


def write(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def read16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def say(voice, text: str, speed: float) -> np.ndarray:
    from piper import SynthesisConfig
    chunks = list(voice.synthesize(text, SynthesisConfig(length_scale=speed)))
    audio = np.concatenate([np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]).astype(np.float32)
    return to_16k(audio / 32768, chunks[0].sample_rate)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phrase", default="hey jeeves")
    ap.add_argument("--noise", type=Path, default=Path.home() / ".cache/homeai/wake-train/audioset_16k")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    from piper import PiperVoice
    rng = random.Random(args.seed)
    noises = sorted(args.noise.glob("*.wav"))
    out = bench_path("wake-phrases")
    index = []
    for kind, texts in (("positive", requests(args.phrase)), ("negative", SOUND_ALIKES)):
        (out / kind).mkdir(parents=True)
        for vpath in voices():
            voice = PiperVoice.load(str(vpath))
            for ti, text in enumerate(texts):
                for speed in SPEEDS:
                    speech = say(voice, text, speed)
                    for below in NOISE_DB:
                        if below is not None and not noises:
                            continue
                        noise_file = rng.choice(noises) if below is not None else None
                        audio = mix(speech, read16k(noise_file), below) if noise_file else \
                            np.concatenate([np.zeros(8000, np.float32), speech, np.zeros(8000, np.float32)])
                        name = f"{vpath.stem}-{ti}-{speed}-{'clean' if below is None else f'{below:g}db'}.wav"
                        write(out / kind / name, audio)
                        index.append({"kind": kind, "file": name, "voice": vpath.stem, "text": text, "speed": speed,
                                      "noise_db_below": below, "noise": noise_file.name if noise_file else None})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{sum(1 for i in index if i['kind'] == 'positive')} positive, "
          f"{sum(1 for i in index if i['kind'] == 'negative')} negative -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
