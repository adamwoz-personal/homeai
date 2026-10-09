#!/usr/bin/env python3
"""Does a Whisper prompt ("Hey Jarvis.") help wake verification, or hurt it?

Problem (2026-10-09): real wakes were rejected because Whisper wrote the name
as "jargon" or "George". whisper.cpp's --prompt biases decoding toward
words in the prompt, which should fix that. The risk is the opposite error:
Whisper "hearing" Jarvis in household speech, which is exactly the false wake
the second-stage check exists to stop.

Method: synthesise wake phrases (several Piper voices) and non-wake phrases
(the real false wakes from the logs, plus close names), degrade them with
noise and a simple room reverb, transcribe with and without each prompt, and
score with the daemon's own mentions_wake_word().

    .venv/bin/python tools/eval_wake_prompt.py
    .venv/bin/python tools/eval_wake_prompt.py --snr 5 --prompts "" "Hey Jarvis." "Jarvis"

Caveat: Piper through a synthetic room is not a webcam mic in a kitchen.
Treat the comparison between prompts as the result, not the absolute rates.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from homeai.config import load  # noqa: E402
from homeai.stt import clean_transcript  # noqa: E402
from homeai.wake_verify import mentions_wake_word  # noqa: E402
from piper_voice_samples import find_voices, synth  # noqa: E402

WAKES = [
    "Hey Jarvis. Set a pasta timer for five minutes.",
    "Hey Jarvis, stop the pasta timer.",
    "Hey Jarvis, turn off the foyer lights.",
    "Jarvis, what's the weather tomorrow?",
    "Hey Jarvis! Play some jazz in the kitchen.",
    "Hey Jarvis. How long is left on the timer?",
]
# Real false wakes from the journal, and names Whisper must not turn into Jarvis.
NOT_WAKES = [
    "Love you, Valor.",
    "David.",
    "Cody will be back in a moment.",
    "Thank you. You're welcome.",
    "Hey George, can you pass the salt?",
    "Hey Travis, are you coming tonight?",
    "Marvin, dinner is ready.",
    "That's a lot of jargon for one meeting.",
    "Harvest is in the fall this year.",
    "Okay, perfect. I'm going to put it in for five minutes.",
]


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768, w.getframerate()


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    n = int(len(audio) * 16000 / rate)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)


def degrade(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Room reverb (decaying noise impulse) plus pink-ish background noise."""
    ir_len = int(0.25 * 16000)
    ir = rng.standard_normal(ir_len) * np.exp(-np.linspace(0, 8, ir_len))
    ir[0] = 1.0
    wet = np.convolve(audio, ir * 0.35)[: len(audio)]
    pad = np.zeros(int(0.5 * 16000))
    wet = np.concatenate([pad, wet, pad])
    noise = np.cumsum(rng.standard_normal(len(wet)))
    noise -= np.convolve(noise, np.ones(400) / 400, mode="same")  # drop the DC drift
    noise *= np.sqrt(np.mean(wet ** 2) / np.mean(noise ** 2) / 10 ** (snr_db / 10))
    out = wet + noise
    return out / max(1.0, np.abs(out).max())


def write16k(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def transcribe(cfg, wav: Path, prompt: str) -> str:
    cmd = [str(cfg.stt.binary), "-m", str(cfg.stt.model), "-f", str(wav),
           "-t", str(cfg.stt.threads), "-nt", "--no-prints"]
    if prompt:
        cmd += ["--prompt", prompt]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    return clean_transcript(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+", default=["", "Hey Jarvis."])
    ap.add_argument("--snr", type=float, nargs="+", default=[20, 8])
    ap.add_argument("--show", action="store_true", help="print every transcript")
    args = ap.parse_args()

    cfg = load()
    piper = str(cfg.tts.binary) if cfg.tts.binary.exists() else "piper"
    voices = find_voices([ROOT / "vendor/piper", ROOT / "vendor/piper/voices"])
    rng = np.random.default_rng(7)
    totals = {p: {"wake_hit": 0, "wake_n": 0, "false": 0, "not_n": 0, "missed": [], "falses": []}
              for p in args.prompts}

    with tempfile.TemporaryDirectory(prefix="wake-eval-") as tmp:
        for voice in voices:
            for text, is_wake in [(t, True) for t in WAKES] + [(t, False) for t in NOT_WAKES]:
                raw = Path(tmp) / "raw.wav"
                synth(piper, voice, text, raw)
                audio = to_16k(*load_wav(raw))
                for snr in args.snr:
                    wav = Path(tmp) / "x.wav"
                    write16k(wav, degrade(audio, snr, rng))
                    for prompt in args.prompts:
                        heard = transcribe(cfg, wav, prompt)
                        hit = mentions_wake_word(heard)
                        t = totals[prompt]
                        tag = f"{voice.stem}@{snr:g}dB: {heard!r}"
                        if is_wake:
                            t["wake_n"] += 1
                            t["wake_hit"] += hit
                            if not hit:
                                t["missed"].append(tag)
                        else:
                            t["not_n"] += 1
                            t["false"] += hit
                            if hit:
                                t["falses"].append(tag)
                        if args.show:
                            print(f"{'W' if is_wake else '-'} {'HIT' if hit else '   '} [{prompt!r}] {tag}")
            print(f"done {voice.stem}", file=sys.stderr, flush=True)

    for prompt, t in totals.items():
        print(f"\nprompt {prompt!r}: wakes accepted {t['wake_hit']}/{t['wake_n']}, "
              f"false wakes accepted {t['false']}/{t['not_n']}")
        for m in t["missed"]:
            print(f"   missed   {m}")
        for f in t["falses"]:
            print(f"   FALSE    {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
