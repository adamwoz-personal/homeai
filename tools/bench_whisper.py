#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Benchmark Whisper models for accuracy and speed.

Motivation: the live assistant transcribed "What is the capital of France?" as
"What **if** the capital of France?" using ``ggml-base.en``. That is a word
error in a five-word sentence, so STT - not the LLM - was the weakest link.

Method: synthesise each phrase with Piper, transcribe with each model, report
word error rate and wall-clock time.

Caveat, stated plainly: Piper speech is cleaner than a real room, so absolute
WER here is optimistic. The **ranking** and the **relative latency cost** are
the useful outputs, not the absolute numbers.

Usage:
    .venv/bin/python tools/bench_whisper.py
    .venv/bin/python tools/bench_whisper.py --models base.en small.en
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeai.config import load  # noqa: E402

PHRASES = [
    "What is the capital of France?",
    "Turn off the lights in the living room.",
    "Set a timer for twenty five minutes.",
    "What is the weather going to be like tomorrow?",
    "How many tablespoons are in a quarter cup?",
    "Add milk and eggs to the shopping list.",
    "What time is my first meeting on Monday?",
    "Play something quiet in the kitchen.",
    "How tall is Mount Everest in feet?",
    "Remind me to take the bins out tonight.",
]


# Whisper legitimately writes "twenty five" as "25". Counting that as an error
# inflates every model's WER by the same amount and hides the real differences,
# so digits are folded to words on both sides before comparison.
_NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
}


def _expand_number(token: str) -> list[str]:
    if not token.isdigit():
        return [token]
    if token in _NUM_WORDS:
        return [_NUM_WORDS[token]]
    n = int(token)
    if 21 <= n <= 99:
        tens, ones = divmod(n, 10)
        return [_NUM_WORDS[str(tens * 10)], _NUM_WORDS[str(ones)]]
    return [token]


def normalise(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    out: list[str] = []
    for token in text.split():
        out.extend(_expand_number(token))
    return out


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by reference length."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def degrade(wav: Path, reverb: int, snr_db: float) -> bool:
    """Approximate a real room: reverberation plus steady background noise.

    Clean Piper audio makes every model look perfect, which tells us nothing -
    the observed live failure happened on far-field microphone audio. Adding
    reverb and noise is a crude but discriminating proxy.
    """
    try:
        wet = wav.with_suffix(".wet.wav")
        r = subprocess.run(
            ["sox", str(wav), str(wet), "reverb", str(reverb), "50", "100",
             "gain", "-n", "-3"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return False

        noise = wav.with_suffix(".noise.wav")
        dur = subprocess.run(["soxi", "-D", str(wet)], capture_output=True, text=True, timeout=30)
        seconds = float(dur.stdout.strip() or 3.0)
        # Brown noise approximates HVAC / room rumble better than white noise.
        r = subprocess.run(
            ["sox", "-n", "-r", "16000", "-c", "1", str(noise),
             "synth", f"{seconds:.2f}", "brownnoise", "gain", f"-{abs(snr_db):.0f}"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return False

        mixed = wav.with_suffix(".mix.wav")
        r = subprocess.run(
            ["sox", "-m", str(wet), str(noise), str(mixed), "rate", "16000", "channels", "1"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return False
        mixed.replace(wav)
        return True
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def synthesise(cfg, text: str, out_wav: Path) -> bool:
    """Render text to a 16 kHz mono WAV, which is what whisper.cpp requires."""
    try:
        piper = subprocess.run(
            [str(cfg.tts.binary), "-m", str(cfg.tts.model_path), "--output_file", str(out_wav)],
            input=text, capture_output=True, text=True, timeout=60,
        )
        if piper.returncode != 0:
            print(f"  piper failed: {piper.stderr.strip()[:200]}")
            return False
        resampled = out_wav.with_suffix(".16k.wav")
        sox = subprocess.run(
            ["sox", str(out_wav), "-r", "16000", "-c", "1", str(resampled)],
            capture_output=True, text=True, timeout=60,
        )
        if sox.returncode != 0:
            print(f"  sox failed: {sox.stderr.strip()[:200]}")
            return False
        resampled.replace(out_wav)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  synthesis error: {exc}")
        return False


def transcribe(cfg, model: Path, wav: Path) -> tuple[str, float]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [str(cfg.stt.binary), "-m", str(model), "-f", str(wav),
             "-nt", "--no-prints", "-t", "8"],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.SubprocessError as exc:
        return f"(error: {exc})", time.monotonic() - start
    elapsed = time.monotonic() - start
    return (proc.stdout.strip() if proc.returncode == 0 else ""), elapsed


def main() -> int:
    cfg = load()
    models_dir = Path(cfg.stt.model).parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--degrade", action="store_true",
                    help="add reverb and background noise to mimic a real room")
    ap.add_argument("--reverb", type=int, default=40)
    ap.add_argument("--snr", type=float, default=22.0,
                    help="lower = noisier; 22 is a quiet room, 12 is noisy")
    args = ap.parse_args()

    if args.models:
        candidates = [models_dir / f"ggml-{m}.bin" for m in args.models]
    else:
        candidates = sorted(
            p for p in models_dir.glob("ggml-*.bin") if not p.name.startswith("for-tests")
        )
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        print("no models found in", models_dir)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        print(f"synthesising {len(PHRASES)} phrases with Piper...")
        wavs: list[tuple[str, Path]] = []
        for i, phrase in enumerate(PHRASES):
            wav = tmpdir / f"p{i}.wav"
            if not synthesise(cfg, phrase, wav):
                continue
            if args.degrade and not degrade(wav, args.reverb, args.snr):
                print("  degradation failed; using clean audio")
            wavs.append((phrase, wav))
        if not wavs:
            print("no audio synthesised; cannot benchmark")
            return 1
        mode = f"degraded (reverb {args.reverb}, SNR {args.snr:.0f}dB)" if args.degrade else "clean"
        print(f"  {len(wavs)} usable clips [{mode}]\n")

        results = []
        for model in candidates:
            total_wer = total_time = 0.0
            mistakes: list[tuple[str, str]] = []
            for phrase, wav in wavs:
                text, elapsed = transcribe(cfg, model, wav)
                wer = word_error_rate(phrase, text)
                total_wer += wer
                total_time += elapsed
                if wer > 0:
                    mistakes.append((phrase, text))
            n = len(wavs)
            r = {
                "name": model.name.replace("ggml-", "").replace(".bin", ""),
                "mb": model.stat().st_size / 1e6,
                "wer": total_wer / n * 100,
                "sec": total_time / n,
                "mistakes": mistakes,
            }
            results.append(r)
            print(f"{r['name']:<18} WER {r['wer']:5.1f}%   {r['sec']:5.2f}s/clip   {r['mb']:6.0f} MB")

        print("\n" + "=" * 58)
        print(f"{'model':<18} {'WER':>7} {'sec/clip':>10} {'size':>9}")
        print("-" * 58)
        for r in sorted(results, key=lambda x: x["wer"]):
            print(f"{r['name']:<18} {r['wer']:6.1f}% {r['sec']:9.2f}s {r['mb']:8.0f}MB")

        print("\nerrors by model:")
        for r in results:
            if not r["mistakes"]:
                print(f"  {r['name']}: none")
                continue
            print(f"  {r['name']}:")
            for want, got in r["mistakes"][:4]:
                print(f"    want: {want}")
                print(f"    got : {got or '(nothing)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
