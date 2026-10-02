#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Synthesize the same Jarvis-style reply with every Piper voice, time it, and play it.

    .venv/bin/python tools/piper_voice_samples.py            # synth + timings
    .venv/bin/python tools/piper_voice_samples.py --play     # ...and play them in turn
    .venv/bin/python tools/piper_voice_samples.py --play --only ryan,alba

Each sample opens with "Voice N, <name>" spoken in that voice, so you can
tell them apart while listening. Timing runs piper exactly as Jarvis does:
a fresh process per utterance with streamed raw PCM. So "first audio"
includes model load and is the delay you would hear. WAVs are written to
~/homeai-bench/voices/ and timings are appended to
~/homeai-bench/voices.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from homeai.config import AudioConfig, TtsConfig  # noqa: E402

SAMPLE = ("Good evening. The kettle needs about four minutes, so you have time for a thought. "
          "Seneca said we suffer more often in imagination than in reality. "
          "I suspect he never waited for a kettle. Want me to keep going?")


def find_voices(dirs: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for d in dirs:
        for onnx in sorted(d.glob("*.onnx")):
            if onnx.with_suffix(".onnx.json").exists():
                seen.setdefault(onnx.stem, onnx)
    return list(seen.values())


def short_name(stem: str) -> str:
    # en_US-hfc_female-medium -> "hfc female, medium, US"
    lang, name, quality = stem.split("-", 2)
    return f"{name.replace('_', ' ')}, {quality}, {lang.split('_')[-1]}"


def sample_rate(onnx: Path) -> int:
    data = json.loads(onnx.with_suffix(".onnx.json").read_text())
    return int(data.get("audio", {}).get("sample_rate", 22050))


def synth(piper: str, onnx: Path, text: str, out: Path) -> dict:
    rate = sample_rate(onnx)
    start = time.monotonic()
    proc = subprocess.Popen([piper, "--model", str(onnx), "--output-raw"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    proc.stdin.write(text.encode())
    proc.stdin.close()
    first = None
    chunks = []
    while chunk := proc.stdout.read(4096):
        if first is None:
            first = time.monotonic()
        chunks.append(chunk)
    proc.wait(timeout=120)
    total = time.monotonic() - start
    pcm = b"".join(chunks)
    if proc.returncode != 0 or not pcm:
        raise RuntimeError(f"piper failed for {onnx.name} (exit {proc.returncode})")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    audio_s = len(pcm) / 2 / rate
    return {"voice": onnx.stem, "rate": rate, "first_audio_s": round(first - start, 3),
            "synth_s": round(total, 3), "audio_s": round(audio_s, 2),
            "rtf": round(total / audio_s, 3), "size_mb": round(onnx.stat().st_size / 1e6, 1)}


def main(argv: list[str] | None = None) -> int:
    tts = TtsConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", nargs="+", type=Path,
                    default=[tts.model_dir, tts.model_dir / "voices"])
    ap.add_argument("--out", type=Path, default=Path.home() / "homeai-bench" / "voices")
    ap.add_argument("--text", default=SAMPLE)
    ap.add_argument("--only", help="comma-separated substrings of voice names")
    ap.add_argument("--play", action="store_true", help="play each sample after synthesis")
    ap.add_argument("--device", help="ALSA device (default: what Jarvis uses)")
    args = ap.parse_args(argv)

    piper = str(tts.binary) if tts.binary.exists() else "piper"
    voices = find_voices(args.dirs)
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        voices = [v for v in voices if any(w in v.stem for w in wanted)]
    # Current voice first, as the reference.
    voices.sort(key=lambda v: (v.stem != tts.voice, v.stem))
    if not voices:
        print("no voices found in", *args.dirs, file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device or AudioConfig().output_device

    rows = []
    for n, onnx in enumerate(voices, 1):
        label = short_name(onnx.stem) + (" (current)" if onnx.stem == tts.voice else "")
        wav = args.out / f"{n:02d}-{onnx.stem}.wav"
        row = synth(piper, onnx, f"Voice {n}, {short_name(onnx.stem)}. {args.text}", wav)
        row.update(n=n, label=label, wav=str(wav), ts=time.time())
        rows.append(row)
        print(f"{n}. {label:34s} first audio {row['first_audio_s']:.2f}s  "
              f"synth {row['synth_s']:.2f}s for {row['audio_s']:.1f}s of speech "
              f"(RTF {row['rtf']:.2f})", flush=True)
    with open(args.out.parent / "voices.jsonl", "a") as log:
        for row in rows:
            log.write(json.dumps(row) + "\n")

    if args.play:
        for row in rows:
            print(f"playing {row['n']}. {row['label']}", flush=True)
            subprocess.run(["aplay", "-q", "-D", device, row["wav"]], check=False)
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
