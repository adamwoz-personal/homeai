#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Avatar Phase 0 spike: is in-process Piper faster to first audio?

Why: today ``homeai/tts.py`` starts a ``piper`` subprocess per utterance, and
that process reloads the voice model every time. AVATAR_PLAN §3.3 proposes
loading ``PiperVoice`` once in the daemon. Pass criterion (plan §10, Phase 0):
in-process first audio is at least 300 ms faster, with the same audio.

What it measures, per sentence and per engine, without playing anything:
- first_ms: call -> first PCM bytes available (what tts.py times today,
  minus aplay, which is the same for both engines);
- total_ms: call -> all PCM bytes;
- samples: audio length. VITS adds random noise inside the model, so audio
  is not bit-identical between runs; "same audio" here means the same length
  within --tolerance and both engines using the same model and settings.

The subprocess path runs the exact command tts.py runs. The in-process path
loads the voice once (load time reported separately; the daemon pays it once
at start-up).

Usage:
    .venv/bin/python tools/avatar/tts_latency_bench.py [--runs 10] [--tolerance 0.10]
Writes ~/homeai-bench/tts_latency-<stamp>.json. Exit 0 pass, 1 fail, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SENTENCES = [
    "Done.",
    "The den light is on.",
    "It's 7 degrees outside with light rain, so take a coat.",
    "Good night. I've turned off the lights, and the front door is locked.",
]
BENCH_DIR = Path.home() / "homeai-bench"
REQUIRED_GAIN_MS = 300.0


def run_subprocess(binary: str, model: Path, text: str) -> tuple[float, float, int]:
    """Time the exact pipeline tts.py starts: piper --output-raw, text on stdin."""
    started = time.monotonic()
    proc = subprocess.Popen(
        [binary, "--model", str(model), "--output-raw"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    proc.stdin.write(text.encode("utf-8"))
    proc.stdin.close()
    first = 0.0
    nbytes = 0
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        if not first:
            first = time.monotonic()
        nbytes += len(chunk)
    rc = proc.wait(timeout=60)
    total = time.monotonic()
    if rc != 0 or not first:
        err = proc.stderr.read().decode(errors="replace").strip()[-300:]
        raise RuntimeError(f"piper exited {rc}: {err}")
    return (first - started) * 1000, (total - started) * 1000, nbytes // 2


def run_inprocess(voice, text: str) -> tuple[float, float, int]:
    """Same synthesis settings as the CLI defaults, on an already-loaded voice."""
    started = time.monotonic()
    first = 0.0
    samples = 0
    for chunk in voice.synthesize(text):
        data = chunk.audio_int16_bytes
        if not first and data:
            first = time.monotonic()
        samples += len(data) // 2
    total = time.monotonic()
    if not first:
        raise RuntimeError("in-process synthesis produced no audio")
    return (first - started) * 1000, (total - started) * 1000, samples


def summarise(values: list[float]) -> dict:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {"p50": round(statistics.median(ordered), 1), "p95": round(p95, 1),
            "min": round(ordered[0], 1), "max": round(ordered[-1], 1)}


def evaluate(rows: list[dict], tolerance: float) -> tuple[bool, list[str]]:
    """Pure verdict over per-sentence results; returns (passed, reasons)."""
    reasons = []
    for row in rows:
        gain = row["subprocess"]["first_ms"]["p50"] - row["inprocess"]["first_ms"]["p50"]
        row["gain_ms"] = round(gain, 1)
        if gain < REQUIRED_GAIN_MS:
            reasons.append(f"{row['text'][:30]!r}: gain {gain:.0f} ms < {REQUIRED_GAIN_MS:.0f}")
        a = row["subprocess"]["samples_p50"]
        b = row["inprocess"]["samples_p50"]
        ratio = abs(a - b) / max(a, 1)
        row["length_diff"] = round(ratio, 3)
        if ratio > tolerance:
            reasons.append(f"{row['text'][:30]!r}: length differs {ratio:.1%}")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()

    try:
        from piper import PiperVoice
        from homeai.config import TtsConfig
    except ImportError as exc:
        print(f"setup error: {exc}")
        return 2
    cfg = TtsConfig()
    binary = str(cfg.binary) if cfg.binary.exists() else None
    if not binary or not cfg.model_path.exists():
        print(f"setup error: piper {cfg.binary} or model {cfg.model_path} missing")
        return 2

    t0 = time.monotonic()
    voice = PiperVoice.load(str(cfg.model_path))
    load_ms = (time.monotonic() - t0) * 1000
    run_inprocess(voice, "Warm up.")  # first ONNX run allocates; the daemon pays this at start

    rows = []
    for text in SENTENCES:
        per = {"subprocess": [], "inprocess": []}
        # Interleave so drift in machine load hits both engines equally.
        for _ in range(args.runs):
            per["subprocess"].append(run_subprocess(binary, cfg.model_path, text))
            per["inprocess"].append(run_inprocess(voice, text))
        row = {"text": text}
        for engine, results in per.items():
            row[engine] = {
                "first_ms": summarise([r[0] for r in results]),
                "total_ms": summarise([r[1] for r in results]),
                "samples_p50": int(statistics.median(r[2] for r in results)),
            }
        rows.append(row)

    passed, reasons = evaluate(rows, args.tolerance)
    report = {"when": datetime.now().isoformat(timespec="seconds"), "voice": cfg.model_path.name,
              "runs": args.runs, "inprocess_load_ms": round(load_ms, 1),
              "required_gain_ms": REQUIRED_GAIN_MS, "passed": passed, "reasons": reasons,
              "sentences": rows}
    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"tts_latency-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"voice {cfg.model_path.name}; in-process load {load_ms:.0f} ms (once); {args.runs} runs each")
    print(f"{'sentence':32} {'sub first p50':>14} {'inproc p50':>11} {'gain':>7} {'len diff':>9}")
    for row in rows:
        print(f"{row['text'][:32]:32} {row['subprocess']['first_ms']['p50']:>12.0f}ms"
              f" {row['inprocess']['first_ms']['p50']:>9.0f}ms {row['gain_ms']:>5.0f}ms"
              f" {row['length_diff']:>8.1%}")
    print(("PASS" if passed else "FAIL: " + "; ".join(reasons)) + f"  -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
