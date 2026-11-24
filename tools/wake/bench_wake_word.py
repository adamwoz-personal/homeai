#!/usr/bin/env python3
"""How often does a wake model fire when nobody said its name? (plans/WAKE_WORD_PLAN.md)

Why: before "Hey Jeeves" replaces "Hey Jarvis" it must not wake on household talk. Jarvis's listen probe
(homeai/listen_probe.py) has saved every stretch of room speech since 2026-11-13 (~33 h, real family voices, TV,
through the real microphone), each with its transcript. This plays every clip through each model exactly as
Jarvis does (openWakeWord, 80 ms frames, threshold, 1.5 s refractory, a silence flush between clips), counts
fires per hour, and lists every fire with its transcript, so a fire on a clip that said the name can be told
apart from a false one. Optional --positives: a folder of WAVs that do say the phrase (recall).

Usage:
    .venv/bin/python tools/wake/bench_wake_word.py --model hey_jarvis --model PATH/hey_jeeves.onnx
        [--threshold 0.5] [--clips ~/.local/share/homeai/listen-probe] [--limit N] [--workers 12]
        [--positives DIR]
Output: ~/homeai-bench/wake-bench-<time>/result.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.benchio import bench_path  # noqa: E402

FRAME = 1280                     # 80 ms at 16 kHz: what Jarvis feeds openWakeWord
FRAME_S = FRAME / 16000
FLUSH_S = 2.0                    # silence between clips, so one clip's sound never carries into the next
REFRACTORY_S = 1.5               # homeai/config.py WakeConfig.refractory_s


def fires(scores: list[float], threshold: float, refractory_s: float = REFRACTORY_S,
          frame_s: float = FRAME_S) -> list[tuple[float, float]]:
    """(time in s, score) of each wake: a score at or over the threshold, then nothing for the refractory time."""
    out, quiet_until = [], -1.0
    for i, score in enumerate(scores):
        t = i * frame_s
        if score >= threshold and t >= quiet_until:
            out.append((round(t, 2), round(float(score), 3)))
            quiet_until = t + refractory_s
    return out


def says(text: str, name_words: list[str]) -> bool:
    """The transcript has any of the name's words ("jarvis"), so a fire there may be a real wake."""
    return any(re.search(rf"\b{re.escape(w)}", text.lower()) for w in name_words)


def name_words(model: str) -> list[str]:
    stem = Path(model).stem.lower()
    stem = re.sub(r"_v\d.*$", "", stem)
    return [w for w in stem.split("_") if w not in ("hey", "ok", "okay")]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: not 16 kHz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


_MODEL = None


def _load(model: str):
    global _MODEL
    from homeai.wake import OpenWakeWordDetector, quiet_missing_cuda
    from openwakeword.model import Model
    path = OpenWakeWordDetector.resolve_model_path(model)
    if not path:
        raise SystemExit(f"no wake model {model!r}")
    quiet_missing_cuda()
    _MODEL = Model(wakeword_model_paths=[path])


def scores_of(audio: np.ndarray) -> list[float]:
    silence = np.zeros(FRAME, dtype=np.int16)
    for _ in range(int(FLUSH_S / FRAME_S)):
        _MODEL.predict(silence)
    _MODEL.reset()
    out = []
    padded = np.concatenate([audio, np.zeros(int(0.5 * 16000), dtype=np.int16)])  # let a phrase at the end finish
    for start in range(0, len(padded) - FRAME + 1, FRAME):
        out.append(float(max(_MODEL.predict(padded[start:start + FRAME]).values(), default=0.0)))
    return out


def _run(job: tuple[str, list[str], float]) -> list[dict]:
    model, paths, threshold = job
    _load(model)
    rows = []
    for p in map(Path, paths):
        try:
            audio = read_wav(p)
        except (ValueError, wave.Error, EOFError) as exc:
            rows.append({"clip": p.name, "error": str(exc)})
            continue
        s = scores_of(audio)
        rows.append({"clip": p.name, "seconds": round(len(audio) / 16000, 2),
                     "max": round(float(max(s, default=0.0)), 3), "fires": fires(s, threshold)})
    return rows


def transcript(wav: Path) -> str:
    meta = wav.with_suffix(".json")
    try:
        return (json.loads(meta.read_text()).get("stt") or {}).get("text", "") or ""
    except (OSError, ValueError):
        return ""


def bench(model: str, wavs: list[Path], threshold: float, workers: int) -> list[dict]:
    chunks = [wavs[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        parts = pool.map(_run, [(model, [str(p) for p in c], threshold) for c in chunks if c])
    return [row for part in parts for row in part]


def summary(model: str, rows: list[dict], texts: dict[str, str]) -> dict:
    words = name_words(model)
    good = [r for r in rows if "error" not in r]
    hours = sum(r["seconds"] for r in good) / 3600
    fired = [r for r in good if r["fires"]]
    named = [r for r in fired if says(texts.get(r["clip"], ""), words)]
    false = [r for r in fired if r not in named]
    n_false = sum(len(r["fires"]) for r in false)
    return {"model": model, "clips": len(good), "hours": round(hours, 2), "errors": len(rows) - len(good),
            "name_words": words, "clips_saying_name": sum(1 for r in good if says(texts.get(r["clip"], ""), words)),
            "fires_on_name": sum(len(r["fires"]) for r in named), "false_fires": n_false,
            "false_per_hour": round(n_false / hours, 3) if hours else None,
            "false": [{"clip": r["clip"], "fires": r["fires"], "text": texts.get(r["clip"], "")} for r in false],
            "near_misses": sorted(({"clip": r["clip"], "max": r["max"], "text": texts.get(r["clip"], "")}
                                   for r in good if not r["fires"] and r["max"] >= 0.2),
                                  key=lambda x: -x["max"])[:30]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", required=True, help="a bundled name (hey_jarvis) or an .onnx path")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--clips", type=Path, default=Path.home() / ".local/share/homeai/listen-probe")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--positives", type=Path, default=None, help="WAVs that say the phrase: recall")
    args = ap.parse_args(argv)
    wavs = sorted(args.clips.glob("*.wav"))[: args.limit or None]
    texts = {w.name: transcript(w) for w in wavs}
    out = bench_path("wake-bench")
    out.mkdir(parents=True)
    result = {"threshold": args.threshold, "clips_dir": str(args.clips), "models": []}
    for model in args.model:
        rows = bench(model, wavs, args.threshold, args.workers)
        s = summary(model, rows, texts)
        if args.positives:
            pos = bench(model, sorted(args.positives.glob("*.wav")), args.threshold, args.workers)
            hit = [r for r in pos if r.get("fires")]
            s["recall"] = {"clips": len(pos), "woke": len(hit), "rate": round(len(hit) / len(pos), 3) if pos else None,
                           "missed": [r["clip"] for r in pos if not r.get("fires")]}
        result["models"].append(s)
        print(json.dumps({k: v for k, v in s.items() if k not in ("false", "near_misses")}), flush=True)
        for f in s["false"][:15]:
            print("   false:", f["fires"], repr(f["text"][:90]), flush=True)
        (out / "result.json").write_text(json.dumps(result, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
