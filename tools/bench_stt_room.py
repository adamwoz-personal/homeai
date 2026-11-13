"""Which whisper model hears the room best? Real listen-probe clips through each installed model, timed.

Why: the 2026-11 bake-off (tools/bench_whisper.py, ROUTING_GUIDE "Benchmarks lie") used Piper speech plus fake
reverb, and picked small.en for near-field voice turns. Interactive mode hears people across the room, and on
2026-11-13 Adam saw small.en mishear real room talk. This runs a random sample of real clips (some that small.en
heard words in, some it called noise) through each model on the CPU, as Jarvis would, and writes:

  ~/homeai-bench/stt-room-<stamp>.jsonl   one row per clip x model: text, wall_s, rtf
  ~/homeai-bench/stt-room-<stamp>.md      side by side, for Adam to say which line is right

There is no ground truth here; "agree" is word agreement with the biggest model, a hint, not a score.

Models: whisper ggml-<name>.bin, or "parakeet" (Parakeet TDT 0.6B v3 via parakeet-cli).
Usage: nice -n 10 .venv/bin/python tools/bench_stt_room.py [--clips 40] [--noise 10] [--models small.en ...]
Reads ~/.local/share/homeai/listen-probe (only clips the listen worker already transcribed).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from homeai import listen_probe  # noqa: E402
from homeai.config import SttConfig  # noqa: E402
from homeai.stt import Transcriber, Transcript, clean_transcript  # noqa: E402

BENCH = Path.home() / "homeai-bench"
MODELS = ["small.en", "medium.en", "parakeet", "large-v3-turbo"]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def agreement(a: str, b: str) -> float | None:
    """Share of words in common (order-free, by count): 1.0 same words, 0.0 nothing shared."""
    wa, wb = words(a), words(b)
    if not wa and not wb:
        return 1.0
    common = sum(min(wa.count(w), wb.count(w)) for w in set(wa))
    return round(2 * common / (len(wa) + len(wb)), 3)


def pick(folder: Path, clips: int, noise: int, seed: int = 13) -> list[tuple[Path, dict]]:
    heard, quiet = [], []
    for path in sorted(folder.glob("*.json")):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        stt = meta.get("stt")
        if not stt or not path.with_suffix(".wav").exists():
            continue
        (heard if stt.get("ok") else quiet).append((path, meta))
    rng = random.Random(seed)
    return rng.sample(heard, min(clips, len(heard))) + rng.sample(quiet, min(noise, len(quiet)))


class Parakeet:
    """NVIDIA Parakeet TDT 0.6B v3 through whisper.cpp's parakeet-cli (built beside whisper-cli), CPU only."""

    def __init__(self, binary: Path, model: Path, threads: int) -> None:
        self.cmd = [str(binary), "-m", str(model), "-t", str(threads), "-ng", "-np", "-f"]

    def transcribe_file(self, wav: Path) -> Transcript:
        proc = subprocess.run([*self.cmd, str(wav)], capture_output=True, encoding="utf-8", errors="replace",
                              timeout=60)
        text = clean_transcript(proc.stdout)
        return Transcript(ok=proc.returncode == 0 and bool(text), text=text, error=proc.stderr[-200:])


def model_for(name: str, base: SttConfig):
    if name == "parakeet":
        path = base.model.parent / "ggml-parakeet-tdt-0.6b-v3-f16.bin"
        return path, Parakeet(base.binary.parent / "parakeet-cli", path, base.threads)
    path = base.model.parent / f"ggml-{name}.bin"
    return path, Transcriber(replace(base, model=path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--noise", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--folder", type=Path, default=listen_probe.DIR)
    args = ap.parse_args(argv)
    base = SttConfig()
    models = {}
    for name in args.models:
        path, models[name] = model_for(name, base)
        if not path.exists():
            print(f"missing model {path}", file=sys.stderr)
            return 2
    sample = pick(args.folder, args.clips, args.noise)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = BENCH / f"stt-room-{stamp}.jsonl"
    table = []
    with out.open("w") as f:
        for path, meta in sample:
            wav, row = path.with_suffix(".wav"), {"clip": path.stem, "duration": meta.get("duration")}
            for name, stt in models.items():
                t0 = time.monotonic()
                r = stt.transcribe_file(wav)
                wall = time.monotonic() - t0
                row[name] = {"text": r.text if r.ok else "", "wall_s": round(wall, 2),
                             "rtf": round(wall / max(1e-6, float(meta.get("duration") or 0)), 3)}
            ref = row[args.models[-1]]["text"]
            for name in args.models:
                row[name]["agree"] = agreement(row[name]["text"], ref)
            f.write(json.dumps(row) + "\n")
            table.append(row)
            print(row["clip"], {n: row[n]["text"][:50] for n in args.models}, flush=True)
    md = [f"# Room clips, {len(table)} ({stamp})", "",
          "| model | mean s/clip | rtf p50 | agree with " + args.models[-1] + " |", "|---|---|---|---|"]
    for name in args.models:
        walls = sorted(r[name]["wall_s"] for r in table)
        rtfs = sorted(r[name]["rtf"] for r in table)
        agree = [r[name]["agree"] for r in table]
        md.append(f"| {name} | {sum(walls) / max(1, len(walls)):.2f} | {rtfs[len(rtfs) // 2] if rtfs else '-'} | "
                  f"{sum(agree) / max(1, len(agree)):.2f} |")
    md += ["", "Which line is right? (clip id, then each model)", ""]
    for r in table:
        md.append(f"**{r['clip']}** ({r['duration']} s)")
        md += [f"- {name}: {r[name]['text'] or '(nothing)'}" for name in args.models]
        md.append("")
    out.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md[:2 + 2 + len(args.models)]))
    print(f"-> {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
