#!/usr/bin/env python3
"""Can a local vision model catch extra heads, extra riders and riders standing in the dragon? (ART_FEATURES_PLAN F4)

Why: "generate, check, retry" needs a checker that agrees with a person. This
asks a vision model fixed questions about the rider bench pictures
(tools/art/bench_rider.py) and compares the answers with labels Adam's
reviewer gave by eye, and times each check.

Ollama keeps one model loaded (OLLAMA_MAX_LOADED_MODELS=1), so this unloads
Jarvis's voice model while it runs and re-pins it at the end. Run it when nobody
is talking to Jarvis. Read-only otherwise; nothing is spoken.

Writes ~/homeai-bench/vision-check-<time>/result.json.

Usage:
    .venv/bin/python tools/art/probe_vision_check.py BENCH_DIR [--model minicpm-v]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai import gpu_mode as G  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"
# Labels by eye, checked at full size (thumbnails misled: a first pass called old-005 one-headed, and all
# three vision models disagreed correctly). Per picture: riders on the dragon, dragon heads, seated astride.
LABELS = {
    "old-005-1": {"riders": 2, "heads": 2, "seated": True}, "old-005-2": {"riders": 3, "heads": 2, "seated": True},
    "old-005-3": {"riders": 2, "heads": 2, "seated": True}, "old-005-4": {"riders": 2, "heads": 1, "seated": True},
    **{f"{case}-{n}": {"riders": 1, "heads": 1, "seated": n == 1} for case in ("old-006", "new-006") for n in (1, 2, 3, 4)},
    **{f"counts-{n}": {"riders": 1, "heads": 1, "seated": True} for n in (1, 2, 3, 4)},
}
QUESTIONS = {
    "riders": "How many people are riding the dragon? Answer with one number only.",
    "heads": "How many heads does the dragon have? Answer with one number only.",
    "seated": "Is the person sitting on the dragon's back, with their legs on either side of it? Answer yes or no only.",
}


def ask(model: str, image: Path, question: str) -> str:
    payload = {"model": model, "stream": False, "options": {"temperature": 0}, "keep_alive": "5m",
               "messages": [{"role": "user", "content": question,
                             "images": [base64.b64encode(image.read_bytes()).decode()]}]}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as reply:
        return json.loads(reply.read())["message"]["content"].strip()


def parse(kind: str, answer: str):
    low = answer.lower()
    if kind == "seated":
        return True if low.startswith("yes") else False if low.startswith("no") else None
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "zero": 0, "none": 0}
    match = re.search(r"\d+", low)
    if match:
        return int(match.group())
    return next((v for w, v in words.items() if re.search(rf"\b{w}\b", low)), None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bench", type=Path)
    ap.add_argument("--model", default="minicpm-v")
    args = ap.parse_args()
    out = Path.home() / "homeai-bench" / f"vision-check-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    rows = []
    try:
        for image in sorted(args.bench.glob("*-[0-9].png")):
            case = image.stem
            if case not in LABELS:
                continue
            for kind, question in QUESTIONS.items():
                start = time.monotonic()
                answer = ask(args.model, image, question)
                got = parse(kind, answer)
                rows.append({"image": image.name, "kind": kind, "answer": answer[:80], "got": got,
                             "want": LABELS[case][kind], "ok": got == LABELS[case][kind],
                             "seconds": round(time.monotonic() - start, 2)})
                print(f"{image.name:16} {kind:7} want {LABELS[case][kind]!s:5} got {got!s:5} "
                      f"{'ok' if rows[-1]['ok'] else 'WRONG'} {rows[-1]['seconds']}s  {answer[:50]!r}", flush=True)
    finally:
        switcher._load_voice()                       # put Jarvis's voice model back, pinned
    summary = {kind: f"{sum(r['ok'] for r in rows if r['kind'] == kind)}/{sum(r['kind'] == kind for r in rows)}"
               for kind in QUESTIONS}
    times = sorted(r["seconds"] for r in rows)
    summary.update(model=args.model, median_s=times[len(times) // 2] if times else None,
                   voice_back=switcher.voice_loaded())
    (out / "result.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
