#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Can one recording be split by speaker? (ART_FEATURES_PLAN S1, go/no-go spike)

In Adam's battleship session a child's words and Adam's arrived in one
recording, so a whole-sentence speaker check couldn't separate them. This
bench makes mixed clips (owner speaks, then another voice, or the reverse),
then measures:

- whisper's word timestamps (``-ml 1 -sow``): how far the word nearest the
  switch lands from the true switch time;
- sliding-window similarity to the owner's profile (``--window`` s, 0.25 s
  hop): owner windows vs other windows, and how many windows fall on the
  wrong side of ``different_below``;
- the resulting text: owner words kept, other words dropped;
- time per clip.

Voices are Piper samples (~/homeai-bench/voices), so this is synthetic:
clean, no room noise, adult voices only. It says whether the method can work,
not how it does on the household.

    .venv/bin/python tools/bench_segment_speakers.py [--owner joe] [--window 1.5]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homeai.config import SpeakerConfig, SttConfig  # noqa: E402
from homeai.speaker import SpeakerEmbedder, cosine  # noqa: E402
from homeai.speaker_split import parse_words  # noqa: E402
from homeai.stt import write_wav  # noqa: E402

SR = 16000
VOICES = Path.home() / "homeai-bench" / "voices"


def read_voice(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if rate == SR:
        return raw
    t_old = np.arange(len(raw)) / rate
    t_new = np.arange(int(len(raw) * SR / rate)) / SR
    return np.interp(t_new, t_old, raw).astype(np.float32)


def words_with_times(binary: Path, model: Path, audio: np.ndarray) -> list[tuple[float, float, str]]:
    with tempfile.TemporaryDirectory(prefix="seg-") as tmp:
        wav = Path(tmp) / "a.wav"
        write_wav(wav, audio, SR)
        out = subprocess.run([str(binary), "-m", str(model), "-f", str(wav), "-ml", "1", "-sow", "--no-prints"],
                             capture_output=True, text=True, timeout=60).stdout
    return parse_words(out)


def window_sims(embedder, profile, audio, window, hop=0.25):
    out, n = [], int(window * SR)
    for start in range(0, max(1, len(audio) - n + 1), int(hop * SR)):
        emb = embedder.embed(audio[start:start + n])
        out.append((start / SR, (start + n) / SR, None if emb is None else cosine(emb, profile)))
    return out


def label_words(words, sims, below):
    """Each word: 'other' if every scored window covering its middle is confidently not the owner."""
    labels = []
    for start, end, text in words:
        mid = (start + end) / 2
        covering = [s for a, b, s in sims if a <= mid <= b and s is not None]
        labels.append("other" if covering and max(covering) < below else "owner")
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--owner", default="joe")
    ap.add_argument("--window", type=float, default=1.5)
    args = ap.parse_args()
    files = sorted(VOICES.glob("*.wav"))
    owner_file = next(f for f in files if args.owner in f.name)
    owner = read_voice(owner_file)
    embedder = SpeakerEmbedder(SpeakerConfig().model_path)
    ok, problem = embedder.load()
    if not ok:
        print(problem)
        return 2
    below = SpeakerConfig().different_below
    stt = SttConfig()
    enrol, rest = owner[: 7 * SR], owner[7 * SR:]
    profile = embedder.embed(enrol)
    gap = np.zeros(int(0.4 * SR), dtype=np.float32)
    out = Path.home() / "homeai-bench" / f"segment-speakers-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    rows = []
    for other_file in (f for f in files if f != owner_file):
        other = read_voice(other_file)[: 4 * SR]
        mine = rest[: 4 * SR]
        for order in ("owner-first", "other-first"):
            parts = (mine, gap, other) if order == "owner-first" else (other, gap, mine)
            audio = np.concatenate(parts)
            switch = (len(parts[0]) + len(gap) / 2) / SR
            t0 = time.monotonic()
            words = words_with_times(stt.binary, stt.model, audio)
            t_stt = time.monotonic() - t0
            t0 = time.monotonic()
            sims = window_sims(embedder, profile, audio, args.window)
            t_emb = time.monotonic() - t0
            labels = label_words(words, sims, below)
            truth = [("owner" if (mid < switch) == (order == "owner-first") else "other")
                     for mid in ((a + b) / 2 for a, b, _ in words)]
            own_sims = [s for a, b, s in sims if s is not None and ((b <= switch) == (order == "owner-first"))
                        and not a < switch < b]
            oth_sims = [s for a, b, s in sims if s is not None and ((b <= switch) != (order == "owner-first"))
                        and not a < switch < b]
            nearest = min((abs(a - switch) for a, _, _ in words), default=None)
            row = {"other": other_file.stem, "order": order, "words": len(words),
                   "owner_kept": sum(1 for t, l in zip(truth, labels) if t == "owner" and l == "owner"),
                   "owner_words": truth.count("owner"),
                   "other_dropped": sum(1 for t, l in zip(truth, labels) if t == "other" and l == "other"),
                   "other_words": truth.count("other"),
                   "owner_sim_min": round(min(own_sims), 2) if own_sims else None,
                   "other_sim_max": round(max(oth_sims), 2) if oth_sims else None,
                   "switch_err_s": None if nearest is None else round(nearest, 2),
                   "stt_s": round(t_stt, 2), "embed_s": round(t_emb, 2),
                   "kept_text": " ".join(w for (_, _, w), l in zip(words, labels) if l == "owner")}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "kept_text"}))
    total_owner = sum(r["owner_words"] for r in rows)
    total_other = sum(r["other_words"] for r in rows)
    summary = {"owner_words_kept": f"{sum(r['owner_kept'] for r in rows)}/{total_owner}",
               "other_words_dropped": f"{sum(r['other_dropped'] for r in rows)}/{total_other}",
               "below": below, "window_s": args.window,
               "embed_s_median": statistics.median(r["embed_s"] for r in rows),
               "stt_s_median": statistics.median(r["stt_s"] for r in rows)}
    (out / "result.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
