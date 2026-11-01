#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Recall through the real request builder, on a scratch database (plan §11, Phase 1).

Why: unit tests use a hash embedder and fake timings. This seeds a throwaway
database, loads the real ONNX embedder, and sends utterances through the
same path the daemon uses (``ConversationMemory.build_request`` ->
``MemoryService.recall``), checking what reaches the prompt and how long
recall takes. Run it after changing recall, the embedder or the similarity
floor.

    .venv/bin/python tools/memory/recall_live.py [--reps 20] [--out ~/homeai-bench]

Checks: the right memory reaches the request; pleasantries and unknown
voices get none; home commands are taken by the daemon's fast path before
any recall; one person never recalls another's facts; the block passes
the leaked-markup detector. Reports recall p50/p95 (plan target: p95 < 40 ms).
The scratch database ``homeai_recall_live_<pid>`` is dropped at the end.
Exit 0 if every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HEADER = "Things you remember from earlier conversations"
FACTS = {
    "adam": [("profile", "Adam keeps a corn snake called Noodle"),
             ("fact", "Adam's sister Kate lives in Denver"),
             ("preference", "Adam prefers metric units for cooking")],
    "ryan": [("fact", "Ryan's secret hideout is behind the garden shed")],
}
# (speaker, utterance, must appear in the request, must not appear in the recall block)
CASES = [
    ("Adam", "what's my snake called again", "Noodle", None),
    ("Adam", "which city does my sister live in", "Denver", None),
    ("Adam", "what beginner snake did you suggest yesterday", "corn snake", None),
    ("Adam", "thank you", None, HEADER),                     # pleasantry: no round trip
    ("Adam", "where is the secret hideout behind the garden shed", None, "hideout"),
    (None, "what's my snake called again", None, HEADER),
]
# Home commands never reach the request builder: the daemon's fast path takes them first.
HOME = ("turn off the kitchen lights", "set a timer for ten minutes")


def seed(store) -> None:
    from homeai.memory_store.store import Episode

    store.ensure_person("adam", "Adam", is_admin=True, voice_handle="Adam")
    store.ensure_person("ryan", "Ryan", voice_handle="Ryan")
    for user, facts in FACTS.items():
        for kind, summary in facts:
            store.add_fact(user, summary, kind=kind, actor="recall_live")
    store.add_episode(Episode("adam", "what's a good beginner snake",
                              "A corn snake is gentle and easy to keep.",
                              occurred_at=datetime.now(timezone.utc) - timedelta(days=1)))


def run(dsn: str, reps: int) -> dict:
    from homeai.config import Config
    from homeai.memory import ConversationMemory
    from homeai.memory_store.db import Db, migrate
    from homeai.memory_store.embed import DEFAULT_MODEL, OnnxEmbedder
    from homeai.memory_store.maintain import percentile
    from homeai.memory_store.service import MemoryService
    from homeai.memory_store.store import PostgresStore
    from homeai.safety import detect_leaked_markup

    migrate(dsn)
    base = Config().memory
    cfg = replace(base, enabled=True, dsn=dsn, extract=False, ignore_speakers=(),
                  admins=("adam",))
    db = Db(dsn, role="homeai_app").open()
    store = PostgresStore(db, OnnxEmbedder(cfg.embed_model or DEFAULT_MODEL))
    seed(store)
    svc = MemoryService(cfg, store_factory=lambda: store).start()
    try:
        if not svc.wait_ready(30):
            raise RuntimeError("memory service did not become ready")
        results = []
        for speaker, text, want, never in CASES:
            mem = ConversationMemory(long_term=svc.recall)
            mem.switch_speaker(speaker)
            request = mem.build_request(text)
            block = svc.recall(speaker, text)
            ok = (want is None or want in request) and (never is None or never not in block) \
                and not detect_leaked_markup(block)
            ms = []
            if want is not None:
                for _ in range(reps):
                    svc.recall(speaker, text)
                    ms.append(svc.last_recall_ms)
            results.append({"speaker": speaker, "utterance": text, "ok": ok, "want": want,
                            "never": never, "block_chars": len(block),
                            "p50_ms": percentile(ms, 0.5), "p95_ms": percentile(ms, 0.95)})
        from homeai import home_intents

        for text in HOME:
            results.append({"speaker": "Adam", "utterance": text, "want": None,
                            "never": "recall (home fast path)", "block_chars": 0,
                            "ok": home_intents.parse(text) is not None,
                            "p50_ms": None, "p95_ms": None})
        every = [r for r in results if r["p95_ms"] is not None]
        return {"at": datetime.now().astimezone().isoformat(), "embed_model": store.embedder.name,
                "min_similarity": cfg.min_similarity, "recall_timeout_ms": cfg.recall_timeout_ms,
                "reps": reps, "cases": results,
                "worst_p95_ms": max((r["p95_ms"] for r in every), default=None)}
    finally:
        svc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=20, help="timed recalls per case")
    ap.add_argument("--out", default="~/homeai-bench", help="directory for the JSON result")
    a = ap.parse_args()
    name = f"homeai_recall_live_{os.getpid()}"
    made = subprocess.run(["createdb", name], capture_output=True, text=True)
    if made.returncode:
        print(f"cannot create {name}: {made.stderr.strip()}", file=sys.stderr)
        return 1
    try:
        t0 = time.monotonic()
        report = run(f"dbname={name}", a.reps)
        report["wall_s"] = round(time.monotonic() - t0, 1)
    finally:
        subprocess.run(["dropdb", "--if-exists", "--force", name], capture_output=True)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"recall_live-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=1))
    for r in report["cases"]:
        t = f"  p50 {r['p50_ms']:.1f} / p95 {r['p95_ms']:.1f} ms" if r["p95_ms"] else ""
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['speaker'] or 'unknown':7} "
              f"{r['utterance']!r}{t}")
    print(f"worst p95 {report['worst_p95_ms']} ms over {a.reps} reps; wrote {path}")
    return 0 if all(r["ok"] for r in report["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
