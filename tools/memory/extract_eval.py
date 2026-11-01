#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Try fact extraction on real conversations, without touching the database.

Reads the voice transcript, splits each enrolled speaker's exchanges into
sessions (a gap of --gap minutes ends one), runs the helper model on each
session exactly as the worker would, and writes:

* ~/homeai-bench/extract-eval-<date>.json  timings, failures, every note
* ~/homeai-bench/extract-eval-<date>.md    a review sheet: each note beside
  what was said, with a [ ] box to tick if it is true and worth keeping

    .venv/bin/python tools/memory/extract_eval.py [--sessions 10] [--model M]

Precision target (plan §11 Phase 2): at least 80% of notes ticked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from homeai.config import Config  # noqa: E402
from homeai.memory_store import extract  # noqa: E402
from homeai.memory_store.cli import _read, backfill_episodes, enrolled_speakers  # noqa: E402
from homeai.memory_store.helper import HelperError, OllamaHelper  # noqa: E402

OUT = Path("~/homeai-bench").expanduser()
FIXTURE = ROOT / "tests" / "fixtures" / "extract_eval.json"


def fixture_sessions() -> list[tuple[str, str, list[dict]]]:
    now = datetime.now().astimezone()
    out = []
    for s in json.loads(FIXTURE.read_text()):
        eps = [{"at": now, "user": u, "assistant": r, "trust": "owner"} for u, r in s["exchanges"]]
        out.append(("adam", "Adam", eps, s))
    return out


def score(todo: list, results: list[dict]) -> dict:
    """Recall: expected facts kept. Noise: notes from sessions that should give none."""
    hits = wanted = noise = 0
    for t, r in zip(todo, results):
        spec, kept = t[3], [n["summary"].lower() for n in r["notes"]]
        if not spec["expect"]:
            noise += len(kept)
            if kept:
                print(f"  NOISE {spec['name']}: {kept}")
        for alts in spec["expect"]:
            wanted += 1
            if any(alt in note for note in kept for alt in alts):
                hits += 1
            else:
                print(f"  MISSED {spec['name']}: wanted one of {alts}, kept {kept}")
    return {"recall": f"{hits}/{wanted}", "noise_notes": noise}


def sessions(cfg: Config, gap_min: float, batch: int) -> list[tuple[str, str, list[dict]]]:
    eps = backfill_episodes(_read(cfg.transcript.path),
                            enrolled=enrolled_speakers(cfg.speaker.registry_path),
                            ignore=cfg.memory.ignore_speakers, include_unknown=False)
    by_user: dict[str, list] = {}
    names: dict[str, str] = {}
    for ep, display in eps:
        by_user.setdefault(ep.username, []).append(ep)
        names[ep.username] = display
    out = []
    for user, items in by_user.items():
        items.sort(key=lambda e: e.occurred_at)
        cur: list[dict] = []
        for ep in items:
            if cur and (ep.occurred_at - cur[-1]["at"] > timedelta(minutes=gap_min)
                        or len(cur) >= batch):
                out.append((user, names[user], cur))
                cur = []
            cur.append({"at": ep.occurred_at, "user": ep.user, "assistant": ep.assistant,
                        "trust": ep.trust})
        if cur:
            out.append((user, names[user], cur))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=10, help="most recent N sessions")
    ap.add_argument("--gap", type=float, default=10, help="minutes of silence between sessions")
    ap.add_argument("--model", help="helper model (default: HOMEAI_LTM_HELPER_MODEL)")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--fixture", action="store_true",
                    help=f"score made-up sessions with known answers ({FIXTURE.name})")
    a = ap.parse_args()

    cfg = Config()
    m = cfg.memory
    helper = OllamaHelper(m.helper_url, a.model or m.helper_model,
                          threads=a.threads or m.helper_threads)
    if a.fixture:
        todo = fixture_sessions()
    else:
        todo = sorted(sessions(cfg, a.gap, m.extract_batch), key=lambda s: s[2][-1]["at"])
        todo = todo[-a.sessions:]
    print(f"{len(todo)} sessions, model {helper.model}, {helper.threads} threads", flush=True)

    results, failures, secs = [], 0, []
    for user, display, eps, *_ in todo:
        payload = extract.extract_payload(display, eps, today=datetime.now().astimezone())
        t0 = time.perf_counter()
        try:
            raw = helper.structured(extract.extract_task(False), extract.extract_schema(),
                                    payload)
            notes = extract.parse_notes(raw, len(eps), is_child=False, exchanges=eps,
                                        person=display)
            for item in raw.get("notes", []):
                if b := extract.borrowed(item.get("summary", ""), eps, display):
                    print(f"    dropped (Jarvis's words {sorted(b)}): {item.get('summary')}")
            error = None
        except HelperError as exc:
            raw, notes, error = None, [], str(exc)
            failures += 1
        dt = time.perf_counter() - t0
        secs.append(dt)
        print(f"  {user} {eps[0]['at']:%Y-%m-%d %H:%M} {len(eps):2} exchanges -> "
              f"{len(notes)} notes  {dt:5.1f}s{'  ERROR ' + error if error else ''}", flush=True)
        results.append({"user": user, "start": eps[0]["at"].isoformat(), "seconds": round(dt, 1),
                        "error": error, "raw_count": len((raw or {}).get("notes", [])),
                        "exchanges": [{"n": i + 1, "said": e["user"], "reply": e["assistant"]}
                                      for i, e in enumerate(eps)],
                        "notes": [{**n.__dict__, "sources": [s + 1 for s in n.sources]}
                                  for n in notes]})

    OUT.mkdir(exist_ok=True)
    stem = OUT / f"extract-eval-{date.today().isoformat()}"
    summary = {"model": helper.model, "threads": helper.threads, "sessions": len(results),
               "failures": failures, "notes": sum(len(r["notes"]) for r in results),
               "seconds_mean": round(sum(secs) / max(len(secs), 1), 1),
               "seconds_max": round(max(secs, default=0), 1)}
    if a.fixture:
        summary.update(score(todo, results))
    stem.with_suffix(".json").write_text(json.dumps({"summary": summary, "sessions": results},
                                                    indent=1, default=str))
    lines = [f"# Extraction review ({helper.model})", "",
             "Tick each note that is TRUE (you said it) and WORTH KEEPING.", ""]
    for r in results:
        lines.append(f"## {r['user']} {r['start'][:16]}  ({r['seconds']} s)")
        for e in r["exchanges"]:
            lines.append(f"{e['n']}. **said:** {e['said']}  \n   *jarvis:* {e['reply'][:160]}")
        lines.append("")
        for n in r["notes"]:
            lines.append(f"- [ ] ({n['kind']}, {n['importance']:.1f}, from {n['sources']}) "
                         f"{n['summary']}")
        if r["error"]:
            lines.append(f"- ERROR: {r['error']}")
        lines.append("")
    stem.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(summary))
    print(f"review: {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
