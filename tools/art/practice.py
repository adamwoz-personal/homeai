#!/usr/bin/env python3
"""Practice nights: klein draws Adam's settings unattended, the checker marks each picture, failing wording is
rewritten and redrawn, and what worked goes to Claude's review queue as a glossary candidate (ART_FEATURES_PLAN
F9 step 2).

Why: the glossary (homeai/art/glossary.json) is how klein gets a chimera right (0/8 -> most), but each entry was
found by hand. Practice does the volume locally: for each topic (homeai/art/practice_topics.json: creatures,
weapons, gear, places per setting, least recently practised first) the voice model writes one scene, klein9
draws it on ``--seeds`` seeds with the glossary's words, and every picture is asked the glossary's questions,
or the voice model's own (F8) for a term the glossary doesn't know. Under ``PASS_OF`` of them right, the words
are rewritten (wording.reword, kid-safe rules) and drawn on fresh seeds, up to ``--rewrites`` times. A rewrite
that passes goes to the review queue as a "candidate"; a term nothing fixed goes there as "best". Nothing
reaches the live glossary without Claude's review (report_review.py).

Yields: it starts only if nobody has spoken to Jarvis for 5 minutes, and stops before the next picture once
anyone does ("heard:" in Jarvis's log) or at ``--until``. Art mode: started if off, then back to voice mode
afterwards, unless Jarvis entered art mode for someone meanwhile (he leaves it himself). Never from coding mode.

Writes ~/homeai-bench/practice-<time>/{results.jsonl,summary.json,<term>/*.png}; remembers what was practised
when in ~/.local/share/homeai/art/practice-history.json.

Usage:
    .venv/bin/python tools/art/practice.py [--until 06:30] [--terms 0] [--seeds 4] [--rewrites 3]
                                           [--setting D&D ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.art import checker as CK  # noqa: E402
from homeai.art import wording as W  # noqa: E402
from homeai.fsutil import atomic_write_json  # noqa: E402

log = logging.getLogger("practice")

TOPICS = REPO / "homeai/art/practice_topics.json"
HISTORY = Path.home() / ".local/share/homeai/art/practice-history.json"
QUIET_S = 300                    # nobody spoke to Jarvis for this long before starting
PASS_OF = 3                      # right pictures out of --seeds (4) for wording to count as working
MAX_SCENE = 160


class Yielded(Exception):
    """Someone spoke to Jarvis, or the night is over."""


@dataclass(frozen=True)
class Topic:
    setting: str
    kind: str
    term: str

    @property
    def key(self) -> str:
        return f"{self.setting}/{self.term}"


@dataclass
class Round:
    wording: str
    passed: int = 0
    total: int = 0
    issues: list[str] = field(default_factory=list)
    pictures: list[tuple[Path, bool]] = field(default_factory=list)


def topics(path: Path = TOPICS, settings: list[str] | None = None) -> list[Topic]:
    data = json.loads(Path(path).read_text())
    out = []
    for setting, kinds in data["settings"].items():
        if settings and setting not in settings:
            continue
        for kind, terms in kinds.items():
            out += [Topic(setting, kind, t) for t in terms]
    return out


def order(found: list[Topic], history: dict, rng: random.Random) -> list[Topic]:
    """Never practised first (shuffled), then the longest ago."""
    shuffled = list(found)
    rng.shuffle(shuffled)
    return sorted(shuffled, key=lambda t: (history.get(t.key) or {}).get("when", ""))


SCENE_SCHEMA = {"type": "object", "properties": {"scene": {"type": "string"}}, "required": ["scene"]}


def scene_messages(topic: Topic) -> list[dict]:
    return [
        {"role": "system", "content":
            "You write one-sentence picture requests, the way a person asks for a picture. Name the thing given "
            "exactly as written, as the clear main subject, in a simple scene of the given setting. Under 25 "
            "words, family friendly, no names of real people or brands. Answer as JSON: {\"scene\": \"...\"}"},
        {"role": "user", "content": f"Setting: {topic.setting}. Thing ({topic.kind}): {topic.term}"},
    ]


def scene(topic: Topic, chat: Callable[[list[dict], dict], str], allowed: Callable[[str], bool]) -> str:
    """The voice model's sentence for ``topic``, or a plain one if its answer is unusable: it must name the term,
    be short, and pass the kid-safe rules."""
    plain = f"{topic.term[0].upper()}{topic.term[1:]}, {topic.setting} style."
    try:
        text = re.sub(r"\s+", " ", str(json.loads(chat(scene_messages(topic), SCENE_SCHEMA)).get("scene") or ""))
        text = text.strip()
    except (ValueError, AttributeError, TypeError, OSError) as exc:
        log.warning("scene for %s failed: %s", topic.key, exc)
        return plain
    bare = re.sub(r"^(?:a|an|the|a pair of)\s+", "", topic.term, flags=re.I).lower()
    if not text or len(text) > MAX_SCENE or bare not in text.lower() or not allowed(text):
        log.info("scene refused for %s: %r", topic.key, text[:120])
        return plain
    return text


def checks_for(text: str, chat: Callable[[list[dict], dict], str] | None) -> list[CK.Check]:
    """The glossary's questions for ``text``; if it names nothing the glossary knows, the voice model's own."""
    asks = W.expand(text)[1]
    if not asks and chat is not None:
        asks = W.asked(text, chat)
    return [CK.Check(a.kind, a.question, a.expect, a.kind, "A", a.wrong) for a in asks]


class Practice:
    def __init__(self, draw: Callable[[str, int], Path], ask: Callable[[Path, str], str],
                 chat: Callable[[list[dict], dict], str], allowed: Callable[[str], bool], review, out: Path,
                 should_stop: Callable[[], str | None], seeds: int = 4, rewrites: int = 3,
                 rng: random.Random | None = None) -> None:
        self.draw, self.ask, self.chat, self.allowed, self.review = draw, ask, chat, allowed, review
        self.out, self.should_stop, self.seeds, self.rewrites = Path(out), should_stop, seeds, rewrites
        self.rng = rng or random.Random()

    def _round(self, topic: Topic, wording: str, checks: list[CK.Check], tag: str) -> Round:
        result = Round(wording)
        folder = self.out / re.sub(r"[^a-z0-9]+", "-", topic.key.lower()).strip("-")
        folder.mkdir(parents=True, exist_ok=True)
        for _ in range(self.seeds):
            why = self.should_stop()
            if why:
                raise Yielded(why)
            seed = self.rng.randrange(1, 2**31)
            image = self.draw(wording, seed)
            target = folder / f"{tag}-{seed}.png"
            shutil.copyfile(image, target)
            failures = CK.failed(target, checks, self.ask)
            result.total += 1
            result.passed += not failures
            result.issues += [said for _, said in failures]
            result.pictures.append((target, not failures))
        return result

    def term(self, topic: Topic) -> dict:
        """Practise one topic; a summary row. Raises Yielded between pictures."""
        text = scene(topic, self.chat, self.allowed)
        checks = checks_for(text, self.chat)
        row = {"topic": topic.key, "kind": topic.kind, "scene": text, "checks": [c.kind for c in checks],
               "rounds": []}
        if not checks:
            row["outcome"] = "nothing to check"
            return row
        current = W.expand(text)[0]
        rounds = [self._round(topic, current, checks, "r0")]
        while rounds[-1].passed < min(PASS_OF, self.seeds) and len(rounds) <= self.rewrites:
            last = rounds[-1]
            wrong = next((p for p, ok in last.pictures if not ok), None)
            try:
                seen = self.ask(wrong, "Describe what is in this picture in one sentence.") if wrong else ""
            except Exception as exc:  # noqa: BLE001 - the rewrite can do without it
                log.warning("describing %s failed: %s", wrong, exc)
                seen = ""
            common = [s for s, _ in collections.Counter(last.issues).most_common(3)]
            new = W.reword(text, current, common, seen, self.chat)
            if not new or not self.allowed(new):
                break
            current = new
            rounds.append(self._round(topic, current, checks, f"r{len(rounds)}"))
        row["rounds"] = [{"wording": r.wording, "passed": r.passed, "total": r.total,
                          "issues": dict(collections.Counter(r.issues))} for r in rounds]
        first, best = rounds[0], max(rounds, key=lambda r: r.passed)
        measured = any(c.kind != "asked" for c in checks)
        if first.passed >= min(PASS_OF, self.seeds) and not measured:
            # Live 2026-11-19: "Is there a glaive visible?" passed 4/4; by eye one was a glaive (the rest spears
            # and an axe). Unmeasured questions can't say the words work, so Claude looks at one every time.
            row["outcome"] = "unmeasured pass"
            self.review.add(first.pictures[0][0], "pass-sample", prompt=first.wording, checks=checks,
                            source="practice", extra={"topic": topic.key, "scene": text,
                                                      "why": "passed only the voice model's own questions"})
        elif first.passed >= min(PASS_OF, self.seeds):
            row["outcome"] = "fine as it is"
        elif best is not first and best.passed >= min(PASS_OF, self.seeds):
            row["outcome"] = "candidate"
            self._keep(topic, text, checks, best, "candidate", first)
        else:
            row["outcome"] = "not fixed"
            self._keep(topic, text, checks, best, "best", first)
        for r in rounds:            # the checker's misses on passes show up in review too
            for picture, ok in r.pictures:
                if ok:
                    self.review.passed(picture, prompt=r.wording, checks=checks, source="practice",
                                       extra={"topic": topic.key, "scene": text})
        return row

    def _keep(self, topic: Topic, text: str, checks, best: Round, reason: str, first: Round) -> None:
        pictures = [p for p, ok in best.pictures if ok] + [p for p, ok in best.pictures if not ok]
        self.review.add(pictures[0], reason, prompt=best.wording, checks=checks, issues=best.issues,
                        source="practice", pictures=pictures[1:],
                        extra={"topic": topic.key, "scene": text,
                               "candidate": {"term": topic.term, "setting": topic.setting, "wording": best.wording,
                                             "before": f"{first.passed}/{first.total}",
                                             "after": f"{best.passed}/{best.total}"}})


# -- the night around it --------------------------------------------------------------------------------------

def journal(since: str, run=subprocess.run) -> str:
    argv = ["journalctl", "--user", "-u", "homeai", "--since", since, "--no-pager", "-q", "-o", "cat"]
    return run(argv, capture_output=True, text=True, check=False).stdout


class Night:
    """When to stop: someone spoke to Jarvis since ``start``, or it's past ``until``."""

    def __init__(self, start: datetime, until: datetime, read: Callable[[str], str] = journal,
                 now: Callable[[], datetime] = datetime.now) -> None:
        self.start, self.until, self._read, self._now = start, until, read, now

    def _since(self) -> str:
        return self.start.strftime("%Y-%m-%d %H:%M:%S")

    def __call__(self) -> str | None:
        if self._now() >= self.until:
            return "the night is over"
        if "heard:" in self._read(self._since()):
            return "someone spoke to Jarvis"
        return None

    def art_taken(self) -> bool:
        """Jarvis entered art mode for someone since the start: it's theirs now, don't switch it off."""
        return "art: entered" in self._read(self._since())


def hand_back(switcher, started: bool, night: Night) -> None:
    """Back to voice mode if practice turned art mode on, unless Jarvis entered it for someone meanwhile."""
    if started and not night.art_taken():
        switcher.voice()


def until_time(text: str, now: datetime) -> datetime:
    hour, minute = (int(x) for x in text.split(":"))
    end = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return end if end > now else end + timedelta(days=1)


def read_history(path: Path = HISTORY) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--until", default="06:30", help="stop at this time (HH:MM, next occurrence)")
    ap.add_argument("--terms", type=int, default=0, help="at most this many topics (0: until --until)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rewrites", type=int, default=3)
    ap.add_argument("--setting", action="append", help="only these settings (repeatable)")
    ap.add_argument("--topics", type=Path, default=TOPICS)
    ap.add_argument("--dry-run", action="store_true", help="print the order and the scenes' glossary hits only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    history = read_history()
    queue = order(topics(args.topics, args.setting), history, random.Random())
    if args.terms:
        queue = queue[:args.terms]
    if args.dry_run:
        for t in queue:
            print(t.key, t.kind, "glossary" if W.expand(t.term)[1] else "")
        return 0
    from homeai import gpu_mode as G
    from homeai.art import comfy as C
    from homeai.art import patcher, policy, workflows
    from homeai.art.review import Review
    from homeai.benchio import bench_path
    start = datetime.now()
    night = Night(start, until_time(args.until, start))
    if "heard:" in journal(f"-{QUIET_S}s"):
        print("someone spoke to Jarvis in the last 5 minutes: not starting", file=sys.stderr)
        return 2
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() == "coding":
        print("not from coding mode", file=sys.stderr)
        return 1
    started = switcher.mode() != "art"
    out = bench_path("practice")
    out.mkdir(parents=True)
    rules = policy.load("kid-safe")
    client = None

    def draw(prompt: str, seed: int) -> Path:
        return client.run(workflows.klein9(prompt, seed, prefix="practice/p")).image

    practice = Practice(draw, CK.ollama_ask, patcher.ollama_chat,
                        lambda text: rules.check_request(text, "practice").allowed, Review(), out, night,
                        seeds=args.seeds, rewrites=args.rewrites)
    rows, why = [], "done"
    try:
        if started:
            switcher.art()
        client = C.ComfyClient()
        for topic in queue:
            try:
                row = practice.term(topic)
            except Yielded as stop:
                why = str(stop)
                break
            rows.append(row)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            history[topic.key] = {"when": datetime.now().isoformat(timespec="seconds"), "outcome": row["outcome"],
                                  "rounds": [f"{r['passed']}/{r['total']}" for r in row["rounds"]]}
            atomic_write_json(HISTORY, history, indent=1)
            print(f"{topic.key}: {row['outcome']} {history[topic.key]['rounds']}", flush=True)
        else:
            why = "every topic practised"
    except Exception as exc:  # noqa: BLE001 - logged; voice mode still comes back below
        log.exception("practice stopped")
        why = f"error: {exc}"
    finally:
        hand_back(switcher, started, night)
    tally = collections.Counter(r["outcome"] for r in rows)
    atomic_write_json(out / "summary.json", {"stopped": why, "started": start.isoformat(timespec="seconds"),
                                             "ended": datetime.now().isoformat(timespec="seconds"),
                                             "topics": len(rows), "outcomes": tally,
                                             "pictures": sum(r["total"] for x in rows for r in x["rounds"])},
                      indent=1)
    print(f"stopped: {why}; {dict(tally)}; {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
