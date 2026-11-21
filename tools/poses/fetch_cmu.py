#!/usr/bin/env python3
"""Find and download clips from the CMU Graphics Lab Motion Capture Database (pose library, ART_FEATURES_PLAN F10).

Why: the pose library's forms come from real human motion, so the poses are ones a body can make. CMU's FAQ
(checked 2026-11-20): "The motion capture data may be copied, modified, or redistributed without permission."
Credit: "The data used in this project was obtained from mocap.cs.cmu.edu."

The site has no API: ``index`` reads its full search listing into subject/trial/description rows; ``search``
filters them by words; ``fetch`` downloads the ASF skeleton and AMC motion of chosen trials. Everything lands in
~/.cache/homeai/cmu-mocap/ (not in the repo).

Usage:
    .venv/bin/python tools/poses/fetch_cmu.py index
    .venv/bin/python tools/poses/fetch_cmu.py search sword fight punch
    .venv/bin/python tools/poses/fetch_cmu.py fetch 02_05 13_17
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

SITE = "http://mocap.cs.cmu.edu"          # its https certificate chain fails verification (2026-11-20)
CACHE = Path.home() / ".cache/homeai/cmu-mocap"
CREDIT = "The data used in this project was obtained from mocap.cs.cmu.edu."
AGENT = {"User-Agent": "homeai-pose-library/1"}
TRIAL = re.compile(r"^\d{2,3}_\d{2}$")


def _get(url: str, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=timeout) as reply:
        return reply.read()


def parse_listing(page: str) -> list[dict]:
    """Rows {trial: "02_05", subject: 2, description} from the site's "view all motions" page.

    The page lists "Subject #N (what)" headings, then per trial its number and description in table cells."""
    lines = [html.unescape(s).strip() for s in re.sub(r"<[^>]+>", "\n", page).splitlines()]
    lines = [s for s in lines if s]
    rows, subject = [], None
    for i, line in enumerate(lines):
        head = re.match(r"^Subject #(\d+)\b", line)
        if head:
            subject = int(head.group(1))
            continue
        if subject is None or not line.isdigit() or i + 1 >= len(lines):
            continue
        description = lines[i + 1]
        if description.isdigit() or description in ("Motion Description", "Feedback", "Animated"):
            continue
        rows.append({"trial": f"{subject:02d}_{int(line):02d}", "subject": subject, "description": description})
    seen, out = set(), []
    for row in rows:                                    # each trial once, first description wins
        if row["trial"] not in seen:
            seen.add(row["trial"])
            out.append(row)
    return out


def search(rows: list[dict], words: list[str]) -> list[dict]:
    """Rows whose description has any of the words (whole word, any case)."""
    found = [re.compile(rf"\b{re.escape(w)}", re.I) for w in words]
    return [r for r in rows if any(f.search(r["description"]) for f in found)]


def clip_urls(trial: str) -> dict[str, str]:
    if not TRIAL.match(trial):
        raise ValueError(f"not a trial like 02_05: {trial!r}")
    subject = trial.split("_")[0]
    return {"asf": f"{SITE}/subjects/{subject}/{subject}.asf", "amc": f"{SITE}/subjects/{subject}/{trial}.amc"}


def fetch(trial: str, cache: Path = CACHE, get=_get) -> dict[str, Path]:
    """The ASF and AMC files of a trial, downloaded once."""
    out = {}
    for kind, url in clip_urls(trial).items():
        path = cache / url.rsplit("/", 1)[1]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            data = get(url)
            if not data or data[:1] == b"<":
                raise OSError(f"{url} did not give a motion file")
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        out[kind] = path
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("index")
    s = sub.add_parser("search")
    s.add_argument("words", nargs="+")
    f = sub.add_parser("fetch")
    f.add_argument("trials", nargs="+")
    args = ap.parse_args(argv)
    index = CACHE / "index.json"
    if args.step == "index":
        rows = parse_listing(_get(f"{SITE}/search.php?subjectnumber=%25&motion=%25", 120).decode("utf-8", "replace"))
        CACHE.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps({"credit": CREDIT, "rows": rows}, indent=1))
        print(f"{len(rows)} trials -> {index}")
    elif args.step == "search":
        for row in search(json.loads(index.read_text())["rows"], args.words):
            print(row["trial"], row["description"])
    else:
        for trial in args.trials:
            print(trial, {k: str(v) for k, v in fetch(trial).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
