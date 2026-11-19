"""Pictures for Claude to review (ART_FEATURES_PLAN F9 step 1).

The local checker misses placement and proportion (live node 002 passed every check with horns on the lion and
a hilt as long as the blade), so what it says isn't the last word. Kept here, each as ``<folder>/picture.png``
plus ``item.json`` (what was asked, what the image model was told, the checks and what they found, what the
person said):

- every picture the person complained about ("it has two heads");
- every picture shown with "Best I could do" (it failed its checks every try);
- 1 in ``sample`` of the pictures that passed, so the checker's misses show up too;
- practice-night glossary candidates (tools/art/practice.py).

``tools/art/report_review.py`` makes a numbered contact sheet and a labels file from them; labelled items move
to ``done/``. Capped at ``limit`` waiting items so a broken night can't fill the disk. Never raises: a review
item is worth less than the picture it came from.
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Callable

from homeai.fsutil import atomic_copy, atomic_write_json

log = logging.getLogger("homeai.art.review")

ROOT = Path.home() / ".local/share/homeai/art/review"
SAMPLE = 10
LIMIT = 2000
REASONS = ("complaint", "best", "pass-sample", "candidate")


def _feature_rows(features) -> list[dict]:
    return [{"id": f.id, "kind": f.kind, "text": f.text} for f in features]


def _check_rows(checks) -> list[dict]:
    return [{"kind": c.kind, "question": c.question, "expect": c.expect, "feature": getattr(c, "feature", "")}
            for c in checks]


class Review:
    def __init__(self, root: Path = ROOT, sample: int = SAMPLE, limit: int = LIMIT,
                 clock: Callable[[], float] = time.time, rng: Callable[[], float] = random.random) -> None:
        self.root, self.sample, self.limit = Path(root), sample, limit
        self._clock, self._rng = clock, rng

    def waiting(self) -> list[Path]:
        """Folders not yet labelled, oldest first."""
        try:
            return sorted(p.parent for p in self.root.glob("*/item.json"))
        except OSError:
            return []

    def add(self, image: Path | None, reason: str, *, prompt: str = "", features=(), checks=(), issues=(),
            said: str = "", source: str = "live", extra: dict | None = None, pictures=()) -> Path | None:
        """Keep one item; its folder, or None (full, or it couldn't be written)."""
        if reason not in REASONS:
            raise ValueError(f"unknown review reason {reason!r}")
        if len(self.waiting()) >= self.limit:
            log.warning("review queue full (%d): %s not kept", self.limit, reason)
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            n = 1
            while (folder := self.root / f"{stamp}-{reason}-{n}").exists():
                n += 1
            folder.mkdir()
            images = [Path(image)] if image is not None else []
            images += [Path(p) for p in pictures]
            names = []
            for i, src in enumerate(images):
                name = "picture.png" if i == 0 else f"picture-{i + 1}.png"
                atomic_copy(src, folder / name)
                names.append(name)
            atomic_write_json(folder / "item.json", {
                "reason": reason, "when": stamp, "source": source, "said": said, "prompt": prompt,
                "features": _feature_rows(features), "checks": _check_rows(checks), "issues": list(issues),
                "pictures": names, **(extra or {})}, indent=1)
        except OSError as exc:
            log.warning("could not keep a review item (%s): %s", reason, exc)
            return None
        log.info("art: kept for review: %s", folder.name)
        return folder

    def passed(self, image: Path, **kw) -> Path | None:
        """A picture that passed its checks: kept 1 time in ``sample``."""
        if self.sample <= 0 or self._rng() >= 1 / self.sample:
            return None
        return self.add(image, "pass-sample", **kw)

    def done(self, folder: Path, label: dict) -> Path:
        """A labelled item, with its label, out of the waiting list into ``done/``."""
        folder = Path(folder)
        item = json.loads((folder / "item.json").read_text())
        atomic_write_json(folder / "item.json", {**item, "label": label}, indent=1)
        target = self.root / "done" / folder.name
        target.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(target)
        return target


def read(folder: Path) -> dict | None:
    try:
        item = json.loads((Path(folder) / "item.json").read_text())
        return item if isinstance(item, dict) else None
    except (OSError, ValueError):
        return None

