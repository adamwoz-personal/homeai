# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Carrying out spoken memory commands (plan §7.5).

``SpokenCommands.handle(username, display, utterance)`` returns ``(reply,
listen)`` or None when the utterance isn't a memory command. ``listen``
means "ask for an answer without the wake word" (a yes/no confirmation).

Deleting more than one thing always asks first; the question names an
example, so "yes" is an informed yes. A pending question expires after
``PENDING_S``. "That's wrong, ..." only counts right after Jarvis listed
memories, so ordinary disagreement never rewrites anything.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from . import commands, recall
from .commands import Command

log = logging.getLogger(__name__)

PENDING_S = 60.0
LISTED_S = 120.0
RECENT_S = 1800.0          # "forget that" reaches back this far
MAX_SPOKEN = 5

MSG_UNKNOWN = "I don't recognise your voice, so I can't keep memories for you."
MSG_OFF = ("You asked me not to remember things about you. Say: start remembering "
           "things about me, if you'd like me to again.")
MSG_BROKEN = "Sorry, my long-term memory isn't working right now."


@dataclass
class _Pending:
    kind: str                       # forget | forget_all | listed
    facts: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    listed: list = field(default_factory=list)   # [(id, summary)] for corrections
    expires: float = 0.0


class SpokenCommands:
    def __init__(self, store, *, min_similarity: float = 0.6,
                 flush: Callable[[], Any] = lambda: None,
                 clock: Callable[[], datetime] | None = None,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.min_similarity = min_similarity
        self._flush = flush
        self._now = clock or (lambda: datetime.now().astimezone())
        self._mono = monotonic
        self._pending: dict[str, _Pending] = {}
        self._last_note: dict[str, tuple[Any, float]] = {}

    # -- entry point -------------------------------------------------------------------

    def handle(self, username: str | None, display: str | None,
               utterance: str) -> tuple[str, bool] | None:
        pending = self._take_pending(username)
        if pending is not None:
            answer = self._answer(username, pending, utterance)
            if answer is not None:
                return answer
        cmd = commands.parse(utterance)
        if cmd is None:
            return None
        if username is None:
            return MSG_UNKNOWN, False
        person = self.store.person(username)
        if person is None:
            return MSG_UNKNOWN, False
        log.info("memory command %s for %s", cmd.kind, username)
        return getattr(self, f"_{cmd.kind}")(username, display or person["display_name"],
                                             person, cmd)

    # -- pending questions -------------------------------------------------------------

    def _take_pending(self, username: str | None) -> _Pending | None:
        p = self._pending.pop(username, None) if username else None
        return p if p is not None and p.expires > self._mono() else None

    def _answer(self, username: str, p: _Pending, utterance: str) -> tuple[str, bool] | None:
        if p.kind == "listed":
            fixed = commands.correction(utterance)
            if fixed is None or not p.listed:
                return None
            return self._correct(username, p, fixed)
        yes = commands.confirmation(utterance)
        if yes is None:
            return None          # something else: the question lapses
        if not yes:
            return "Okay, I'll keep them.", False
        if p.kind == "forget_all":
            n = self.store.forget_person(username, actor="voice")
            return f"Done. I've forgotten all {n} of my memories about you.", False
        eps = list(dict.fromkeys(p.episodes + self.store.sources_of(username, p.facts)))
        self.store.forget_items(username, facts=p.facts, episodes=eps, actor="voice")
        return "Done, I've forgotten that.", False

    def _ask(self, username: str, pending: _Pending, question: str) -> tuple[str, bool]:
        pending.expires = self._mono() + PENDING_S
        self._pending[username] = pending
        return question, True

    # -- commands -----------------------------------------------------------------------

    def _remember(self, username, display, person, cmd: Command, pinned: bool = False):
        if not person["memory_enabled"]:
            return MSG_OFF, False
        summary = commands.first_person_to_third(cmd.text, display)
        fid = self.store.add_fact(username, summary, kind="fact", pinned=pinned,
                                  importance=0.9 if pinned else 0.7, source="explicit",
                                  actor="voice")
        self._last_note[username] = (fid, self._mono())
        return ("I'll always remember that." if pinned else "Okay, I'll remember that."), False

    def _remember_always(self, username, display, person, cmd):
        return self._remember(username, display, person, cmd, pinned=True)

    def _recall_me(self, username, display, person, cmd):
        facts = self.store.facts(username)
        if not facts:
            return "I don't have anything saved about you yet.", False
        # Profile lines read best first; then pinned and important facts.
        facts.sort(key=lambda f: (f["kind"] != "profile", not f["pinned"], -f["importance"]))
        top = facts[:MAX_SPOKEN]
        more = len(facts) - len(top)
        text = "Here's what I remember about you. " + " ".join(_sentence(f["summary"]) for f in top)
        if more:
            text += f" And {more} more {'thing' if more == 1 else 'things'}."
        return self._list(username, top, text)

    def _recall_about(self, username, display, person, cmd):
        found = self._search(username, cmd.text)
        if not found:
            return f"I don't remember anything about {cmd.text}.", False
        now = self._now()
        parts = []
        for c in found[:3]:
            if c.kind == "fact":
                parts.append(_sentence(c.summary))
            else:
                parts.append(_sentence(f"{recall.ago(c.at, now).capitalize()}, you said: "
                                       f"{c.text_user}"))
        facts = [{"id": c.id, "summary": c.summary} for c in found[:3] if c.kind == "fact"]
        return self._list(username, facts, " ".join(parts))

    def _forget_last(self, username, display, person, cmd):
        self._flush()
        now = self._mono()
        note = self._last_note.get(username)
        ep = self.store.last_episode(username)
        ep_age = (self._now() - ep["at"]).total_seconds() if ep else None
        if note and now - note[1] < RECENT_S and (ep_age is None or now - note[1] <= ep_age):
            self.store.forget_items(username, facts=[note[0]], actor="voice")
            self._last_note.pop(username, None)
            return "Okay, I've forgotten that.", False
        if ep is not None and ep_age is not None and ep_age < RECENT_S:
            self.store.forget_items(username, episodes=[ep["id"]], actor="voice")
            return "Okay, I've forgotten that.", False
        return ("I'm not sure what to forget. Say: forget what I said about, "
                "and then the topic."), False

    def _forget_about(self, username, display, person, cmd):
        found = self._search(username, cmd.text)
        if not found:
            return f"I don't have anything about {cmd.text}.", False
        facts = [c.id for c in found if c.kind == "fact"]
        eps = [c.id for c in found if c.kind == "episode"]
        example = found[0].summary or found[0].text_user
        n = len(found)
        return self._ask(username, _Pending("forget", facts, eps),
                         f"I found {n} {'memory' if n == 1 else 'memories'} about {cmd.text}, "
                         f"like: {_clip(example)}. Shall I forget "
                         f"{'it' if n == 1 else 'them'}?")

    def _forget_thread(self, username, display, person, cmd):
        facts = self.store.facts(username)
        if cmd.text:
            hits = self._search(username, cmd.text)
            seed = next((c.id for c in hits if c.kind == "fact"), None)
            seed_fact = next((f for f in facts if f["id"] == seed), None)
        else:
            seed_fact = max((f for f in facts if f["thread_id"]), default=None,
                            key=lambda f: f["updated_at"])
        if seed_fact is None:
            if cmd.text:
                return self._forget_about(username, display, person, cmd)
            return "I'm not sure which idea you mean. Tell me what it was about.", False
        thread = seed_fact["thread_id"]
        chosen = [f for f in facts if thread and f["thread_id"] == thread] or [seed_fact]
        n = len(chosen)
        return self._ask(username, _Pending("forget", [f["id"] for f in chosen]),
                         f"That idea has {n} {'memory' if n == 1 else 'memories'}, like: "
                         f"{_clip(chosen[0]['summary'])}. Shall I forget "
                         f"{'it' if n == 1 else 'all of it'}?")

    def _forget_all(self, username, display, person, cmd):
        n = self.store.count_for(username)
        if not n:
            return "I don't have any memories about you.", False
        return self._ask(username, _Pending("forget_all"),
                         f"That will delete all {n} of my memories about you, and I can't undo "
                         f"it. Are you sure?")

    def _stop(self, username, display, person, cmd):
        self.store.set_memory_enabled(username, False, actor="voice")
        n = self.store.count_for(username)
        tail = (f" I still have {n} older memories; say: forget everything about me, to delete "
                f"them.") if n else ""
        return "Okay, I've stopped remembering things about you." + tail, False

    def _start(self, username, display, person, cmd):
        self.store.set_memory_enabled(username, True, actor="voice")
        return "Okay, I'll remember things about you again.", False

    # -- helpers ---------------------------------------------------------------------------

    def _list(self, username: str, facts: list[dict], text: str) -> tuple[str, bool]:
        if facts:
            self._pending[username] = _Pending(
                "listed", listed=[(f["id"], f["summary"]) for f in facts],
                expires=self._mono() + LISTED_S)
        return text, False

    def _correct(self, username: str, p: _Pending, fixed: str) -> tuple[str, bool]:
        person = self.store.person(username)
        summary = commands.first_person_to_third(fixed, person["display_name"])
        if len(p.listed) == 1:
            target = p.listed[0][0]
        else:
            vecs = self.store.embedder.embed([summary] + [s for _, s in p.listed], kind="doc")
            sims = vecs[1:] @ vecs[0]
            target = p.listed[int(sims.argmax())][0]
        self.store.supersede_fact(username, target, summary, actor="voice")
        return "Thanks, I've corrected that.", False

    def _search(self, username: str, text: str) -> list[recall.Candidate]:
        """The person's own memories about ``text`` (never household ones)."""
        now = self._now()
        vec = self.store.embedder.embed([text], kind="query")[0]
        with self.store.db.scoped(username, "person") as conn:
            rows = recall.fetch(conn, username, vec, text, None)
        kept = recall.keep(recall.fuse(rows, now), self.min_similarity)
        return [c for c in kept if c.username == username and not (c.kind == "fact" and c.standing
                                                                    and c.sim < self.min_similarity)]


def _sentence(text: str) -> str:
    text = " ".join(text.split())
    return text if text[-1:] in ".!?" else text + "."


def _clip(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."
