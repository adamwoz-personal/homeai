# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Background jobs: extract facts from episodes, rebuild profiles.

``Worker.step()`` claims one due job and runs it. The service calls it from
its own thread, only while nobody is talking to Jarvis; ``homeai-memory
extract`` calls it by hand.

Failure rules (plan §7.3):
* helper unreachable -> the job is retried later (backoff, then ``failed``);
* helper output invalid twice -> those episodes are marked read and skipped:
  retrying the same input would only burn CPU on the same answer.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Callable

from . import extract, jobs
from .helper import HelperInvalid

log = logging.getLogger(__name__)

KINDS = ("extract", "reflect")
TRUST_ORDER = ("outside", "household", "owner")   # weakest first


def extract_key(username: str) -> str:
    return f"extract:{username}"


def reflect_key(username: str) -> str:
    return f"reflect:{username}"


def weakest_trust(trusts) -> str:
    return min(trusts, key=TRUST_ORDER.index)


def input_hash(facts: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for f in sorted(facts, key=lambda f: str(f["id"])):
        h.update(f"{f['id']}|{f['updated_at'].isoformat()}|{f['summary']}\n".encode())
    return h.hexdigest()[:16]


class Worker:
    def __init__(self, store, helper, *, batch: int = 20, reflect_delay_s: float = 1800,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.store, self.helper = store, helper
        self.batch, self.reflect_delay_s = batch, reflect_delay_s
        self._now = clock or (lambda: datetime.now().astimezone())

    # -- queue -----------------------------------------------------------------------

    def recover(self) -> int:
        with self.store.db.scoped(None, "admin") as conn:
            return jobs.recover(conn)

    def step(self) -> bool:
        """Run one due job. False when there was nothing to do."""
        with self.store.db.scoped(None, "admin") as conn:
            job = jobs.claim(conn, KINDS)
        if job is None:
            return False
        try:
            username = job.payload["username"]
            result = self.extract(username) if job.kind == "extract" else self.reflect(username)
        except Exception as exc:  # noqa: BLE001 - a job failure is recorded, not raised
            with self.store.db.scoped(None, "admin") as conn:
                state = jobs.fail(conn, job, f"{type(exc).__name__}: {exc}")
            log.warning("memory job %s %s failed (%s): %s", job.kind, job.payload, state, exc)
            return True
        with self.store.db.scoped(None, "admin") as conn:
            jobs.finish(conn, job.id)
        if job.kind == "extract" and result.get("episodes") \
                and self.store.count_unextracted(username):
            # More than one batch, or turns that arrived while this ran. Only
            # after finish: a running twin would swallow the enqueue.
            with self.store.db.scoped(None, "admin") as conn:
                jobs.enqueue(conn, "extract", {"username": username}, key=extract_key(username))
        log.info("memory job %s %s: %s", job.kind, username, result)
        return True

    def run_all(self, limit: int = 100) -> int:
        n = 0
        while n < limit and self.step():
            n += 1
        return n

    # -- extraction --------------------------------------------------------------------

    def extract(self, username: str) -> dict[str, Any]:
        person = self.store.person(username)
        if person is None or not person["memory_enabled"]:
            return {"skipped": "memory off"}
        episodes = self.store.unextracted(username, self.batch)
        if not episodes:
            return {"episodes": 0}
        payload = extract.extract_payload(person["display_name"], episodes, today=self._now(),
                                          known_topics=self.store.known_topics())
        try:
            raw = self.helper.structured(extract.extract_task(person["is_child"]),
                                         extract.extract_schema(), payload)
        except HelperInvalid as exc:
            log.warning("extraction for %s: helper output invalid twice (%s); skipping %d"
                        " exchanges", username, exc, len(episodes))
            self.store.mark_extracted(username, episodes)
            return {"episodes": len(episodes), "invalid": True}
        notes = extract.parse_notes(raw, len(episodes), is_child=person["is_child"],
                                    exchanges=episodes, person=person["display_name"])
        counts: dict[str, int] = {}
        for note in notes:
            action = self._apply(username, note, episodes)
            counts[action] = counts.get(action, 0) + 1
        self.store.mark_extracted(username, episodes)
        changed = sum(v for k, v in counts.items() if k in ("add", "update", "supersede"))
        if changed:
            with self.store.db.scoped(None, "admin") as conn:
                jobs.enqueue(conn, "reflect", {"username": username}, key=reflect_key(username),
                             delay_s=self.reflect_delay_s)
        return {"episodes": len(episodes), "notes": len(notes), **counts}

    def _apply(self, username: str, note: extract.Note, episodes: list[dict[str, Any]]) -> str:
        trust = weakest_trust(episodes[i]["trust"] for i in note.sources)
        vec = self.store.embedder.embed([note.summary], kind="doc")[0]
        neighbours = self.store.nearest_facts(username, vec, 5)
        decision = extract.triage([n["sim"] for n in neighbours])
        if decision is None:
            try:
                raw = self.helper.structured(extract.RECONCILE_TASK, extract.reconcile_schema(),
                                             extract.reconcile_payload(note, neighbours))
                decision = extract.parse_decision(raw, note, neighbours, trust)
            except HelperInvalid:
                decision = extract.Decision("add")
        if decision.action == "add" and not decision.summary:
            decision = extract.Decision("add", None, note.summary)
        if decision.action in ("update", "supersede", "add") and decision.summary != note.summary:
            vec = self.store.embedder.embed([decision.summary], kind="doc")[0]
        self.store.apply_note(username, note, decision, neighbours, vec, episodes,
                              trust=trust, actor=f"extractor:{self.helper.name}")
        return decision.action

    # -- reflection ----------------------------------------------------------------------

    def reflect(self, username: str) -> dict[str, Any]:
        person = self.store.person(username)
        if person is None or not person["memory_enabled"]:
            return {"skipped": "memory off"}
        facts = extract.reflection_inputs(self.store.facts(username))
        digest = input_hash(facts)
        if digest == self.store.profile_hash(username):
            return {"unchanged": True}
        if len(facts) < 3:
            # Too little to summarise; keep any hand-written profile, drop a stale one.
            self.store.replace_profile(username, [], digest, actor="reflection")
            return {"lines": 0, "facts": len(facts)}
        raw = self.helper.structured(extract.REFLECT_TASK, extract.reflect_schema(),
                                     extract.reflect_payload(person["display_name"], facts))
        lines = [(s, [facts[i]["id"] for i in src])
                 for s, src in extract.parse_reflection(raw, len(facts))]
        n = self.store.replace_profile(username, lines, digest,
                                       actor=f"reflection:{self.helper.name}")
        return {"lines": n, "facts": len(facts)}
