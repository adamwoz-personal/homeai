# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Re-extract facts from kept episodes with a newer model or prompt (plan §8.5).

Episodes are kept verbatim so that a better helper can re-read them later.
``rederive(..., apply=False)`` (the default) is a diff and changes nothing:

* ``new``   - a note no current fact covers;
* ``known`` - a note an existing fact already says (similarity >= NOOP_SIM);
* ``near``  - related to a fact; on apply the helper decides add/update/supersede;
* ``gone``  - extractor facts from these episodes that no note re-found.

``apply=True`` sends each note through the normal extraction path
(``Worker.apply_note``): pinned, explicit, import, reflection and mcp facts
are never rewritten, and nothing is deleted - ``gone`` facts are only
reported, for a person to forget if they agree.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from . import extract
from .helper import HelperInvalid
from .store import StoreError
from .worker import Worker, reflect_key

log = logging.getLogger(__name__)


def classify(store, username: str, note: extract.Note) -> dict[str, Any]:
    vec = store.embedder.embed([note.summary], kind="doc")[0]
    near = store.nearest_facts(username, vec, 1)
    best = near[0] if near else None
    sim = best["sim"] if best else 0.0
    status = "new" if sim < extract.ASK_SIM else "known" if sim >= extract.NOOP_SIM else "near"
    return {"status": status, "note": note.summary, "kind": note.kind,
            "closest": best["summary"] if best and status != "new" else None,
            "closest_id": best["id"] if best and status != "new" else None,
            "protected": bool(best and status == "near" and extract.protected(best, "owner")),
            "sim": round(float(sim), 3)}


def rederive(store, helper, username: str, *, since: datetime | None = None,
             apply: bool = False, batch: int = 20) -> dict[str, Any]:
    person = store.person(username)
    if person is None:
        raise StoreError(f"no such person {username!r}")
    if not person["memory_enabled"]:
        return {"skipped": "memory off"}
    episodes = store.episodes_for(username, since)
    worker = Worker(store, helper, batch=batch)
    findings: list[dict[str, Any]] = []
    actions: dict[str, int] = {}
    notes = invalid = 0
    for i in range(0, len(episodes), batch):
        chunk = episodes[i:i + batch]
        try:
            batch_notes = worker.notes_for(person, chunk)
        except HelperInvalid as exc:
            log.warning("rederive %s: helper output invalid for %d exchanges: %s",
                        username, len(chunk), exc)
            invalid += 1
            continue
        notes += len(batch_notes)
        for note in batch_notes:
            if apply:
                act = worker.apply_note(username, note, chunk)
                actions[act] = actions.get(act, 0) + 1
            else:
                findings.append(classify(store, username, note))
    report: dict[str, Any] = {"person": username, "episodes": len(episodes), "notes": notes,
                              "invalid_batches": invalid, "model": helper.name, "applied": apply}
    if apply:
        report["actions"] = actions
        if any(actions.get(k) for k in ("add", "update", "supersede")):
            from . import jobs

            with store.db.scoped(None, "admin") as conn:
                jobs.enqueue(conn, "reflect", {"username": username}, key=reflect_key(username))
        return report
    refound = {f["closest_id"] for f in findings if f["closest_id"] is not None}
    gone = [f for f in store.facts_from_episodes(username, [e["id"] for e in episodes])
            if f["id"] not in refound]
    report["findings"] = findings
    report["gone"] = gone
    report["counts"] = {s: sum(f["status"] == s for f in findings)
                        for s in ("new", "known", "near")} | {"gone": len(gone)}
    return report
