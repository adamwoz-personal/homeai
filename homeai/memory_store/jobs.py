# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""A persistent job queue in ``memory.jobs``, safe for several workers.

* ``enqueue`` de-duplicates by key: a second request for a queued job moves
  its start time (debounce) instead of adding another.
* ``claim`` takes one due job with ``FOR UPDATE SKIP LOCKED``, so two workers
  never get the same job.
* ``fail`` retries with backoff, then gives up (state ``failed``).
* ``recover`` re-queues jobs left ``running`` by a crash.

Every function takes a connection inside the caller's transaction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MAX_ATTEMPTS = 3
BACKOFF_S = (60, 600, 3600)
STALE_RUNNING_S = 1800


@dataclass(frozen=True)
class Job:
    id: int
    kind: str
    payload: dict[str, Any]
    attempts: int


def enqueue(conn, kind: str, payload: dict[str, Any], *, key: str, delay_s: float = 0,
            debounce: bool = True) -> None:
    """Queue ``kind`` unless an equal job (same key) is queued or running.

    With ``debounce`` a queued twin is pushed back to now + delay (a new
    turn restarts the "session ended" clock); otherwise it keeps its time,
    except that a shorter delay always wins (a full batch runs now).
    """
    later = "EXCLUDED.not_before" if debounce else \
        "LEAST(memory.jobs.not_before, EXCLUDED.not_before)"
    conn.execute(
        "INSERT INTO memory.jobs (kind, payload, dedupe_key, not_before)"
        " VALUES (%s, %s, %s, now() + make_interval(secs => %s))"
        " ON CONFLICT (dedupe_key) WHERE state IN ('queued','running') DO UPDATE"
        f" SET not_before = {later}, updated_at = now() WHERE memory.jobs.state = 'queued'",
        (kind, json.dumps(payload), key, float(delay_s)))


def run_now(conn, key: str) -> int:
    """Make a queued job due immediately. Returns how many changed."""
    return conn.execute("UPDATE memory.jobs SET not_before = now() WHERE dedupe_key = %s"
                        " AND state = 'queued'", (key,)).rowcount


def claim(conn, kinds: tuple[str, ...]) -> Job | None:
    row = conn.execute(
        "UPDATE memory.jobs SET state = 'running', attempts = attempts + 1, updated_at = now()"
        " WHERE id = (SELECT id FROM memory.jobs WHERE state = 'queued' AND not_before <= now()"
        "             AND kind = ANY(%s) ORDER BY not_before, id"
        "             FOR UPDATE SKIP LOCKED LIMIT 1)"
        " RETURNING id, kind, payload, attempts", (list(kinds),)).fetchone()
    return Job(*row) if row else None


def finish(conn, job_id: int) -> None:
    conn.execute("UPDATE memory.jobs SET state = 'done', last_error = NULL, updated_at = now()"
                 " WHERE id = %s", (job_id,))


def fail(conn, job: Job, error: str) -> str:
    """Retry later, or give up after MAX_ATTEMPTS. Returns the new state."""
    if job.attempts >= MAX_ATTEMPTS:
        state, delay = "failed", 0
    else:
        state, delay = "queued", BACKOFF_S[min(job.attempts, len(BACKOFF_S)) - 1]
    conn.execute("UPDATE memory.jobs SET state = %s, last_error = %s, updated_at = now(),"
                 " not_before = now() + make_interval(secs => %s) WHERE id = %s",
                 (state, error[:500], delay, job.id))
    return state


def recover(conn, stale_s: int = STALE_RUNNING_S) -> int:
    """Re-queue jobs a crashed worker left running; tidy old finished ones."""
    n = conn.execute("UPDATE memory.jobs SET state = 'queued', updated_at = now()"
                     " WHERE state = 'running'"
                     " AND updated_at < now() - make_interval(secs => %s)", (stale_s,)).rowcount
    conn.execute("DELETE FROM memory.jobs WHERE state = 'done' AND updated_at < now() - interval '7 days'")
    return n


def pending(conn) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT id, kind, dedupe_key, state, attempts, not_before, last_error"
                        " FROM memory.jobs WHERE state <> 'done' ORDER BY not_before").fetchall()
    keys = ("id", "kind", "key", "state", "attempts", "not_before", "last_error")
    return [dict(zip(keys, r)) for r in rows]
