# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Reads and writes for people, episodes and facts. No recall ranking here.

Every method takes the acting ``username`` and ``scope`` and runs under
``Db.scoped``, so row-level security decides visibility. Admin-only
operations check ``people.is_admin`` too: RLS sees the scope claim, the
store checks that the claim is true.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .db import Db
from .embed import Embedder
from .policy import USERNAME_RE, redact

log = logging.getLogger(__name__)

SEARCH_TEXT_MAX = 4000
FACT_KINDS = ("fact", "preference", "event", "question", "answer", "thought", "opinion",
              "plan", "summary", "profile", "instruction")


class StoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class Episode:
    username: str
    user: str
    assistant: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0
    channel: str = "voice"
    trust: str = "owner"
    kind: str = "exchange"
    source_ref: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def content(self) -> dict[str, Any]:
        return {"v": 1, "user": self.user, "assistant": self.assistant, **self.extra}

    def search_text(self) -> str:
        return f"{self.user}\n{self.assistant}"[:SEARCH_TEXT_MAX]


class PostgresStore:
    def __init__(self, db: Db, embedder: Embedder) -> None:
        self.db = db
        self.embedder = embedder

    # -- people --------------------------------------------------------------

    def ensure_person(self, username: str, display_name: str, *, is_admin: bool = False,
                      voice_handle: str | None = None) -> bool:
        """Create the person if new. True if created. Never demotes or renames."""
        if not USERNAME_RE.match(username):
            raise StoreError(f"bad username {username!r}")
        with self.db.scoped(None, "admin") as conn:
            cur = conn.execute(
                "INSERT INTO memory.people (username, display_name, is_admin) VALUES (%s, %s, %s) "
                "ON CONFLICT (username) DO NOTHING", (username, display_name, is_admin))
            created = cur.rowcount == 1
            if voice_handle:
                conn.execute("INSERT INTO memory.identities (channel, handle, username, verified) "
                             "VALUES ('voice', %s, %s, true) ON CONFLICT DO NOTHING",
                             (voice_handle, username))
            if created:
                self._audit(conn, None, username, "person.create", "system", {})
        return created

    def people(self) -> list[dict[str, Any]]:
        with self.db.scoped(None, "admin") as conn:
            rows = conn.execute("SELECT username, display_name, is_child, memory_enabled, is_admin "
                                "FROM memory.people ORDER BY username").fetchall()
        keys = ("username", "display_name", "is_child", "memory_enabled", "is_admin")
        return [dict(zip(keys, r)) for r in rows]

    def person(self, username: str) -> dict[str, Any] | None:
        return next((p for p in self.people() if p["username"] == username), None)

    def is_admin(self, username: str) -> bool:
        p = self.person(username)
        return bool(p and p["is_admin"])

    # -- episodes --------------------------------------------------------------

    def add_episode(self, ep: Episode) -> uuid.UUID | None:
        """Store one exchange. None if skipped (memory off, or a duplicate)."""
        p = self.person(ep.username)
        if p is None or not p["memory_enabled"]:
            return None
        ep = Episode(**{**ep.__dict__, "user": redact(ep.user), "assistant": redact(ep.assistant)})
        vec = self.embedder.embed([ep.search_text()], kind="doc")[0]
        with self.db.scoped(ep.username, "person") as conn:
            row = conn.execute(
                "INSERT INTO memory.episodes (occurred_at, username, confidence, channel, trust, kind,"
                " content, text_for_search, embedding, embed_model, source_ref)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (source_ref, occurred_at) WHERE source_ref IS NOT NULL DO NOTHING"
                " RETURNING id",
                (ep.occurred_at, ep.username, ep.confidence, ep.channel, ep.trust, ep.kind,
                 json.dumps(ep.content()), ep.search_text(), vec, self.embedder.name,
                 ep.source_ref)).fetchone()
            if row is None:
                return None
            self._audit(conn, row[0], ep.username, "episode.add", "recorder", {"kind": ep.kind})
        return row[0]

    def list_episodes(self, actor: str, scope: str = "person", *, owner: str | None = None,
                      limit: int = 20) -> list[dict[str, Any]]:
        sql = ("SELECT id, occurred_at, username, content FROM memory.episodes"
               " WHERE state = 'active'" + (" AND username = %s" if owner else "") +
               " ORDER BY occurred_at DESC LIMIT %s")
        args = (owner, limit) if owner else (limit,)
        with self.db.scoped(actor, scope) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [{"id": r[0], "at": r[1], "username": r[2], **r[3]} for r in rows]

    # -- facts -----------------------------------------------------------------

    def add_fact(self, username: str, summary: str, *, kind: str = "fact",
                 topics: tuple[str, ...] = (), importance: float = 0.5, pinned: bool = False,
                 trust: str = "owner", source: str = "explicit", actor: str = "cli",
                 detail: dict[str, Any] | None = None) -> uuid.UUID:
        if kind not in FACT_KINDS:
            raise StoreError(f"unknown fact kind {kind!r}")
        summary = redact(" ".join((summary or "").split()))
        if not 1 <= len(summary) <= 300:
            raise StoreError("a fact summary must be 1-300 characters")
        vec = self.embedder.embed([summary], kind="doc")[0]
        with self.db.scoped(username, "person") as conn:
            fid = conn.execute(
                "INSERT INTO memory.facts (username, topics, detail, kind, summary, embedding,"
                " embed_model, importance, pinned, trust, source) VALUES"
                " (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (username, list(topics), json.dumps({"v": 1, **(detail or {})}), kind, summary, vec,
                 self.embedder.name, importance, pinned, trust, source)).fetchone()[0]
            self._audit(conn, fid, username, "fact.add", actor, {"kind": kind})
        return fid

    # -- forgetting ------------------------------------------------------------

    def forget(self, object_id: uuid.UUID | str, actor: str, scope: str = "person") -> str | None:
        """Delete an episode or fact the actor can see. Returns what was deleted.

        The audit row records that something was deleted, never its text.
        """
        oid = uuid.UUID(str(object_id))
        if scope == "admin" and not self.is_admin(actor):
            raise StoreError(f"{actor} is not an admin")
        with self.db.scoped(actor, scope) as conn:
            # People delete only their own; shared household rows need an admin.
            own = "" if scope == "admin" else " AND username = %s"
            args = (oid,) if scope == "admin" else (oid, actor)
            for table, what in (("episodes", "episode"), ("facts", "fact")):
                row = conn.execute(f"DELETE FROM memory.{table} WHERE id = %s{own} RETURNING username",
                                   args).fetchone()
                if row:
                    if what == "episode":
                        # Facts left with no evidence were derived from this alone.
                        conn.execute("DELETE FROM memory.facts f WHERE f.source = 'extractor'"
                                     " AND NOT EXISTS (SELECT 1 FROM memory.fact_sources s"
                                     " WHERE s.fact_id = f.id)")
                    self._audit(conn, oid, row[0], f"{what}.forget", actor, {})
                    return what
        return None

    # -- stats -------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self.db.scoped(None, "admin") as conn:
            ep = dict(conn.execute("SELECT username, count(*) FROM memory.episodes"
                                   " GROUP BY username").fetchall())
            facts = dict(conn.execute("SELECT username, count(*) FROM memory.facts"
                                      " WHERE state = 'active' GROUP BY username").fetchall())
            size = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
        return {"episodes": ep, "facts": facts, "db_bytes": size}

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _audit(conn, object_id, username, op: str, actor: str, detail: dict[str, Any]) -> None:
        conn.execute("INSERT INTO memory.events (object_id, username, op, actor, detail)"
                     " VALUES (%s,%s,%s,%s,%s)", (object_id, username, op, actor, json.dumps(detail)))
