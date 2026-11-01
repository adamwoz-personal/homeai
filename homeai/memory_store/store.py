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
MAX_DERIVATION_DEPTH = 8   # summaries of summaries of ... facts
FACT_KINDS = ("fact", "preference", "event", "question", "answer", "thought", "opinion",
              "plan", "summary", "profile", "instruction")


class StoreError(RuntimeError):
    pass


_THREADS = uuid.UUID("6f1f9a52-3c1e-4c55-9a59-6d0f3c1a7e10")


def thread_id(username: str, name: str) -> uuid.UUID:
    """Same person + same thread name -> same id, across extraction runs."""
    return uuid.uuid5(_THREADS, f"{username}:{name}")


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
                    break
            else:
                return None
        self._prune_topics()
        return what

    # -- people settings ---------------------------------------------------------

    def set_memory_enabled(self, username: str, on: bool, actor: str) -> bool:
        with self.db.scoped(None, "admin") as conn:
            n = conn.execute("UPDATE memory.people SET memory_enabled = %s WHERE username = %s",
                             (on, username)).rowcount
            if n:
                self._audit(conn, None, username, "person.memory_" + ("on" if on else "off"),
                            actor, {})
        return bool(n)

    # -- extraction (Phase 2) -------------------------------------------------------

    def unextracted(self, username: str, limit: int = 20) -> list[dict[str, Any]]:
        """Oldest episodes of ``username`` the extractor hasn't read."""
        with self.db.scoped(username, "person") as conn:
            rows = conn.execute(
                "SELECT id, occurred_at, content, trust FROM memory.episodes"
                " WHERE username = %s AND extracted_at IS NULL AND state = 'active'"
                " ORDER BY occurred_at LIMIT %s", (username, limit)).fetchall()
        return [{"id": r[0], "at": r[1], "user": r[2].get("user", ""),
                 "assistant": r[2].get("assistant", ""), "trust": r[3]} for r in rows]

    def count_unextracted(self, username: str) -> int:
        with self.db.scoped(username, "person") as conn:
            return conn.execute("SELECT count(*) FROM memory.episodes WHERE username = %s"
                                " AND extracted_at IS NULL AND state = 'active'",
                                (username,)).fetchone()[0]

    def mark_extracted(self, username: str, episodes: list[dict[str, Any]]) -> None:
        with self.db.scoped(username, "person") as conn:
            conn.execute("UPDATE memory.episodes SET extracted_at = now()"
                         " WHERE username = %s AND (id, occurred_at) IN"
                         " (SELECT * FROM unnest(%s::uuid[], %s::timestamptz[]))",
                         (username, [e["id"] for e in episodes], [e["at"] for e in episodes]))

    def known_topics(self, limit: int = 40) -> list[str]:
        with self.db.scoped(None, "admin") as conn:
            return [r[0] for r in conn.execute("SELECT name FROM memory.topics"
                                               " ORDER BY use_count DESC, name LIMIT %s", (limit,))]

    def nearest_facts(self, username: str, vec, k: int = 5) -> list[dict[str, Any]]:
        """The person's own active facts closest to ``vec``, with similarity."""
        with self.db.scoped(username, "person") as conn:
            rows = conn.execute(
                "SELECT id, summary, 1 - (embedding <=> %s::vector), pinned, source, trust,"
                " created_at, topics, thread_id, kind FROM memory.facts"
                " WHERE username = %s AND state = 'active' AND embedding IS NOT NULL"
                " ORDER BY embedding <=> %s::vector LIMIT %s", (vec, username, vec, k)).fetchall()
        keys = ("id", "summary", "sim", "pinned", "source", "trust", "created_at", "topics",
                "thread_id", "kind")
        return [dict(zip(keys, r)) for r in rows]

    def apply_note(self, username: str, note, decision, neighbours: list[dict[str, Any]],
                   vec, episodes: list[dict[str, Any]], *, trust: str, actor: str) -> uuid.UUID | None:
        """Apply one reconcile decision in one transaction, with provenance.

        Returns the fact that now holds the note (None for skip).
        """
        if decision.action == "skip":
            return None
        srcs = [episodes[i] for i in note.sources]
        thread = thread_id(username, note.thread) if note.thread else None
        target = neighbours[decision.target] if decision.target is not None else None
        with self.db.scoped(username, "person") as conn:
            if decision.action == "noop":
                fid = target["id"]
                conn.execute("UPDATE memory.facts SET access_count = access_count + 1,"
                             " strength = strength + 0.1 WHERE id = %s", (fid,))
            elif decision.action == "update":
                fid = target["id"]
                conn.execute(
                    "UPDATE memory.facts SET summary = %s, embedding = %s,"
                    " topics = ARRAY(SELECT DISTINCT unnest(topics || %s::text[])),"
                    " importance = GREATEST(importance, %s), updated_at = now(),"
                    " thread_id = coalesce(thread_id, %s) WHERE id = %s",
                    (decision.summary, vec, list(note.topics), note.importance, thread, fid))
                self._audit(conn, fid, username, "fact.update", actor, {"kind": note.kind})
            else:   # add / supersede
                fid = conn.execute(
                    "INSERT INTO memory.facts (username, topics, detail, kind, summary, embedding,"
                    " embed_model, importance, trust, source, derived_by, thread_id, supersedes)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'extractor',%s,%s,%s) RETURNING id",
                    (username, list(note.topics), json.dumps({"v": 1, "thread": note.thread}),
                     note.kind, decision.summary or note.summary, vec, self.embedder.name,
                     note.importance, trust, actor, thread,
                     target["id"] if target else None)).fetchone()[0]
                if decision.action == "supersede":
                    conn.execute("UPDATE memory.facts SET state = 'superseded', valid_to = now(),"
                                 " updated_at = now() WHERE id = %s", (target["id"],))
                    conn.execute("INSERT INTO memory.fact_links (src, dst, rel)"
                                 " VALUES (%s, %s, 'contradicts')", (fid, target["id"]))
                    self._drop_derived(conn, target["id"], username)
                self._audit(conn, fid, username, f"fact.{decision.action}", actor,
                            {"kind": note.kind})
            for e in srcs:
                conn.execute("INSERT INTO memory.fact_sources (fact_id, episode_id, episode_at)"
                             " VALUES (%s,%s,%s) ON CONFLICT DO NOTHING", (fid, e["id"], e["at"]))
            for t in note.topics:
                conn.execute("INSERT INTO memory.topics (name, use_count) VALUES (%s, 1)"
                             " ON CONFLICT (name) DO UPDATE SET use_count = memory.topics.use_count + 1",
                             (t,))
        return fid

    # -- facts: reading, profile, bulk forget ----------------------------------------

    def facts(self, username: str, *, kinds: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """The person's own active facts, most important first."""
        sql = ("SELECT id, kind, summary, topics, importance, pinned, trust, source, thread_id,"
               " created_at, updated_at, detail FROM memory.facts"
               " WHERE username = %s AND state = 'active'"
               + (" AND kind = ANY(%s)" if kinds else "") +
               " ORDER BY pinned DESC, importance DESC, updated_at DESC")
        args = (username, list(kinds)) if kinds else (username,)
        with self.db.scoped(username, "person") as conn:
            rows = conn.execute(sql, args).fetchall()
        keys = ("id", "kind", "summary", "topics", "importance", "pinned", "trust", "source",
                "thread_id", "created_at", "updated_at", "detail")
        return [dict(zip(keys, r)) for r in rows]

    def replace_profile(self, username: str, lines: list[tuple[str, list]], input_hash: str,
                        *, actor: str) -> int:
        """Swap the person's reflection profile for ``lines`` atomically.

        ``lines``: (summary, [source fact ids]). Each line is linked to its
        sources (derived_from), so deleting a source deletes the line.
        """
        vecs = self.embedder.embed([s for s, _ in lines], kind="doc") if lines else []
        with self.db.scoped(username, "person") as conn:
            conn.execute("DELETE FROM memory.facts WHERE username = %s AND source = 'reflection'",
                         (username,))
            for (summary, sources), vec in zip(lines, vecs):
                fid = conn.execute(
                    "INSERT INTO memory.facts (username, kind, summary, embedding, embed_model,"
                    " importance, trust, source, derived_by, detail) VALUES"
                    " (%s,'profile',%s,%s,%s,0.7,'owner','reflection',%s,%s) RETURNING id",
                    (username, summary, vec, self.embedder.name, actor,
                     json.dumps({"v": 1, "input_hash": input_hash}))).fetchone()[0]
                with conn.cursor() as cur:
                    cur.executemany("INSERT INTO memory.fact_links (src, dst, rel)"
                                    " VALUES (%s, %s, 'derived_from')",
                                    [(fid, s) for s in sources])
            self._audit(conn, None, username, "profile.rebuild", actor, {"lines": len(lines)})
        return len(lines)

    def profile_hash(self, username: str) -> str | None:
        with self.db.scoped(username, "person") as conn:
            row = conn.execute("SELECT detail->>'input_hash' FROM memory.facts WHERE username = %s"
                               " AND source = 'reflection' LIMIT 1", (username,)).fetchone()
        return row[0] if row else None

    def last_episode(self, username: str) -> dict[str, Any] | None:
        with self.db.scoped(username, "person") as conn:
            row = conn.execute("SELECT id, occurred_at, content FROM memory.episodes"
                               " WHERE username = %s ORDER BY occurred_at DESC LIMIT 1",
                               (username,)).fetchone()
        return {"id": row[0], "at": row[1], **row[2]} if row else None

    def sources_of(self, username: str, fact_ids: list) -> list[uuid.UUID]:
        """Episodes the given facts were extracted from."""
        if not fact_ids:
            return []
        with self.db.scoped(username, "person") as conn:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT s.episode_id FROM memory.fact_sources s JOIN memory.facts f"
                " ON f.id = s.fact_id WHERE f.username = %s AND s.fact_id = ANY(%s)",
                (username, list(fact_ids)))]

    def supersede_fact(self, username: str, old_id, summary: str, *, actor: str) -> uuid.UUID:
        """The person corrected a fact: a new explicit fact replaces it."""
        summary = redact(" ".join((summary or "").split()))[:300]
        if len(summary) < 1:
            raise StoreError("empty correction")
        vec = self.embedder.embed([summary], kind="doc")[0]
        with self.db.scoped(username, "person") as conn:
            old = conn.execute("SELECT kind, topics, pinned, importance, thread_id FROM memory.facts"
                               " WHERE id = %s AND username = %s AND state = 'active'",
                               (old_id, username)).fetchone()
            if old is None:
                raise StoreError("that memory is gone")
            kind = "fact" if old[0] in ("profile", "summary") else old[0]
            fid = conn.execute(
                "INSERT INTO memory.facts (username, topics, kind, summary, embedding, embed_model,"
                " importance, pinned, trust, source, thread_id, supersedes) VALUES"
                " (%s,%s,%s,%s,%s,%s,%s,%s,'owner','explicit',%s,%s) RETURNING id",
                (username, old[1], kind, summary, vec, self.embedder.name, max(old[3], 0.7),
                 old[2], old[4], old_id)).fetchone()[0]
            if old[0] == "profile":
                # A wrong profile line is rebuilt from facts; just drop it.
                conn.execute("DELETE FROM memory.facts WHERE id = %s", (old_id,))
            else:
                conn.execute("UPDATE memory.facts SET state = 'superseded', valid_to = now(),"
                             " updated_at = now() WHERE id = %s", (old_id,))
                conn.execute("INSERT INTO memory.fact_links (src, dst, rel)"
                             " VALUES (%s, %s, 'contradicts')", (fid, old_id))
                self._drop_derived(conn, old_id, username)
            self._audit(conn, fid, username, "fact.correct", actor, {})
        return fid

    def forget_items(self, username: str, *, facts: list = (), episodes: list = (),
                     actor: str) -> int:
        """Delete the person's own facts and episodes in one transaction.

        Facts left without any source episode go too (they came only from
        what was forgotten); summaries built from any of them go via the
        ``facts_drop_derived`` trigger. Returns rows deleted (not counting
        those cascades).
        """
        n = 0
        with self.db.scoped(username, "person") as conn:
            if episodes:
                n += conn.execute("DELETE FROM memory.episodes WHERE username = %s AND id = ANY(%s)",
                                  (username, list(episodes))).rowcount
                conn.execute("DELETE FROM memory.facts f WHERE f.username = %s"
                             " AND f.source = 'extractor' AND NOT EXISTS"
                             " (SELECT 1 FROM memory.fact_sources s WHERE s.fact_id = f.id)",
                             (username,))
            if facts:
                n += self._delete_facts(conn, "f.username = %s AND f.id = ANY(%s)",
                                        (username, list(facts)))
            self._audit(conn, None, username, "bulk.forget", actor,
                        {"facts": len(facts), "episodes": len(episodes)})
        self._prune_topics()
        return n

    def forget_person(self, username: str, actor: str) -> int:
        """Everything the person owns. The people row (and settings) stays."""
        with self.db.scoped(username, "person") as conn:
            n = self._delete_facts(conn, "f.username = %s", (username,))
            n += conn.execute("DELETE FROM memory.episodes WHERE username = %s",
                              (username,)).rowcount
            self._audit(conn, None, username, "person.forget_all", actor, {"rows": n})
        self._prune_topics()
        return n

    def count_for(self, username: str) -> int:
        """Episodes plus active facts the person owns."""
        with self.db.scoped(username, "person") as conn:
            return conn.execute(
                "SELECT (SELECT count(*) FROM memory.episodes WHERE username = %s)"
                " + (SELECT count(*) FROM memory.facts WHERE username = %s AND state = 'active')",
                (username, username)).fetchone()[0]

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

    def _prune_topics(self) -> int:
        """Drop topic names no fact uses any more: a forgotten fact's topic
        must not outlive it. Admin scope so everyone's facts count as uses."""
        with self.db.scoped(None, "admin") as conn:
            return conn.execute("DELETE FROM memory.topics t WHERE NOT EXISTS"
                                " (SELECT 1 FROM memory.facts f WHERE f.topics @> ARRAY[t.name])"
                                ).rowcount

    @staticmethod
    def _delete_facts(conn, where: str, params: tuple) -> int:
        """DELETE the facts matching ``where``, summaries before their sources.

        The facts_drop_derived trigger deletes a fact's summaries; if the same
        statement also targets one of them Postgres refuses ("tuple already
        modified by an operation triggered by the current command"). Deleting
        top-down, rows nothing else is derived from first, avoids that.
        """
        n = 0
        for _ in range(MAX_DERIVATION_DEPTH):
            step = conn.execute(
                f"DELETE FROM memory.facts f WHERE ({where}) AND NOT EXISTS"
                " (SELECT 1 FROM memory.fact_links l WHERE l.dst = f.id"
                " AND l.rel = 'derived_from')", params).rowcount
            n += step
            if not step:
                break
        return n + conn.execute(f"DELETE FROM memory.facts f WHERE {where}", params).rowcount

    @staticmethod
    def _drop_derived(conn, fact_id, username: str) -> None:
        """A superseded fact's summaries are stale: delete them and queue a
        rebuild (deletes do this in the facts_drop_derived trigger)."""
        n = conn.execute("DELETE FROM memory.facts WHERE id IN (SELECT src FROM memory.fact_links"
                         " WHERE dst = %s AND rel = 'derived_from')", (fact_id,)).rowcount
        if n:
            from . import jobs
            jobs.enqueue(conn, "reflect", {"username": username}, key=f"reflect:{username}")

    @staticmethod
    def _audit(conn, object_id, username, op: str, actor: str, detail: dict[str, Any]) -> None:
        conn.execute("INSERT INTO memory.events (object_id, username, op, actor, detail)"
                     " VALUES (%s,%s,%s,%s,%s)", (object_id, username, op, actor, json.dumps(detail)))
