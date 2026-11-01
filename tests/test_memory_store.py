# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""People, episodes, facts and forgetting (memory_store/store.py), against a
throwaway database. Skipped without Postgres."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("psycopg_pool")

from homeai.memory_store.store import Episode, StoreError  # noqa: E402


def ep(user="what's a good beginner snake", who="adam", **kw):
    return Episode(username=who, user=user, assistant="A corn snake.", **kw)


def events(store):
    with store.db.scoped(None, "admin") as conn:
        return conn.execute("SELECT op, actor, detail::text FROM memory.events ORDER BY id").fetchall()


def test_ensure_person_is_idempotent_and_never_demotes(mem_store):
    assert not mem_store.ensure_person("adam", "Someone Else", is_admin=False)
    p = mem_store.person("adam")
    assert p["display_name"] == "Adam" and p["is_admin"]
    assert mem_store.is_admin("adam") and not mem_store.is_admin("ryan")
    assert not mem_store.is_admin("nobody")


def test_ensure_person_rejects_bad_names(mem_store):
    with pytest.raises(StoreError):
        mem_store.ensure_person("Bad Name", "x")


def test_voice_identity_recorded(mem_store):
    with mem_store.db.scoped(None, "admin") as conn:
        rows = conn.execute("SELECT handle, username FROM memory.identities ORDER BY 1").fetchall()
    assert rows == [("Adam", "adam"), ("Ryan", "ryan")]


def test_add_and_list_episode(mem_store):
    eid = mem_store.add_episode(ep())
    assert isinstance(eid, uuid.UUID)
    [e] = mem_store.list_episodes("adam")
    assert e["id"] == eid and e["user"] == "what's a good beginner snake"
    assert e["assistant"] == "A corn snake." and e["v"] == 1
    assert ("episode.add", "recorder") == events(mem_store)[-1][:2]


def test_episode_secrets_are_redacted_before_storage(mem_store):
    mem_store.add_episode(ep(user="the gate code is 4491"))
    with mem_store.db.scoped(None, "admin") as conn:
        stored = conn.execute("SELECT content::text || text_for_search FROM memory.episodes"
                              ).fetchone()[0]
    assert "4491" not in stored and "[redacted]" in stored


def test_backfill_is_idempotent(mem_store):
    at = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    assert mem_store.add_episode(ep(occurred_at=at, source_ref="transcript:x")) is not None
    assert mem_store.add_episode(ep(occurred_at=at, source_ref="transcript:x")) is None
    assert len(mem_store.list_episodes("adam")) == 1


def test_skips_unknown_and_memory_disabled_people(mem_store):
    assert mem_store.add_episode(ep(who="stranger")) is None
    assert mem_store.add_episode(ep(who="guest")) is None   # seeded with memory off


def test_list_visibility(mem_store):
    mem_store.add_episode(ep(who="adam"))
    mem_store.add_episode(ep(who="ryan", user="dragons"))
    assert {e["username"] for e in mem_store.list_episodes("ryan")} == {"ryan"}
    assert {e["username"] for e in mem_store.list_episodes("adam", "admin")} == {"adam", "ryan"}
    assert {e["username"] for e in mem_store.list_episodes("adam", "admin", owner="ryan")} == {"ryan"}


def test_add_fact_validates(mem_store):
    fid = mem_store.add_fact("adam", "  Adam   likes corn snakes ", topics=("snakes",))
    with mem_store.db.scoped("adam") as conn:
        row = conn.execute("SELECT summary, topics, embed_model, source FROM memory.facts"
                           " WHERE id = %s", (fid,)).fetchone()
    assert row == ("Adam likes corn snakes", ["snakes"], "hash-test", "explicit")
    with pytest.raises(StoreError):
        mem_store.add_fact("adam", "x", kind="nonsense")
    with pytest.raises(StoreError):
        mem_store.add_fact("adam", "   ")
    with pytest.raises(StoreError):
        mem_store.add_fact("adam", "word " * 70)


def test_forget_own(mem_store):
    eid = mem_store.add_episode(ep())
    fid = mem_store.add_fact("adam", "likes snakes")
    assert mem_store.forget(eid, "adam") == "episode"
    assert mem_store.forget(str(fid), "adam") == "fact"
    assert mem_store.list_episodes("adam") == []
    assert mem_store.forget(eid, "adam") is None


def test_cannot_forget_someone_elses(mem_store):
    eid = mem_store.add_episode(ep(who="adam"))
    assert mem_store.forget(eid, "ryan") is None
    assert len(mem_store.list_episodes("adam")) == 1


def test_household_needs_an_admin_to_forget(mem_store):
    eid = mem_store.add_episode(ep(who="household"))
    assert mem_store.forget(eid, "ryan") is None
    assert mem_store.forget(eid, "adam", "admin") == "episode"


def test_admin_scope_is_checked(mem_store):
    eid = mem_store.add_episode(ep(who="adam"))
    with pytest.raises(StoreError):
        mem_store.forget(eid, "ryan", "admin")
    assert mem_store.forget(eid, "adam", "admin") == "episode"


def test_forget_audit_keeps_no_text(mem_store):
    eid = mem_store.add_episode(ep(user="a very private question"))
    mem_store.forget(eid, "adam")
    op, actor, detail = events(mem_store)[-1]
    assert (op, actor) == ("episode.forget", "adam")
    assert "private" not in detail
    with mem_store.db.scoped(None, "admin") as conn:
        assert "private" not in str(conn.execute("SELECT * FROM memory.events").fetchall())


def test_forgetting_an_episode_removes_facts_derived_only_from_it(mem_store):
    at = datetime.now(timezone.utc) - timedelta(minutes=1)
    eid = mem_store.add_episode(ep(occurred_at=at))
    derived = mem_store.add_fact("adam", "derived fact", source="extractor")
    explicit = mem_store.add_fact("adam", "explicit fact")
    with mem_store.db.scoped("adam") as conn:
        conn.execute("INSERT INTO memory.fact_sources VALUES (%s, %s, %s)", (derived, eid, at))
    mem_store.forget(eid, "adam")
    with mem_store.db.scoped("adam") as conn:
        left = {r[0] for r in conn.execute("SELECT id FROM memory.facts")}
    assert left == {explicit}


def test_stats(mem_store):
    mem_store.add_episode(ep())
    mem_store.add_fact("ryan", "likes dragons")
    s = mem_store.stats()
    assert s["episodes"] == {"adam": 1} and s["facts"] == {"ryan": 1}
    assert s["db_bytes"] > 0
