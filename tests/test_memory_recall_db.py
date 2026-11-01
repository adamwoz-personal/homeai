# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Recall SQL against a throwaway database (memory_store/recall.fetch)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("psycopg_pool")

from homeai.memory_store import recall  # noqa: E402
from homeai.memory_store.store import Episode  # noqa: E402

NOW = datetime.now(timezone.utc)


def find(store, who, text, window=None):
    vec = store.embedder.embed([text], kind="query")[0]
    with store.db.scoped(who) as conn:
        return recall.fetch(conn, who, vec, text, window)


@pytest.fixture
def seeded(mem_store):
    add = mem_store.add_episode
    add(Episode("adam", "what is a good beginner snake", "A corn snake is gentle.",
                occurred_at=NOW - timedelta(days=2)))
    add(Episode("ryan", "my favourite snake is the king cobra", "King cobras are amazing.",
                occurred_at=NOW - timedelta(days=1)))
    add(Episode("household", "the garage door code changed", "Noted.",
                occurred_at=NOW - timedelta(days=30)))
    mem_store.add_fact("adam", "Adam is an engineer who likes snakes", kind="profile")
    mem_store.add_fact("ryan", "Ryan wants to be a herpetologist", pinned=True)
    return mem_store


def test_finds_own_episode_by_vector_and_text(seeded):
    rows = find(seeded, "adam", "beginner snake")
    hits = {(c.lists[0], c.text_user) for c in rows if c.kind == "episode"}
    assert ("episode_vec", "what is a good beginner snake") in hits
    assert ("episode_fts", "what is a good beginner snake") in hits


def test_never_returns_another_persons_rows(seeded):
    for who, other in (("adam", "ryan"), ("ryan", "adam")):
        rows = find(seeded, who, "snake cobra herpetologist engineer")
        assert rows and {c.username for c in rows} <= {who, "household"}, other


def test_household_rows_are_shared(seeded):
    assert any(c.username == "household" for c in find(seeded, "ryan", "garage door code"))


def test_standing_facts_only_for_the_speaker(seeded):
    standing = [c for c in find(seeded, "adam", "anything at all") if c.lists == ("standing",)]
    assert [c.summary for c in standing] == ["Adam is an engineer who likes snakes"]
    ryan = [c for c in find(seeded, "ryan", "anything at all") if c.lists == ("standing",)]
    assert [c.summary for c in ryan] == ["Ryan wants to be a herpetologist"]


def test_time_window_limits_episodes_not_facts(seeded):
    window = (NOW - timedelta(days=3), NOW - timedelta(days=1, hours=12))
    rows = find(seeded, "adam", "snake", window)
    assert {c.text_user for c in rows if c.kind == "episode"} == {"what is a good beginner snake"}
    assert find(seeded, "adam", "snake", (NOW - timedelta(hours=1), NOW)) and all(
        c.kind == "fact" for c in find(seeded, "adam", "snake", (NOW - timedelta(hours=1), NOW)))


def test_similarity_is_cosine(seeded):
    # Episodes embed the question and the answer together.
    rows = find(seeded, "adam", "what is a good beginner snake A corn snake is gentle.")
    exact = [c for c in rows if c.text_user == "what is a good beginner snake"]
    assert exact[0].sim == pytest.approx(1.0, abs=1e-4)


def test_archived_rows_are_not_recalled(seeded):
    with seeded.db.scoped(None, "admin") as conn:
        conn.execute("UPDATE memory.episodes SET state = 'archived'")
        conn.execute("UPDATE memory.facts SET state = 'archived'")
    assert find(seeded, "adam", "beginner snake") == []
