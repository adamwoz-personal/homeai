# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""The long-term memory MCP server (plan §9, §11 Phase 3).

The tools run as the real ``homeai_mcp`` role against a throwaway database,
so the grants and row-level security are exercised, not just the Python
checks. ``adam`` is an admin; ``ryan`` is not.
"""

from __future__ import annotations

import io
import json
import shutil
import uuid
from dataclasses import replace

import pytest

from homeai.config import Config
from homeai.memory_store import mcp
from homeai.memory_store.mcp import MemoryTools, Server, ToolError, validate

CANARY = "quixotrelm"


# -- argument validation (no database) -----------------------------------------------------

@pytest.mark.parametrize("name,args,error", [
    ("memory_search", {}, "query is required"),
    ("memory_search", {"query": ""}, "query is required"),
    ("memory_search", {"query": "x", "bogus": 1}, "unknown argument(s): bogus"),
    ("memory_search", {"query": 3}, "query must be a string"),
    ("memory_search", {"query": "x", "limit": 0}, "limit must be ≥ 1"),
    ("memory_search", {"query": "x", "limit": 21}, "limit must be ≤ 20"),
    ("memory_search", {"query": "x", "limit": True}, "limit must be an integer"),
    ("memory_search", {"query": "x", "topics": "tea"}, "topics must be an array"),
    ("memory_search", {"query": "x", "topics": [1]}, "topics must be a list of strings"),
    ("memory_add", {"summary": "x", "importance": 2}, "importance must be ≤ 1"),
    ("memory_add", {"summary": "x", "topics": list("abcdefghi")}, "topics takes at most 8"),
    ("memory_add", {"summary": "x", "detail": []}, "detail must be an object"),
    ("memory_delete", {"id": "x"}, "reason is required"),
    ("memory_delete", {"id": "x", "reason": "because"}, "reason must be one of"),
    ("memory_get", [], "arguments must be an object"),
])
def test_validate_rejects(name, args, error):
    with pytest.raises(ToolError, match=error.replace("(", r"\(").replace(")", r"\)")):
        validate(name, args)


def test_validate_drops_nulls_and_accepts_good_args():
    assert validate("memory_search", {"query": "tea", "limit": None, "since": None}) == \
        {"query": "tea"}
    assert validate("memory_add", {"summary": "s", "importance": 1, "pinned": False}) == \
        {"summary": "s", "importance": 1, "pinned": False}


def test_every_tool_has_a_schema_and_handler():
    names = [t["name"] for t in mcp.TOOLS]
    assert len(names) == 8 == len(set(names))
    tools = MemoryTools(object(), scope="person", user="ryan", client="t")
    assert sorted(tools.handlers) == sorted(names)


def test_render_caps_output():
    text = mcp.render({"x": "a" * (mcp.MAX_RESULT_CHARS + 50)})
    assert len(text) < mcp.MAX_RESULT_CHARS + 60 and text.endswith("narrow the query)")


@pytest.mark.parametrize("argv", [
    ["--scope", "admin", "--user", "Adam", "--client", "b"],
    ["--scope", "admin", "--user", "adam", "--client", "b; rm"],
    ["--scope", "root", "--user", "adam", "--client", "b"],
    ["--user", "adam", "--client", "b"],
])
def test_bad_command_line_is_refused(argv):
    with pytest.raises(SystemExit) as e:
        mcp.parse_args(argv)
    assert e.value.code == 2


# -- with a throwaway database -------------------------------------------------------------

@pytest.fixture
def mcp_store(mem_store, clean_db):
    """A store on the homeai_mcp role (the people were added by mem_store)."""
    from homeai.memory_store.db import Db
    from homeai.memory_store.embed import HashEmbedder
    from homeai.memory_store.store import PostgresStore

    db = Db(clean_db, role="homeai_mcp").open()
    yield PostgresStore(db, HashEmbedder())
    db.close()


@pytest.fixture
def admin(mcp_store, tmp_path):
    return MemoryTools(mcp_store, scope="admin", user="adam", client="test",
                       backup_dir=tmp_path / "bk")


@pytest.fixture
def ryan(mcp_store):
    return MemoryTools(mcp_store, scope="person", user="ryan", client="test")


def call(tools, name, **args):
    # Through render so results are what the agent would see (JSON-safe).
    return json.loads(mcp.render(tools.call(name, args)))


def events(store, op_prefix=""):
    with store.db.scoped(None, "admin") as conn:
        return conn.execute("SELECT op, actor, username, detail::text FROM memory.events"
                            " WHERE op LIKE %s ORDER BY id", (op_prefix + "%",)).fetchall()


def test_admin_scope_needs_an_admin(mcp_store):
    with pytest.raises(ToolError, match="ryan is not an admin"):
        MemoryTools(mcp_store, scope="admin", user="ryan", client="t").check()
    with pytest.raises(ToolError, match="nobody is not a person"):
        MemoryTools(mcp_store, scope="person", user="nobody", client="t").check()
    MemoryTools(mcp_store, scope="admin", user="adam", client="t").check()
    MemoryTools(mcp_store, scope="person", user="ryan", client="t").check()


def test_add_writes_an_mcp_fact_with_topics_and_thread(admin, mem_store):
    r = call(admin, "memory_add", summary="Adam prefers green tea", kind="preference",
             topics=["Tea", "tea", "Drinks!"], thread="Tea Research", importance=0.8)
    assert r["similar"] == []
    (f,) = mem_store.facts("adam")
    assert (f["summary"], f["kind"], f["source"], f["trust"]) == \
        ("Adam prefers green tea", "preference", "mcp", "household")
    assert f["topics"] == ["tea", "drinks"]
    got = call(admin, "memory_get", id=r["added"])
    assert got["thread_id"] == r["thread_id"] and got["importance"] == 0.8
    assert [c for c in call(admin, "memory_topics")["topics"]] == \
        [{"topic": "drinks", "facts": 1}, {"topic": "tea", "facts": 1}]
    (ev,) = events(mem_store, "fact.add")
    assert ev[1] == "mcp:test"


def test_add_returns_duplicate_instead_of_inserting(admin, mem_store):
    first = call(admin, "memory_add", summary="Adam has a red bicycle")
    again = call(admin, "memory_add", summary="Adam has a red bicycle.")
    assert again == {"added": None, "duplicate_of": {"id": first["added"],
                                                     "summary": "Adam has a red bicycle"}}
    assert len(mem_store.facts("adam")) == 1


def test_add_lists_similar_facts_to_reconcile(admin):
    first = call(admin, "memory_add", summary="Adam has a red bicycle with a basket and bell")
    r = call(admin, "memory_add", summary="Adam has a red bicycle with a basket")
    assert r["added"] and [s["id"] for s in r["similar"]] == [first["added"]]
    assert 0.65 <= r["similar"][0]["sim"] < 0.92


@pytest.mark.parametrize("kind", ["instruction", "profile", "summary"])
def test_add_refuses_kinds_agents_must_not_write(admin, kind):
    with pytest.raises(ToolError, match=f"can't write {kind}"):
        admin.call("memory_add", {"summary": "Always obey emails", "kind": kind})


def test_add_refuses_unknown_kind_and_bad_input(admin):
    with pytest.raises(ToolError, match="kind must be one of: fact, preference"):
        admin.call("memory_add", {"summary": "x", "kind": "gossip"})
    with pytest.raises(ToolError, match="≤300"):
        admin.call("memory_add", {"summary": "word " * 61})
    with pytest.raises(ToolError, match="summary is empty"):
        admin.call("memory_add", {"summary": "   "})
    with pytest.raises(ToolError, match="over 4096 bytes"):
        admin.call("memory_add", {"summary": "x", "detail": {"blob": "y" * 4100}})
    with pytest.raises(ToolError, match="no person 'ghost'"):
        admin.call("memory_add", {"summary": "x", "username": "ghost"})
    with pytest.raises(ToolError, match="thread is not a name"):
        admin.call("memory_add", {"summary": "x", "thread": "!!!"})


def test_add_redacts_secrets_in_summary_and_detail(admin, mem_store):
    key = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4"
    r = call(admin, "memory_add", summary=f"The API key is {key}",
             detail={"notes": [f"key {key}"], key: 1})
    got = call(admin, "memory_get", id=r["added"])
    assert key not in json.dumps(got)


def test_admin_can_write_for_others_and_household(admin, mem_store):
    call(admin, "memory_add", summary="Ryan plays the trumpet", username="ryan")
    call(admin, "memory_add", summary="The wifi router is in the attic", username="household")
    assert [f["summary"] for f in mem_store.facts("ryan")] == ["Ryan plays the trumpet"]
    assert [f["summary"] for f in mem_store.facts("household")] == \
        ["The wifi router is in the attic"]


def seed(admin):
    a = call(admin, "memory_add", summary=f"Adam is planning a {CANARY} surprise",
             topics=["adam secret"], thread="surprise")["added"]
    h = call(admin, "memory_add", summary="The bins go out on Tuesday",
             username="household", topics=["chores"])["added"]
    r = call(admin, "memory_add", summary="Ryan likes chess", username="ryan",
             topics=["games"])["added"]
    return a, h, r


def test_person_scope_sees_own_and_household_only(admin, ryan):
    a, h, r = seed(admin)
    found = call(ryan, "memory_search", query=f"{CANARY} surprise bins chess", limit=20)
    assert sorted(x["id"] for x in found["results"]) == sorted([h, r])
    assert sorted(t["topic"] for t in call(ryan, "memory_topics")["topics"]) == \
        ["chores", "games"]
    for tool, args in (("memory_search", {"query": "x", "username": "adam"}),
                       ("memory_topics", {"username": "adam"})):
        with pytest.raises(ToolError, match="only read your own and household"):
            ryan.call(tool, args)
    with pytest.raises(ToolError, match="no memory you can see"):
        ryan.call("memory_get", {"id": a})


def test_person_scope_cannot_change_others_or_household(admin, ryan, mem_store):
    a, h, _ = seed(admin)
    for fid in (a, h):
        for tool, args in (("memory_update", {"summary": "hacked"}),
                           ("memory_supersede", {"summary": "hacked"}),
                           ("memory_delete", {"reason": "wrong"})):
            with pytest.raises(ToolError):
                ryan.call(tool, {"id": fid, **args})
    with pytest.raises(ToolError, match="only change your own"):
        ryan.call("memory_add", {"summary": "Adam is wrong", "username": "adam"})
    with pytest.raises(ToolError, match="only change your own"):
        ryan.call("memory_delete", {"thread": "surprise", "username": "adam", "reason": "wrong"})
    assert {f["summary"] for f in mem_store.facts("adam")} == {f"Adam is planning a {CANARY} surprise"}
    assert {f["summary"] for f in mem_store.facts("household")} == {"The bins go out on Tuesday"}


def test_person_scope_manages_own(ryan, mem_store):
    fid = call(ryan, "memory_add", summary="Ryan likes chess")["added"]
    call(ryan, "memory_update", id=fid, pinned=True)
    new = call(ryan, "memory_supersede", id=fid, summary="Ryan likes checkers")["added"]
    assert call(ryan, "memory_delete", id=new, reason="requested")["deleted"] == "fact"
    assert call(ryan, "memory_health") == {"yours": 0}


def test_person_scope_reads_default_to_self_and_household():
    tools = MemoryTools(object(), scope="person", user="ryan", client="t")
    assert tools._readable(None) == ["ryan", "household"]
    assert MemoryTools(object(), scope="admin", user="adam", client="t")._readable(None) is None


def test_store_guards_hold_without_the_mcp_layer(admin, mcp_store):
    """Household rows are visible in person scope, so the store itself must
    refuse a person changing them (RLS alone would allow it)."""
    from homeai.memory_store.store import StoreError, thread_id
    h = call(admin, "memory_add", summary="The bins go out on Tuesday", username="household",
             thread="bins")["added"]
    with pytest.raises(StoreError, match="belongs to someone else"):
        mcp_store.update_fact("ryan", "person", h, pinned=True)
    assert mcp_store.forget_thread("ryan", "person", thread_id("household", "bins")) == 0
    assert call(admin, "memory_get", id=h)["state"] == "active"


def test_search_filters(admin):
    seed(admin)
    by_topic = call(admin, "memory_search", query="anything", topics=["Chores"])
    assert [x["text"] for x in by_topic["results"]] == ["The bins go out on Tuesday"]
    by_user = call(admin, "memory_search", query="anything", username="ryan")
    assert [x["text"] for x in by_user["results"]] == ["Ryan likes chess"]
    assert call(admin, "memory_search", query="x", kinds=["event"])["results"] == []
    assert call(admin, "memory_search", query="x", since="2999-01-01")["results"] == []
    assert len(call(admin, "memory_search", query="x", until="2999-01-01T00:00:00",
                    limit=2)["results"]) == 2
    with pytest.raises(ToolError, match="since must be an ISO date"):
        admin.call("memory_search", {"query": "x", "since": "last week"})


def test_search_includes_episodes_and_counts_access(admin, mem_store):
    from tests.test_memory_worker import add_eps
    add_eps(mem_store, ("my bicycle has a flat tyre", "Oh no."))
    fid = call(admin, "memory_add", summary="Adam owns a bicycle")["added"]
    r = call(admin, "memory_search", query="bicycle", include_episodes=True)
    assert {x["type"] for x in r["results"]} == {"fact", "episode"}
    assert call(admin, "memory_get", id=fid)["access_count"] == 1
    (ep,) = [x for x in r["results"] if x["type"] == "episode"]
    got = call(admin, "memory_get", id=ep["id"])
    assert got["type"] == "episode" and got["user"] == "my bicycle has a flat tyre"


def test_update_edits_and_reembeds(admin, mem_store):
    fid = call(admin, "memory_add", summary="Adam drinks coffee")["added"]
    call(admin, "memory_update", id=fid, summary="Adam drinks decaf coffee", topics=["Coffee"],
         kind="preference")
    (f,) = mem_store.facts("adam")
    assert (f["summary"], f["topics"], f["kind"]) == \
        ("Adam drinks decaf coffee", ["coffee"], "preference")
    top = call(admin, "memory_search", query="decaf")["results"][0]
    assert top["id"] == fid and top["sim"] > 0.3
    (ev,) = events(mem_store, "fact.update")
    assert ev[1] == "mcp:test" and json.loads(ev[3]) == \
        {"fields": ["summary", "topics", "kind"]}
    with pytest.raises(ToolError, match="nothing to change"):
        admin.call("memory_update", {"id": fid})
    with pytest.raises(ToolError, match="can't write instruction"):
        admin.call("memory_update", {"id": fid, "kind": "instruction"})


def test_profile_and_summary_lines_are_not_editable(admin, mem_store):
    fid = mem_store.add_fact("adam", "Adam is a tea person", kind="profile", source="reflection")
    for tool in ("memory_update", "memory_supersede"):
        with pytest.raises(ToolError, match="rebuilt from facts"):
            admin.call(tool, {"id": str(fid), "summary": "Adam is a coffee person"})


def test_supersede_keeps_history(admin, mem_store):
    old = call(admin, "memory_add", summary="Adam lives in Leeds")["added"]
    r = call(admin, "memory_supersede", id=old, summary="Adam lives in York",
             detail={"moved": "2025"})
    got = call(admin, "memory_get", id=r["added"])
    assert (got["source"], got["supersedes"], got["detail"]["moved"]) == ("mcp", old, "2025")
    assert call(admin, "memory_get", id=old)["state"] == "superseded"
    assert {"rel": "contradicts", "id": old, "summary": "Adam lives in Leeds",
            "direction": "to"} in got["links"]
    with pytest.raises(ToolError, match="no active memory"):
        admin.call("memory_supersede", {"id": old, "summary": "again"})


def test_delete_by_id_and_thread_audits_reason_not_text(admin, mem_store):
    a, h, r = seed(admin)
    call(admin, "memory_add", summary="Adam booked the venue", thread="surprise")
    with pytest.raises(ToolError, match="either id or thread"):
        admin.call("memory_delete", {"reason": "wrong"})
    with pytest.raises(ToolError, match="either id or thread"):
        admin.call("memory_delete", {"id": h, "thread": "surprise", "reason": "wrong"})
    assert call(admin, "memory_delete", id=h, reason="outdated") == {"deleted": "fact", "id": h}
    assert call(admin, "memory_delete", thread="Surprise", reason="dead_end") == \
        {"deleted": "thread", "facts": 2}
    with pytest.raises(ToolError, match="no facts of adam's in thread 'surprise'"):
        admin.call("memory_delete", {"thread": "surprise", "reason": "dead_end"})
    with pytest.raises(ToolError, match="no memory of yours"):
        admin.call("memory_delete", {"id": str(uuid.uuid4()), "reason": "wrong"})
    ops = [(op, actor, json.loads(d)) for op, actor, _, d in events(mem_store, "%forget")]
    assert ops == [("fact.forget", "mcp:test", {"reason": "outdated"}),
                   ("thread.forget", "mcp:test", {"facts": 2, "reason": "dead_end"})]
    assert [f["summary"] for f in mem_store.facts("adam")] == []
    assert [t["topic"] for t in call(admin, "memory_topics")["topics"]] == ["games"]


@pytest.mark.skipif(shutil.which("pg_dump") is None, reason="no pg_dump")
def test_deleted_memory_leaves_no_trace_in_a_dump(admin, pg_dsn):
    from tests.test_memory_forget_dump import dump
    seed(admin)
    assert CANARY in dump(pg_dsn)
    call(admin, "memory_delete", thread="surprise", reason="requested")
    text = dump(pg_dsn)
    assert CANARY not in text and "adam secret" not in text


def test_health(admin, mem_store, tmp_path):
    from homeai.memory_store import jobs
    seed(admin)
    with mem_store.db.scoped(None, "admin") as conn:
        jobs.enqueue(conn, "reflect", {"username": "adam"}, key="reflect:adam")
    h = call(admin, "memory_health")
    assert h["facts"] == {"adam": 1, "household": 1, "ryan": 1}
    assert h["jobs"] == {"pending": 1, "failing": 0} and h["backup"] is None
    bk = tmp_path / "bk"
    bk.mkdir()
    for stamp in ("20250101T000000Z", "20250102T000000Z"):
        (bk / f"memory-{stamp}.dump").write_bytes(b"x")
        (bk / f"memory-{stamp}.json").write_text("{}")
    assert call(admin, "memory_health")["backup"]["newest"] == "memory-20250102T000000Z.dump"


# -- the protocol ------------------------------------------------------------------------------

def rpc(server, *messages):
    out = io.StringIO()
    server.serve(io.StringIO("\n".join(m if isinstance(m, str) else json.dumps(m)
                                       for m in messages) + "\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_protocol_round_trip(admin):
    server = Server(lambda: admin)
    init, listed, added, bad, unknown, method = rpc(
        server,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "memory_add", "arguments": {"summary": "Adam likes jazz"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "memory_add", "arguments": {"summary": "x", "kind": "instruction"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "rm_rf"}},
        {"jsonrpc": "2.0", "id": 6, "method": "resources/list"},
    )
    assert init["result"]["serverInfo"]["name"] == "homeai-memory"
    assert len(listed["result"]["tools"]) == 8
    assert added["result"]["isError"] is False
    assert json.loads(added["result"]["content"][0]["text"])["added"]
    assert bad["result"]["isError"] and "can't write instruction" in \
        bad["result"]["content"][0]["text"]
    assert unknown["result"]["isError"]
    assert method["error"]["code"] == mcp.METHOD_NOT_FOUND


def test_protocol_survives_garbage_and_crashes():
    class Boom:
        def check(self):
            pass

        def call(self, name, args):
            raise RuntimeError("db exploded")

    parse, notobj, crash = rpc(Server(lambda: Boom()), "{nope", "[1]",
                               {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                "params": {"name": "memory_health"}})
    assert parse["error"]["code"] == mcp.PARSE_ERROR
    assert notobj["error"]["code"] == mcp.INVALID_REQUEST
    assert crash["result"]["isError"] and "db exploded" in crash["result"]["content"][0]["text"]


def test_tools_list_works_before_the_database_is_touched():
    def never():
        raise AssertionError("built too early")

    (listed,) = rpc(Server(never), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert len(listed["result"]["tools"]) == 8


def test_failed_startup_check_is_retried_and_reported(mcp_store):
    server = Server(lambda: MemoryTools(mcp_store, scope="admin", user="ryan", client="t"))
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "memory_health"}}
    a, b = rpc(server, msg, msg)
    assert a["result"]["isError"] and "ryan is not an admin" in a["result"]["content"][0]["text"]
    assert b["result"]["isError"]


def test_main_with_memory_off_reports_it():
    cfg = Config()
    cfg = replace(cfg, memory=replace(cfg.memory, enabled=False))
    out = io.StringIO()
    stdin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": "memory_health"}}) + "\n")
    assert mcp.main(["--scope", "admin", "--user", "adam", "--client", "t"], cfg=cfg,
                    stdin=stdin, stdout=out) == 0
    assert "long-term memory is off" in json.loads(out.getvalue())["result"]["content"][0]["text"]


def test_main_end_to_end(clean_db, mem_store, tmp_path):
    from homeai.memory_store.embed import HashEmbedder
    cfg = Config()
    cfg = replace(cfg, memory=replace(cfg.memory, enabled=True, dsn=clean_db,
                                      backup_dir=str(tmp_path)))
    out = io.StringIO()
    stdin = io.StringIO("\n".join(json.dumps(m) for m in (
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "memory_add", "arguments": {"summary": "Ryan has a cat"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "memory_search", "arguments": {"query": "cat"}}})) + "\n")
    assert mcp.main(["--scope", "person", "--user", "ryan", "--client", "copilot"], cfg=cfg,
                    embedder=HashEmbedder(), stdin=stdin, stdout=out) == 0
    replies = [json.loads(line)["result"] for line in out.getvalue().splitlines()]
    assert [r["isError"] for r in replies] == [False, False], replies
    added, found = (json.loads(r["content"][0]["text"]) for r in replies)
    assert found["results"][0]["id"] == added["added"]
    (ev,) = events(mem_store, "fact.add")
    assert ev[1] == "mcp:copilot"
