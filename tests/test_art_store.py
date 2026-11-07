"""Art project store (homeai/art/store.py): tree, navigation, atomic writes, damage isolation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from homeai.art import store as S


@pytest.fixture
def st(tmp_path):
    return S.Store(tmp_path / "projects", clock=lambda: 1793000000.0)


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "out.png"
    path.write_bytes(b"\x89PNG fake")
    return path


def chain(project):
    return [n.id for n in project.lineage()]


def test_new_nodes_chain_from_current(st):
    p = st.create("general", "adam", "a fox")
    a = p.add("draw a fox", {"subject": "fox"})
    b = p.add("make it blue", {"subject": "fox", "palette": "blue"})
    assert (a.parent, b.parent, p.current) == (None, "001", "002")
    assert chain(p) == ["001", "002"]


def test_back_then_new_instruction_branches(st):
    p = st.create("general", "adam")
    for text in ("fox", "blue", "at night"):
        p.add(text, {})
    assert p.back(2).id == "001"
    branch = p.add("in snow", {})
    assert branch.parent == "001" and branch.id == "004"
    assert [n.id for n in p.children("001")] == ["002", "004"]
    assert "003" in p.nodes, "going back never deletes"


def test_back_stops_at_the_first_picture(st):
    p = st.create("general", "adam")
    p.add("fox", {})
    p.add("blue", {})
    assert p.back(10).id == "001"
    with pytest.raises(S.StoreError):
        p.back(0)


def test_back_on_an_empty_project_is_explained(st):
    with pytest.raises(S.StoreError, match="nothing to go back"):
        st.create("general", "adam").back()


def test_everything_survives_reopening(st, picture):
    p = st.create("general", "adam", "Fox!")
    p.add("fox", {"subject": "fox"})
    p.finish("001", picture, seed=7, seconds=6.6)
    p.add("blue", {})
    p.pin("001")
    q = st.open(p.id)
    assert q.current == "002" and q.info["pins"] == ["001"]
    assert q.nodes["001"].state == "done" and q.nodes["001"].meta == {"seed": 7, "seconds": 6.6}
    assert q.image("001").read_bytes() == picture.read_bytes()
    assert p.id.endswith("-fox")


def test_fail_keeps_the_reason(st):
    p = st.create("general", "adam")
    p.add("fox", {})
    p.fail("001", "I ran out of graphics memory")
    assert st.open(p.id).nodes["001"].meta["error"] == "I ran out of graphics memory"


def test_pending_after_a_crash(st, picture):
    p = st.create("general", "adam")
    p.add("fox", {})
    p.add("blue", {})
    p.finish("001", picture)
    assert [n.id for n in st.open(p.id).pending()] == ["002"]


def test_find_the_newest_matching_instruction(st):
    p = st.create("general", "adam")
    for text in ("draw a fox", "make it blue", "make it BLUE again", "add snow"):
        p.add(text, {})
    assert p.find("blue").id == "003"
    assert p.find("dragon") is None and p.find("  ") is None


def test_soft_delete_moves_current_and_undo_restores(st):
    p = st.create("general", "adam")
    p.add("fox", {})
    p.add("blue", {})
    p.delete("002")
    assert p.current == "001" and [n.id for n in p.live()] == ["001"]
    assert p.find("blue") is None
    with pytest.raises(S.StoreError, match="deleted"):
        p.select("002")
    p.delete("002", undo=True)
    assert p.select("002").id == "002"


def test_back_skips_deleted_ancestors(st):
    p = st.create("general", "adam")
    for text in ("a", "b", "c"):
        p.add(text, {})
    p.delete("002")
    p.select("003")
    assert p.back().id == "001"


def test_unpin(st):
    p = st.create("general", "adam")
    p.add("a", {})
    p.pin("001")
    p.pin("001")
    assert p.info["pins"] == ["001"]
    p.pin("001", on=False)
    assert p.info["pins"] == []


def test_unknown_parent_or_node_is_an_error(st):
    p = st.create("general", "adam")
    with pytest.raises(S.StoreError):
        p.add("x", {}, parent="999")
    with pytest.raises(S.StoreError):
        p.fail("999", "x")


def test_damaged_sidecar_is_isolated(st):
    p = st.create("general", "adam")
    p.add("fox", {})
    p.add("blue", {})
    (p.path / "nodes" / "001.json").write_text("{not json")
    q = st.open(p.id)
    assert list(q.nodes) == ["002"] and q.broken and q.broken[0].startswith("001.json")


def test_bad_state_counts_as_damage(st):
    p = st.create("general", "adam")
    p.add("fox", {})
    data = json.loads((p.path / "nodes/001.json").read_text())
    data["state"] = "exploded"
    (p.path / "nodes/001.json").write_text(json.dumps(data))
    assert st.open(p.id).broken


def test_current_pointing_at_damage_is_cleared(st):
    p = st.create("general", "adam")
    p.add("fox", {})
    (p.path / "nodes" / "001.json").write_text("{")
    assert st.open(p.id).current is None


def test_newer_schema_is_refused(st):
    p = st.create("general", "adam")
    info = json.loads((p.path / "project.json").read_text())
    info["schema"] = 2
    (p.path / "project.json").write_text(json.dumps(info))
    with pytest.raises(S.StoreError, match="schema 2"):
        st.open(p.id)


def test_missing_and_damaged_projects(st):
    with pytest.raises(S.StoreError, match="no project"):
        st.open("nope")
    p = st.create("general", "adam")
    (p.path / "project.json").write_text("{")
    with pytest.raises(S.StoreError, match="damaged"):
        st.open(p.id)


def test_same_second_projects_get_their_own_folder(st):
    a, b = st.create("general", "adam", "fox"), st.create("general", "adam", "fox")
    assert a.id != b.id and b.id.endswith("-2")


def test_projects_by_owner_newest_first(tmp_path):
    times = iter([100.0, 100.0, 200.0, 200.0, 300.0, 300.0])  # create reads the clock twice
    st = S.Store(tmp_path, clock=lambda: next(times))
    a, b, c = st.create("general", "adam", "a"), st.create("general", "ani", "b"), st.create("general", "adam", "c")
    assert st.projects("adam") == [c.id, a.id]
    assert st.projects() == [c.id, b.id, a.id]


def test_write_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "w" / "x.json"
    S.write_json(target, {"v": 1})

    def crash(*args):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError):
        S.write_json(target, {"v": 2})
    assert json.loads(target.read_text()) == {"v": 1}
    assert [p.name for p in target.parent.iterdir()] == ["x.json"], "no temp file left behind"


def test_lineage_survives_a_parent_loop(st):
    p = st.create("general", "adam")
    p.add("a", {})
    p.add("b", {})
    p.nodes["001"].parent = "002"
    assert len(p.lineage()) == 2


def test_slug():
    assert S.slug("A Fox, in the SNOW!") == "a-fox-in-the-snow"
    assert S.slug("!!!") == "untitled"
    assert S.slug("x" * 50) == "x" * 30
