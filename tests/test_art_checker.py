"""Checking pictures before they're shown (homeai/art/checker.py, ART_FEATURES_PLAN F4)."""
from __future__ import annotations

import json
import random

import pytest

from homeai.art import checker as CK
from homeai.art import features as F
from homeai.art.patcher import features_patch_for
from homeai.art.session import ArtSession
from homeai.art.store import Store
from tests.test_daemon import assistant  # noqa: F401 - fixture


def feats(*pairs):
    return [F.Feature(chr(65 + i), kind, text) for i, (kind, text) in enumerate(pairs)]


RIDER = feats(("subject", "a single red dragon with one head and four legs"),
              ("element", "one samurai in full armor sitting on the dragon's back"), ("setting", "the ocean"))


def test_the_checklist_comes_from_the_features():
    checks = CK.checklist(RIDER)
    assert [(c.kind, c.expect, c.noun) for c in checks] == [("heads", 1, "dragon"), ("riders", 1, "dragon")]
    assert "dragon" in checks[0].question and checks[1].question.startswith("How many people are riding")


@pytest.mark.parametrize("subject, heads", [
    ("a single flying dragon", 1), ("a dragon with two heads", 2), ("a hydra dragon", None),
    ("a two-headed dragon", None), ("two dragons fighting", None), ("a castle on a hill", None),
    ("dragons in the sky", None), ("an undead wyvern", 1)])
def test_heads_only_when_one_creature_is_promised(subject, heads):
    got = [c.expect for c in CK.checklist(feats(("subject", subject))) if c.kind == "heads"]
    assert got == ([heads] if heads else [])


def test_riders():
    two = feats(("subject", "a dragon"), ("element", "two knights riding the dragon"))
    assert [c.expect for c in CK.checklist(two) if c.kind == "riders"] == [2]
    standing = feats(("subject", "a dragon"), ("element", "a knight standing beside it"))
    assert [c for c in CK.checklist(standing) if c.kind == "riders"] == []
    nobody = feats(("subject", "a samurai riding a horse"))
    assert [(c.kind, c.expect) for c in CK.checklist(nobody)] == [("heads", 1), ("riders", 1)]


@pytest.mark.parametrize("answer, n", [("1", 1), ("One.", 1), ("There are two heads.", 2), ("3", 3),
                                       ("I can't tell", None), ("none", 0)])
def test_parse_count(answer, n):
    assert CK.parse_count(answer) == n


def test_problems_name_what_is_wrong(tmp_path):
    answers = {"heads": "2", "riders": "1"}
    ask = lambda image, q: answers["heads" if "heads" in q else "riders"]  # noqa: E731
    assert CK.problems(tmp_path / "x.png", CK.checklist(RIDER), ask) == ["2 heads"]
    answers.update(heads="1", riders="3")
    assert CK.problems(tmp_path / "x.png", CK.checklist(RIDER), ask) == ["3 riders"]
    answers.update(riders="who knows")
    assert CK.problems(tmp_path / "x.png", CK.checklist(RIDER), ask) == []     # a shrug is not a failure


# -- in art mode: check, redraw, show -------------------------------------------------------------

@pytest.fixture
def art(tmp_path):
    from tests.test_art_voice import FakeJobs, FakeSwitcher, FakeViewer
    from homeai.art import voice as AV
    said = json.dumps({"ops": [{"op": "add", "kind": "subject", "text": "a single dragon"},
                               {"op": "add", "kind": "element", "text": "one samurai sitting on its back"}]})
    patch = lambda spec, text: features_patch_for(spec, text, chat=lambda m, s: said)  # noqa: E731

    def build(owner, notify):
        jobs = FakeJobs(notify)
        return ArtSession(Store(tmp_path / "p"), jobs, owner, patch=patch, rng=random.Random(1)), jobs

    answers, warmed = [], []

    def ask(image, question):
        value = answers.pop(0) if answers else "1"
        if isinstance(value, Exception):
            raise value
        return value
    a = AV.ArtVoice(FakeSwitcher(), build, lambda t, d: None, viewer=FakeViewer(), pictures=tmp_path / "Pictures",
                    background=lambda fn: fn(), checker=ask, warm_voice=lambda: warmed.append(1))
    a._start_idle_watch = lambda: None
    a.answers, a.warmed, a.tmp = answers, warmed, tmp_path
    a.handle("draw a dragon with a samurai on its back", "Adam")
    return a


def finish(art, node_id):
    project = art.session.project
    image = art.tmp / f"{node_id}.png"
    image.write_bytes(b"png")
    project.finish(node_id, image)
    art._on_job({"state": "done", "project": project.id, "node": node_id, "image": str(project.image(node_id))})


def exported(art):
    return sorted(p.name for p in (art.tmp / "Pictures").rglob("*.png"))


def test_a_wrong_picture_is_redrawn_before_anyone_sees_it(art):
    art.answers += ["2", "1"]                    # heads 2 (wrong), riders 1
    finish(art, "001")
    project = art.session.project
    assert project.nodes["001"].state == "failed" and "2 heads" in project.nodes["001"].meta["error"]
    assert project.current == "002" and project.nodes["002"].meta["redraw_of"] == "001"
    assert project.nodes["002"].meta["tries"] == 1 and project.nodes["002"].spec["seed"] != \
        project.nodes["001"].spec["seed"]
    assert exported(art) == [] and art.status.startswith("Redrawing") and art.warmed == []
    finish(art, "002")                            # all answers "1": right
    assert exported(art) == [f"{project.id}-002.png"] and art.status == "Done." and art.warmed == [1]


def test_after_three_redraws_the_last_try_is_shown(art):
    art.answers += ["2", "1"] * 4
    for node in ("001", "002", "003", "004"):
        finish(art, node)
    project = art.session.project
    assert [project.nodes[n].state for n in ("001", "002", "003")] == ["failed"] * 3
    assert exported(art) == [f"{project.id}-004.png"] and art.status == "Best I could do: 2 heads."
    assert "005" not in project.nodes


def test_a_broken_checker_shows_the_picture(art):
    art.answers.append(OSError("ollama is down"))
    finish(art, "001")
    assert len(exported(art)) == 1 and art.status == "Done." and art.warmed == [1]


def test_a_redraw_keeps_its_place_among_the_options(art):
    session = art.session
    art.handle("give me four", "Adam")
    grid = list(session.grid)
    art.answers += ["1", "1", "3", "1"]           # option 1 right, option 2 has three heads
    finish(art, grid[0])
    finish(art, grid[1])
    assert session.grid[0] == grid[0] and session.grid[1] not in grid
    assert session.project.nodes[session.grid[1]].meta["redraw_of"] == grid[1]
    assert session.project.current == grid[0]     # the person is still looking at option 1


def test_no_checker_no_check(art):
    art.checker = None
    art.answers.append("5")
    finish(art, "001")
    assert len(exported(art)) == 1 and art.answers == ["5"]


def test_a_picture_checked_as_art_mode_ends_is_still_saved(art):
    session = art.session
    seen = []

    def ask(image, question):                     # "I'm done drawing" arrives while the check runs
        seen.append(art.status)
        art.session = None
        return "1"
    art.checker = ask
    project = session.project
    image = art.tmp / "x.png"
    image.write_bytes(b"png")
    project.finish("001", image)
    art.session = session
    art._on_job({"state": "done", "project": project.id, "node": "001", "image": str(project.image("001"))})
    assert seen[0] == "Checking..." and exported(art) == [f"{project.id}-001.png"]
    assert (art.tmp / "Pictures" / "adam" / f"{project.id}-001.json").exists()      # with its description


def test_daemon_checks_pictures_unless_turned_off(assistant, monkeypatch):
    import dataclasses
    from homeai.config import ArtConfig
    assistant.cfg = dataclasses.replace(assistant.cfg, art=ArtConfig(voice=True, check=True))
    assert assistant._build_art().checker is CK.ollama_ask
    assistant.cfg = dataclasses.replace(assistant.cfg, art=ArtConfig(voice=True, check=False))
    assert assistant._build_art().checker is None
