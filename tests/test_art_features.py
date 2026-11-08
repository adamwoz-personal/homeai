"""Lettered features (homeai/art/features.py), their patcher, session, saving and window (ART_FEATURES_PLAN F1/F2)."""
from __future__ import annotations

import json
import random

import pytest

from homeai.art import features as F
from homeai.art import policy
from homeai.art import viewer as V
from homeai.art.patcher import MSG_NOT_CAUGHT, features_patch_for
from homeai.art.prompt_spec import SpecError
from homeai.art.session import ArtSession
from homeai.art.store import Store

DRAGON = [F.Feature("A", "subject", "a single green dragon with one head"),
          F.Feature("B", "element", "one samurai sitting on the dragon's back"),
          F.Feature("C", "setting", "a neon city at night"),
          F.Feature("D", "style", "cyberpunk")]


# -- the model of features -------------------------------------------------------------------

def test_change_keeps_the_letter_and_count():
    after, changed = F.apply_ops(DRAGON, [{"op": "change", "id": "a", "text": "a single red dragon with one head"}])
    assert [f.id for f in after] == ["A", "B", "C", "D"] and after[0].text == "a single red dragon with one head"
    assert changed == ["A"]


def test_add_takes_the_next_letter_never_a_removed_one():
    after, _ = F.apply_ops(DRAGON, [{"op": "remove", "id": "C"}])
    after, changed = F.apply_ops(after, [{"op": "add", "kind": "setting", "text": "mountains"}], used={"C"})
    assert changed == ["E"] and [f.id for f in after] == ["A", "B", "D", "E"]


def test_bad_operations_change_nothing():
    for ops in ([{"op": "change", "id": "Z", "text": "x"}], [{"op": "remove", "id": "A"}],
                [{"op": "add", "kind": "nonsense", "text": "x"}], [{"op": "add", "text": ""}], [{"op": "paint"}]):
        with pytest.raises(SpecError):
            F.apply_ops(DRAGON, ops)


def test_at_most_twelve_features():
    many = [F.Feature("A", "subject", "x")] + [F.Feature(chr(66 + i), "element", "y") for i in range(11)]
    with pytest.raises(SpecError):
        F.apply_ops(many, [{"op": "add", "text": "one more"}])


def test_render_puts_the_main_subject_first_and_the_look_last():
    shuffled = [DRAGON[3], DRAGON[2], DRAGON[1], DRAGON[0]]
    out = F.render(shuffled, "landscape")
    assert out.text == ("A single green dragon with one head. One samurai sitting on the dragon's back. "
                        "A neon city at night. Cyberpunk.")
    assert (out.width, out.height) == (1216, 832)


def test_describe_is_short():
    after, changed = F.apply_ops(DRAGON, [{"op": "change", "id": "A", "text": "red"}, {"op": "remove", "id": "C"},
                                          {"op": "add", "kind": "lighting", "text": "moonlight"}])
    assert F.describe(changed, DRAGON, after) == "Changed A, added E and removed C."
    assert F.describe([], DRAGON, DRAGON) == "Nothing changed."


def test_load_rejects_bad_stored_features():
    for raw in ("x", [{"id": "A", "kind": "subject"}], [{"id": "AA", "kind": "subject", "text": "x"}],
                [{"id": "A", "kind": "subject", "text": "x"}, {"id": "A", "kind": "element", "text": "y"}]):
        with pytest.raises(SpecError):
            F.load(raw)


def test_older_pictures_become_features():
    spec = {"subject": "a red fox", "details": "ears up", "palette": "blue", "seed": 3}
    assert [(f.id, f.kind, f.text) for f in F.features_of(spec)] == [
        ("A", "subject", "a red fox"), ("B", "element", "ears up"), ("C", "colour", "blue")]


@pytest.mark.parametrize("said, ids", [("remove C", ["C"]), ("delete feature b", ["B"]), ("get rid of C and D", ["C", "D"]),
                                       ("remove the moon", None)])
def test_letter_commands(said, ids):
    ops = F.letter_command(said)
    assert (ops is None) if ids is None else [o["id"] for o in ops] == ids


# -- the patcher -------------------------------------------------------------------------------

def spec_of(features, **extra):
    return {"features": F.dump(features), "aspect": "landscape", "letters_used": [f.id for f in features], **extra}


def test_patch_merges_a_change_into_its_letter():
    said = json.dumps({"ops": [{"op": "change", "id": "A", "text": "a single red dragon with one head"}]})
    result = features_patch_for(spec_of(DRAGON, seed=7), "make the dragon red", chat=lambda m, s: said)
    assert result.ok and result.changed == ["A"] and result.spec["seed"] == 7
    assert F.features_of(result.spec)[0].text == "a single red dragon with one head"
    assert result.spec["aspect"] == "landscape"


def test_patch_removes_by_letter_without_the_model():
    def no_model(*a):
        raise AssertionError("the model was asked")
    result = features_patch_for(spec_of(DRAGON), "remove C", chat=no_model)
    assert result.ok and [f.id for f in F.features_of(result.spec)] == ["A", "B", "D"]
    assert "C" in result.spec["letters_used"]


@pytest.mark.parametrize("said", ["not json", json.dumps({"ops": []}), json.dumps({"ops": [{"op": "remove", "id": "Q"}]}),
                                  json.dumps({"nothing": 1})])
def test_patch_failures_keep_the_picture(said):
    result = features_patch_for(spec_of(DRAGON), "do something", chat=lambda m, s: said)
    assert not result.ok and result.message == MSG_NOT_CAUGHT and result.spec == spec_of(DRAGON)


def test_new_picture_from_nothing_and_aspect():
    said = json.dumps({"ops": [{"op": "add", "kind": "subject", "text": "a wyvern"}], "aspect": "wide"})
    result = features_patch_for({}, "draw a wide picture of a wyvern", chat=lambda m, s: said)
    assert result.ok and result.spec["aspect"] == "wide" and result.changed == ["A"]


def test_the_model_is_shown_the_letters():
    seen = []
    features_patch_for(spec_of(DRAGON), "make B a knight",
                       chat=lambda m, s: seen.append(m) or json.dumps({"ops": [{"op": "change", "id": "B", "text": "k"}]}))
    assert "B (element): one samurai sitting on the dragon's back" in seen[0][1]["content"]


# -- the session ---------------------------------------------------------------------------------

class Queue:
    def __init__(self):
        self.jobs = []

    def submit(self, project, node, graph, priority=1):
        self.jobs.append(graph)
        return type("J", (), {"id": f"j{len(self.jobs)}"})()

    def cancel(self, job):
        return True


def scripted(*answers):
    answers = list(answers)
    return lambda spec, said: features_patch_for(spec, said, chat=lambda m, s: json.dumps(answers.pop(0)))


@pytest.fixture
def session(tmp_path):
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a single dragon"},
                              {"op": "add", "kind": "element", "text": "one samurai on its back"}]},
                     {"ops": [{"op": "change", "id": "A", "text": "a single red dragon"}]})
    s = ArtSession(Store(tmp_path / "p"), Queue(), "adam", patch=patch, rng=random.Random(1))
    s.speaker = "Adam"
    return s


def test_session_draws_and_changes_by_letter(session):
    assert session.handle("draw a dragon with a samurai on its back").kind == "new"
    assert session.project.info["title"] == "a single dragon"
    reply = session.handle("make the dragon red")
    assert reply.spoken == "Changed A." and session.last_changed == ["A"]
    prompt = session.queue.jobs[-1]["pos"]["inputs"]["text"]
    assert prompt == "A single red dragon. One samurai on its back."
    node = session.project.nodes["002"]
    assert node.meta["prompt"] == prompt and node.meta["by"] == "Adam" and node.meta["model"] == "klein9"
    assert node.meta["seed"] == node.spec["seed"] == session.project.nodes["001"].spec["seed"]
    assert session.handle("what prompt made this").spoken == prompt


def test_kid_rules_steer_features(tmp_path):
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a knight on a horse"}]})
    s = ArtSession(Store(tmp_path), Queue(), "guest", rules=policy.load("kid-safe"), patch=patch)
    s.handle("draw a knight on a horse")
    feats = F.features_of(s.project.nodes["001"].spec)
    assert [(f.kind, f.text) for f in feats[1:]] == [("element", "everyone fully clothed"),
                                                      ("style", "family-friendly")] or \
        {(f.kind, f.text) for f in feats[1:]} == {("element", "everyone fully clothed"), ("style", "family-friendly")}


def test_kid_rules_refuse_adult_features(tmp_path):
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a fox"},
                              {"op": "add", "kind": "element", "text": "a bloody battlefield"}]})
    s = ArtSession(Store(tmp_path), Queue(), "guest", rules=policy.load("kid-safe"), patch=patch)
    assert s.handle("draw a fox somewhere").spoken == policy.MSG_KID_DENIED and s.queue.jobs == []


# -- saving and the window state ------------------------------------------------------------------

def test_saved_pictures_go_to_the_askers_folder_with_their_description(tmp_path):
    from tests.test_art_voice import FakeJobs, FakeSwitcher, FakeViewer
    from homeai.art import voice as AV
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a dragon"}]},
                     {"ops": [{"op": "add", "kind": "subject", "text": "a cat"}]})

    def build(owner, notify):
        jobs = FakeJobs(notify)
        return ArtSession(Store(tmp_path / "p"), jobs, owner, patch=patch, rng=random.Random(1)), jobs

    published = []
    art = AV.ArtVoice(FakeSwitcher(), build,
                      lambda t, d: published.append((t, d)), viewer=FakeViewer(), pictures=tmp_path / "Pictures",
                      background=lambda fn: fn())
    art._start_idle_watch = lambda: None
    art.handle("draw a dragon", "Adam")
    project = art.session.project
    image = tmp_path / "out.png"
    image.write_bytes(b"png")
    project.finish("001", image)
    art._on_job({"state": "done", "project": project.id, "node": "001", "image": str(project.image("001"))})
    saved = tmp_path / "Pictures" / "adam" / AV.export_name(project.id, "001")
    sidecar = json.loads(saved.with_suffix(".json").read_text())
    assert saved.read_bytes() == b"png" and sidecar["by"] == "adam" and sidecar["prompt"] == "A dragon."
    assert sidecar["description"]["features"] == [{"id": "A", "kind": "subject", "text": "a dragon"}]
    assert sidecar["seed"] == project.nodes["001"].spec["seed"] and sidecar["model"] == "klein9"
    state = published[-1][1]
    assert state["features"] == [{"id": "A", "kind": "subject", "text": "a dragon"}] and state["changed"] == ["A"]
    # an unknown voice's picture goes to guest/
    art.handle("new picture", None)
    art.handle("draw a cat", None)
    project = art.session.project
    project.finish("001", image)
    art._on_job({"state": "done", "project": project.id, "node": "001", "image": str(project.image("001"))})
    assert (tmp_path / "Pictures" / "guest" / AV.export_name(project.id, "001")).exists()


@pytest.mark.parametrize("person, folder", [("Adam", "adam"), (None, "guest"), ("../etc", "etc"), ("Guest", "guest"),
                                            ("", "guest")])
def test_folder_names_are_safe(person, folder):
    from homeai.art.voice import folder_name
    assert folder_name(person) == folder


def test_feature_rows_for_the_window():
    state = {"features": [{"id": "A", "kind": "subject", "text": "a single red dragon with one head"},
                          {"id": "B", "kind": "element", "text": "a rider"}], "changed": ["A"]}
    rows = V.feature_rows(state, 20)
    assert rows[0] == ("A", "subject", True) and ("", "a single red dragon", True) in rows
    assert ("B", "element", False) in rows and ("", "a rider", False) in rows
    assert V.feature_rows({}) == []
    assert V.wrap("one two three", 7) == ["one two", "three"]
