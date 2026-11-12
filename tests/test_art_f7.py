"""Art mode F7 (ART_FEATURES_PLAN): show the last N, compare, save by name, a 2x copy, a JSON beside every picture.

Adam's decisions (2026-11-12): "last N" is the session's last N pictures (not the lineage); the upscale is 2x
and the original stays; every saved picture has a JSON with prompt, seed and model.
"""
from __future__ import annotations

import json
import random

import pytest

from homeai.art import features as F
from homeai.art import grammar, session as S, viewer as V, voice as AV, workflows
from homeai.art.store import Store
from tests.test_art_session import FakeQueue, fake_patch
from tests.test_art_voice import art, finished, last_art  # noqa: F401 - fixture


@pytest.fixture
def session(tmp_path):
    s = S.ArtSession(Store(tmp_path / "p"), FakeQueue(), "adam", patch=fake_patch, rng=random.Random(1))
    s.input_dir = tmp_path / "input"
    return s


def finish(session, tmp_path, node_id=None, data=b"png"):
    """Finish the current (or given) picture as the queue does: through its own copy of the project, so the
    session's in-memory copy still says pending (the live run's bug, 2026-11-12)."""
    project = session.project
    node_id = node_id or project.current
    image = tmp_path / f"out-{project.id}-{node_id}.png"
    image.write_bytes(data)
    session.store.open(project.id).finish(node_id, image)
    job = next(j["id"] for j in session.queue.jobs if j["project"] == project.id and j["node"] == node_id)
    assert session.on_event({"state": "done", "project": project.id, "node": node_id, "job": job,
                             "image": str(project.image(node_id))}) is None
    return node_id


def three_pictures(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    session.handle("make it blue")
    finish(session, tmp_path)
    session.handle("again")
    finish(session, tmp_path)


# -- grammar ---------------------------------------------------------------------------

@pytest.mark.parametrize("said, n", [("show me the last four", 4), ("show the last 3 pictures", 3),
                                     ("show me the last eight images", 8), ("show me the last 2 ones", 2)])
def test_show_last_is_heard(said, n):
    assert grammar.parse(said) == grammar.Intent("show_last", n=n)


@pytest.mark.parametrize("said, picks", [("compare two and four", (2, 4)), ("compare the first and the third", (1, 3)),
                                         ("show 2 and 4 side by side", (2, 4)), ("compare 5 with 8", (5, 8)),
                                         ("compare number one and number two", (1, 2))])
def test_compare_two_numbers_is_heard(said, picks):
    assert grammar.parse(said) == grammar.Intent("compare", picks=picks)


@pytest.mark.parametrize("said", ["compare", "compare it", "side by side", "show them side by side",
                                  "compare it with the one before", "compare it to before"])
def test_compare_with_before_is_heard(said):
    assert grammar.parse(said) == grammar.Intent("compare")


def test_show_two_and_four_without_side_by_side_is_not_a_comparison():
    assert (grammar.parse("show me two and four") or grammar.Intent("none")).kind != "compare"


@pytest.mark.parametrize("said", ["upscale it", "upscale", "print size", "make it print size",
                                  "higher resolution", "make it higher resolution", "a print version"])
def test_upscale_is_heard(said):
    assert grammar.parse(said) == grammar.Intent("upscale")


def test_make_it_bigger_stays_a_change_to_the_picture():
    assert grammar.parse("make it bigger").kind == "edit"


@pytest.mark.parametrize("said, n", [("number five", 5), ("the fifth one", 5), ("option 8", 8), ("eight", None), ("option 9", None),
                                     ("the eighth one", 8), ("pick seven", None)])
def test_picking_goes_up_to_eight(said, n):
    intent = grammar.parse(said)
    if n is None:
        assert intent is None or intent.kind != "select" or intent.n <= 8
    else:
        assert intent == grammar.Intent("select", n=n)


def test_picking_nine_is_not_a_pick():
    for said in ("number nine", "option 9"):
        assert (grammar.parse(said) or grammar.Intent("none")).kind != "select"


# -- show the last N ---------------------------------------------------------------------

def test_show_last_puts_the_last_pictures_on_a_numbered_sheet(session, tmp_path):
    three_pictures(session, tmp_path)
    reply = session.handle("show me the last two")
    assert reply.spoken == "Here are the last 2. Say a number to pick one."
    assert [(e["node"], e["n"]) for e in session.sheet] == [("002", 1), ("003", 2)] and not session.comparing


def test_show_last_is_capped_at_eight(session, tmp_path):
    for _ in range(10):
        session.handle("draw a red fox") if session.project is None else session.handle("again")
        finish(session, tmp_path)
    session._on_show_last(grammar.Intent("show_last", n=50), "")
    assert len(session.sheet) == grammar.MAX_SHOWN == 8


def test_show_last_reaches_back_into_earlier_pictures_of_the_session(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    fox = session.project.id
    session.handle("draw a grey cat")
    finish(session, tmp_path)
    session.handle("show me the last two")
    assert [e["project"] for e in session.sheet] == [fox, session.project.id]
    assert session.handle("number one").spoken == "That one."
    assert session.project.id == fox and session.project.current == "001" and session.sheet == []


def test_a_picture_from_an_earlier_project_that_finishes_late_still_counts(session, tmp_path):
    session.handle("draw a red fox")
    fox = session.project
    session.handle("draw a grey cat")
    fox.finish("001", tmp_path.joinpath("x.png") if tmp_path.joinpath("x.png").write_bytes(b"p") else None)
    assert session.on_event({"state": "done", "project": fox.id, "node": "001", "job": "j1", "image": "x"}) is None
    assert (fox.id, "001") in session.seen
    session.on_event({"state": "done", "project": fox.id, "node": "001", "job": "not-ours", "image": "x"})
    assert session.seen.count((fox.id, "001")) == 1


def test_show_last_skips_failed_pictures_bigger_copies_and_repeats(session, tmp_path):
    three_pictures(session, tmp_path)
    session.store.open(session.project.id).fail("002", "checked: 2 heads")
    session.seen.append((session.project.id, "003"))
    session.handle("upscale it")
    finish(session, tmp_path, "004")
    session.handle("show me the last five")
    assert [e["node"] for e in session.sheet] == ["001", "003"]


def test_show_last_with_nothing_drawn(session):
    assert session.handle("show me the last four").spoken == S.MSG_NOTHING_YET


def test_a_new_picture_clears_the_sheet(session, tmp_path):
    three_pictures(session, tmp_path)
    session.handle("show me the last two")
    session.handle("again")
    assert session.sheet == [] and not session.comparing


# -- compare ----------------------------------------------------------------------------

def test_compare_alone_is_before_and_now(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    session.handle("make it blue")
    finish(session, tmp_path)
    reply = session.handle("compare")
    assert reply.spoken == "Before on the left, now on the right." and session.comparing
    assert [(e["node"], e["n"]) for e in session.sheet] == [("001", 1), ("002", 2)]


def test_compare_skips_a_failed_step_before(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    session.handle("make it blue")
    finish(session, tmp_path)
    session.handle("make it blue")
    finish(session, tmp_path)
    session.store.open(session.project.id).fail("002", "x")
    session.handle("compare")
    assert [e["node"] for e in session.sheet] == ["001", "003"]


def test_compare_with_nothing_before(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    assert session.handle("compare").spoken == S.MSG_NOTHING_BEFORE and session.sheet == []


def test_compare_two_from_the_sheet_keeps_their_numbers(session, tmp_path):
    three_pictures(session, tmp_path)
    session.handle("show me the last three")
    assert session.handle("compare two and three").spoken == "Side by side: 2 and 3."
    assert [(e["node"], e["n"]) for e in session.sheet] == [("002", 2), ("003", 3)] and session.comparing
    session.handle("number two")                     # the number on screen, not the place in the pair
    assert session.project.current == "002"


def test_compare_two_options(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    session.handle("give me four")
    session.handle("compare two and four")
    assert [e["node"] for e in session.sheet] == [session.grid[1], session.grid[3]]


def test_compare_a_number_that_is_not_there(session, tmp_path):
    three_pictures(session, tmp_path)
    session.handle("show me the last two")
    assert session.handle("compare one and five").spoken == "There's no number 5."
    assert not session.comparing


def test_compare_numbers_with_nothing_numbered(session, tmp_path):
    session.handle("draw a red fox")
    assert session.handle("compare one and two").spoken == S.MSG_NO_GRID


def test_picking_a_number_that_is_not_on_the_sheet(session, tmp_path):
    three_pictures(session, tmp_path)
    session.handle("show me the last two")
    assert session.handle("number five").spoken == "There's no number 5."
    assert session.handle("the last one").spoken == "That one." and session.project.current == "003"


# -- upscale ----------------------------------------------------------------------------

def test_upscale_queues_a_2x_copy_and_keeps_the_picture_current(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    reply = session.handle("upscale it")
    assert reply.spoken == "Making a 2 times bigger copy. The original stays." and reply.jobs
    job = session.queue.jobs[-1]
    assert job["graph"] == workflows.upscale(f"art/{session.project.id}-001.png", prefix=f"art/{session.project.id}")
    assert (session.input_dir / "art" / f"{session.project.id}-001.png").read_bytes() == b"png"
    copy = session.project.nodes[job["node"]]
    assert copy.kind == "upscale" and copy.parent == "001" and session.project.current == "001"
    assert copy.meta["upscale_of"] == "001" and copy.meta["scale"] == 2 and copy.meta["size"] == [2048, 2048]
    assert copy.meta["upscaler"] == workflows.UPSCALER and copy.meta["prompt"] and copy.meta["seed"]


def test_upscale_once_per_picture(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    session.handle("upscale it")
    assert session.handle("upscale it").spoken == S.MSG_UPSCALED_ALREADY and len(session.queue.jobs) == 2


def test_upscale_again_after_the_copy_failed(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    session.handle("upscale it")
    session.store.open(session.project.id).fail(session.queue.jobs[-1]["node"], "out of memory")
    assert session.handle("upscale it").kind == "upscale" and len(session.queue.jobs) == 3


def test_upscale_waits_for_the_picture(session):
    session.handle("draw a red fox")
    assert session.handle("upscale it").spoken == S.MSG_STILL_DRAWING and len(session.queue.jobs) == 1


def test_edits_go_on_from_the_picture_not_the_copy(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    session.handle("upscale it")
    session.handle("make it blue")
    assert session.queue.jobs[-1]["graph"]["noise"]["inputs"]["noise_seed"] == session.project.nodes["001"].spec["seed"]
    assert session.project.nodes[session.project.current].parent == "001"


def test_the_upscale_graph():
    g = workflows.upscale("art/x.png", prefix="art/p")
    assert g["load"]["inputs"]["image"] == "art/x.png"
    assert g["model"]["inputs"]["model_name"] == "RealESRGAN_x2plus.pth" and workflows.UPSCALE == 2
    assert g["up"]["inputs"] == {"upscale_model": ["model", 0], "image": ["load", 0]}
    assert g["save"]["inputs"] == {"images": ["up", 0], "filename_prefix": "art/p"}


def test_stage_copies_into_comfyui_input(tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(b"abc")
    assert workflows.stage(src, "p-001.png", tmp_path / "in") == "art/p-001.png"
    assert (tmp_path / "in" / "art" / "p-001.png").read_bytes() == b"abc"


# -- keep by name (session side) ----------------------------------------------------------

def test_save_by_name_asks_the_saver(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    calls = []
    session.save_as = lambda project, node, name: calls.append((project.id, node.id, name)) or "ember 2"
    assert session.handle("save that as ember").spoken == "Saved as ember 2."
    assert calls == [(session.project.id, "001", "ember")] and session.project.info["pins"] == ["001"]


@pytest.mark.parametrize("error, said", [(ValueError("x"), S.MSG_BAD_NAME), (OSError("x"), S.MSG_SAVE_FAILED)])
def test_save_by_name_failures_are_said(session, tmp_path, error, said):
    session.handle("draw a red fox")
    finish(session, tmp_path)

    def boom(*a):
        raise error
    session.save_as = boom
    assert session.handle("save that as ember").spoken == said


def test_save_by_name_without_a_saver(session, tmp_path):
    session.handle("draw a red fox")
    finish(session, tmp_path)
    assert session.handle("save that as ember").spoken == S.MSG_NO_SAVE


def test_save_by_name_waits_for_the_picture(session):
    session.handle("draw a red fox")
    session.save_as = lambda *a: pytest.fail("saved a picture that isn't drawn")
    assert session.handle("save that as ember").spoken == S.MSG_STILL_DRAWING


def test_keep_without_a_name_still_pins(session):
    session.handle("draw a red fox")
    assert session.handle("keep that").spoken == "Kept." and session.project.info["pins"] == ["001"]


# -- the deck request (Adam 2026-11-12) ------------------------------------------------------

def test_a_new_picture_that_fails_says_so_without_mentioning_a_picture(session):
    session._patch = lambda spec, text: type("R", (), {"ok": False})()
    assert session.handle_free("let's make assembly instructions for a deck").spoken == S.MSG_NOT_CAUGHT_NEW


def test_a_new_picture_of_only_a_setting_gets_a_subject():
    ops = [{"op": "add", "kind": "setting", "text": "an 8 foot by 8 foot deck on the ground for a tent"}]
    out = F.subject_first([], ops)
    assert out[0]["kind"] == "subject" and ops[0]["kind"] == "setting"
    features, _ = F.apply_ops([], out, set())
    assert features[0].kind == "subject"


def test_subject_first_skips_a_placeholder_and_leaves_edits_and_subjects_alone():
    placeholder = [{"op": "add", "kind": "setting", "text": "none"}, {"op": "add", "kind": "style", "text": "ink"}]
    assert [o["kind"] for o in F.subject_first([], placeholder)] == ["setting", "subject"]
    has = [{"op": "add", "kind": "setting", "text": "a cliff"}, {"op": "add", "kind": "subject", "text": "a fox"}]
    assert F.subject_first([], has) == has
    before = [F.Feature("A", "subject", "a fox")]
    edit = [{"op": "add", "kind": "setting", "text": "a cliff"}]
    assert F.subject_first(before, edit) == edit


def test_the_patcher_promotes_the_subject_for_a_new_picture():
    from homeai.art.patcher import features_patch_for
    answer = json.dumps({"about_picture": True, "ops": [{"op": "add", "id": "", "kind": "setting",
                                                         "text": "a wooden deck on the ground"}]})
    result = features_patch_for({}, "let's make a deck on the ground", chat=lambda m, s: answer)
    assert result.ok and F.features_of(result.spec)[0].kind == "subject"


# -- voice: saving, JSON always, the window state ---------------------------------------------

@pytest.mark.parametrize("name, slug", [("Ember", "ember"), ("Ember's Lair", "ember-s-lair"), ("  my  fox ", "my-fox"),
                                        ("../../etc/passwd", "etc-passwd"), ("!!!", ""), ("a" * 60, "a" * 40)])
def test_name_slug(name, slug):
    assert AV.name_slug(name) == slug


def test_free_name_counts_up(tmp_path):
    assert AV.free_name(tmp_path, "ember") == ("ember", 1)
    (tmp_path / "ember.png").write_bytes(b"x")
    assert AV.free_name(tmp_path, "ember") == ("ember-2", 2)
    (tmp_path / "ember-2.json").write_text("{}")
    assert AV.free_name(tmp_path, "ember") == ("ember-3", 3)


def test_saving_by_name_writes_the_picture_and_its_json(art, tmp_path):
    art.handle("draw a red fox", "Adam")
    finished(art, tmp_path)
    assert art.handle("save that as Ember", "Adam") == "Saved as ember."
    folder = tmp_path / "Pictures" / "adam"
    assert (folder / "ember.png").read_bytes() == b"png"
    data = json.loads((folder / "ember.json").read_text())
    assert data["name"] == "ember" and data["prompt"] and data["seed"] and data["model"] == "klein9"
    assert art.handle("save that as ember", "Adam") == "Saved as ember 2."
    assert (folder / "ember-2.png").exists() and (folder / "ember.png").read_bytes() == b"png"


def test_a_kid_s_named_picture_goes_in_her_folder(art, tmp_path):
    art.handle("draw a red fox", "Ani")
    finished(art, tmp_path)
    art.handle("save that as ember", "Ani")
    assert (tmp_path / "Pictures" / "ani" / "ember.png").exists()
    assert not (tmp_path / "Pictures" / "adam").exists()


def test_a_name_with_no_letters_is_refused(art, tmp_path):
    art.handle("draw a red fox", "Adam")
    finished(art, tmp_path)
    assert art.handle("save that as ---", "Adam") == S.MSG_BAD_NAME


def test_every_saved_picture_has_its_json_even_from_an_earlier_project(art, tmp_path):
    art.handle("draw a red fox", "Adam")
    fox = art.session.project
    art.handle("draw a grey cat", "Adam")
    image = tmp_path / "fox.png"
    image.write_bytes(b"png")
    fox.finish("001", image)
    art._on_job({"state": "done", "project": fox.id, "node": "001", "job": "j1", "image": str(fox.image("001"))})
    data = json.loads((tmp_path / "Pictures" / "adam" / AV.export_name(fox.id, "001")).with_suffix(".json").read_text())
    assert data["prompt"] and data["seed"] and data["model"] == "klein9" and data["node"] == "001"


def test_a_picture_with_no_project_still_gets_a_json(art, tmp_path):
    art.handle("draw a red fox", "Adam")
    image = tmp_path / "x.png"
    image.write_bytes(b"png")
    art._export({"state": "done", "project": "gone", "node": "009", "image": str(image)})
    target = tmp_path / "Pictures" / "guest" / AV.export_name("gone", "009")
    assert target.exists() and json.loads(target.with_suffix(".json").read_text())["project"] == "gone"


def test_the_bigger_copy_is_saved_beside_the_original_and_not_checked(art, tmp_path):
    asked = []
    art.checker = lambda image, question: asked.append(question) or "1"
    art.handle("draw a red fox", "Adam")
    project = finished(art, tmp_path)
    art.session.input_dir = tmp_path / "input"
    art.handle("upscale it", "Adam")
    copy = art.queue.submitted[-1]["node"]
    # A subject the checker would count, so a check on the copy would show up in ``asked``.
    project.nodes[copy].spec = {"features": F.dump([F.Feature("A", "subject", "a knight and a wizard")])}
    project.finish(copy, tmp_path / "out.png")
    art._on_job({"state": "done", "project": project.id, "node": copy, "image": str(project.image(copy))})
    target = tmp_path / "Pictures" / "adam" / f"{project.id}-001-2x.png"
    data = json.loads(target.with_suffix(".json").read_text())
    assert target.exists() and data["kind"] == "upscale" and data["upscale_of"] == "001" and data["scale"] == 2
    assert data["upscaler"] == workflows.UPSCALER and data["prompt"] and data["seed"]
    assert asked == [] and last_art(art)["status"] == "The 2 times bigger copy is saved."
    assert last_art(art)["current"]["node"] == "001"


def test_the_window_gets_the_sheet_and_whether_it_compares(art, tmp_path):
    art.handle("draw a red fox", "Adam")
    finished(art, tmp_path)
    art.handle("make it blue", "Adam")
    project = art.session.project
    project.finish("002", tmp_path / "out.png")
    art._on_job({"state": "done", "project": project.id, "node": "002", "image": str(project.image("002"))})
    art.handle("compare", "Adam")
    state = last_art(art)
    assert state["compare"] is True and [o["n"] for o in state["options"]] == [1, 2]
    assert [o["node"] for o in state["options"]] == ["001", "002"] and state["options"][0]["image"]
    assert state["options"][0]["project"] == project.id and state["current"]["project"] == project.id


def test_a_missing_picture_on_the_sheet_is_an_empty_box(art, tmp_path):
    art.handle("draw a red fox", "Adam")
    entry = AV.ArtVoice._sheet_entry(art.session, {"project": "gone", "node": "001", "n": 1})
    assert entry["image"] is None and entry["n"] == 1


# -- the window ------------------------------------------------------------------------------

def test_compare_layout_is_two_halves_side_by_side():
    boxes = V.compare_layout(1000, 1000)
    (lx, ly, lw, lh), (rx, ry, rw, rh) = boxes["pair"]
    assert ly == ry and lw == rw and lh == rh and lx + lw < rx and rx + rw <= 1000 - V.PAD
    assert lw >= 470 and boxes["status"][1] > ly + lh


def test_the_window_compares_only_two_pictures_when_asked():
    two = [{"n": 1}, {"n": 2}]
    assert V.comparing({"compare": True, "options": two}) == two
    assert V.comparing({"compare": False, "options": two}) == []
    assert V.comparing({"compare": True, "options": two + [{"n": 3}]}) == []


def test_current_is_matched_by_project_too():
    current = {"node": "001", "project": "fox"}
    assert V.is_current({"node": "001", "project": "fox"}, current)
    assert not V.is_current({"node": "001", "project": "cat"}, current)
    assert V.is_current({"node": "001"}, current) and not V.is_current({"node": "002"}, current)
    assert not V.is_current({"node": "001"}, {})
