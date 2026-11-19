"""The review queue (homeai/art/review.py, ART_FEATURES_PLAN F9 step 1): what's kept for Claude, from live art
mode and practice, and the report that turns Claude's labels into checker labels; plus the glossary as data."""
from __future__ import annotations

import json
import logging

import pytest

from homeai.art import checker as CK
from homeai.art import review as R
from homeai.art import wording as W
from tests.test_art_checker import art, finish  # noqa: F401 - fixture
from tests.test_daemon import assistant  # noqa: F401 - fixture
from tools.art import report_review


def png(path):
    path.write_bytes(b"png")
    return path


def test_an_item_keeps_the_picture_and_what_was_asked_said_and_found(tmp_path):
    review = R.Review(tmp_path / "q", clock=lambda: 0)
    check = CK.Check("goat", "Is there a goat?", True, "goat", "B", "no goat")
    folder = review.add(png(tmp_path / "a.png"), "best", prompt="a chimera, a lion", issues=["no goat"],
                        checks=[check], said="", extra={"node": "003"}, pictures=[png(tmp_path / "b.png")])
    item = R.read(folder)
    assert (folder / "picture.png").read_bytes() == b"png" and item["pictures"] == ["picture.png", "picture-2.png"]
    assert item["reason"] == "best" and item["prompt"] == "a chimera, a lion" and item["issues"] == ["no goat"]
    assert item["checks"] == [{"kind": "goat", "question": "Is there a goat?", "expect": True, "feature": "B"}]
    assert item["node"] == "003" and review.waiting() == [folder]
    again = review.add(png(tmp_path / "a.png"), "best")
    assert again != folder and len(review.waiting()) == 2          # same second: a new folder, not overwritten


def test_an_unknown_reason_is_a_bug(tmp_path):
    with pytest.raises(ValueError):
        R.Review(tmp_path).add(png(tmp_path / "a.png"), "whatever")


def test_a_full_queue_keeps_nothing_more(tmp_path):
    review = R.Review(tmp_path / "q", limit=2)
    assert review.add(png(tmp_path / "a.png"), "best") and review.add(tmp_path / "a.png", "best")
    assert review.add(tmp_path / "a.png", "best") is None and len(review.waiting()) == 2


def test_a_picture_that_cant_be_copied_keeps_nothing_and_doesnt_raise(tmp_path):
    assert R.Review(tmp_path / "q").add(tmp_path / "missing.png", "best") is None


@pytest.mark.parametrize("roll, kept", [(0.0, True), (0.099, True), (0.1, False), (0.9, False)])
def test_one_pass_in_ten_is_kept(tmp_path, roll, kept):
    review = R.Review(tmp_path / "q", rng=lambda: roll)
    assert bool(review.passed(png(tmp_path / "a.png"))) is kept


def test_done_moves_the_item_out_of_the_waiting_list_with_its_label(tmp_path):
    review = R.Review(tmp_path / "q")
    folder = review.add(png(tmp_path / "a.png"), "complaint", said="it has two heads")
    done = review.done(folder, {"right": False})
    assert review.waiting() == [] and R.read(done)["label"] == {"right": False}
    assert done.parent == tmp_path / "q" / "done"


# -- from art mode -------------------------------------------------------------------------------------------

@pytest.fixture
def reviewed(art, tmp_path):  # noqa: F811
    art.review = art.session.review = R.Review(tmp_path / "review", rng=lambda: 0.0)
    return art


def test_a_picture_shown_wrong_is_kept_for_review(reviewed, monkeypatch):
    monkeypatch.setattr(CK, "MAX_REDRAWS", 0)
    reviewed.answers += ["2", "1"]                      # two heads
    finish(reviewed, "001")
    assert reviewed.status.startswith("Best I could do")
    item = R.read(reviewed.review.waiting()[0])
    assert item["reason"] == "best" and item["issues"] == ["2 heads"] and item["prompt"]
    assert [c["kind"] for c in item["checks"]] == ["heads", "riders"]


def test_a_picture_being_redrawn_is_not_kept(reviewed):
    reviewed.answers += ["2", "1"]
    finish(reviewed, "001")
    assert reviewed.status.startswith("Redrawing") and reviewed.review.waiting() == []


def test_a_right_picture_is_sampled(reviewed):
    finish(reviewed, "001")
    assert R.read(reviewed.review.waiting()[0])["reason"] == "pass-sample"


def test_a_review_that_fails_still_shows_the_picture(reviewed):
    def broken(*a, **k):
        raise RuntimeError("disk")
    reviewed.review.passed = broken
    finish(reviewed, "001")
    assert reviewed.status == "Done."


def test_a_complaint_keeps_the_picture_and_the_words(reviewed):
    finish(reviewed, "001")
    reviewed.review.waiting()[0].rename(reviewed.tmp / "moved")          # drop the pass sample
    reviewed.handle("it has two heads", "Adam")
    item = R.read(reviewed.review.waiting()[0])
    assert item["reason"] == "complaint" and item["said"] == "it has two heads" and item["by"] == "Adam"
    assert item["node"] == "001" and [f["text"] for f in item["features"]][0] == "a single dragon"


def test_entering_art_mode_is_logged_for_practice_nights_to_see(art, caplog):  # noqa: F811
    art.leave()
    with caplog.at_level(logging.INFO, logger="homeai.art.voice"):
        art.handle("let's draw", "Adam")
    assert "art: entered for Adam" in caplog.text


def test_the_daemon_keeps_pictures_for_review(assistant):  # noqa: F811
    import dataclasses
    from homeai.config import ArtConfig
    assistant.cfg = dataclasses.replace(assistant.cfg, art=ArtConfig(voice=True, check=True))
    assert isinstance(assistant._build_art().review, R.Review)


# -- the report ------------------------------------------------------------------------------------------------

def test_labels_become_checker_labels_and_move_the_item(tmp_path):
    review = R.Review(tmp_path / "q")
    checks = [CK.Check("goat", "Goat?", True, "goat"), CK.Check("blades", "Blades?", 1, "blades")]
    from PIL import Image
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    review.add(tmp_path / "a.png", "best", checks=checks)
    review.add(tmp_path / "b.png", "pass-sample", checks=checks)
    out, n = report_review.build(review, tmp_path / "rep")
    assert n == 2 and (out / "sheet.png").exists()
    labels = json.loads((out / "labels.json").read_text())
    first = labels["items"][0]
    assert first["n"] == 1 and first["label"] == {"right": None, "parts": {}, "notes": ""}
    first["label"] = {"right": False, "parts": {"goat": False, "blades": 2, "wings": True}, "notes": "no goat"}
    (out / "labels.json").write_text(json.dumps(labels))
    assert report_review.apply(out / "labels.json", review) == (1, 2)     # unlabelled item 2 stays waiting
    assert len(review.waiting()) == 1
    rows = json.loads((out / "vision-labels.json").read_text())["labels"]
    assert [(r["type"], r["question"], r["expect"]) for r in rows] == [("goat", "Goat?", False),
                                                                        ("blades", "Blades?", 2)]
    assert rows[0]["image"].startswith(str(tmp_path / "q" / "done"))
    assert report_review.apply(out / "labels.json", review) == (0, 0)     # applying twice adds nothing
    assert len(json.loads((out / "vision-labels.json").read_text())["labels"]) == 2


# -- the glossary is data (F9 steps 1 and 3) -------------------------------------------------------------------

def glossary(tmp_path, labelled=30, q="goat", pattern=r"\bchimera\b", visual="a lion with a goat's head"):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"min_labelled": 30,
                                "questions": {"goat": {"question": "Goat?", "labelled": labelled}},
                                "terms": [{"name": "chimera", "pattern": pattern, "visual": visual,
                                           "asks": [{"q": q, "expect": True, "wrong": "no goat"}]}]}))
    return path


def test_a_measured_question_is_asked(tmp_path):
    questions, terms = W.load(glossary(tmp_path))
    assert questions == {"goat": "Goat?"} and [a.kind for a in terms[0].asks] == ["goat"]


def test_a_question_measured_on_too_few_pictures_is_not_asked(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="homeai.art.wording"):
        _, terms = W.load(glossary(tmp_path, labelled=29))
    assert terms[0].asks == () and "not asked" in caplog.text
    assert terms[0].visual == "a lion with a goat's head"            # the words still help


def test_an_ask_naming_no_question_is_dropped(tmp_path):
    assert W.load(glossary(tmp_path, q="wings"))[1][0].asks == ()


def test_a_term_that_cant_be_read_is_skipped(tmp_path):
    assert W.load(glossary(tmp_path, pattern="(unclosed"))[1] == ()


def test_an_unreadable_glossary_means_the_words_as_said(tmp_path):
    (tmp_path / "g.json").write_text("{")
    assert W.load(tmp_path / "g.json") == ({}, ())


def test_a_visual_with_an_unknown_blank_leaves_the_words_alone(tmp_path, monkeypatch):
    _, terms = W.load(glossary(tmp_path, visual="a {colour} lion"))
    monkeypatch.setattr(W, "TERMS", terms)
    assert W.expand("fighting a chimera") == ("fighting a chimera", [])


def test_every_question_in_the_shipped_glossary_is_measured_and_asked():
    data = json.loads(W.GLOSSARY.read_text())
    assert data["min_labelled"] >= 30
    named = [a["q"] for t in data["terms"] for a in t["asks"]]
    asked = [a.kind for t in W.TERMS for a in t.asks]
    assert asked == named                       # nothing dropped: a check can't vanish by a data edit
    assert all(data["questions"][q]["labelled"] >= data["min_labelled"] for q in named)
