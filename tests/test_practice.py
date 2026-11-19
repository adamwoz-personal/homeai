"""Practice nights (tools/art/practice.py, ART_FEATURES_PLAN F9 step 2): what's drawn, when a wording counts
as working, when it's rewritten, what reaches Claude's review queue, and when the night gives way."""
from __future__ import annotations

import json
import random
from datetime import datetime

import pytest

from homeai.art import review as R
from tools.art import practice as P

DRAGONISH = P.Topic("D&D", "creatures", "an owlbear")
CHIMERA = P.Topic("fantasy", "creatures", "a chimera")


def test_the_shipped_topics_cover_adam_s_settings():
    found = P.topics()
    assert {t.setting for t in found} >= {"D&D", "fantasy", "steampunk", "cyberpunk", "anime"}
    assert {t.kind for t in found} == {"creatures", "weapons", "gear", "places"}
    assert len({t.key for t in found}) == len(found)
    assert [t.setting for t in P.topics(settings=["anime"])] == ["anime"] * len(P.topics(settings=["anime"]))


def test_never_practised_first_then_the_longest_ago():
    a, b, c = (P.Topic("s", "k", t) for t in "abc")
    history = {"s/a": {"when": "2026-11-18T01:00:00"}, "s/b": {"when": "2026-11-17T01:00:00"}}
    assert P.order([a, b, c], history, random.Random(1)) == [c, b, a]


def chat(scene="An owlbear in a dark forest.", questions=("Does the creature have an owl's head?",),
         rewrites=("an owlbear: a bear's body with an owl's head and feathered face",)):
    rewrites, said = list(rewrites), []

    def answer(messages, schema):
        said.append(messages)
        key = schema["required"][0]
        if key == "scene":
            if isinstance(scene, Exception):
                raise scene
            return json.dumps({"scene": scene})
        if key == "questions":
            return json.dumps({"questions": list(questions)})
        return json.dumps({"wording": rewrites.pop(0) if rewrites else ""})
    answer.said = said
    return answer


@pytest.mark.parametrize("scene, used", [
    ("An owlbear in a dark forest.", True),
    ("A bear with feathers in a forest.", False),             # doesn't name the term
    ("An owlbear " + "very " * 40 + "big.", False),           # too long
    (ValueError("bad json"), False),
])
def test_the_scene_names_the_term_or_a_plain_one_is_used(scene, used):
    text = P.scene(DRAGONISH, chat(scene), lambda t: True)
    assert text == (scene if used else "An owlbear, D&D style.")


def test_a_scene_the_kid_safe_rules_refuse_is_not_used():
    assert P.scene(DRAGONISH, chat(), lambda t: False) == "An owlbear, D&D style."


def test_the_glossary_s_questions_first_else_the_model_s_own():
    assert [c.kind for c in P.checks_for("A knight fighting a chimera.", chat())] == ["goat", "snake", "horns"]
    own = P.checks_for("An owlbear in a dark forest.", chat())
    assert [c.kind for c in own] == ["asked"] and "owl's head" in own[0].question


class Rig:
    """Fake klein and checker: ``verdicts`` says per picture whether its one question gets the right answer."""

    def __init__(self, tmp_path, verdicts, stop_after=None, allowed=lambda t: True, chat_=None):
        self.verdicts, self.drawn, self.n = list(verdicts), [], 0
        self.review = R.Review(tmp_path / "review", rng=lambda: 0.99)
        self.chat = chat_ or chat(rewrites=[f"owlbear words {n}" for n in range(1, 6)])

        def draw(prompt, seed):
            self.drawn.append(prompt)
            image = tmp_path / f"d{len(self.drawn)}.png"
            image.write_bytes(b"png")
            return image

        def ask(image, question):
            if question.startswith("Describe"):
                return "a bear"
            right = self.verdicts.pop(0)
            if question.startswith("How many"):
                return "1" if right else "2"
            return "Yes" if right else "No"

        def stop():
            self.n += 1
            return "someone spoke to Jarvis" if stop_after is not None and self.n > stop_after else None
        self.practice = P.Practice(draw, ask, self.chat, allowed, self.review, tmp_path / "out", stop,
                                   seeds=4, rewrites=3, rng=random.Random(1))


def test_wording_that_already_works_is_left_alone(tmp_path):
    rig = Rig(tmp_path, [True, True, True, False])
    rig.practice.chat = chat(scene="A knight wielding a sword in an owlbear's den.")
    row = rig.practice.term(DRAGONISH)
    assert row["checks"] == ["blades"]
    assert row["outcome"] == "fine as it is" and [r["passed"] for r in row["rounds"]] == [3]
    assert rig.review.waiting() == []


def test_a_pass_on_the_model_s_own_questions_alone_always_goes_to_review(tmp_path):
    rig = Rig(tmp_path, [True, True, True, False])
    row = rig.practice.term(DRAGONISH)
    assert row["outcome"] == "unmeasured pass" and [r["passed"] for r in row["rounds"]] == [3]
    [folder] = rig.review.waiting()
    assert R.read(folder)["why"] == "passed only the voice model's own questions"


def test_a_rewrite_that_passes_3_of_4_is_a_candidate_for_claude(tmp_path):
    rig = Rig(tmp_path, [False] * 4 + [True, False, False, False] + [True, True, True, False])
    row = rig.practice.term(DRAGONISH)
    assert row["outcome"] == "candidate" and [r["passed"] for r in row["rounds"]] == [0, 1, 3]
    assert rig.drawn[-1] == "owlbear words 2" and len(set(rig.drawn[:4])) == 1
    [folder] = rig.review.waiting()
    item = R.read(folder)
    assert item["reason"] == "candidate" and item["source"] == "practice"
    assert item["candidate"] == {"term": "an owlbear", "setting": "D&D", "wording": "owlbear words 2",
                                 "before": "0/4", "after": "3/4"}
    assert len(item["pictures"]) == 4 and item["prompt"] == "owlbear words 2"


def test_at_most_three_rewrites_then_the_best_try_goes_to_review(tmp_path):
    rig = Rig(tmp_path, [False] * 4 + [True, True, False, False] + [False] * 8)
    row = rig.practice.term(DRAGONISH)
    assert row["outcome"] == "not fixed" and len(row["rounds"]) == 4 and len(rig.drawn) == 16
    item = R.read(rig.review.waiting()[0])
    assert item["reason"] == "best" and item["candidate"]["wording"] == "owlbear words 1"


def test_a_rewrite_the_rules_refuse_ends_the_rewrites(tmp_path):
    rig = Rig(tmp_path, [False] * 4, allowed=lambda t: "words" not in t)
    row = rig.practice.term(DRAGONISH)
    assert row["outcome"] == "not fixed" and len(row["rounds"]) == 1


def test_the_rewriter_hears_what_went_wrong_and_what_the_picture_shows(tmp_path):
    rig = Rig(tmp_path, [False] * 4 + [True] * 4)
    rig.practice.term(DRAGONISH)
    asked = rig.chat.said[-1][1]["content"]
    assert "owl's head" in asked and "What the picture shows: a bear" in asked


def test_it_stops_before_the_next_picture_once_someone_speaks(tmp_path):
    rig = Rig(tmp_path, [True] * 8, stop_after=2)
    with pytest.raises(P.Yielded, match="spoke"):
        rig.practice.term(DRAGONISH)
    assert len(rig.drawn) == 2


def test_nothing_to_check_draws_nothing(tmp_path):
    rig = Rig(tmp_path, [], chat_=chat(questions=()))
    assert rig.practice.term(DRAGONISH)["outcome"] == "nothing to check" and rig.drawn == []


def test_passes_are_sampled_for_review_too(tmp_path):
    rig = Rig(tmp_path, [True] * 4)
    rig.review.passed = lambda image, **kw: rig.drawn.append(("sampled", kw["source"]))
    rig.practice.term(DRAGONISH)
    assert rig.drawn.count(("sampled", "practice")) == 4


# -- the night --------------------------------------------------------------------------------------------------

def test_the_night_ends_at_the_hour_or_when_jarvis_hears_someone():
    start = datetime(2026, 11, 19, 1, 0)
    log = {"text": "art: checked x"}
    night = P.Night(start, datetime(2026, 11, 19, 6, 30), read=lambda since: log["text"],
                    now=lambda: datetime(2026, 11, 19, 3, 0))
    assert night() is None
    log["text"] += "\nheard: what's the weather"
    assert night() == "someone spoke to Jarvis"
    late = P.Night(start, datetime(2026, 11, 19, 6, 30), read=lambda since: "", now=lambda: datetime(2026, 11, 19, 6, 30))
    assert late() == "the night is over"


def test_the_journal_is_read_from_the_start_of_the_night():
    seen = []
    P.Night(datetime(2026, 11, 19, 1, 2, 3), datetime(2026, 11, 20), read=lambda since: seen.append(since) or "",
            now=lambda: datetime(2026, 11, 19, 2))()
    assert seen == ["2026-11-19 01:02:03"]


@pytest.mark.parametrize("scope, user", [("system", False), ("user", True)])
def test_jarvis_s_log_is_read_where_his_service_runs(scope, user):
    seen = []

    class Done:
        stdout = "heard: hi"
    assert P.journal("-300s", run=lambda argv, **kw: seen.append(argv) or Done(), scope=lambda: scope) == "heard: hi"
    assert ("--user" in seen[0]) is user and seen[0][seen[0].index("-u") + 1] == "homeai"


def test_art_mode_entered_for_someone_is_left_on():
    night = P.Night(datetime(2026, 11, 19), datetime(2026, 11, 20), read=lambda since: "art: entered for Adam")
    assert night.art_taken()
    assert not P.Night(datetime(2026, 11, 19), datetime(2026, 11, 20), read=lambda since: "heard: hi").art_taken()


class Switcher:
    def __init__(self):
        self.calls = []

    def voice(self):
        self.calls.append("voice")


@pytest.mark.parametrize("started, log, back", [
    (True, "heard: hi", True), (True, "art: entered for Adam", False), (False, "", False)])
def test_voice_mode_comes_back_only_if_practice_started_art_and_nobody_took_it(started, log, back):
    switcher = Switcher()
    P.hand_back(switcher, started, P.Night(datetime(2026, 11, 19), datetime(2026, 11, 20), read=lambda since: log))
    assert switcher.calls == (["voice"] if back else [])


@pytest.mark.parametrize("now, end", [
    (datetime(2026, 11, 19, 23, 0), datetime(2026, 11, 20, 6, 30)),
    (datetime(2026, 11, 19, 1, 0), datetime(2026, 11, 19, 6, 30)),
])
def test_until_is_the_next_time_it_comes_round(now, end):
    assert P.until_time("06:30", now) == end


def test_dry_run_lists_topics_without_touching_the_gpu(capsys):
    assert P.main(["--dry-run", "--setting", "steampunk", "--terms", "3"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
