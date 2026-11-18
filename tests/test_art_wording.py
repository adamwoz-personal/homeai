"""Look, notice, reword (homeai/art/wording.py, ART_FEATURES_PLAN F8): the glossary, learned and rewritten wording,
the voice model's own check questions, and the loop in art mode."""
from __future__ import annotations

import json

import pytest

from homeai.art import checker as CK
from homeai.art import features as F
from homeai.art import wording as W
from homeai.art.patcher import features_patch_for
from tests.test_art_checker import art, exported, feats, finish  # noqa: F401 - fixture
from tests.test_daemon import assistant  # noqa: F401 - fixture

ADAM = feats(("subject", "an anime girl wielding a katana"), ("element", "fighting a chimera"))


# -- the glossary ------------------------------------------------------------------------------------

def test_a_chimera_is_said_as_what_it_looks_like():
    text, asks = W.expand("fighting a chimera")
    assert text.startswith("fighting a chimera, a lion with a golden mane")
    assert "a goat's head, rising from the middle of its back" in text and "a living snake for a tail" in text
    assert [(a.kind, a.expect) for a in asks] == [("goat", True), ("snake", True)]


@pytest.mark.parametrize("said, noun, curve", [
    ("an anime girl wielding a katana", "katana", " curved"),
    ("an anime girl wielding one katana with one blade", "katana", " curved"),     # Adam's own fix, swallowed
    ("a knight holding a sword", "sword", ""),
    ("a samurai with his katana", "katana", " curved")])
def test_one_sword_is_said_as_one_hilt_held_with_both_hands(said, noun, curve):
    text, asks = W.expand(said)
    assert f"holding a single {noun} with both hands on its one hilt, its one long{curve} blade" in text
    assert "with one blade" not in text and text.count(noun) == 1
    assert [(a.kind, a.expect, a.question) for a in asks] == [("blades", 1, W.BLADES_Q)]


@pytest.mark.parametrize("said", ["a girl wielding two katanas", "a sword in a stone", "a chimney", "a dragon"])
def test_what_the_glossary_doesnt_know_is_left_alone(said):
    assert W.expand(said) == (said, [])


def test_features_stay_as_said_and_the_prompt_says_what_they_look_like():
    from homeai.art.session import render_spec
    spec = {"features": F.dump(ADAM), "aspect": "square"}
    prompt = render_spec(spec).text
    assert prompt.startswith("An anime girl holding a single katana with both hands on its one hilt")
    assert "Fighting a chimera, a lion with a golden mane" in prompt
    assert [f.text for f in F.features_of(spec)] == ["an anime girl wielding a katana", "fighting a chimera"]


# -- which words win -----------------------------------------------------------------------------------

def test_own_rewrite_then_learned_then_glossary(tmp_path):
    learned = W.Learned(tmp_path / "wording.json")
    katana = ADAM[0]
    assert W.prompt_text(katana, {}, learned).startswith("an anime girl holding a single katana")
    learned.remember("An anime girl, wielding a katana!", "LEARNED")              # same words, any case and marks
    assert W.prompt_text(katana, {}, learned) == "LEARNED"
    own = {"wording": {"A": {"for": katana.text, "text": "OWN"}}}
    assert W.prompt_text(katana, own, learned) == "OWN"
    stale = {"wording": {"A": {"for": "an anime boy wielding a katana", "text": "OWN"}}}
    assert W.prompt_text(katana, stale, learned) == "LEARNED"                       # the feature changed since


def test_a_broken_learned_file_is_ignored_and_replaced(tmp_path):
    path = tmp_path / "wording.json"
    path.write_text("not json")
    learned = W.Learned(path)
    assert learned.get("x") is None
    learned.remember("x", "y")
    assert learned.get("x") == "y"


# -- rewording ----------------------------------------------------------------------------------------

def chat_says(answer):
    seen = []

    def chat(messages, schema):
        seen.append(messages)
        return answer if isinstance(answer, str) else json.dumps(answer)
    chat.seen = seen
    return chat


def test_reword_gives_the_model_what_failed_and_what_the_picture_shows():
    chat = chat_says({"wording": "an anime girl gripping one katana, both hands together on its single hilt."})
    new = W.reword("an anime girl wielding a katana", "old words", ["2 sword blades"], "a girl with two swords",
                   chat)
    assert new == "an anime girl gripping one katana, both hands together on its single hilt"
    user = chat.seen[0][1]["content"]
    assert "2 sword blades" in user and "a girl with two swords" in user and "old words" in user


@pytest.mark.parametrize("answer", [
    {"wording": "a girl with a katana, not two"},          # klein draws what it's told about
    {"wording": "x" * (W.MAX_WORDING + 1)},
    {"wording": "old words"},                              # nothing new
    {"wording": ""}, {"other": 1}, "not json", ["a list"]])
def test_a_bad_rewrite_keeps_the_old_words(answer):
    assert W.reword("a girl", "old words", ["2 sword blades"], "", chat_says(answer)) is None


def test_the_model_writes_yes_no_checks_for_what_the_glossary_doesnt_know():
    chat = chat_says({"questions": ["Describe the horn.", "Does the creature have a single horn?",
                                    "Is there no dragon?", "Does it have the body of a horse and the tail of an ox?",
                                    "Does it have hooves?", "Is it a qilin?"]})
    asks = W.asked("fighting a ki-rin", chat)      # not yes/no, negative, two things at once, past the 2: dropped
    assert [a.question for a in asks] == ["Does the creature have a single horn? Answer yes or no.",
                                          "Does it have hooves? Answer yes or no."]
    assert all(a.expect is True and a.kind == "asked" for a in asks)
    assert asks[0].wrong == "no to 'Does the creature have a single horn?'"
    assert W.asked("x", chat_says("garbage")) == []


# -- the checker ------------------------------------------------------------------------------------

def test_the_checklist_asks_about_blades_and_the_chimera_per_feature():
    checks = CK.checklist(ADAM)
    assert [(c.kind, c.expect, c.feature) for c in checks] == [
        ("people", 1, "A"), ("blades", 1, "A"), ("goat", True, "B"), ("snake", True, "B")]


@pytest.mark.parametrize("answer, yes", [("Yes.", True), ("yes, there is", True), ("No", False),
                                          ("  No, it isn't", False), ("I can't tell", None), ("nope", None)])
def test_parse_yes(answer, yes):
    assert CK.parse_yes(answer) is yes


def test_failed_says_what_is_wrong_per_check(tmp_path):
    answers = {"people": "1", "blades": "2", "goat": "No.", "snake": "Yes"}

    def ask(image, question):
        return next(v for k, v in answers.items() if (k == "people" and "people" in question)
                    or (k == "blades" and "blades" in question) or (k == "goat" and "goat" in question)
                    or (k == "snake" and "snake" in question))
    got = CK.failed(tmp_path / "x.png", CK.checklist(ADAM), ask)
    assert [(c.feature, said) for c, said in got] == [("A", "2 sword blades"), ("B", "no goat's head on its back")]
    answers.update(blades="1", goat="maybe")
    assert CK.problems(tmp_path / "x.png", CK.checklist(ADAM), ask) == []


def test_the_checker_is_jarvis_s_own_voice_model_and_never_unloads_it(monkeypatch, tmp_path):
    from homeai import httpjson, voice_model
    sent = []
    monkeypatch.setattr(voice_model, "current", lambda: "qwen35-voice")
    monkeypatch.setattr(httpjson, "ollama_chat", lambda model, messages, **kw: sent.append((model, kw)) or "1")
    image = tmp_path / "x.png"
    image.write_bytes(b"png")
    CK.ollama_ask(image, "How many?")
    CK.ollama_ask(image, "How many?", "minicpm-v")
    assert sent[0][0] == "qwen35-voice" and sent[0][1]["keep_alive"] == -1 and sent[0][1]["think"] is False
    assert sent[1][0] == "minicpm-v" and sent[1][1]["keep_alive"] == "2m"


# -- session ------------------------------------------------------------------------------------------

def test_an_edit_keeps_the_picture_s_rewrites():
    spec = {"features": F.dump(ADAM), "wording": {"A": {"for": ADAM[0].text, "text": "OWN"}}}
    said = json.dumps({"ops": [{"op": "add", "kind": "colour", "text": "red hair"}]})
    result = features_patch_for(spec, "make her hair red", chat=lambda m, s: said)
    assert result.ok and result.spec["wording"] == spec["wording"]


def test_a_rewrite_must_pass_the_kid_safe_rules(tmp_path):
    from homeai.art import policy
    from homeai.art.session import ArtSession
    from homeai.art.store import Store
    kid = ArtSession(Store(tmp_path / "p"), None, "ani", rules=policy.load("kid-safe"))
    assert kid.allowed_wording("a girl holding one katana")
    assert not kid.allowed_wording("a girl covered in blood and gore")
    assert kid.learned.path == tmp_path / "wording.json"


# -- in art mode: notice, reword, redraw, learn ------------------------------------------------------------

@pytest.fixture
def katana_art(art):
    """Adam's picture: the girl with a katana, checked for people (1) and blades (1)."""
    art.reword_chat = chat_says({"wording": "an anime girl gripping one katana with both hands on a single hilt"})
    art.session.project.nodes["001"].spec["features"] = F.dump(feats(("subject", "an anime girl wielding a katana")))
    return art


def test_the_first_failure_redraws_the_second_rewords(katana_art):
    art = katana_art
    art.answers += ["1", "2"]                                   # people 1, blades 2
    finish(art, "001")
    project = art.session.project
    assert "wording" not in project.nodes["002"].spec and art.reword_chat.seen == []
    art.answers += ["1", "2", "a girl holding two swords"]      # fails again; then the picture is described
    finish(art, "002")
    spec = project.nodes["003"].spec
    assert spec["wording"] == {"A": {"for": "an anime girl wielding a katana",
                                     "text": "an anime girl gripping one katana with both hands on a single hilt"}}
    assert project.nodes["003"].meta["prompt"].startswith("An anime girl gripping one katana")
    assert "a girl holding two swords" in art.reword_chat.seen[0][1]["content"]
    assert art.status.endswith("Trying other words for it.")
    assert all("check pictures" not in m[0]["content"] for m in art.reword_chat.seen)  # A is checked already
    art.answers += ["1", "1"]                                   # the rewrite worked
    finish(art, "003")
    assert exported(art) == [f"{project.id}-003.png"]
    assert art.session.learned.get("an anime girl wielding a katana").startswith("an anime girl gripping")


def test_a_rewrite_the_rules_refuse_is_not_used(katana_art):
    art = katana_art
    art.session.allowed_wording = lambda text: False
    art.answers += ["1", "2", "1", "2", "seen"]
    finish(art, "001")
    finish(art, "002")
    assert "wording" not in art.session.project.nodes["003"].spec


def test_no_reword_chat_no_rewrite_and_no_asked_checks(katana_art):
    art = katana_art
    art.reword_chat = None
    art.answers += ["1", "2", "1", "2"]
    finish(art, "001")
    finish(art, "002")
    assert "wording" not in art.session.project.nodes["003"].spec


def model_says(questions, wording):
    """The voice model: check questions for the questions schema, a rewrite for the rewrite schema."""
    seen = []

    def chat(messages, schema):
        seen.append(messages)
        if "questions" in schema["properties"]:
            return json.dumps({"questions": questions})
        return json.dumps({"wording": wording})
    chat.seen = seen
    return chat


def test_a_feature_nothing_checks_gets_the_model_s_own_questions_once(art):
    art.reword_chat = model_says(["Does the creature have a single horn?"], "a golden horse with one horn")
    art.session.project.nodes["001"].spec["features"] = F.dump(feats(("subject", "a ki-rin")))
    art.answers += ["No.", "a horse"]                           # fails; then the picture is described
    finish(art, "001")
    project = art.session.project
    assert "no to 'Does the creature have a single horn?'" in project.nodes["001"].meta["error"]
    art.answers += ["Yes."]
    finish(art, "002")
    questions = [m for m in art.reword_chat.seen if "check pictures" in m[0]["content"]]
    assert len(questions) == 1                                  # asked once per feature text, not per picture
    assert exported(art) == [f"{project.id}-002.png"]


def test_only_the_model_s_own_questions_failing_rewords_at_once_and_redraws_once(art):
    art.reword_chat = model_says(["Does the creature have a single horn?"], "a golden horse with one horn")
    art.session.project.nodes["001"].spec["features"] = F.dump(feats(("subject", "a ki-rin")))
    art.answers += ["No.", "a horse"]
    finish(art, "001")
    project = art.session.project
    assert project.nodes["002"].spec["wording"]["A"]["text"] == "a golden horse with one horn"   # no waiting
    art.answers += ["No."]
    finish(art, "002")                                          # still no: shown, no third try
    assert exported(art) == [f"{project.id}-002.png"] and "003" not in project.nodes
    assert art.status.startswith("Best I could do: no to")


def test_daemon_rewords_with_jarvis_s_own_model_when_checking_is_on(assistant):  # noqa: F811
    import dataclasses
    from homeai.art import patcher
    from homeai.config import ArtConfig
    assistant.cfg = dataclasses.replace(assistant.cfg, art=ArtConfig(voice=True, check=True))
    assert assistant._build_art().reword_chat is patcher.ollama_chat
    assistant.cfg = dataclasses.replace(assistant.cfg, art=ArtConfig(voice=True, check=False))
    assert assistant._build_art().reword_chat is None


def test_a_check_expecting_no_fails_on_yes(tmp_path):
    check = CK.Check("wings", "Does the lion have wings? Answer yes or no.", False, "wings", "A", "a lion with wings")
    assert CK.problems(tmp_path / "x.png", [check], lambda image, q: "Yes.") == ["a lion with wings"]
    assert CK.problems(tmp_path / "x.png", [check], lambda image, q: "No.") == []
