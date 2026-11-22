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


def test_kid_rules_steer_the_prompt_not_the_features(tmp_path):
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a knight on a horse"}]},
                     {"ops": [{"op": "add", "kind": "subject", "text": "a red fox"}]})
    s = ArtSession(Store(tmp_path), Queue(), "guest", rules=policy.load("kid-safe"), patch=patch)
    s.handle("draw a knight on a horse")
    node = s.project.nodes["001"]
    assert [f.text for f in F.features_of(node.spec)] == ["a knight on a horse"]
    assert node.meta["prompt"] == "A knight on a horse. Everyone fully clothed, family-friendly."
    s.handle("new picture")
    s.handle("draw a red fox")
    assert s.project.nodes["001"].meta["prompt"] == "A red fox. Family-friendly."     # clothes only with people
    s.rules = policy.load("allow-all")
    s._draw("again", s.project.nodes["001"].spec, None)
    assert s.queue.jobs[-1]["pos"]["inputs"]["text"] == "A red fox."


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
    # an unknown voice's own session goes to guest/
    art.session, art.owner = None, None
    art.handle("draw a picture of a cat", None)
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


# -- Adam's battleship session (2026-11-07) ------------------------------------------------------

def test_a_letterless_change_is_skipped():
    after, changed = F.apply_ops(DRAGON, [{"op": "change", "text": "A to a steampunk battleship"},
                                          {"op": "add", "kind": "element", "text": "a steampunk battleship"}])
    assert changed == ["E"] and after[0] == DRAGON[0] and after[-1].text == "a steampunk battleship"
    with pytest.raises(SpecError):
        F.apply_ops(DRAGON, [{"op": "change", "id": "Q", "text": "x"}])      # a named letter must exist


def test_one_main_subject():
    after, changed = F.apply_ops(DRAGON, [{"op": "add", "kind": "subject", "text": "a steampunk battleship"}])
    assert [f.kind for f in after].count("subject") == 1 and after[-1].kind == "element"


def test_a_complaint_never_writes_the_defect_in(tmp_path):
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a single dragon"}]},
                     {"ops": [{"op": "change", "id": "A", "text": "a single dragon with two heads, one on each side"}]})
    s = ArtSession(Store(tmp_path), Queue(), "adam", patch=patch, rng=random.Random(1))
    s.handle("draw a dragon")
    reply = s.handle("again. it has two, a little head on this side and a big head on this side")
    assert reply.spoken.startswith("Fixing it") and len(s.grid) == 4
    for node in s.grid:
        assert F.features_of(s.project.nodes[node].spec)[0].text == "a single dragon"


def test_a_question_never_changes_the_picture(tmp_path):
    from homeai.art.session import MSG_QUESTION

    def no_model(*a):
        raise AssertionError("patched a question")
    s = ArtSession(Store(tmp_path), Queue(), "adam", patch=scripted({"ops": [{"op": "add", "kind": "subject",
                                                                              "text": "a dragon"}]}))
    s.handle("draw a dragon")
    s._patch = no_model
    assert s.handle_free("what is that in his hand?").spoken == MSG_QUESTION


def test_what_prompt_includes_hidden_steering_words(tmp_path):
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a knight"}]})
    s = ArtSession(Store(tmp_path), Queue(), "guest", rules=policy.load("kid-safe"), patch=patch)
    s.handle("draw a knight")
    assert s.handle("what prompt made this").spoken == s.project.nodes["001"].meta["prompt"]


def test_a_dropped_colour_is_put_back():
    lost = [F.Feature("A", "subject", "a dragon"), F.Feature("B", "element", "a samurai sitting on the dragon's back")]
    after, changed = F.keep_details("draw a samurai with red armor riding a dragon", lost, ["A", "B"])
    assert after[1].text == "a samurai sitting on the dragon's back, in red armor" and changed == ["A", "B"]
    kept = [F.Feature("A", "subject", "a dragon"), F.Feature("B", "element", "a samurai in red armor")]
    assert F.keep_details("draw a samurai with red armor riding a dragon", kept, [])[0] == kept
    assert F.keep_details("a castle with blue towers", lost, [])[0] == lost           # nothing names the castle
    after, changed = F.keep_details("make the knight wearing gold armor", [F.Feature("A", "subject", "a knight")], [])
    assert after[0].text == "a knight, in gold armor" and changed == ["A"]


def test_the_patcher_keeps_the_colour(tmp_path):
    said = json.dumps({"about_picture": True, "ops": [
        {"op": "add", "id": "", "kind": "subject", "text": "a dragon"},
        {"op": "add", "id": "", "kind": "element", "text": "a samurai sitting on the dragon's back"}]})
    result = features_patch_for({}, "draw a samurai with red armor riding a dragon", chat=lambda m, s: said)
    assert "red armor" in F.features_of(result.spec)[1].text


def test_chatter_is_not_a_change():
    said = json.dumps({"about_picture": False, "ops": [{"op": "change", "id": "A", "kind": "subject", "text": "x"}]})
    result = features_patch_for(spec_of(DRAGON), "oh my gosh, what's that", chat=lambda m, s: said)
    assert not result.ok and result.spec == spec_of(DRAGON)


def test_placeholders_are_not_features():
    after, changed = F.apply_ops(DRAGON[:1], [{"op": "add", "kind": "setting", "text": "nowhere specific"},
                                              {"op": "add", "kind": "style", "text": "None"}])
    assert after == DRAGON[:1] and changed == []


# -- a complaint about what the picture shows (Adam's paladin, 2026-11-20) -----------------------------------------

PALADIN_SAID = ("that troll looks a whole lot like an orc and it seems to be wielding swords and the paladin seems to "
                "be wielding two swords instead of having a sword and a shield buckled to his other arm")


@pytest.mark.parametrize("said, named", [
    (PALADIN_SAID, {"orc", "swords"}),
    ("it looks like a horse", {"horse"}),
    ("make the troll so it looks like a cave troll", set()),             # a request, not a mistake
    ("can it look more like a dragon", set()),
    ("make the dragon red", set()),
])
def test_what_the_picture_wrongly_shows(said, named):
    assert F.mistakes_named(said) == named


@pytest.mark.parametrize("said, complaint", [
    (PALADIN_SAID, True), ("it looks like an orc", True), ("make it so it looks like a cave troll", False),
    ("give the paladin a shield", False)])
def test_seeing_it_wrong_is_a_complaint(said, complaint):
    from homeai.art import grammar
    assert grammar.is_complaint(said) is complaint


def test_a_complaint_keeps_the_right_part_of_the_patch_and_never_the_mistake():
    before = [F.Feature("A", "subject", "a paladin wielding a longsword in his right hand and a shield in his left hand"),
              F.Feature("B", "element", "a troll"), F.Feature("C", "element", "the paladin facing off against the troll")]
    after = [F.Feature("A", "subject", "a paladin with one longsword in his right hand and a shield on his left arm"),
             F.Feature("B", "element", "an orc wielding two swords"),
             F.Feature("C", "element", "the paladin facing off against the orc, wielding two swords in both hands"),
             F.Feature("D", "element", "a second orc")]
    kept, changed = F.keep_mistakes_out(before, after, ["A", "B", "C", "D"], PALADIN_SAID)
    assert [f.text for f in kept] == [after[0].text, "a troll", before[2].text]
    assert changed == ["A"]


def test_a_mistake_already_asked_for_stays():
    before = [F.Feature("A", "subject", "a knight with two swords")]
    after = [F.Feature("A", "subject", "a knight with two swords, one in each hand")]
    assert F.keep_mistakes_out(before, after, ["A"], "it has three swords")[0] == after


def test_a_removal_in_a_complaint_stands():
    before = [F.Feature("A", "subject", "a troll"), F.Feature("B", "element", "an extra goblin")]
    kept, changed = F.keep_mistakes_out(before, before[:1], ["B"], "it has an extra goblin, it looks like a mess")
    assert kept == before[:1] and changed == ["B"]


def test_the_paladin_complaint_redraws_with_the_troll_kept(tmp_path):
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a paladin with a longsword and a shield"},
                              {"op": "add", "kind": "element", "text": "a troll"}]},
                     {"ops": [{"op": "change", "id": "B", "text": "an orc wielding two swords"}]})
    s = ArtSession(Store(tmp_path), Queue(), "adam", patch=patch, rng=random.Random(1))
    s.handle("draw a paladin with a longsword and a shield fighting a troll")
    reply = s.handle_free(PALADIN_SAID)
    assert reply.spoken.startswith("Fixing it") and len(s.grid) == 4
    for node in s.grid:
        assert [f.text for f in F.features_of(s.project.nodes[node].spec)][1] == "a troll"


def test_the_patcher_is_told_a_complaint_names_the_mistake():
    from homeai.art.patcher import FEATURE_SYSTEM
    assert "that is the mistake: never write it into a" in FEATURE_SYSTEM
    assert "Never rename a thing to what it was mistaken for." in FEATURE_SYSTEM
    assert "or says something in it came out wrong" in FEATURE_SYSTEM      # still about the picture (eval 2026-11-20)


def test_a_complaint_about_two_swords_never_asks_for_two_swords():
    before = [F.Feature("A", "subject", "a knight holding a longsword")]
    after = [F.Feature("A", "subject", "a knight holding two swords")]
    assert F.keep_mistakes_out(before, after, ["A"], "it has two blades now") == (before, [])


# -- "the sword sucks": a judgement becomes a better description (Adam, 2026-11-21) ---------------------------------

SAMURAI = [F.Feature("A", "subject", "a samurai in black armor holding a katana"),
           F.Feature("B", "setting", "a bamboo forest at dawn")]


@pytest.mark.parametrize("said, words", [
    ("the sword sucks", {"suck"}), ("his armor looks really bad", {"bad"}), ("that forest is lame", {"lame"}),
    ("the wings look kind of crappy", {"crappy"}), ("ugh, this sword looks like garbage", {"garbage"}),
    ("the sword is terrible", {"terrible"}),
    ("make him look ugly", set()), ("draw an ugly troll", set()), ("the troll should look bad", set()),
    ("make the dragon red", set()), ("oh gosh, it's fake again", set())])
def test_a_judgement_is_found_and_a_request_is_not(said, words):
    from homeai.art import grammar
    assert F.judged(said) == words
    assert grammar.is_complaint(said) is bool(words)


def chats(*answers):
    """A pretend voice model giving these answers in turn; .asked keeps each system prompt."""
    answers = list(answers)

    def chat(messages, schema):
        chat.asked.append((messages[0]["content"], schema))
        return json.dumps(answers.pop(0))
    chat.asked = []
    return chat


def test_a_judged_thing_left_alone_gets_the_narrow_ask_and_only_that_thing_changes():
    from homeai.art.patcher import BETTER_SYSTEM
    paladin = [F.Feature("A", "subject", "a paladin wielding a longsword in his right hand and a shield in his left hand"),
               F.Feature("B", "element", "a troll")]
    chat = chats({"about_picture": True, "ops": [{"op": "change", "id": "A", "kind": "subject", "text": paladin[0].text}]},
                 {"id": "A", "thing": "longsword", "described": "longsword of gleaming steel with an engraved hilt"})
    result = features_patch_for(spec_of(paladin), "the sword is terrible", chat=chat)
    assert result.ok and result.changed == ["A"]
    assert F.features_of(result.spec)[0].text == ("a paladin wielding a longsword of gleaming steel with an engraved "
                                                  "hilt in his right hand and a shield in his left hand")
    assert F.features_of(result.spec)[1].text == "a troll"
    assert chat.asked[1][0] == BETTER_SYSTEM
    assert chat.asked[1][1]["properties"]["id"]["enum"] == ["A", "B"]      # only real letters (it said "1")


def test_a_judgement_called_chatter_still_gets_the_narrow_ask():
    chat = chats({"about_picture": False, "ops": []},
                 {"id": "A", "thing": "katana", "described": "a katana with a long polished blade"})
    result = features_patch_for(spec_of(SAMURAI), "the sword sucks", chat=chat)
    assert result.ok and F.features_of(result.spec)[0].text == \
        "a samurai in black armor holding a katana with a long polished blade"


def test_a_first_answer_that_rewrites_needs_no_second_ask():
    chat = chats({"about_picture": True, "ops": [{"op": "change", "id": "B", "kind": "setting",
                                                  "text": "a dense bamboo forest at dawn with tall stalks"}]})
    result = features_patch_for(spec_of(SAMURAI), "that forest is lame", chat=chat)
    assert F.features_of(result.spec)[1].text == "a dense bamboo forest at dawn with tall stalks"
    assert len(chat.asked) == 1


@pytest.mark.parametrize("described, text", [
    ("of dark steel with a gold guard", "a samurai in black armor holding a katana of dark steel with a gold guard"),
    ("blazing steel blade", "a samurai in black armor holding a katana with a blazing steel blade"),
    ("engraved blade", "a samurai in black armor holding a katana with an engraved blade"),
    ("the katana with a ridged blade.", "a samurai in black armor holding a katana with a ridged blade")])
def test_the_described_thing_reads_as_a_phrase(described, text):
    chat = chats({"about_picture": False, "ops": []}, {"id": "A", "thing": "katana", "described": described})
    assert F.features_of(features_patch_for(spec_of(SAMURAI), "the sword sucks", chat=chat).spec)[0].text == text


def test_a_thing_of_two_words_is_described_in_place():
    chat = chats({"about_picture": False, "ops": []},
                 {"id": "A", "thing": "black armor", "described": "black armor of lacquered iron plates"})
    assert F.features_of(features_patch_for(spec_of(SAMURAI), "his armor looks bad", chat=chat).spec)[0].text == \
        "a samurai in black armor of lacquered iron plates holding a katana"


@pytest.mark.parametrize("answer", [
    {"id": "A", "thing": "sword", "described": "sword of steel"},        # not a word in the feature
    {"id": "A", "thing": "two words", "described": "two words of x"},   # not in the feature either
    {"id": "A", "thing": "", "described": "katana of steel"},
    {"id": "A", "thing": "katana", "described": ""},
    {"id": "A", "thing": "katana", "described": "katana"},               # says nothing more
    {"id": "Z", "thing": "katana", "described": "katana of steel"},
    ["not", "a", "dict"]])
def test_a_useless_narrow_answer_changes_nothing(answer):
    chat = chats({"about_picture": False, "ops": []}, answer)
    assert not features_patch_for(spec_of(SAMURAI), "the sword sucks", chat=chat).ok


def test_chatter_is_still_not_a_change_when_nothing_is_judged():
    chat = chats({"about_picture": False, "ops": []})
    assert not features_patch_for(spec_of(SAMURAI), "oh gosh, it's fake again", chat=chat).ok
    assert len(chat.asked) == 1


def test_a_judgement_word_is_never_written_in():
    after = [F.Feature("A", "subject", "a samurai in black armor holding a katana that doesn't suck"), SAMURAI[1]]
    assert F.keep_mistakes_out(SAMURAI, after, ["A"], "the sword sucks") == (SAMURAI, [])
    sucking = [F.Feature("A", "subject", "a samurai sucking on his katana"), SAMURAI[1]]
    assert F.keep_mistakes_out(SAMURAI, sucking, ["A"], "the sword sucks") == (SAMURAI, [])
    fine = [F.Feature("A", "subject", "a samurai in black armor holding a polished katana"), SAMURAI[1]]
    assert F.keep_mistakes_out(SAMURAI, fine, ["A"], "the sword sucks") == (fine, ["A"])


def test_the_sword_sucks_redraws_four_tries_of_a_better_sword(tmp_path):
    patch = scripted({"ops": [{"op": "add", "kind": "subject", "text": "a samurai in black armor holding a katana"}]},
                     {"about_picture": False, "ops": []},
                     {"id": "A", "thing": "katana", "described": "katana with a long, polished steel blade"})
    s = ArtSession(Store(tmp_path), Queue(), "guest", rules=policy.load("kid-safe"), patch=patch, rng=random.Random(1))
    s.handle("draw a samurai in black armor holding a katana")
    reply = s.handle_free("the sword sucks")
    assert reply.spoken.startswith("Fixing it") and len(s.grid) == 4
    for node in s.grid:
        assert F.features_of(s.project.nodes[node].spec)[0].text == \
            "a samurai in black armor holding a katana with a long, polished steel blade"


def test_the_patcher_is_told_a_judgement_means_draw_it_better():
    from homeai.art.patcher import FEATURE_SYSTEM
    assert "When the person says something looks bad" in FEATURE_SYSTEM
    assert "Never write their judgement word into a feature." in FEATURE_SYSTEM
