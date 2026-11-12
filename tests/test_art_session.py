"""Art session (homeai/art/session.py), policy hooks (policy.py) and the default workflow (workflows.py)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from homeai.art import policy, prompt_spec, workflows
from homeai.art.patcher import MSG_NOT_CAUGHT, PatchResult
from homeai.art.session import MSG_NOTHING_TO_STOP, MSG_NOTHING_YET, ArtSession
from homeai.art.store import Store

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "art"))
import comfy_common  # noqa: E402


class FakeQueue:
    def __init__(self):
        self.jobs = []
        self.cancelled = []

    def submit(self, project, node, graph, priority=1):
        job = type("Job", (), {"id": f"j{len(self.jobs) + 1}"})()
        self.jobs.append({"id": job.id, "project": project, "node": node, "graph": graph, "priority": priority})
        return job

    def cancel(self, job_id):
        if job_id in self.cancelled:
            return False
        self.cancelled.append(job_id)
        return True


def fake_patch(spec, instruction):
    """A stand-in for the voice model with a few fixed answers."""
    if instruction.startswith(("draw", "paint", "make a")):
        what = instruction.split(" ", 1)[1].removeprefix("me ")
        return PatchResult(prompt_spec.validate({"subject": what}), {"subject": what}, True)
    if "blue" in instruction:
        return PatchResult(prompt_spec.apply_patch(spec, {"palette": "blue"}), {"palette": "blue"}, True)
    return PatchResult(spec, {}, False, MSG_NOT_CAUGHT)


@pytest.fixture
def session(tmp_path):
    return ArtSession(Store(tmp_path / "p"), FakeQueue(), "adam", patch=fake_patch, rng=random.Random(1))


def prompt_of(job):
    return job["graph"]["pos"]["inputs"]["text"]


def seed_of(job):
    return job["graph"]["noise"]["inputs"]["noise_seed"]


def test_ordinary_talk_is_passed_on(session):
    assert session.handle("turn off the lights") is None
    assert session.handle("what's the weather") is None


def test_new_picture(session):
    reply = session.handle("draw a red fox")
    assert reply.kind == "new" and reply.spoken == "Drawing a red fox."
    job = session.queue.jobs[0]
    assert prompt_of(job) == "A red fox." and job["project"] == session.project.id and job["node"] == "001"
    assert session.project.nodes["001"].spec["seed"] == seed_of(job)


def test_edit_keeps_the_seed_and_says_what_changed(session):
    session.handle("draw a red fox")
    reply = session.handle("make it blue")
    first, second = session.queue.jobs
    assert reply.spoken == "Changing palette to blue."
    assert seed_of(first) == seed_of(second) and "Blue colours" in prompt_of(second)
    assert session.project.nodes["002"].parent == "001"


def test_edit_not_caught_draws_nothing(session):
    session.handle("draw a red fox")
    assert session.handle("make it sing").spoken == MSG_NOT_CAUGHT
    assert len(session.queue.jobs) == 1


def test_commands_before_any_picture(session):
    for said in ("make it blue", "again", "go back", "keep that", "what prompt made this", "give me four"):
        assert session.handle(said).spoken == MSG_NOTHING_YET, said


def test_again_is_a_sibling_with_a_new_seed(session):
    session.handle("draw a red fox")
    session.handle("again")
    first, second = session.queue.jobs
    assert seed_of(first) != seed_of(second) and prompt_of(first) == prompt_of(second)
    assert session.project.nodes["002"].parent is None


def test_variations_then_pick(session):
    session.handle("draw a red fox")
    reply = session.handle("give me three")
    assert reply.kind == "variations" and len(reply.jobs) == 3
    assert len({seed_of(j) for j in session.queue.jobs[1:]}) == 3
    assert session.handle("the second one").spoken == "That one."
    assert session.project.current == session.grid[1]
    session.handle("the last one")
    assert session.project.current == session.grid[2]
    assert session.handle("the fourth one").spoken == "There are only 3."


def test_select_without_options(session):
    session.handle("draw a red fox")
    assert session.handle("the first one").kind == "select" and session.project.current == "001"


def test_back_and_branch(session):
    session.handle("draw a red fox")
    session.handle("make it blue")
    assert session.handle("go back").spoken == "Back to: draw a red fox."
    session.handle("again")
    assert session.project.current == "003"


def test_keep_pins(session):
    session.handle("draw a red fox")
    assert session.handle("keep that").spoken == "Kept."
    assert session.project.info["pins"] == ["001"]


def test_what_reads_the_prompt(session):
    session.handle("draw a red fox")
    assert session.handle("what prompt made this").spoken == "A red fox."


def test_cancel(session):
    assert session.handle("stop").spoken == MSG_NOTHING_TO_STOP
    session.handle("draw a red fox")
    assert session.handle("never mind").spoken == "Stopped." and session.queue.cancelled == ["j1"]


def test_an_intent_with_no_handler_is_said_plainly(session):
    from homeai.art import grammar
    session.handle("draw a red fox")
    grammar_parse = grammar.parse
    try:
        grammar.parse = lambda text: grammar.Intent("teleport")
        assert session.handle("teleport it").spoken == "I can't do that one yet."
    finally:
        grammar.parse = grammar_parse


class Denying:
    def __init__(self, at):
        self.at = at

    def check_request(self, text, profile):
        return policy.Decision("deny", "Not that one.") if self.at == "request" else policy.ALLOW

    def check_spec(self, spec, profile):
        if self.at == "rewrite":
            return policy.Decision("rewrite", spec={**spec, "subject": "a friendly fox"})
        return policy.Decision("deny", "Not that one.") if self.at == "spec" else policy.ALLOW

    def check_image(self, path, profile):
        return policy.Decision("deny", "That picture came out wrong.") if self.at == "image" else policy.ALLOW


@pytest.mark.parametrize("at", ["request", "spec"])
def test_policy_deny_is_spoken_and_draws_nothing(tmp_path, at):
    s = ArtSession(Store(tmp_path), FakeQueue(), "ani", rules=Denying(at), patch=fake_patch)
    reply = s.handle("draw a red fox")
    assert (reply.kind, reply.spoken) == ("denied", "Not that one.") and s.queue.jobs == []


def test_policy_rewrite_is_drawn(tmp_path):
    s = ArtSession(Store(tmp_path), FakeQueue(), "ani", rules=Denying("rewrite"), patch=fake_patch)
    s.handle("draw a red fox")
    assert prompt_of(s.queue.jobs[0]) == "A friendly fox."


def test_image_policy_hides_a_finished_picture(tmp_path):
    s = ArtSession(Store(tmp_path), FakeQueue(), "ani", rules=Denying("image"), patch=fake_patch)
    s.handle("draw a red fox")
    event = {"state": "done", "project": s.project.id, "node": "001", "image": "/x.png"}
    assert s.on_event(event) == "That picture came out wrong."
    assert s.store.open(s.project.id).nodes["001"].state == "failed"
    assert s.on_event({**event, "state": "running"}) is None
    assert s.on_event({**event, "project": "other"}) is None


def test_allowed_picture_is_shown(session):
    session.handle("draw a red fox")
    assert session.on_event({"state": "done", "project": session.project.id, "node": "001", "image": "x"}) is None


def test_unknown_policy_fails_closed():
    rules = policy.load("kid-safe-typo")
    for check, arg in ((rules.check_request, "a fox"), (rules.check_spec, {}), (rules.check_image, Path("x"))):
        decision = check(arg, "ani")
        assert not decision.allowed and decision.reason == policy.MSG_UNKNOWN_POLICY
    assert policy.load("allow-all").check_request("x", "adam").allowed


def test_klein9_graph_is_the_bake_off_graph():
    ours = workflows.klein9("a fox", 7, 832, 1216, prefix="p")
    assert ours == comfy_common.workflow("klein9", "a fox", 7, 832, 1216, prefix="p")


# -- kid-safe policy -------------------------------------------------------------------

@pytest.mark.parametrize("said", ["draw a naked lady", "a bloody battlefield with dead bodies", "a man smoking a cigarette",
                                  "a sexy elf", "kill the dragon"])
def test_kid_safe_refuses_adult_words(said):
    rules = policy.load("kid-safe")
    decision = rules.check_request(said, "ani")
    assert not decision.allowed and decision.reason == policy.MSG_KID_DENIED


@pytest.mark.parametrize("said", ["a killer whale", "a bloodhound puppy", "a samurai with a sword", "a scary dragon",
                                  "a knight fighting a monster"])
def test_kid_safe_allows_kid_things(said):
    assert policy.load("kid-safe").check_request(said, "ani").allowed


def test_kid_safe_steers_the_spec_family_friendly():
    rules = policy.load("kid-safe")
    decision = rules.check_spec({"subject": "a samurai", "style": "anime"}, "ani")
    assert decision.verdict == "rewrite"
    assert decision.spec["style"] == "anime, family-friendly"
    assert decision.spec["details"] == "everyone fully clothed"
    animal = rules.check_spec({"subject": "a fox"}, "ani")
    assert animal.spec == {"subject": "a fox", "style": "family-friendly"}
    again = rules.check_spec(decision.spec, "ani")
    assert again.verdict == "allow", "already steered: no change"


def test_kid_safe_refuses_a_spec_the_model_made_adult():
    decision = policy.load("kid-safe").check_spec({"subject": "a fox", "details": "in lingerie"}, "ani")
    assert not decision.allowed


def test_kid_safe_session_draws_the_steered_spec(tmp_path):
    s = ArtSession(Store(tmp_path), FakeQueue(), "ani", rules=policy.load("kid-safe"), patch=fake_patch)
    s.handle("draw a red fox")
    assert prompt_of(s.queue.jobs[0]) == "A red fox. Family-friendly style."
