"""Art mode by voice (homeai/art/voice.py), the picture window's pure parts (viewer.py) and the daemon hook."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from homeai.art import viewer as V
from homeai.art import voice as AV
from homeai.art.session import ArtSession
from homeai.art.store import Store
from tests.test_art_session import fake_patch
from tests.test_daemon import assistant  # noqa: F401 - fixture


class FakeSwitcher:
    def __init__(self, mode="voice", fail=None):
        self._mode, self.fail, self.calls = mode, fail, []

    def mode(self):
        return self._mode

    def art(self):
        self.calls.append("art")
        if self.fail:
            raise RuntimeError(self.fail)
        self._mode = "art"

    def voice(self):
        self.calls.append("voice")
        self._mode = "voice"


class FakeJobs:
    def __init__(self, notify):
        self.notify, self.submitted, self.cancelled = notify, [], []
        self.started = self.stopped = False
        self.states = {}

    def submit(self, project, node, graph, priority=1):
        job = type("Job", (), {"id": f"j{len(self.submitted) + 1}"})()
        self.submitted.append({"id": job.id, "project": project, "node": node})
        self.states[job.id] = "queued"
        return job

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        self.states[job_id] = "cancelled"
        return True

    def waiting(self):
        return [j for j, s in self.states.items() if s == "queued"]

    def get(self, job_id):
        return type("J", (), {"state": self.states[job_id]})() if job_id in self.states else None

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeViewer:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def art(tmp_path):
    published, states = [], []

    def build(owner, notify):
        jobs = FakeJobs(notify)
        return ArtSession(Store(tmp_path / "p"), jobs, owner, patch=fake_patch, rng=random.Random(1)), jobs

    a = AV.ArtVoice(FakeSwitcher(), build, lambda etype, data: published.append((etype, data)),
                    viewer=FakeViewer(), pictures=tmp_path / "Pictures", clock=Clock(), on_state=states.append,
                    background=lambda fn: fn())
    a._start_idle_watch = lambda: None             # tests drive check_idle themselves
    a.published, a.states = published, states
    return a


def last_art(a):
    return [d for t, d in a.published if t == "art"][-1]


# -- outside art mode ------------------------------------------------------------

@pytest.mark.parametrize("said", ["what's the weather", "make me a sandwich", "turn off the lights",
                                  "I'm done drawing", "make it blue", "again"])
def test_ordinary_talk_outside_art_mode_goes_to_jarvis(art, said):
    assert art.handle(said, "Adam") is None
    assert art.switcher.calls == [] and not art.active


@pytest.mark.parametrize("said", ["art mode", "switch to art mode", "Switch to art mode.", "let's draw",
                                  "go into art mode", "start art mode"])
def test_entering_art_mode(art, said):
    assert art.handle(said, "Adam") == AV.MSG_READY
    assert art.switcher.calls == ["art"] and art.active
    assert art.viewer.calls == ["start"] and art.queue.started
    assert art.states == ["mode_switching"] and last_art(art)["active"] is True


def test_a_drawing_request_enters_art_mode_and_draws(art):
    reply = art.handle("make a picture of a samurai with red armor", "Adam")
    assert reply.startswith("Art mode. Drawing") and art.active
    assert len(art.queue.submitted) == 1


def test_already_in_art_mode_from_the_command_line_is_not_switched_again(art):
    art.switcher._mode = "art"
    assert art.handle("draw a fox", "Adam").startswith("Art mode.")
    assert art.switcher.calls == []


@pytest.mark.parametrize("speaker", [None, "Ani", "Beth"])
def test_kids_and_unknown_voices_draw_under_kid_rules(art, speaker):
    assert art.handle("draw a fox", speaker).startswith("Art mode.") and art.active
    assert art.session.rules.name == "kid-safe"
    assert "family-friendly" in art.session.project.nodes["001"].spec["style"]


def test_adults_draw_without_limits_matched_without_case(art):
    art.handle("draw a fox", "adam")
    assert art.session.rules.name == "allow-all"
    assert "family-friendly" not in (art.session.project.nodes["001"].spec.get("style") or "")


def test_rules_follow_whoever_is_speaking(art):
    from homeai.art.policy import MSG_KID_DENIED
    art.handle("art mode", "Adam")
    assert art.handle("draw a bloody knight", None) == MSG_KID_DENIED and art.queue.submitted == []
    assert art.handle("draw a bloody knight", "Adam") == "Drawing a bloody knight."


def test_a_failed_switch_is_spoken_and_stays_in_voice(art):
    art.switcher.fail = "the coder is busy; try again when it's idle"
    assert art.handle("art mode", "Adam") == "I couldn't start art mode: the coder is busy"
    assert not art.active and art.viewer.calls == []


# -- in art mode -------------------------------------------------------------------

def test_in_art_mode_sentences_go_to_the_session(art):
    art.handle("art mode", "Adam")
    assert art.handle("draw a red fox", "Adam") == "Drawing a red fox."
    assert art.handle("make it blue", "Adam") == "Changing palette to blue."
    assert len(art.queue.submitted) == 2


def test_house_talk_in_art_mode_still_goes_to_jarvis(art):
    art.handle("art mode", "Adam")
    assert art.handle("turn off the lights", "Adam") is None
    assert art.handle("what's the weather", "Adam") is None


def test_house_talk_from_anyone_in_art_mode_goes_to_jarvis_and_keeps_the_idle_clock(art):
    art.handle("art mode", "Adam")
    art._clock.now += AV.IDLE_EXIT_S + 1
    assert art.handle("turn off the lights", None) is None
    assert art.check_idle()


def test_saying_art_mode_again(art):
    art.handle("art mode", "Adam")
    assert art.handle("art mode", "Adam") == AV.MSG_ALREADY and art.switcher.calls == ["art"]


def test_leaving(art):
    art.handle("art mode", "Adam")
    art.handle("draw a red fox", "Adam")
    jobs = art.queue
    assert art.handle("I'm done drawing", "Adam") == AV.MSG_LEFT
    assert not art.active and jobs.stopped and jobs.cancelled == ["j1"]
    assert art.viewer.calls == ["start", "stop"] and art.switcher.calls == ["art", "voice"]
    assert last_art(art) == {"active": False}
    assert art.handle("make it blue", "Adam") is None


# -- idle exit ------------------------------------------------------------------------

def test_idle_exit_after_twenty_quiet_minutes(art):
    art.handle("art mode", "Adam")
    art._clock.now += AV.IDLE_EXIT_S - 1
    assert not art.check_idle() and art.active
    art._clock.now += 2
    assert art.check_idle() and not art.active and art.switcher.calls == ["art", "voice"]


def test_talking_resets_the_idle_clock(art):
    art.handle("art mode", "Adam")
    art._clock.now += AV.IDLE_EXIT_S - 1
    art.handle("what prompt made this", "Adam")
    art._clock.now += 2
    assert not art.check_idle()


def test_no_idle_exit_while_something_waits_or_draws(art):
    art.handle("art mode", "Adam")
    art.handle("draw a red fox", "Adam")
    art._clock.now += AV.IDLE_EXIT_S + 1
    assert not art.check_idle()                     # queued
    art.queue.states["j1"] = "running"
    assert not art.check_idle()                     # drawing
    art.queue.states["j1"] = "done"
    assert art.check_idle()


# -- pictures arriving -------------------------------------------------------------------

def finished(art, tmp_path, node="001"):
    image = tmp_path / "out.png"
    image.write_bytes(b"png")
    project = art.session.project
    project.finish(node, image)
    art.queue.states["j1"] = "done"
    art._on_job({"state": "done", "project": project.id, "node": node, "image": str(project.image(node))})
    return project


def test_a_finished_picture_is_saved_to_pictures_and_shown(art, tmp_path):
    art.handle("draw a red fox", "Adam")
    project = finished(art, tmp_path)
    saved = tmp_path / "Pictures" / "adam" / AV.export_name(project.id, "001")
    assert saved.read_bytes() == b"png"
    state = last_art(art)
    assert state["current"]["image"] == str(project.image("001")) and state["status"] == "Done."
    assert state["current"]["instruction"] == "draw a red fox"


def test_a_refused_picture_is_neither_saved_nor_shown(art, tmp_path):
    from tests.test_art_session import Denying
    art.handle("draw a red fox", "Adam")
    art.session.rules = Denying("image")
    finished(art, tmp_path)
    assert not (tmp_path / "Pictures").exists()
    assert last_art(art)["status"] == "That picture came out wrong."


def test_saving_failure_does_not_stop_drawing(art, tmp_path):
    art.handle("draw a red fox", "Adam")
    art.pictures = tmp_path / "file"
    art.pictures.write_text("not a folder")
    finished(art, tmp_path)
    assert last_art(art)["status"] == "Done."


def test_state_while_drawing(art):
    art.handle("draw a red fox", "Adam")
    art.queue.states["j1"] = "running"
    art._on_job({"state": "running", "project": art.session.project.id, "node": "001"})
    state = last_art(art)
    assert state["drawing"] and state["status"] == "Drawing..." and state["current"]["image"] is None


def test_failed_job_message_is_shown(art):
    art.handle("draw a red fox", "Adam")
    art._on_job({"state": "failed", "project": art.session.project.id, "node": "001", "message": "Out of memory."})
    assert last_art(art)["status"] == "Out of memory."


# -- the window -------------------------------------------------------------------

def test_viewer_command_is_a_transient_unit():
    cmd = AV.Viewer(run=lambda c: 0).command()
    assert cmd[:3] == ["systemd-run", "--user", "--collect"] and f"--unit={AV.VIEWER_UNIT}" in cmd
    assert cmd[-2:] == ["-m", "homeai.art.viewer"]


def test_viewer_can_follow_another_bus():
    assert "--setenv=HOMEAI_AVATAR_SOCKET=/x.sock" in AV.Viewer(socket="/x.sock").command()
    assert not any("HOMEAI_AVATAR_SOCKET" in c for c in AV.Viewer().command())


def test_viewer_start_replaces_an_old_window():
    ran = []
    AV.Viewer(run=lambda c: ran.append(c[:3]) or 0).start()
    assert ran[0] == ["systemctl", "--user", "stop"] and ran[1][0] == "systemd-run"


@pytest.mark.parametrize("w, h, bw, bh, want", [(1024, 1024, 500, 300, (300, 300)), (832, 1216, 1000, 1000, (684, 1000)),
                                               (0, 10, 50, 50, (50, 50))])
def test_fit(w, h, bw, bh, want):
    assert V.fit(w, h, bw, bh) == want


def test_layout_without_and_with_thumbnails():
    alone = V.layout(1000, 1000, 1)
    assert alone["thumbs"] == [] and alone["main"][3] == 1000 - V.STATUS_H - 3 * V.PAD
    assert alone["status"][1] + alone["status"][3] == 1000 - V.PAD
    boxes = V.layout(1000, 1000, 12)
    assert len(boxes["thumbs"]) == V.MAX_THUMBS
    right = max(x + w for x, y, w, h in boxes["thumbs"])
    assert right <= 1000 and boxes["main"][1] + boxes["main"][3] < boxes["status"][1]
    assert boxes["status"][1] + boxes["status"][3] < boxes["thumbs"][0][1]


def test_status_line():
    assert V.status_line({"active": False}) == "Art mode is off."
    line = V.status_line({"active": True, "current": {"instruction": "make it blue"}, "drawing": True,
                          "waiting": 2, "status": "Changing it."})
    assert line == "“make it blue”  ·  drawing  ·  2 waiting  ·  Changing it."
    assert V.status_line({"active": True}) == "Say what to draw."


def test_art_state_from_events_and_snapshot():
    state = {"active": True, "status": "x"}
    assert V.art_state({"type": "state", "data": {"state": "idle"}}, state) is state
    assert V.art_state({"type": "art", "data": {"active": False}}, state) == {"active": False}
    snap = {"type": "snapshot", "data": {"events": {"art": {"type": "art", "data": {"active": True, "status": "s"}}}}}
    assert V.art_state(snap, {}) == {"active": True, "status": "s"}
    assert V.art_state({"type": "snapshot", "data": {"events": {}}}, state) is state


def test_thumbs_are_the_latest_steps():
    lineage = [{"node": f"{i:03d}"} for i in range(12)]
    assert [t["node"] for t in V.thumbs_of({"lineage": lineage})] == [f"{i:03d}" for i in range(4, 12)]


# -- the daemon hook ------------------------------------------------------------------

class FakeArt:
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def handle(self, text, speaker):
        self.seen.append((text, speaker))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _turn(quick, heard, art_reply, name="Adam"):
    from unittest.mock import MagicMock
    from homeai.stt import Transcript
    quick.art = FakeArt(art_reply)
    quick.stt.transcribe_audio.return_value = Transcript(ok=True, text=heard)
    quick.agent.ask.reset_mock()
    quick.tts.say_safe.reset_mock()
    reply = quick.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "Six.", 1, ""
    quick.tts.say_chunked.return_value.interrupted = False
    quick.tts.say_chunked.return_value.first_audio_ms = 100.0
    quick.transcript = MagicMock()
    ident = type("I", (), {"name": name, "score": 0.6, "best": name, "margin": 0.5, "reason": ""})()
    quick._identify = lambda audio, turn: (ident, None)
    audio = np.zeros(16000, dtype=np.float32)
    quick._handle(audio, "wake", np.zeros(800, dtype=np.float32))
    return quick.transcript.write.call_args.args[0], [c.args[0] for c in quick.tts.say_safe.call_args_list]


@pytest.fixture
def quick_art(assistant, monkeypatch):
    from tests.test_daemon import _with_wake
    monkeypatch.setattr("homeai.daemon.FOLLOWUP_SETTLE_S", 0.0)
    _with_wake(assistant, barge_in=False, verify=False)
    return assistant


def test_art_is_off_in_tests_by_default(assistant):
    assert assistant.art is None


def test_daemon_speaks_the_art_reply_and_skips_the_agent(quick_art):
    turn, spoken = _turn(quick_art, "draw a red fox", "Art mode. Drawing a red fox.")
    quick_art.agent.ask.assert_not_called()
    assert spoken == ["Art mode. Drawing a red fox."] and turn.verdict == "art"
    assert quick_art.art.seen == [("draw a red fox", "Adam")]


def test_daemon_passes_non_art_talk_to_the_agent(quick_art):
    turn, _ = _turn(quick_art, "what is two plus four", None)
    quick_art.agent.ask.assert_called_once()


def test_a_question_from_art_mode_opens_a_followup(quick_art):
    _turn(quick_art, "art mode", AV.MSG_READY)
    assert quick_art._followup_armed.is_set()


def test_art_crash_is_spoken_not_fatal(quick_art):
    turn, spoken = _turn(quick_art, "draw a fox", RuntimeError("boom"))
    assert spoken == ["Something went wrong in art mode."]
    quick_art.agent.ask.assert_not_called()


# -- Adam's first live session (2026-11-07): every sentence that went wrong -----------------

@pytest.mark.parametrize("said", ["Hey Jervis, let's draw. Wait, bye, ma'am one.", "I'm Adam, switch to art mode.",
                                  "okay switch to art mode now"])
def test_entering_with_extra_words(art, said):
    assert art.handle(said, "Adam") == AV.MSG_READY and art.active


def test_lets_draw_something_enters_and_draws(art):
    reply = art.handle("let's draw a dragon with a samurai riding on its back", "Adam")
    assert reply == "Art mode. Drawing a dragon with a samurai riding on its back."
    assert len(art.queue.submitted) == 1


def anything_patch(spec, instruction):
    """A voice model that takes any sentence as the subject of a new picture."""
    from homeai.art.patcher import PatchResult
    from homeai.art import prompt_spec
    if spec:
        return PatchResult(spec, {}, False, "not caught")
    return PatchResult(prompt_spec.validate({"subject": instruction}), {"subject": instruction}, True)


@pytest.mark.parametrize("said", ["Let's start with the dragon.", "a red fox in the snow"])
def test_in_art_mode_a_sentence_without_a_verb_is_drawn(art, said):
    art.handle("art mode", "Adam")
    art.session._patch = anything_patch
    reply = art.handle(said, "Adam")
    assert reply is not None and len(art.queue.submitted) == 1


def test_in_art_mode_a_free_sentence_changes_the_picture(art):
    art.handle("art mode", "Adam")
    art.handle("draw a red fox", "Adam")
    assert art.handle("let's make it blue", "Adam") == "Changing palette to blue."
    assert art.session.project.nodes["002"].parent == "001"


@pytest.mark.parametrize("said", ["Are we in art mode right now?", "what mode are we in"])
def test_asking_if_in_art_mode(art, said):
    art.handle("art mode", "Adam")
    assert art.handle(said, "Adam") == AV.MSG_IN_ART_MODE


@pytest.mark.parametrize("said", ["what time is it", "how tall is mount everest", "who won the game last night"])
def test_other_questions_in_art_mode_go_to_jarvis(art, said):
    art.handle("art mode", "Adam")
    assert art.handle(said, "Adam") is None and art.queue.submitted == []


@pytest.mark.parametrize("said", ["what prompt made this", "is it too dark", "where is the fox", "does it look right"])
def test_questions_about_the_picture_stay_in_art_mode(art, said):
    art.handle("art mode", "Adam")
    art.handle("draw a red fox", "Adam")
    assert art.handle(said, "Adam") is not None


def test_drawing_sentence_uses_the_askers_words():
    from homeai.art.session import drawing_sentence
    assert drawing_sentence("draw a dragon with a samurai riding on its back") == \
        "Drawing a dragon with a samurai riding on its back."
    assert drawing_sentence("make a picture of a samurai") == "Drawing a samurai."
    assert drawing_sentence("let's start with the dragon") == "Drawing it now."


def test_long_fields_are_trimmed_not_refused():
    from homeai.art.patcher import clip_long_text, patch_for
    clipped = clip_long_text({"lighting": "glowing " * 20, "palette": "red"})
    assert len(clipped["lighting"]) <= 80 and clipped["lighting"].endswith("glowing") and clipped["palette"] == "red"
    long = '{"lighting": "' + "neon " * 30 + '"}'
    result = patch_for({"subject": "a dragon"}, "make it neon", chat=lambda m, s: long)
    assert result.ok and len(result.spec["lighting"]) <= 80



# -- Adam's second session (2026-11-07 evening) --------------------------------------------

def test_descriptions_filed_as_references_are_dropped():
    from homeai.art.patcher import patch_for
    said = '{"subject": "undead dragon", "references": ["cyberpunk samurai", "undead creature"]}'
    result = patch_for({"subject": "a dragon"}, "make it an undead dragon", chat=lambda m, s: said)
    assert result.ok and result.spec["subject"] == "undead dragon" and not result.spec.get("references")
    named = patch_for({"subject": "a dragon"}, "use Mira", chat=lambda m, s: '{"references": ["Mira"]}')
    assert named.spec["references"] == ["Mira"]


def test_none_is_not_a_value():
    from homeai.art.patcher import patch_for
    result = patch_for({}, "draw a dragon", chat=lambda m, s: '{"subject": "a dragon", "negative": "none"}')
    assert result.ok and not result.spec.get("negative")


def test_patch_instructions_cover_complaints_and_invented_fields():
    from homeai.art.patcher import SYSTEM
    assert "complains" in SYSTEM and "never repeat the complaint" in SYSTEM
    assert "nobody asked for" in SYSTEM and "never descriptions" in SYSTEM


# -- picking up after a restart ------------------------------------------------------------

def test_reentering_within_the_hour_resumes_the_latest_project(art):
    art.handle("draw a red fox", "Adam")
    project = art.session.project.id
    art.leave()
    assert art.handle("art mode", "Adam") == AV.MSG_READY
    assert art.session.project is not None and art.session.project.id == project
    assert art.handle("make it blue", "Adam") == "Changing palette to blue."


def test_an_old_project_is_not_resumed(art):
    import time as _time
    art.handle("draw a red fox", "Adam")
    art.leave()
    art._wall = lambda: _time.time() + AV.RESUME_WITHIN_S + 60
    art.handle("art mode", "Adam")
    assert art.session.project is None


def test_someone_elses_project_is_not_resumed(art):
    art.handle("draw a red fox", "Adam")
    art.leave()
    art.handle("art mode", None)
    assert art.session.project is None


def test_adopting_art_mode_left_on_goes_back_to_voice_when_idle(art, monkeypatch):
    monkeypatch.setattr(AV, "IDLE_CHECK_S", 0.01)
    art.switcher._mode = "art"
    art._clock.now = 0.0
    art.idle_s = 0.0
    art.adopt()
    import time as _time
    deadline = _time.monotonic() + 2
    while art.switcher.calls != ["voice"] and _time.monotonic() < deadline:
        _time.sleep(0.01)
    assert art.switcher.calls == ["voice"]


def test_adopt_does_nothing_in_voice_mode(art, monkeypatch):
    monkeypatch.setattr(AV, "IDLE_CHECK_S", 0.01)
    art.idle_s = 0.0
    art.adopt()
    import time as _time
    _time.sleep(0.05)
    assert art.switcher.calls == []


# -- F0: fresh starts and complaints (Adam's dragon session) -----------------------------------

def test_new_picture_with_what_to_draw_starts_a_new_project(art):
    art.handle("art mode", "Adam")
    art.handle("draw a red fox", "Adam")
    first = art.session.project.id
    reply = art.handle("let's start with a new picture. let's go with a medieval frigate on the open ocean", "Adam")
    assert reply == "Drawing a medieval frigate on the open ocean."
    assert art.session.project.id != first and art.session.project.nodes["001"].parent is None


def test_new_picture_alone_asks_what_to_draw(art):
    from homeai.art.session import MSG_FRESH
    art.handle("art mode", "Adam")
    art.handle("draw a red fox", "Adam")
    assert art.handle("new picture", "Adam") == MSG_FRESH and art.session.project is None
    assert art.handle("make it blue", "Adam").startswith("There's no picture yet")


def blue_patch(spec, instruction):
    from homeai.art.patcher import PatchResult
    from homeai.art import prompt_spec
    if not spec:
        return PatchResult(prompt_spec.validate({"subject": "a dragon"}), {"subject": "a dragon"}, True)
    new = prompt_spec.apply_patch(spec, {"details": "a single head, one rider"})
    return PatchResult(new, {"details": "a single head, one rider"}, True)


@pytest.mark.parametrize("said", ["we ended up with a second head and a second rider this time",
                                  "the samurai's leg is merged with the dragon's leg",
                                  "there's an extra head"])
def test_a_complaint_draws_four_new_tries_to_pick_from(art, said):
    art.handle("art mode", "Adam")
    art.session._patch = blue_patch
    art.handle("draw a dragon", "Adam")
    seed = art.session.project.nodes["001"].spec["seed"]
    reply = art.handle(said, "Adam")
    assert reply == "Fixing it: 4 new tries. Say which one you like."
    tries = [art.session.project.nodes[n] for n in art.session.grid]
    assert len(tries) == 4 and all(t.parent == "001" for t in tries)
    assert seed not in {t.spec["seed"] for t in tries} and len({t.spec["seed"] for t in tries}) == 4
    state = art.state()
    assert [o["n"] for o in state["options"]] == [1, 2, 3, 4]
    assert art.handle("the third one", "Adam") == "That one." and art.session.project.current == tries[2].id


def test_an_ordinary_change_keeps_the_seed_and_shows_no_options(art):
    art.handle("art mode", "Adam")
    art.session._patch = blue_patch
    art.handle("draw a dragon", "Adam")
    art.handle("make it a bit darker", "Adam")
    nodes = art.session.project.nodes
    assert nodes["002"].spec["seed"] == nodes["001"].spec["seed"] and "options" not in art.state()


def test_viewer_numbers_the_options():
    state = {"options": [{"n": 1, "node": "002"}, {"n": 2, "node": "003"}], "lineage": [{"node": "001"}]}
    assert [t["n"] for t in V.thumbs_of(state)] == [1, 2]


# -- F3: describing at length ------------------------------------------------------------------------

@pytest.mark.parametrize("said, tidy", [
    ("a red dragon on a cliff at sunset. That's it, go ahead and draw it.", "a red dragon on a cliff at sunset"),
    ("draw a castle. okay that's all", "draw a castle"),
    ("a fox in the snow, you can draw it now", "a fox in the snow"),
    ("draw it", "draw it"),
    ("that's it", "that's it")])
def test_sign_offs_are_trimmed(said, tidy):
    from homeai.art import grammar
    assert grammar._tidy(said) == tidy


def test_daemon_listens_longer_in_art_mode(assistant):
    assistant.art = type("A", (), {"active": False})()
    assert not assistant._long_capture()
    assistant.art.active = True
    assert assistant._long_capture()
    assistant.art = None
    assert not assistant._long_capture()
