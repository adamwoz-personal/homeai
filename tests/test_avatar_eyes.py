"""Eye-follow (homeai/avatar/eyes.py): the avatar looks at whoever the webcam sees, and only while awake."""
from __future__ import annotations

import types

import pytest

from homeai.avatar import eyes
from homeai.avatar.eyes import EyeFollow, biggest_face, look_from, parse, track
from homeai.avatar.face import FACE_STATES, FOLLOW_HEAD, FaceMotion
from homeai.config import AvatarConfig


class FakeProc:
    def __init__(self) -> None:
        self.stdout = iter(())
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def rig():
    clock, started = Clock(), []

    def start():
        started.append(FakeProc())
        return started[-1]

    return EyeFollow(start, clock), clock, started


def test_off_unless_opted_in(monkeypatch):
    monkeypatch.delenv("HOMEAI_EYE_FOLLOW", raising=False)
    assert AvatarConfig().eye_follow is False
    monkeypatch.setenv("HOMEAI_EYE_FOLLOW", "1")
    assert AvatarConfig().eye_follow is True


def test_gaze_points_toward_the_person():
    right_of_screen = look_from(0.2, 0.5)               # the camera faces us: image left is screen right
    assert right_of_screen[0] > 0 and right_of_screen[1] == 0
    assert look_from(0.8, 0.5)[0] < 0
    assert look_from(0.5, 0.2)[1] > 0                   # high in the image: look up
    assert look_from(0.5, 0.5) == (0.0, 0.0)


def test_gaze_stays_inside_the_eye():
    assert look_from(0.0, 0.0) == (eyes.LIMIT, eyes.LIMIT)
    assert look_from(1.0, 1.0) == (-eyes.LIMIT, -eyes.LIMIT)


def points(*xs, nose=(0.5, 0.5)):
    p = [types.SimpleNamespace(x=x, y=0.5) for x in xs]
    return [p[0], types.SimpleNamespace(x=nose[0], y=nose[1]), *p[1:]]


def test_the_biggest_face_wins():
    near = points(0.1, 0.6, nose=(0.3, 0.4))
    far = points(0.7, 0.8, nose=(0.75, 0.2))
    assert biggest_face([far, near]) == (0.3, 0.4)
    assert biggest_face([near, far]) == (0.3, 0.4)
    assert biggest_face([]) is None


def test_malformed_output_is_ignored():
    assert parse("0.25 0.75\n") == (0.25, 0.75)
    assert parse("-\n") is None
    assert parse("") is None
    assert parse("1 2 3") is None
    assert parse("left right") is None


def test_track_reports_faces_and_no_face():
    frames = iter(["a", "b"])
    seen = []
    code = track(lambda: next(frames), lambda frame, ms: [points(0.1, 0.9, nose=(0.2, 0.3))] if frame == "a" else [],
                 seen.append, sleep=lambda s: None, running=lambda: len(seen) < 2)
    assert code == 0 and seen == ["0.2000 0.3000", eyes.NO_FACE]


def test_a_dead_camera_ends_the_process_with_a_failure():
    reads = []

    def read():
        reads.append(1)

    code = track(read, lambda f, ms: [], lambda line: None, sleep=lambda s: None,
                 running=lambda: len(reads) < 10 * eyes.MAX_MISSED_READS)      # bounded, so a mutant fails, not hangs
    assert code == 1 and len(reads) == eyes.MAX_MISSED_READS


def test_camera_opens_when_awake(rig):
    follow, clock, started = rig
    follow.update(clock.now, "idle")
    follow.update(clock.now, "speaking")
    assert len(started) == 1 and not started[0].terminated


@pytest.mark.parametrize("state", ["sleep", "offline", "muted", "busy", "mode_switching"])
def test_camera_closes_when_she_is_not_awake(rig, state):
    follow, clock, started = rig
    follow.update(clock.now, "idle")
    assert follow.update(clock.now, state) is None
    assert started[0].terminated and len(started) == 1
    follow.update(clock.now, state)
    assert len(started) == 1


def test_camera_never_opens_asleep(rig):
    follow, clock, started = rig
    follow.update(clock.now, "sleep")
    assert not started


def test_stopping_forgets_the_face(rig):
    follow, clock, _ = rig
    follow.update(clock.now, "idle")
    follow.see("0.2 0.5")
    follow.stop()
    assert follow.update(clock.now, "idle") is None


def test_dead_camera_process_waits_before_retrying(rig):
    follow, clock, started = rig
    follow.update(clock.now, "idle")
    started[0].returncode = 1
    follow.update(clock.now, "idle")
    clock.now += eyes.RETRY_S - 1
    follow.update(clock.now, "idle")
    assert len(started) == 1
    clock.now += 1
    follow.update(clock.now, "idle")
    assert len(started) == 2


def test_she_looks_where_the_face_is(rig):
    follow, clock, _ = rig
    follow.see("0.2 0.5")
    assert follow.update(clock.now, "idle") == look_from(0.2, 0.5)


def test_lost_face_gives_her_gaze_back(rig):
    follow, clock, _ = rig
    follow.see("0.2 0.5")
    clock.now += eyes.LOST_S - 0.1
    assert follow.update(clock.now, "idle") is not None
    follow.see(eyes.NO_FACE)                            # "no face" doesn't refresh it
    clock.now += 0.2
    assert follow.update(clock.now, "idle") is None


def test_small_moves_do_not_make_the_eyes_tremble(rig):
    follow, clock, _ = rig
    follow.see("0.300 0.5")
    first = follow.update(clock.now, "idle")
    follow.see("0.305 0.5")
    assert follow.update(clock.now, "idle") == first
    follow.see("0.400 0.5")
    assert follow.update(clock.now, "idle") == look_from(0.4, 0.5)


def settle(motion, state, seconds=2.0, dt=1 / 60):
    out, now = {}, 10.0
    for _ in range(int(seconds / dt)):
        out = motion.update(now, dt, state)
        now += dt
    return out


@pytest.mark.parametrize("state", ["idle", "listening", "speaking", "followup"])
def test_gaze_goes_to_the_person(state):
    motion = FaceMotion(seed=1)
    motion.look_at = (0.7, -0.4)
    out = settle(motion, state)
    jitter = 0.31 * FACE_STATES[state].gaze[2]
    assert abs(out["gaze_x"] - 0.7) <= jitter + 0.02 and out["gaze_y"] < -0.3


def test_thinking_keeps_her_own_gaze():
    a, b = FaceMotion(seed=1), FaceMotion(seed=1)
    a.look_at = (-0.9, -0.9)
    assert settle(a, "thinking")["gaze_y"] == pytest.approx(settle(b, "thinking")["gaze_y"])


def test_head_turns_a_little_toward_the_person():
    a, b = FaceMotion(seed=1), FaceMotion(seed=1)
    a.look_at = (0.8, 0.5)
    following, own = settle(a, "idle"), settle(b, "idle")
    assert following["head_yaw"] - own["head_yaw"] == pytest.approx(FOLLOW_HEAD[0] * 0.8, abs=0.005)
    assert own["head_pitch"] - following["head_pitch"] == pytest.approx(FOLLOW_HEAD[1] * 0.5, abs=0.005)
