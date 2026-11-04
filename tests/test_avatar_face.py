"""Face motion (homeai/avatar/face.py): visemes, look-ahead, springs, blinks, gaze, states."""
from __future__ import annotations

import math

import pytest

from homeai.avatar import face
from homeai.avatar.scene import Scene, Utterance


def utt(visemes, start=10.0, seconds=1.0, level=255, hz=60):
    n = int(seconds * hz)
    return Utterance.from_event({"id": "u", "start_mono": start, "level_hz": hz, "level": [level] * n,
                                 "duration_ms": seconds * 1000, "visemes": visemes})


def run(motion, start, end, state="speaking", utterances=(), level=1.0, dt=1 / 60):
    t, out = start, []
    while t < end - 1e-9:
        t += dt
        out.append((t, motion.update(t, dt, state, utterances, level)))
    return out


# ---- hold filter ----

def test_hold_filter_drops_a_one_frame_flicker():
    keys = [(0.0, "AA", 1.0), (0.10, "EH", 1.0), (0.12, "AA", 1.0), (0.3, "rest", 1.0)]
    assert [k[1] for k in face.hold_filter(keys)] == ["AA", "rest"]


def test_hold_filter_keeps_a_short_closure():
    keys = [(0.0, "AA", 1.0), (0.10, "PP", 1.0), (0.12, "AA", 1.0)]
    assert [k[1] for k in face.hold_filter(keys)] == ["AA", "PP", "AA"]


def test_hold_filter_merges_repeats_keeping_the_strongest():
    out = face.hold_filter([(0.0, "AA", 0.4), (0.1, "AA", 0.9), (0.2, "OU", 1.0)])
    assert out == [(0.0, "AA", 0.9), (0.2, "OU", 1.0)]


def test_hold_filter_drops_a_flicker_in_the_scene():
    u = utt([[0, "AA", 1.0], [100, "EH", 1.0], [120, "AA", 1.0]])
    assert u.mouth_cls == ["AA"]
    assert len(u.vis_t) == 3                               # the sphere still sees every viseme


# ---- targets ----

def test_rest_and_silence_give_a_closed_neutral_mouth():
    assert all(v == 0.0 for v in face.mouth_targets([], 10.5).values())
    assert all(v == 0.0 for v in face.mouth_targets([utt([[0, "rest", 1.0]])], 10.5).values())


def test_lips_close_before_a_bilabial_lands():
    u = utt([[0, "AA", 1.0], [500, "PP", 1.0], [700, "AA", 1.0]])
    m = face.FaceMotion()
    frames = run(m, 10.0, 10.5, utterances=[u])           # up to the moment PP is due
    p = frames[-1][1]
    assert p["lips_close"] > 0.6 and p["jaw_open"] < 0.25


def test_quiet_vowels_do_not_open_the_jaw():
    u = utt([[0, "AA", 1.0]], level=int(255 * face.LEVEL_FLOOR * 0.5))
    assert face.mouth_targets([u], 10.5)["jaw_open"] == 0.0


def test_without_visemes_the_jaw_follows_loudness():
    u = utt([], level=255)
    assert face.mouth_targets([u], 10.5)["jaw_open"] == pytest.approx(face.LEVEL_ONLY_JAW, abs=0.01)


def test_unknown_viseme_falls_back_to_a_mid_vowel():
    t = face.mouth_targets([utt([[0, "ZZ", 1.0]])], 10.5)
    assert t["jaw_open"] == pytest.approx(face.VISEME_PARAMS["EH"]["jaw_open"], abs=0.01)


# ---- springs ----

def test_jaw_step_never_overshoots():
    m = face.FaceMotion()
    target = face.mouth_targets([utt([[0, "AA", 1.0]])], 10.5)["jaw_open"]
    peaks = [p["jaw_open"] for _, p in run(m, 10.0, 10.9, utterances=[utt([[0, "AA", 1.0]])])]
    assert max(peaks) <= target + 1e-9
    assert peaks[-1] == pytest.approx(target, rel=0.02)


def test_speech_stop_closes_the_mouth_within_120ms():
    scene = Scene()
    scene.set_connected(True)
    scene.apply({"type": "state", "data": {"state": "speaking"}})
    scene.apply({"type": "utterance", "data": {"id": "a", "start_mono": 10.0, "level_hz": 60,
                                               "level": [255] * 120, "duration_ms": 2000,
                                               "visemes": [[0, "AA", 1.0]]}})
    t = 10.0
    while t < 10.5:
        t += 1 / 60
        scene.frame(t)
    assert scene.frame(t).params["jaw_open"] > 0.5
    scene.apply({"type": "speech_stop", "data": {}})
    end = t + 0.12
    while t < end:
        t += 1 / 60
        jaw = scene.frame(t).params["jaw_open"]
    assert jaw < 0.05


# ---- blinks ----

def test_blinks_come_every_three_to_six_seconds():
    m = face.FaceMotion(seed=3)
    starts, last = [], 0.0
    for t, p in run(m, 0.0, 120.0, state="idle"):
        if p["blink_l"] > 0.9 and t - last > face.BLINK_S:
            starts.append(t)
            last = t
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(starts) > 15
    singles = [g for g in gaps if g > face.BLINK_S + 0.2]
    assert singles and min(singles) >= face.BLINK_EVERY_S[0] - 0.05
    assert max(gaps) <= face.BLINK_EVERY_S[1] + 0.05


def test_blink_profile_closes_fully_and_reopens():
    assert face.blink_profile(-0.01) == 0.0
    assert face.blink_profile(face.BLINK_CLOSE_S + 0.01) == 1.0
    assert face.blink_profile(face.BLINK_S + 0.01) == 0.0


def test_no_blinking_while_asleep():
    m = face.FaceMotion()
    run(m, 0.0, 3.0, state="sleep")
    m.blink_now(3.0)                   # even a forced blink is suppressed with the eyes shut
    assert m._blinks(3.05, 1.0) == 0.0


# ---- eyes / gaze / states ----

def test_sleep_closes_the_eyes_slowly_and_waking_opens_them_fast():
    m = face.FaceMotion()
    run(m, 0.0, 1.0, state="idle")
    shut = run(m, 1.0, 4.0, state="sleep")
    assert shut[6][1]["blink_l"] < 0.5                    # not snapped shut
    assert shut[-1][1]["blink_l"] > 0.95
    woke = run(m, 4.0, 4.3, state="idle")
    assert woke[-1][1]["blink_l"] < 0.15


def test_listening_widens_the_eyes_and_centres_the_gaze():
    m = face.FaceMotion()
    p = run(m, 0.0, 2.0, state="listening")[-1][1]
    assert p["blink_l"] < 0.0
    assert abs(p["gaze_x"]) < 0.1 and abs(p["gaze_y"]) < 0.1


def test_thinking_looks_up_and_away():
    m = face.FaceMotion()
    p = run(m, 0.0, 1.0, state="thinking")[-1][1]
    assert p["gaze_y"] > 0.3 and abs(p["gaze_x"]) > 0.3


def test_gaze_and_head_stay_clamped():
    m = face.FaceMotion()
    m._x["gaze_x"], m._x["head_yaw"] = 9.0, 9.0
    p = m.update(1.0, 1 / 60, "idle")
    assert abs(p["gaze_x"]) <= face.GAZE_LIMIT and abs(p["head_yaw"]) <= face.HEAD_LIMIT


def test_offline_greys_and_dims():
    p = run(face.FaceMotion(), 0.0, 3.0, state="offline")[-1][1]
    assert p["grey"] > 0.9 and p["bright"] < 0.5


def test_unknown_state_moves_like_idle():
    a = run(face.FaceMotion(seed=1), 0.0, 2.0, state="idle")[-1][1]
    b = run(face.FaceMotion(seed=1), 0.0, 2.0, state="who-knows")[-1][1]
    assert a == b


def test_same_seed_same_motion():
    a = run(face.FaceMotion(seed=9), 0.0, 8.0, state="idle")
    b = run(face.FaceMotion(seed=9), 0.0, 8.0, state="idle")
    assert a == b
    assert all(math.isfinite(v) for _, p in a for v in p.values())


def test_every_rig_field_is_driven():
    from homeai.avatar import rig

    p = face.FaceMotion().update(1.0, 1 / 60, "idle")
    assert set(rig.FIELD_PARAMS) <= set(p)


# ---- app: which look to show ----

class _Asset:
    version = "v9"


def _broken():
    raise ValueError("bad hash")


def test_auto_look_prefers_a_chosen_face():
    from homeai.avatar import app

    assert app.choose_look("auto", lambda: _Asset())[0] == "face"


@pytest.mark.parametrize("load", [lambda: None, _broken])
def test_auto_look_falls_back_to_the_sphere(load):
    from homeai.avatar import app

    look, asset, why = app.choose_look("auto", load)
    assert (look, asset) == ("sphere", None) and why


@pytest.mark.parametrize("load", [lambda: None, _broken])
def test_forced_face_look_fails_loudly(load):
    from homeai.avatar import app

    with pytest.raises(ValueError):
        app.choose_look("face", load)


def test_sphere_look_never_loads_the_face():
    from homeai.avatar import app

    assert app.choose_look("sphere", _broken) == ("sphere", None, "asked for the sphere")
