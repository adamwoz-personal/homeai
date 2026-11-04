"""Avatar renderer: pure scene, app helpers, headless GL (AVATAR_PLAN §10 Phase 1)."""
from __future__ import annotations

import json
import math
import queue

import numpy as np
import pytest

from homeai.avatar import app
from homeai.avatar.scene import (BASE_RADIUS, FPS, JAW_OMEGA, LOOKAHEAD_S, OPEN_RADIUS, Scene,
                                 Utterance, spring)


def utt(start, level, visemes=None, uid="u1", hz=60.0):
    data = {"id": uid, "start_mono": start, "level": level, "level_hz": hz,
            "duration_ms": len(level) / hz * 1000}
    if visemes is not None:
        data["visemes"] = visemes
    return {"type": "utterance", "data": data}


def run(scene, t0, t1, fps=60):
    frames = []
    t = t0
    while t < t1:
        frames.append((t, scene.frame(t)))
        t += 1 / fps
    return frames


# ---- spring --------------------------------------------------------------------

def test_spring_converges_without_overshoot():
    x, v, peak = 0.0, 0.0, 0.0
    for _ in range(120):
        x, v = spring(x, v, 1.0, JAW_OMEGA, 1 / 60)
        peak = max(peak, x)
    assert abs(x - 1.0) < 1e-3 and peak <= 1.0 + 1e-9


def test_spring_large_dt_stable_and_zero_dt_noop():
    assert spring(0.3, 0.1, 1.0, JAW_OMEGA, 0) == (0.3, 0.1)
    x, _ = spring(0.0, 0.0, 1.0, JAW_OMEGA, 10.0)
    assert abs(x - 1.0) < 1e-6


def test_spring_lag_is_about_two_over_omega():
    """Lookahead cancels the spring's lag; keep them consistent."""
    x, v, t = 0.0, 0.0, 0.0
    while x < 0.5:
        x, v = spring(x, v, 1.0, JAW_OMEGA, 0.001)
        t += 0.001
    assert abs(t - LOOKAHEAD_S) < 0.015


# ---- utterances ------------------------------------------------------------------

def test_level_only_drives_openness():
    u = Utterance.from_event(utt(10.0, [0, 255, 128])["data"])
    assert u.openness_at(10.0) == 0.0
    assert u.openness_at(10.0 + 1.5 / 60) == 1.0
    assert u.openness_at(9.0) == 0.0 and u.openness_at(11.0) == 0.0


def test_visemes_shape_mouth_and_silence_closes_it():
    level = [255] * 30 + [0] * 30
    u = Utterance.from_event(utt(0.0, level, [[0, "PP", 1.0], [100, "AA", 1.0], [400, "rest", 0.0]])["data"])
    assert u.openness_at(0.05) == 0.0            # bilabial: closed though loud
    assert u.openness_at(0.2) == 1.0             # open vowel, full level
    assert u.openness_at(0.45) == 0.0            # rest
    u2 = Utterance.from_event(utt(0.0, [0] * 60, [[0, "AA", 1.0]])["data"])
    assert u2.openness_at(0.2) == 0.0            # quiet vowel stays shut


def test_bad_utterance_ignored():
    s = Scene()
    s.apply({"type": "utterance", "data": {"id": "x"}})
    s.apply({"type": "utterance", "data": {"start_mono": "soon"}})
    assert s.utterances == []


# ---- scene -----------------------------------------------------------------------

def test_mouth_follows_speech_on_time_and_closes():
    s = Scene()
    s.set_connected(True)
    s.apply({"type": "state", "data": {"state": "speaking"}})
    s.apply(utt(1.0, [0] * 30 + [255] * 30 + [0] * 30))   # loud 1.5-2.0 s
    frames = run(s, 0.9, 3.0)
    opened = [t for t, f in frames if f.openness > 0.5]
    assert opened, "mouth never opened"
    assert abs(opened[0] - 1.5) <= 0.040 and abs(opened[-1] - 2.0) <= 0.040
    assert frames[-1][1].openness < 0.05
    assert max(f.radius for _, f in frames) > BASE_RADIUS + OPEN_RADIUS * 0.8


def test_speech_stop_closes_within_120ms():
    s = Scene()
    s.apply(utt(0.0, [255] * 600))
    run(s, 0.0, 1.0)
    s.apply({"type": "speech_stop", "data": {}})
    after = run(s, 1.0, 1.2)
    assert [f.openness for t, f in after if t >= 1.12][0] < 0.1


def test_snapshot_restores_state_and_disconnect_goes_offline():
    s = Scene()
    s.apply({"type": "snapshot", "data": {"events": {"state": {"type": "state",
                                                               "data": {"state": "thinking"}}}}})
    assert s.state == "thinking"
    s.apply(utt(0.0, [255] * 60))
    s.set_connected(False)
    assert s.state == "offline" and s.utterances == []


def test_unknown_state_ignored_and_caption_kept():
    s = Scene(state="idle", connected=True)
    s.apply({"type": "state", "data": {"state": "dancing"}})
    s.apply({"type": "caption", "t_mono": 5.0, "data": {"who": "user", "text": "hi"}})
    assert s.state == "idle"
    assert s.frame(5.0).captions == [("user", "hi", 1.0)]


def test_fps_policy():
    s = Scene()
    for state, fps in (("speaking", 60), ("listening", 60), ("idle", 30), ("sleep", 10),
                       ("offline", 10)):
        s.state = state
        assert s.fps() == fps
    s.state = "idle"
    s.apply(utt(0.0, [255] * 10))
    assert s.fps() == 60                          # speech always gets full rate
    assert set(FPS) >= {"idle", "listening", "thinking", "speaking", "followup", "sleep",
                        "muted", "offline", "generating", "mode_switching"}


def test_state_colour_eases_not_jumps():
    s = Scene(state="idle", connected=True)
    first = s.frame(0.0).colour
    s.state = "muted"
    nxt = s.frame(1 / 60).colour
    assert nxt != first and nxt[0] < 0.9           # moving toward red, not there yet
    late = run(s, 1 / 60, 2.0)[-1][1].colour
    assert late[0] > 0.95


def test_stale_utterances_dropped_and_spin_integrates():
    s = Scene(state="thinking", connected=True)
    s.apply(utt(0.0, [255] * 6))
    s.frame(0.0)
    f = s.frame(2.0)
    assert s.utterances == []
    assert math.isclose(f.angle, (1.2 * 0.25) % (2 * math.pi))   # dt capped at 0.25 s


# ---- phase 2: shown states, motion, captions ---------------------------------------

def idle_scene(**kw):
    s = Scene(state="idle", connected=True, **kw)
    s.frame(0.0)
    return s


def test_sleeps_after_idle_minutes_and_any_event_wakes_it():
    s = idle_scene(sleep_s=300.0)
    assert s.frame(299.0).state == "idle"
    f = s.frame(300.0)
    assert f.state == "sleep" and f.fps == 10
    s.apply({"type": "caption", "data": {"who": "user", "text": "hey"}})
    assert s.frame(300.1).state == "idle"
    assert s.frame(599.0).state == "idle"         # the timer restarted at the event


def test_only_idle_sleeps_and_zero_disables():
    s = Scene(state="thinking", connected=True, sleep_s=1.0)
    s.frame(0.0)
    assert s.frame(10.0).state == "thinking"
    s = idle_scene(sleep_s=0.0)
    assert s.frame(1e6).state == "idle"


def test_speech_keeps_it_awake():
    s = idle_scene(sleep_s=1.0)
    s.apply(utt(0.5, [200] * 600))                # ten seconds of speech
    s.frame(0.5)
    assert s.frame(9.0).state == "idle"


def test_quiet_hours_sleep_after_thirty_seconds():
    from datetime import time as dtime

    from homeai.avatar.scene import QUIET_SLEEP_S
    window = (dtime(22, 0), dtime(7, 0))
    s = idle_scene(sleep_s=3600.0, quiet=window, clock=lambda: dtime(23, 30))
    assert s.frame(QUIET_SLEEP_S - 1).state == "idle"
    assert s.frame(QUIET_SLEEP_S).state == "sleep"
    s = idle_scene(sleep_s=3600.0, quiet=window, clock=lambda: dtime(12, 0))
    assert s.frame(QUIET_SLEEP_S).state == "idle"


def test_coding_mode_shows_busy_not_offline_and_switching_wins():
    s = Scene()
    s.local_mode = "coding"
    f = s.frame(0.0)
    assert f.state == "busy" and ("status", "coding mode", 1.0) in f.captions
    s.local_mode = None
    assert s.frame(0.1).state == "offline"
    s.set_connected(True)
    s.apply({"type": "state", "data": {"state": "speaking"}})
    s.local_mode = "switching"
    assert s.frame(0.2).state == "mode_switching"


def test_motion_eases_in_per_state():
    from homeai.avatar.scene import RING_PERIOD_S
    s = Scene(state="listening", connected=True)
    first = s.frame(0.0)
    assert first.ripple == 0.0                     # eases in, no jump
    f = run(s, 0.0, 1.5)[-1][1]
    assert f.ripple > 0.95 and f.swirl == 0.0 and f.ring == -1.0
    s.state = "thinking"
    frames = run(s, 1.5, 3.5)
    assert frames[-1][1].ripple < 0.05
    assert frames[-1][1].swirl != frames[-2][1].swirl      # swirling
    s.state = "followup"
    f = run(s, 3.5, 5.0)[-1][1]
    assert f.ring_amp > 0.95 and 0.0 <= f.ring < 1.0
    later = s.frame(5.0 + RING_PERIOD_S / 4).ring
    assert math.isclose((later - f.ring) % 1.0, 0.25, abs_tol=0.02)   # ring travels outward


def test_bands_reach_the_frame():
    s = Scene(state="speaking", connected=True)
    ev = utt(0.0, [255] * 60)
    ev["data"]["bands"] = {"low": [255] * 60, "high": [51] * 60}
    s.apply(ev)
    f = s.frame(0.5)
    assert f.low == 1.0 and f.high == pytest.approx(0.2)
    assert s.frame(5.0).low == 0.0


def test_user_caption_fades_after_hold():
    from homeai.avatar.scene import CAPTION_FADE_S, CAPTION_HOLD_S
    s = idle_scene()
    s.apply({"type": "caption", "t_mono": 1.0, "data": {"who": "user", "text": "hello"}})
    assert s.frame(1.0 + CAPTION_HOLD_S - 0.01).captions == [("user", "hello", 1.0)]
    mid = s.frame(1.0 + CAPTION_HOLD_S + CAPTION_FADE_S / 2).captions[0][2]
    assert 0.4 < mid < 0.6
    assert s.frame(1.0 + CAPTION_HOLD_S + CAPTION_FADE_S).captions == []


def test_jarvis_caption_follows_the_sentence_being_spoken():
    s = idle_scene()
    s.apply({"type": "caption", "t_mono": 1.0, "data": {"who": "user", "text": "moon?"}})
    s.apply({"type": "state", "data": {"state": "speaking"}})
    s.apply({"type": "caption", "t_mono": 1.0,
             "data": {"who": "jarvis", "text": "First sentence here. Second one now."}})
    s.apply(utt(2.0, [200] * 60, uid="a"))         # 2.0-3.0
    s.apply(utt(3.0, [200] * 60, uid="b"))         # 3.0-4.0
    said = lambda t: [c for c in s.frame(t).captions if c[0] == "jarvis"][0][1]  # noqa: E731
    assert said(1.5) == "First sentence here."
    assert said(2.5) == "First sentence here."
    assert said(3.5) == "Second one now."


def test_jarvis_caption_held_while_talking_then_fades():
    from homeai.avatar.scene import CAPTION_FADE_S, CAPTION_HOLD_S
    s = idle_scene()
    s.apply({"type": "state", "data": {"state": "speaking"}})
    s.apply({"type": "caption", "t_mono": 0.0, "data": {"who": "jarvis", "text": "A long reply."}})
    for t in range(1, 20):
        assert s.frame(float(t)).captions[0][2] == 1.0      # 19 s of talking, still shown
    s.apply({"type": "state", "data": {"state": "idle"}})
    s.frame(20.0)
    assert s.frame(20.0 + CAPTION_HOLD_S + CAPTION_FADE_S + 0.01).captions == []


def test_new_question_clears_the_old_answer():
    s = idle_scene()
    s.apply({"type": "caption", "t_mono": 0.0, "data": {"who": "jarvis", "text": "Old answer."}})
    s.apply({"type": "caption", "t_mono": 0.5, "data": {"who": "user", "text": "new question"}})
    assert [c[0] for c in s.frame(1.0).captions] == ["user"]


def test_snapshot_after_renderer_restart_restores_shown_state():
    s = Scene()
    s.set_connected(True)
    s.apply({"type": "snapshot", "data": {"events": {"state": {"type": "state",
                                                               "data": {"state": "listening"}}}}})
    assert s.frame(0.0).state == "listening"


def test_tour_stages_every_state():
    from homeai.avatar.tour import TOUR, staged
    for name in TOUR:
        scene, t = staged(name)
        assert scene.frame(t).state == name, name
    with pytest.raises(ValueError):
        staged("dancing")


def test_tour_index_walks_states_then_ends():
    from homeai.avatar.tour import tour_index
    assert [tour_index(t, 2.0, 3) for t in (-1.0, 0.0, 1.99, 2.0, 5.9, 6.0)] == [0, 0, 0, 1, 2, None]
    with pytest.raises(ValueError):
        tour_index(1.0, 0.0)


# ---- app helpers -----------------------------------------------------------------

def test_prepare_env_defaults_to_xwayland_but_respects_caller():
    assert app.prepare_env({})["SDL_VIDEODRIVER"] == "x11"
    assert app.prepare_env({"SDL_VIDEODRIVER": "wayland"})["SDL_VIDEODRIVER"] == "wayland"


def test_geometry_round_trip_and_bad_files(tmp_path):
    p = tmp_path / "cfg" / "avatar.json"
    assert app.load_geometry(p) == app.DEFAULT_GEOMETRY
    app.save_geometry({"size": [800, 600], "position": [10, 20], "fullscreen": True}, p)
    assert app.load_geometry(p) == {"size": [800, 600], "position": [10, 20], "fullscreen": True}
    assert not list(p.parent.glob("*.tmp"))
    for bad in ("not json", "[1,2]", json.dumps({"size": [5, 5], "position": "x", "fullscreen": 1})):
        p.write_text(bad)
        assert app.load_geometry(p) == app.DEFAULT_GEOMETRY


def test_save_geometry_failure_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    app.save_geometry({"size": [1, 1]}, blocker / "sub" / "avatar.json")


def test_position_saved_without_title_bar_creep():
    # Asked for (100, 200), the window manager put it 37 px lower; the user then moved it.
    assert app.position_to_save([300, 437], [100, 200], [100, 237]) == [300, 400]
    assert app.position_to_save([300, 437], None, None) == [300, 437]


def test_frame_delay():
    assert app.frame_delay(60, 0.0, 0.001) == 0.0
    assert math.isclose(app.frame_delay(30, 0.0, 0.010), 1 / 30 - 0.010)
    assert app.frame_delay(10, 0.0, 0.5) == 0.0


def test_drain_applies_events_and_status():
    q = queue.SimpleQueue()
    q.put({"type": app.STATUS, "connected": True})
    q.put({"type": "state", "data": {"state": "listening"}})
    s = Scene()
    assert app.drain(q, s) == 2
    assert s.connected and s.state == "listening"
    q.put({"type": app.STATUS, "connected": False})
    app.drain(q, s)
    assert s.state == "offline"


def test_lock_held_reads_proc_locks_by_inode(tmp_path):
    lock = tmp_path / "homeai-mode.lock"
    lock.write_text("")
    ino = lock.stat().st_ino
    locks = tmp_path / "locks"
    locks.write_text(f"1: POSIX  ADVISORY  WRITE 9 103:02:{ino} 0 EOF\n"
                     f"2: FLOCK  ADVISORY  WRITE 9 103:02:{ino + 1} 0 EOF\n")
    assert not app.lock_held(lock, locks)           # wrong kind, wrong inode
    locks.write_text(locks.read_text() + f"3: FLOCK  ADVISORY  WRITE 9 103:02:{ino} 0 EOF\n")
    assert app.lock_held(lock, locks)
    assert not app.lock_held(tmp_path / "missing", locks)
    assert not app.lock_held(lock, tmp_path / "no-proc")


def test_lock_held_sees_a_real_flock(tmp_path):
    import fcntl
    lock = tmp_path / "m.lock"
    with open(lock, "w") as fh:
        assert not app.lock_held(lock)
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert app.lock_held(lock)


def test_local_mode(tmp_path, monkeypatch):
    dropin, lock = tmp_path / "coding.conf", tmp_path / "lock"
    assert app.local_mode(dropin, lock) is None
    dropin.write_text("")
    assert app.local_mode(dropin, lock) == "coding"
    monkeypatch.setattr(app, "lock_held", lambda p: True)
    assert app.local_mode(dropin, lock) == "switching"


def test_build_scene_reads_sleep_settings(monkeypatch):
    monkeypatch.setenv("HOMEAI_AVATAR_SLEEP_MIN", "2")
    monkeypatch.setenv("HOMEAI_AVATAR_QUIET_HOURS", "22:00-07:00")
    s = app.build_scene()
    assert s.sleep_s == 120.0 and s.quiet is not None
    monkeypatch.setenv("HOMEAI_AVATAR_QUIET_HOURS", "")
    assert app.build_scene().quiet is None


# ---- headless GL -----------------------------------------------------------------

@pytest.fixture(scope="module")
def gl():
    pytest.importorskip("moderngl")
    from homeai.avatar import gl as glmod
    try:
        ctx, fbo = glmod.headless((240, 240))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no EGL: {exc}")
    yield glmod, ctx, fbo
    ctx.release()


def test_rendered_sphere_grows_with_openness(gl):
    glmod, ctx, fbo = gl
    r = glmod.SphereRenderer(ctx, points=4000)
    s = Scene(state="speaking")
    f = s.frame(0.0)
    sizes = []
    for openness in (0.0, 1.0):
        f.openness = openness
        f.radius = BASE_RADIUS + OPEN_RADIUS * openness
        r.draw(f, 0.0, fbo.size)
        px = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(240, 240, 3)
        sizes.append(glmod.lit_radius(px))
    r.release()
    assert sizes[0] == pytest.approx(BASE_RADIUS, abs=0.04)
    assert sizes[1] - sizes[0] == pytest.approx(OPEN_RADIUS, abs=0.03)


def test_lit_radius_pure():
    from homeai.avatar.gl import fibonacci_sphere, lit_radius
    px = np.zeros((100, 100, 3), dtype=np.uint8)
    assert lit_radius(px) == 0.0
    px[25:75, 40:60] = 200
    assert lit_radius(px) == 0.5
    with pytest.raises(ValueError):
        fibonacci_sphere(0)
    pts = fibonacci_sphere(500)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-5)


def test_point_attributes_deterministic_with_large_dot_share():
    from homeai.avatar.gl import point_attributes
    a = point_attributes(4000, seed=3)
    assert a.shape == (4000, 5) and a.dtype == np.float32
    assert np.array_equal(a, point_attributes(4000, seed=3))
    assert not np.array_equal(a, point_attributes(4000, seed=4))
    large = a[:, 4] >= 1.0
    assert 0.25 < large.mean() < 0.35
    assert a[:, 4].min() >= 0.6 and a[:, 4].max() <= 1.5


def test_captions_render_over_the_sphere(gl):
    from homeai.avatar.captions import Rasteriser
    glmod, ctx, fbo = gl
    r = glmod.SphereRenderer(ctx, points=2000)
    f = Scene(state="idle", connected=True).frame(0.0)
    r.draw(f, 0.0, fbo.size)
    plain = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(240, 240, 3)
    img = Rasteriser().image([("jarvis", "Hello there, Adam.", 1.0)], fbo.size)
    r.draw(f, 0.0, fbo.size, captions=img)
    shown = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(240, 240, 3)
    r.release()
    # fbo.read rows start at the bottom, like the texture: the caption is in the first rows.
    assert np.abs(shown[:40].astype(int) - plain[:40]).sum() > 1000
    assert np.array_equal(shown[100:140], plain[100:140])       # the sphere centre untouched
