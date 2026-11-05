"""Avatar golden images: every shown state renders as recorded (AVATAR_PLAN §10 Phase 2).

Regenerate deliberately with ``tools/avatar/update_golden.py`` after changing the look.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from homeai.avatar import tour

GOLDEN = Path(__file__).parent / "golden" / "avatar"


@pytest.fixture(scope="module")
def renders():
    pytest.importorskip("moderngl")
    Image = pytest.importorskip("PIL.Image")
    try:
        out = dict(tour.render_states(tour.TOUR, tour.GOLDEN_SIZE, tour.GOLDEN_POINTS))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no headless GL: {exc}")
    return out, Image


@pytest.mark.parametrize("state", tour.TOUR)
def test_state_matches_golden(renders, state):
    out, Image = renders
    golden = np.asarray(Image.open(GOLDEN / f"{state}.png").convert("RGB"))
    result = tour.compare(golden, out[state])
    assert result["ok"], result


def test_states_are_distinguishable(renders):
    out, _ = renders
    for a, b in combinations(tour.TOUR, 2):
        assert tour.compare(out[a], out[b])["mean"] > 1.0, (a, b)


def test_compare_tolerance():
    a = np.zeros((20, 20, 3), dtype=np.uint8)
    assert tour.compare(a, a) == {"ok": True, "mean": 0.0, "bad_share": 0.0}
    b = a.copy()
    b[0, :4, 0] = 60                                # 1% of pixels well off: allowed
    assert tour.compare(a, b)["ok"]
    b[0, 4, 0] = 60                                 # 1.25%: too many, though the mean is tiny
    assert not tour.compare(a, b)["ok"]
    assert not tour.compare(a, a + 3)["ok"]         # everything slightly off: mean too high
    assert not tour.compare(a, np.zeros((5, 5, 3), dtype=np.uint8))["ok"]


# ---- face look (Phase 4): one frame per viseme class, blink, sleep, listening, ... ----

@pytest.fixture(scope="module")
def face_renders():
    pytest.importorskip("moderngl")
    Image = pytest.importorskip("PIL.Image")
    try:
        out = dict(tour.render_face_golden())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no headless GL: {exc}")
    return out, Image


@pytest.mark.parametrize("pose", tour.FACE_TOUR)
def test_face_pose_matches_golden(face_renders, pose):
    out, Image = face_renders
    golden = np.asarray(Image.open(GOLDEN / "face" / f"{pose}.png").convert("RGB"))
    result = tour.compare(golden, out[pose])
    assert result["ok"], result


@pytest.mark.parametrize("a,b", tour.FACE_DISTINCT)
def test_face_poses_are_distinguishable(face_renders, a, b):
    out, _ = face_renders
    assert tour.compare(out[a], out[b])["mean"] > 0.3, (a, b)


def test_staged_face_rejects_unknown_pose():
    with pytest.raises(ValueError):
        tour.staged_face("yawn")


def _eye_box(rgb):
    """Mean RGB inside both eyes' boxes (synthetic face, rest geometry) and of the whole frame."""
    import json

    from homeai.avatar import facegl, rig

    data = json.loads((Path(__file__).parents[1] / tour.FACE_FIXTURE).read_text())
    v = np.array(data["landmarks"], dtype=np.float64)
    side = float(data["image_size"][0])
    box = []
    for region in ("eye_r", "eye_l"):
        px = facegl.to_pixels(v[list(rig.REGIONS[region])], side, tour.FACE_GOLDEN_SIZE, facegl.ZOOM)
        (x0, y0), (x1, y1) = px.min(axis=0).astype(int), px.max(axis=0).astype(int) + 1
        box.append(rgb[y0:y1, x0:x1].reshape(-1, 3))
    return np.vstack(box).astype(float).mean(axis=0), rgb.reshape(-1, 3).astype(float).mean(axis=0)


def test_sleeping_eyes_are_shut_not_slits(face_renders):
    out, _ = face_renders
    rest_eye, rest_all = _eye_box(out["rest"])
    sleep_eye, sleep_all = _eye_box(out["sleep"])
    # Dimming the whole face isn't enough: the eyes must go dark relative to the rest of it.
    assert sleep_eye.sum() / sleep_all.sum() < 0.6 * rest_eye.sum() / rest_all.sum()


def test_eyes_take_the_states_colour(face_renders):
    out, _ = face_renders
    rest, _ = _eye_box(out["rest"])                                       # staged speaking: cyan
    thinking, _ = _eye_box(out["thinking"])
    assert thinking[0] / thinking[1] > rest[0] / rest[1] + 0.05           # thinking's lavender


@pytest.fixture(scope="module")
def pulse_asset():
    pytest.importorskip("moderngl")
    return tour.golden_asset()


def _shots(asset, delays, low=0.0):
    try:
        return tour.render_pulse(asset, delays, tour.FACE_GOLDEN_SIZE, low=low)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no headless GL: {exc}")


def test_speech_pulse_travels_down_the_hair(pulse_asset):
    """Adam: flash the roots and let it travel, not the whole hair at once."""
    delays = (0.1, 0.3, 0.6)
    silent, lit, rest, arc = _shots(pulse_asset, delays)
    places = [tour.pulse_arc(lit[d], silent, rest, arc)[0] for d in delays]
    assert places[0] < places[1] < places[2]
    assert places[2] > 3 * places[0]


def test_speech_pulse_finishes_and_leaves_the_hair_quiet(pulse_asset):
    silent, lit, rest, arc = _shots(pulse_asset, (3.0,), low=1.0)
    assert tour.pulse_arc(lit[3.0], silent, rest, arc)[1] == 0.0


def test_speech_pulse_stays_bright_far_down_the_strands():
    """The strand's own fade must not swallow the pulse: the light speech adds, read from the
    shader at fixed points on one strand, so it doesn't depend on the hair's layout."""
    pytest.importorskip("moderngl")
    import moderngl

    from homeai.avatar import facegl, gl

    try:
        ctx, _ = gl.headless((8, 8))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no headless GL: {exc}")
    prog = ctx.program(vertex_shader=facegl.HAIR_VERTEX, varyings=["c"])
    for name, value in {"view": (1.0, 1.0, 0.0, 0.0), "colour": (1.0, 1.0, 1.0), "glow": 0.5, "ring": -1.0,
                        "face_h": 100.0, "pulse": 1}.items():
        facegl.FaceRenderer._set(prog, name, value)
    along = (0.2, 0.8)
    layout = np.array([[0, 0, 1, 100, s, 0, 0.8, 1, 0, 1, 0] for s in along], dtype=np.float32)
    vao = ctx.vertex_array(prog, [(ctx.buffer(layout.tobytes()), "2f 1f 1f 1f 1f 1f 2f 1f 1f", "root", "side", "len",
                                   "s", "phase", "size", "aim", "droop", "bend")])

    def light(speech):
        ring = np.tile(np.array(speech, dtype="f4"), (facegl.HAIR_PULSE_SAMPLES, 1))
        tex = ctx.texture((facegl.HAIR_PULSE_SAMPLES, 1), 4, ring.tobytes(), dtype="f4")
        tex.filter = moderngl.NEAREST, moderngl.NEAREST
        tex.use(1)
        out = ctx.buffer(reserve=len(along) * 12)
        vao.transform(out, moderngl.POINTS)
        return np.frombuffer(out.read(), dtype=np.float32).reshape(-1, 3)[:, 0]

    gain = light((0.0, 1.0, 1.0, 0.0)) - light((0.0, 0.0, 0.0, 0.0))
    ctx.release()
    assert gain[0] > 0
    assert gain[1] > 0.5 * gain[0]                          # 0.62 as built, 0.22 if the fade swallows it


def test_face_poses_do_not_inherit_speech(pulse_asset, monkeypatch):
    from homeai.avatar import facegl

    heard = []
    draw = facegl.FaceRenderer.draw

    def spy(self, frame, *args, **kwargs):
        heard.append(bool(self.pulse.ring.any()))
        return draw(self, frame, *args, **kwargs)

    monkeypatch.setattr(facegl.FaceRenderer, "draw", spy)
    try:
        list(tour.render_face(pulse_asset, ("pulse", "rest"), tour.FACE_GOLDEN_SIZE))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no headless GL: {exc}")
    assert heard == [True, False]


def test_hair_shader_matches_its_cpu_path(pulse_asset):
    """hair_rest and hair_wind are the shader's path on the CPU, for tests and probes:
    read the positions back from the GPU so they can't drift apart."""
    import moderngl

    from homeai.avatar import facegl, gl

    asset = pulse_asset
    layout = facegl.hair_layout(facegl.lit_points(asset.xy, asset.colour), asset.verts, facegl.point_spacing(asset.xy))
    face_h, phase, amp, swirl, wind = facegl.head_geometry(asset.verts)["face_h"], 5.0, 1.3, 0.4, facegl.HAIR_WIND
    try:
        ctx, _ = gl.headless((8, 8))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no headless GL: {exc}")
    prog = ctx.program(vertex_shader=facegl.HAIR_VERTEX, varyings=["gl_Position"])
    for name, value in {"spread": facegl.HAIR_SPREAD, "drop": facegl.HAIR_DROP, "gravity": facegl.HAIR_GRAVITY,
                        "view": (1.0, 1.0, 0.0, 0.0), "pivot": (0.0, 0.0), "depth": 0.0, "head": (0.0, 0.0, 0.0),
                        "hair_phase": phase, "hair_amp": amp, "swirl": swirl, "face_h": face_h,
                        "hair_wind_amount": wind, "pulse": 1}.items():
        facegl.FaceRenderer._set(prog, name, value)
    silence = ctx.texture((facegl.HAIR_PULSE_SAMPLES, 1), 4, np.zeros((facegl.HAIR_PULSE_SAMPLES, 4), "f4").tobytes(),
                          dtype="f4")
    silence.filter = moderngl.NEAREST, moderngl.NEAREST
    silence.use(1)
    vbo = ctx.buffer(layout.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f 1f 1f 1f 1f 1f 2f 1f 1f", "root", "side", "len", "s", "phase", "size",
                                   "aim", "droop", "bend")])
    out = ctx.buffer(reserve=len(layout) * 16)
    vao.transform(out, moderngl.POINTS)
    got = np.frombuffer(out.read(), dtype=np.float32).reshape(-1, 4)[:, :2] * (1.0, -1.0)
    ctx.release()

    ln, s, strand = layout[:, 3:4].astype(float), layout[:, 4:5].astype(float), layout[:, 5:6].astype(float)
    rest = facegl.hair_rest(layout)
    want = rest + ln * wind * amp * s ** 1.5 * facegl.hair_wind(rest / face_h, phase)
    want[:, :1] += ln * 0.05 * np.sin(swirl + strand) * s
    np.testing.assert_allclose(got, want, atol=0.05)


def _opened(asset, monkeypatch, rings):
    """Face points (shader's positions, CPU's positions) with the jaw open, and at rest, with ``rings`` marked."""
    import moderngl

    from homeai.avatar import facegl, gl, rig

    monkeypatch.setattr(facegl, "MOUTH_RINGS", {None: rings})
    idx, attrs = facegl.face_buffers(asset)
    rest = np.asarray(asset.verts, dtype=np.float32)
    opened = rig.deform(rest, rig.default_fields(rest), {"jaw_open": 1.0}).astype(np.float32)
    try:
        ctx, _ = gl.headless((8, 8))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no headless GL: {exc}")
    prog = ctx.program(vertex_shader=facegl.FACE_VERTEX, varyings=["gl_Position", "c"])     # c keeps every input live
    for name, value in {"view": (1.0, 1.0, 0.0, 0.0), "pivot": (0.0, 0.0), "depth": 0.0, "head": (0.0, 0.0, 0.0),
                        "shimmer": 0.0, "verts": 0}.items():
        facegl.FaceRenderer._set(prog, name, value)
    tex = ctx.texture((len(rest), 1), 2, opened.tobytes(), dtype="f4")
    tex.filter = moderngl.NEAREST, moderngl.NEAREST
    tex.use(0)
    vao = ctx.vertex_array(prog, [(ctx.buffer(idx.tobytes()), "3i", "idx"),
                                  (ctx.buffer(attrs.tobytes()), "3f 3f 1f 1f 1f 2f", "w", "col", "size", "phase",
                                   "layer", "rigid")])
    out = ctx.buffer(reserve=len(idx) * 28)
    vao.transform(out, moderngl.POINTS)
    got = np.frombuffer(out.read(), dtype=np.float32).reshape(-1, 7)[:, :2] * (1.0, -1.0)
    ctx.release()
    return got, facegl.face_positions(idx, attrs, opened), facegl.face_positions(idx, attrs, rest)


def test_mouth_rings_keep_their_shape_as_the_jaw_opens(pulse_asset, monkeypatch):
    """Adam: the open mouth stretches the circles beside it. Read back from the shader: a marked
    ring only moves, and the same dots unmarked do stretch, so the check can fail."""
    from homeai.avatar import rig

    v = np.asarray(pulse_asset.verts, dtype=np.float64)
    h = rig.face_height(v)
    centre, radii = (float(v[61, 0] - 0.06 * h), float(v[61, 1])), (0.05 * h, 0.05 * h)
    n = len(pulse_asset.xy)
    ring = ((pulse_asset.xy - centre) ** 2).sum(axis=1) <= radii[0] ** 2
    assert ring.sum() > 10

    got, cpu, rest = _opened(pulse_asset, monkeypatch, ((centre, radii),))
    np.testing.assert_allclose(got, cpu, atol=0.05)
    np.testing.assert_allclose(rest[:n], pulse_asset.xy, atol=0.05)     # marking a ring doesn't move it at rest
    assert np.ptp((got[:n] - rest[:n])[ring], axis=0).max() < 0.05

    loose, _, rest = _opened(pulse_asset, monkeypatch, ())
    assert np.ptp((loose[:n] - rest[:n])[ring], axis=0).max() > 0.02 * rig.face_height(v)


def _eye_readback(asset, gaze):
    """(positions, colours) of every face dot from the real shader, at ``gaze``, and the buffers."""
    import moderngl

    from homeai.avatar import facegl, gl, rig

    idx, attrs = facegl.face_buffers(asset)
    rest = np.asarray(asset.verts, dtype=np.float32)
    moved = rig.deform(rest, rig.default_fields(rest), {"gaze_x": gaze[0], "gaze_y": gaze[1]}).astype(np.float32)
    windows = facegl.eye_windows(rest, asset.xy, attrs[:len(asset.xy), 8])
    try:
        ctx, _ = gl.headless((8, 8))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no headless GL: {exc}")
    prog = ctx.program(vertex_shader=facegl.FACE_VERTEX, varyings=["gl_Position", "c"])
    for name, value in {"view": (1.0, 1.0, 0.0, 0.0), "pivot": (0.0, 0.0), "depth": 0.0, "head": (0.0, 0.0, 0.0),
                        "shimmer": 0.0, "verts": 0, "bright": 1.0, "teeth": 1.0, "lids": (0.0, 0.0),
                        "eye_tint": (1.0, 1.0, 1.0, 0.0), "eye_window_r": windows[0],
                        "eye_window_l": windows[1]}.items():
        facegl.FaceRenderer._set(prog, name, value)
    tex = ctx.texture((len(rest), 1), 2, moved.tobytes(), dtype="f4")
    tex.filter = moderngl.NEAREST, moderngl.NEAREST
    tex.use(0)
    vao = ctx.vertex_array(prog, [(ctx.buffer(idx.tobytes()), "3i", "idx"),
                                  (ctx.buffer(attrs.tobytes()), "3f 3f 1f 1f 1f 2f", "w", "col", "size", "phase",
                                   "layer", "rigid")])
    out = ctx.buffer(reserve=len(idx) * 28)
    vao.transform(out, moderngl.POINTS)
    got = np.frombuffer(out.read(), dtype=np.float32).reshape(-1, 7).copy()
    ctx.release()
    return got[:, :2] * (1.0, -1.0), got[:, 4:7].sum(axis=1), attrs


def test_the_iris_keeps_its_pattern_at_full_gaze(pulse_asset):
    """Adam: the eyes broke up when they looked far to the side. Dots of an iris all move together."""
    from homeai.avatar import facegl

    still, _, attrs = _eye_readback(pulse_asset, (0.0, 0.0))
    looking, _, _ = _eye_readback(pulse_asset, (0.9, 0.9))
    for eye in facegl.EYE_LAYERS.values():
        iris = attrs[:, 8] == eye + facegl.IRIS_LAYER
        assert iris.sum() > 3
        shift = looking[iris] - still[iris]
        assert np.abs(shift).max() > 1.0 and np.ptp(shift, axis=0).max() < 0.05


def test_gaze_moves_nothing_but_the_irises(pulse_asset):
    from homeai.avatar import facegl

    still, _, attrs = _eye_readback(pulse_asset, (0.0, 0.0))
    looking, _, _ = _eye_readback(pulse_asset, (0.9, 0.9))
    others = np.modf(attrs[:, 8])[0] != facegl.IRIS_LAYER
    assert np.abs(looking[others] - still[others]).max() < 0.05


def test_rebinding_leaves_every_dot_where_it_was_at_rest(pulse_asset):
    still, _, _ = _eye_readback(pulse_asset, (0.0, 0.0))
    np.testing.assert_allclose(still[:len(pulse_asset.xy)], pulse_asset.xy, atol=0.05)


def test_the_iris_tucks_under_the_lids(pulse_asset):
    """Adam: the eyes overlapped the eyelids. Whole at rest, dimmed where it moves past the window."""
    from homeai.avatar import facegl

    _, rest_light, attrs = _eye_readback(pulse_asset, (0.0, 0.0))
    _, light, _ = _eye_readback(pulse_asset, (0.0, 0.9))
    iris = np.modf(attrs[:, 8])[0] == facegl.IRIS_LAYER
    expected = attrs[iris, 3:6].sum(axis=1) * 0.85          # the shader's brightness with no glow
    np.testing.assert_allclose(rest_light[iris], expected, rtol=1e-3)
    assert (light[iris] < 0.01).any()
