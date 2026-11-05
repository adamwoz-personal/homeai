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


def test_speech_pulse_stays_bright_far_down_the_strands(pulse_asset):
    """The strand's own fade must not swallow the pulse before it gets anywhere."""
    delays = (0.3, 0.6)
    silent, lit, rest, arc = _shots(pulse_asset, delays)
    near, far = (tour.pulse_light(lit[d], silent) for d in delays)
    assert far > 1.1 * near


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
