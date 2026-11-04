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
