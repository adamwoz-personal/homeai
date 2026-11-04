"""The face build end to end (detection faked) and the landmark editor's helpers (AVATAR_PLAN §5)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from homeai.avatar import assets, facebuild  # noqa: E402
from tests.avatar_face import landmarks  # noqa: E402

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "avatar"


def _load(name):
    import sys

    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                                # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_detect(monkeypatch, tmp_path):
    """A gradient 'face' image with the fixture landmarks; returns the image path."""
    lm, side = landmarks()
    yy, xx = np.mgrid[0:side, 0:side]
    lum = np.clip(1 - np.hypot(xx - side / 2, yy - side / 2) / (side / 2), 0, 1)
    crop = (np.dstack([lum * 0.4, lum, lum * 0.5]) * 255).astype(np.uint8)
    cut = {"x0": 320, "y0": 0, "side": side}
    monkeypatch.setattr(facebuild, "load_crop", lambda image: (crop, cut))
    monkeypatch.setattr(facebuild, "detect_landmarks", lambda rgb: lm.copy())
    image = tmp_path / "art.jpg"
    image.write_bytes(b"art")
    return image


def test_build_writes_a_loadable_reproducible_version(fake_detect, tmp_path):
    kw = {"points": 8000, "iterations": 2}
    m = facebuild.build(fake_detect, tmp_path / "v1", **kw)
    a = assets.load(tmp_path / "v1")
    assert m["counts"]["points"] == len(a.xy) == 8000
    assert {p.name for p in (tmp_path / "v1").iterdir()} == {
        "rig.json", "points.npz", "build.json", "preview.png", "source.png"}
    np.testing.assert_allclose(a.points_for({}), a.xy, atol=0.05)
    facebuild.build(fake_detect, tmp_path / "again", **kw)
    assert m["files"] == json.loads((tmp_path / "again" / "build.json").read_text())["files"]


def test_build_refuses_overwrite_bad_counts_and_foreign_overrides(fake_detect, tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "rig.json").write_text("{}")
    with pytest.raises(facebuild.BuildError, match="never overwritten"):
        facebuild.build(fake_detect, tmp_path / "v1")
    for n in (7999, 20001):
        with pytest.raises(facebuild.BuildError, match="points"):
            facebuild.build(fake_detect, tmp_path / "v2", points=n)
    foreign = {"crop": {"x0": 0, "y0": 0, "side": 768}, "landmarks": {}}
    with pytest.raises(facebuild.BuildError, match="another|crop"):
        facebuild.build(fake_detect, tmp_path / "v3", overrides=foreign, points=8000, iterations=1)
    assert not (tmp_path / "v3").exists()


def test_build_applies_overrides_made_on_the_same_crop(fake_detect, tmp_path):
    ov = {"crop": {"x0": 320, "y0": 0, "side": 768}, "landmarks": {"1": [400.0, 420.0]}}
    facebuild.build(fake_detect, tmp_path / "v1", overrides=ov, points=8000, iterations=1)
    assert assets.load(tmp_path / "v1").verts[1].tolist() == [400.0, 420.0]


# ---- landmark editor ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def editor():
    return _load("landmark_editor")


def test_fit_round_trips_and_centres(editor):
    fit = editor.Fit.of(768, 768, 900)
    pts = np.array([[0.0, 0], [768, 768], [100.5, 300.25]])
    np.testing.assert_allclose(fit.to_image(fit.to_screen(pts)), pts)
    wide = editor.Fit.of(1000, 500, 900)
    assert wide.ox == 0 and wide.oy == pytest.approx(225)


def test_pick_nearest_within_radius(editor):
    pts = np.array([[10.0, 10], [30, 10], [100, 100]])
    assert editor.pick(pts, (18, 10)) == 0
    assert editor.pick(pts, (22, 10)) == 1
    assert editor.pick(pts, (60, 60)) is None
    assert editor.pick(np.zeros((0, 2)), (0, 0)) is None


def test_overrides_keep_only_moved_landmarks_and_round_trip(editor, tmp_path):
    lm, _ = landmarks()
    edited = lm.copy()
    edited[13] += (3, -2)
    edited[14] += (0.1, 0)                                   # jitter: not saved
    crop = {"x0": 320, "y0": 0, "side": 768}
    data = editor.overrides(lm, edited, "abc", crop)
    assert list(data["landmarks"]) == ["13"] and data["crop"] == crop
    path = tmp_path / "ov.json"
    path.write_text(json.dumps(data))
    back = editor.load_existing(path, lm, crop)
    np.testing.assert_allclose(back[13], edited[13], atol=0.01)
    np.testing.assert_array_equal(editor.load_existing(path, lm, {**crop, "x0": 0}), lm)
    np.testing.assert_array_equal(editor.load_existing(tmp_path / "none.json", lm, crop), lm)
