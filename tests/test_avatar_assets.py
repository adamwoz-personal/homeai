"""Face assets: load/validate, hashes, versions, atomic current/previous switching (AVATAR_PLAN §5)."""
import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from homeai.avatar import assets  # noqa: E402
from tests.avatar_face import make_asset  # noqa: E402


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "avatar"
    monkeypatch.setenv("HOMEAI_AVATAR_ASSETS", str(r))
    return r


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return make_asset(tmp_path_factory.mktemp("face") / "v1")


def _copy(src, dst):
    import shutil
    shutil.copytree(src, dst)
    return dst


def test_round_trip_and_points_follow_the_rig(built):
    a = assets.load(built)
    assert a.version == "v1" and len(a.xy) == 600
    np.testing.assert_allclose(a.points_for({}), a.xy, atol=0.05)
    opened = a.points_for({"jaw_open": 1.0})
    below_mouth = a.xy[:, 1] > a.verts[14, 1] + 5
    on_face = below_mouth & (np.abs(a.xy[:, 0] - a.verts[14, 0]) < 20) & (a.xy[:, 1] < a.verts[152, 1])
    assert on_face.any()
    assert (opened[on_face, 1] > a.xy[on_face, 1] + 1).all()


def test_edited_file_fails_the_hash_check(built, tmp_path):
    v = _copy(built, tmp_path / "v1")
    rig_json = json.loads((v / "rig.json").read_text())
    rig_json["vertices"][0][0] += 1
    (v / "rig.json").write_text(json.dumps(rig_json))
    with pytest.raises(assets.AssetError, match="does not match"):
        assets.load(v)
    assets.load(v, verify_hashes=False)


def test_inconsistent_points_fail_validation(built):
    a = assets.load(built)
    a.tri = a.tri.copy()
    a.tri[0] = len(a.triangles)
    with pytest.raises(assets.AssetError, match="triangle"):
        assets.validate(a)
    a = assets.load(built)
    a.bary = a.bary * 2
    with pytest.raises(assets.AssetError, match="sum to 1"):
        assets.validate(a)
    a = assets.load(built)
    a.fields["jaw_open"] = a.fields["jaw_open"][:-1]
    with pytest.raises(assets.AssetError, match="shape"):
        assets.validate(a)


def test_missing_or_wrong_format_is_an_asset_error(built, tmp_path):
    with pytest.raises(assets.AssetError, match="unreadable"):
        assets.load(tmp_path / "nothing")
    v = _copy(built, tmp_path / "v2")
    b = json.loads((v / "build.json").read_text())
    b["format"] = 99
    (v / "build.json").write_text(json.dumps(b))
    with pytest.raises(assets.AssetError, match="format"):
        assets.load(v)


def test_versions_sort_numerically_and_next_version(root, built):
    assert assets.versions(root) == [] and assets.next_version(root) == "v1"
    for name in ("v1", "v2", "v10"):
        _copy(built, root / name)
    (root / "scratch").mkdir()
    assert assets.versions(root) == ["v1", "v2", "v10"]
    assert assets.next_version(root) == "v11"


def test_use_swaps_current_and_previous_and_rollback_returns(root, built):
    _copy(built, root / "v1")
    _copy(built, root / "v2")
    assert assets.load_current(root) is None
    assert assets.use(root, "v1") is None
    assert assets.use(root, "v2") == "v1"
    assert assets.current(root) == "v2" and (root / "previous").resolve().name == "v1"
    assert assets.load_current(root).version == "current"
    assert assets.rollback(root) == "v1"
    assert assets.current(root) == "v1" and assets._link(root, "previous") == "v2"
    assert assets.use(root, "v1") == "v1"                     # already current: nothing moves
    assert assets._link(root, "previous") == "v2"
    assert not list(root.glob(".*.tmp"))


def test_use_refuses_missing_or_broken_versions_and_keeps_current(root, built):
    _copy(built, root / "v1")
    assets.use(root, "v1")
    broken = _copy(built, root / "v2")
    (broken / "points.npz").write_bytes(b"junk")
    for bad in ("v2", "v9", "../etc", "current"):
        with pytest.raises(assets.AssetError):
            assets.use(root, bad)
    assert assets.current(root) == "v1"
    assert assets._link(root, "previous") is None


def test_rollback_with_nothing_before_fails(root):
    root.mkdir()
    with pytest.raises(assets.AssetError, match="nothing"):
        assets.rollback(root)


def test_cli(root, built, capsys):
    assert assets.main(["list"]) == 0
    assert "no faces built" in capsys.readouterr().out
    _copy(built, root / "v1")
    assert assets.main(["check"]) == 1
    assert assets.main(["use", "v1"]) == 0
    assert assets.main(["check"]) == 0
    assert "v1: OK (600 points" in capsys.readouterr().out
    assets.main(["list"])
    assert "* v1" in capsys.readouterr().out
    assert assets.main(["use", "v7"]) == 1
    assert "error: no such version" in capsys.readouterr().out
    assert assets.main(["rollback"]) == 1
