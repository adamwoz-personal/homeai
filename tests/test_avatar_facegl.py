"""Face renderer's pure layout (homeai/avatar/facegl.py) and rig.Deformer: no GL needed."""
from __future__ import annotations

import numpy as np
import pytest

from homeai.avatar import facegl, rig
from tests.avatar_face import mesh


@pytest.fixture(scope="module")
def verts():
    return mesh()[0]


def test_deformer_matches_deform_and_reuses_its_buffer(verts):
    fields = rig.default_fields(verts)
    d = rig.Deformer(verts, fields)
    params = {"jaw_open": 0.8, "blink_l": 1.0, "gaze_x": -0.5}
    a = d(params)
    np.testing.assert_allclose(a, rig.deform(verts, fields, params), atol=1e-4)
    assert d({}) is a                                       # same array every frame
    np.testing.assert_allclose(a, verts, atol=1e-6)


def test_teeth_sit_between_the_inner_lips(verts):
    idx, w, layer = facegl.mouth_points(verts)
    assert len(idx) > 50 and set(np.unique(layer)) == {1.0, 2.0}
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    opened = rig.deform(verts, rig.default_fields(verts), {"jaw_open": 1.0})
    pts = np.einsum("nk,nkd->nd", w, opened[idx])
    top, bottom = opened[13, 1], opened[14, 1]
    assert (pts[:, 1] > top - 2).all() and (pts[:, 1] < bottom + 2).all()


def test_teeth_hidden_until_the_lips_part(verts):
    h = rig.face_height(verts)
    shut = verts.copy()
    shut[14, 1] = shut[13, 1]
    assert facegl.teeth_visibility(shut, 1.0, h) == 0.0
    wide = verts.copy()
    wide[14, 1] = wide[13, 1] + 0.05 * h
    assert facegl.teeth_visibility(wide, 1.0, h) == pytest.approx(1.0)
    assert facegl.teeth_visibility(wide, 0.0, h) == 0.0


def test_hair_layout():
    a = facegl.hair_attributes(10, 5, seed=2)
    assert a.shape == (50, 5)
    assert ((a[:, 1] >= 0) & (a[:, 1] < 1)).all()
    np.testing.assert_array_equal(a, facegl.hair_attributes(10, 5, seed=2))
    with pytest.raises(ValueError):
        facegl.hair_attributes(0, 5)


def test_view_fills_height_and_keeps_aspect():
    side = 1000.0
    px = facegl.to_pixels(np.array([[0, 0], [side, side], [side / 2, side / 2]]), side, (1600, 800))
    np.testing.assert_allclose(px[2], (800, 400))
    np.testing.assert_allclose(px[0], (400, 0))           # square image, centred, full height
    np.testing.assert_allclose(px[1], (1200, 800))


def test_point_spacing_of_a_grid():
    g = np.stack(np.meshgrid(np.arange(30) * 4.0, np.arange(30) * 4.0), -1).reshape(-1, 2)
    assert facegl.point_spacing(g) == pytest.approx(4.0)


def test_aura_roots_surround_the_face(verts):
    geo = facegl.head_geometry(verts)
    oval = verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"])]
    e = ((oval - geo["hc"]) / geo["radii"]) ** 2
    assert (e.sum(axis=1) < 1.0).mean() > 0.8               # face mostly inside the hairline
