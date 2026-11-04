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


def test_hair_hangs_from_the_sides_not_the_crown(verts):
    xy = verts.astype(np.float64)
    spacing = facegl.point_spacing(xy)
    a = facegl.hair_layout(xy, verts, spacing, strands=20, points=400, seed=2)
    assert a.shape[1] == 7 and len(a) >= 400 * 0.9
    np.testing.assert_array_equal(a, facegl.hair_layout(xy, verts, spacing, strands=20, points=400, seed=2))
    top, bottom = min(verts[105, 1], verts[334, 1]), max(verts[172, 1], verts[397, 1])
    assert (a[:, 1] >= top - 1e-3).all() and (a[:, 1] <= bottom + 1e-3).all()   # no crown roots
    assert set(np.unique(a[:, 2])) == {-1.0, 1.0}
    for side in (-1, 1):                                   # roots sit just outside the art, never on it
        r = a[a[:, 2] == side]
        edge = np.array([facegl.silhouette_edge(xy, y, side, 2 * spacing,
                                                centre=verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean(),
                                                gap=facegl.HAIR_GAP * rig.face_height(verts))
                         for y in r[:, 1]])
        gap = (r[:, 0] - edge) * side
        assert (gap > 0).all() and (gap < 2 * spacing).all()
    strands = np.unique(a[:, [1, 3]], axis=0)              # (root y, length) per strand
    third = len(strands) // 3
    by_y = strands[np.argsort(strands[:, 0])]
    assert by_y[-third:, 1].mean() > 3 * by_y[:third, 1].mean()   # short above the ears, long below
    assert ((a[:, 4] >= 0) & (a[:, 4] < 1)).all()


def test_hair_layout_rejects_bad_input(verts):
    with pytest.raises(ValueError):
        facegl.hair_layout(verts, verts, 1.0, strands=1)
    with pytest.raises(ValueError):
        facegl.hair_layout(np.zeros((0, 2)), verts, 1.0, strands=10, points=100)


def test_lit_points_drop_the_dark_surround_unless_nothing_is_lit():
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    colour = np.array([[0.5, 0.2, 0.1], [0.05, 0.05, 0.05], [0.0, 0.3, 0.0]])
    np.testing.assert_array_equal(facegl.lit_points(xy, colour), xy[[0, 2]])
    np.testing.assert_array_equal(facegl.lit_points(xy, colour * 0), xy)


def test_silhouette_edge():
    xy = np.array([[10.0, 5.0], [50.0, 5.0], [30.0, 40.0]])
    assert facegl.silhouette_edge(xy, 6.0, -1, 2.0, gap=100) == 10.0
    assert facegl.silhouette_edge(xy, 6.0, 1, 2.0, gap=100) == 50.0
    assert facegl.silhouette_edge(xy, 20.0, 1, 2.0) is None
    row = np.column_stack([np.r_[np.arange(40, 61, 2.0), 90.0], np.zeros(12)])   # head 40..60, stray dot at 90
    assert facegl.silhouette_edge(row, 0.0, 1, 1.0, centre=50.0, gap=5.0) == 60.0
    assert facegl.silhouette_edge(row, 0.0, -1, 1.0, centre=50.0, gap=5.0) == 40.0


def test_view_fills_height_and_keeps_aspect():
    side = 1000.0
    px = facegl.to_pixels(np.array([[0, 0], [side, side], [side / 2, side / 2]]), side, (1600, 800))
    np.testing.assert_allclose(px[2], (800, 400))
    np.testing.assert_allclose(px[0], (400, 0))           # square image, centred, full height
    np.testing.assert_allclose(px[1], (1200, 800))


def test_point_spacing_of_a_grid():
    g = np.stack(np.meshgrid(np.arange(30) * 4.0, np.arange(30) * 4.0), -1).reshape(-1, 2)
    assert facegl.point_spacing(g) == pytest.approx(4.0)
