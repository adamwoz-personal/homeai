"""Face renderer's pure layout (homeai/avatar/facegl.py) and rig.Deformer: no GL needed."""
from __future__ import annotations

import numpy as np
import pytest

from homeai.avatar import facegl, rig
from tests.avatar_face import head_with_skull, mesh


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


def _neck(verts):
    """The face mesh plus a column of neck dots below it, as the real art has."""
    h = rig.face_height(verts)
    cx, jaw = verts[:, 0].mean(), max(verts[172, 1], verts[397, 1])
    g = np.stack(np.meshgrid(np.linspace(cx - 0.25 * h, cx + 0.25 * h, 25),
                             np.linspace(jaw, jaw + 0.8 * h, 40)), -1).reshape(-1, 2)
    return np.vstack([verts.astype(np.float64), g])


def test_hair_hangs_from_the_sides_and_down_the_neck(verts):
    xy = _neck(verts)
    spacing = facegl.point_spacing(verts)
    a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=0, points=400, seed=2)
    assert a.shape[1] == 10 and len(a) >= 400 * 0.9
    np.testing.assert_array_equal(a, facegl.hair_layout(xy, verts, spacing, strands=20, crown=0, points=400, seed=2))
    h = rig.face_height(verts)
    top, jaw = min(verts[105, 1], verts[334, 1]), max(verts[172, 1], verts[397, 1])
    assert (a[:, 1] >= top - 1e-3).all()                    # crown=0: nothing above the brows
    assert a[:, 1].max() > jaw + 0.6 * h                    # all the way down the neck
    assert set(np.unique(a[:, 2])) == {-1.0, 1.0}
    np.testing.assert_array_equal(a[:, 7], a[:, 2])         # side strands aim straight out
    assert (a[:, 8] == 0).all()
    order = np.argsort(a[:, 1])                            # less gravity up high: frames the face
    assert (np.diff(a[order, 9]) >= -1e-6).all() and a[order[0], 9] < 0.85
    assert (a[a[:, 1] > jaw, 9] == 1.0).all()
    for side in (-1, 1):                                   # roots sit just outside the art, never on it
        r = a[a[:, 2] == side]
        edge = np.array([facegl.silhouette_edge(xy, y, side, 2 * spacing,
                                                centre=verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean(),
                                                gap=facegl.HAIR_GAP * h)
                         for y in r[:, 1]])
        gap = (r[:, 0] - edge) * side
        assert (gap > 0).all() and (gap < 2 * spacing).all()
    strands = np.unique(a[:, [1, 3]], axis=0)              # (root y, length) per strand
    above = strands[strands[:, 0] < jaw]
    third = len(above) // 3
    by_y = above[np.argsort(above[:, 0])]
    assert by_y[-third:, 1].mean() > 3 * by_y[:third, 1].mean()   # short above the ears, long below
    near_jaw = strands[np.abs(strands[:, 0] - jaw) < 0.15 * h, 1].mean()
    neck_end = strands[strands[:, 0] > jaw + 0.6 * h, 1].mean()
    assert neck_end < 0.7 * near_jaw                       # shorter low on the neck: the rest is off screen
    assert ((a[:, 4] >= 0) & (a[:, 4] < 1)).all()


def test_crown_hair_is_long_falls_beside_the_face_and_meets_the_sides(verts):
    xy = head_with_skull(verts)                            # the head alone, without the mesh's frame ring
    spacing = facegl.point_spacing(xy)
    h = rig.face_height(verts)
    a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=40, seed=2)
    top = min(verts[105, 1], verts[334, 1])
    origin = np.array([verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean(), top])
    is_crown = a[:, 1] < top - 1e-3
    crown = np.unique(a[is_crown][:, [0, 1, 3]], axis=0)
    assert len(crown) > 30
    assert crown[:, 2].mean() > h                          # as long as the side hair, not tufts
    out = crown[:, :2] - origin
    for row in out:                                        # roots just outside the skull's outline
        d = np.linalg.norm(row)
        edge = facegl.ray_edge(xy, origin, row / d, 2 * spacing, facegl.HAIR_GAP * h)
        assert 0 < d - edge < 2 * spacing
    assert crown[:, 1].min() < verts[10, 1]                # over the top of the head
    for side in (-1, 1):                                   # no gap between crown and side hair
        s = crown[np.sign(crown[:, 0] - origin[0]) == side]
        assert s[:, 1].max() > top - 0.05 * h
    rest = facegl.hair_rest(a[is_crown])
    assert facegl.inside_outline(xy, verts, spacing, rest).mean() < 0.01   # beside the face, not over it


def test_inside_outline():
    xy = np.stack(np.meshgrid(np.arange(0, 101, 4.0), np.arange(0, 101, 4.0)), -1).reshape(-1, 2)
    verts = np.zeros((478, 2))
    verts[[105, 334], 1] = 50.0                            # brow line across the middle
    verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0] = 50.0
    verts[10, 1], verts[152, 1] = 0.0, 100.0               # face height 100
    p = np.array([[50.0, 20.0], [50.0, -20.0], [50.0, 80.0], [130.0, 80.0], [-30.0, 30.0]])
    np.testing.assert_array_equal(facegl.inside_outline(xy, verts, 4.0, p), [True, False, True, False, False])


def test_hair_points_follow_strand_length_by_default(verts):
    xy = verts.astype(np.float64)
    spacing = facegl.point_spacing(xy)
    a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=10)
    total = np.unique(a[:, [0, 1, 3]], axis=0)[:, 2].sum()
    assert len(a) == pytest.approx(facegl.HAIR_DENSITY * total / rig.face_height(verts), rel=0.1)


def test_hair_layout_rejects_bad_input(verts):
    with pytest.raises(ValueError):
        facegl.hair_layout(verts, verts, 1.0, strands=1)
    with pytest.raises(ValueError):
        facegl.hair_layout(np.zeros((0, 2)), verts, 1.0, strands=10, crown=0, points=100)


def test_ray_edge():
    line = np.column_stack([np.zeros(11), -np.arange(0, 21, 2.0)])          # dots straight up 0..20
    stray = np.vstack([line, [[0.0, -40.0], [5.0, -10.0]]])                # a stray beyond, one off the ray
    assert facegl.ray_edge(stray, (0, 0), (0, -1), 1.0, 5.0) == 20.0
    assert facegl.ray_edge(stray, (0, 0), (0, -1), 1.0, 100.0) == 40.0
    assert facegl.ray_edge(line, (0, 0), (0, 1), 1.0, 5.0) == 0.0          # behind the origin: only itself
    assert facegl.ray_edge(line + [50.0, 0], (0, 0), (0, -1), 1.0, 5.0) is None


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


def test_view_sits_the_image_on_the_window_bottom():
    side = 1000.0
    px = facegl.to_pixels(np.array([[0, 0], [side, side]]), side, (800, 800), zoom=0.8)
    np.testing.assert_allclose(px[:, 1], (160, 800))       # no black strip under the shoulders
    np.testing.assert_allclose(px[:, 0], (80, 720))


def test_point_spacing_of_a_grid():
    g = np.stack(np.meshgrid(np.arange(30) * 4.0, np.arange(30) * 4.0), -1).reshape(-1, 2)
    assert facegl.point_spacing(g) == pytest.approx(4.0)
