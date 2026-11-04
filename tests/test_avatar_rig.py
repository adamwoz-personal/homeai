"""Face rig and mesh: fields, deformation, mesh quality, binding (AVATAR_PLAN §4.5, §5)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from homeai.avatar import facebuild, rig  # noqa: E402
from tests.avatar_face import landmarks, mesh  # noqa: E402


@pytest.fixture(scope="module")
def face():
    verts, tris, side = mesh()
    return verts, tris, side, rig.default_fields(verts)


# ---- fields ------------------------------------------------------------------------

def test_default_fields_are_valid_for_a_real_face(face):
    verts, _, _, fields = face
    assert set(fields) == set(rig.FIELD_PARAMS)
    rig.validate_fields(fields, len(verts), rig.face_height(verts))
    for name, f in fields.items():
        assert np.abs(f).max() > 0, f"{name} moves nothing"


def test_every_region_is_valid():
    lm, _ = landmarks()
    rig.validate_regions({k: list(v) for k, v in rig.REGIONS.items()}, len(lm))


@pytest.mark.parametrize("regions, msg", [
    ({**rig.REGIONS, "jaw": []}, "no landmarks"),
    ({**rig.REGIONS, "nose": [1, 999]}, "out of range"),
])
def test_validate_regions_rejects(regions, msg):
    with pytest.raises(rig.RigError, match=msg):
        rig.validate_regions(regions, rig.N_FACE)


def _bad(fields, name, f):
    return {**fields, name: f}


def test_validate_fields_rejects_missing_shape_nonfinite_huge_and_ring(face):
    verts, _, _, fields = face
    v, h = len(verts), rig.face_height(verts)
    cases = [
        ({k: f for k, f in fields.items() if k != "blink_l"}, "missing"),
        (_bad(fields, "jaw_open", np.zeros((v - 1, 2))), "shape"),
        (_bad(fields, "jaw_open", np.full((v, 2), np.nan)), "finite"),
        (_bad(fields, "jaw_open", np.full((v, 2), h)), "face height"),
    ]
    ring = fields["jaw_open"].copy()
    ring[rig.N_FACE, 0] = 0.5
    cases.append((_bad(fields, "jaw_open", ring), "boundary ring"))
    for bad, msg in cases:
        with pytest.raises(rig.RigError, match=msg):
            rig.validate_fields(bad, v, h)


def test_default_fields_rejects_too_few_vertices():
    with pytest.raises(rig.RigError):
        rig.default_fields(np.zeros((100, 2)))


def test_jaw_open_moves_the_chin_down_not_the_brows(face):
    verts, _, _, fields = face
    f = fields["jaw_open"]
    h = rig.face_height(verts)
    assert f[152, 1] > 0.05 * h                               # chin
    gap = f[14, 1] - f[13, 1]                                 # inner lips part by most of the chin's drop
    assert gap > 0.8 * f[152, 1]
    assert f[list(rig.REGIONS["brow_r"]), :].max() == 0
    assert f[13, 1] == 0                                     # upper inner lip stays


def test_blink_closes_the_lids_and_gaze_moves_only_the_irises(face):
    verts, _, _, fields = face
    for side, upper, lower in (("r", 159, 145), ("l", 386, 374)):
        closed = rig.deform(verts, fields, {f"blink_{side}": 1.0})
        gap_open = verts[lower, 1] - verts[upper, 1]
        gap_closed = closed[lower, 1] - closed[upper, 1]
        assert gap_open > 0 and abs(gap_closed) < 0.2 * gap_open
    moved = np.nonzero(np.abs(fields["gaze_x"]).sum(1))[0]
    assert set(moved) == set(rig.REGIONS["iris_r"] + rig.REGIONS["iris_l"])


def test_deform_blends_linearly_and_ignores_unknown_params(face):
    verts, _, _, fields = face
    half = rig.deform(verts, fields, {"jaw_open": 0.5, "glow": 1.0, "head_yaw": 0.3})
    full = rig.deform(verts, fields, {"jaw_open": 1.0})
    np.testing.assert_allclose(half - verts, (full - verts) / 2, atol=1e-4)
    np.testing.assert_array_equal(rig.deform(verts, fields, {}), verts)


# ---- mesh and binding --------------------------------------------------------------------

def test_mesh_has_no_slit_or_sliver_triangles(face):
    verts, tris, _, _ = face
    assert not any(rig.spans_mouth(t, rig.face_regions(verts)) for t in tris)
    area, angle = facebuild.triangle_quality(verts, tris)
    assert area.min() >= facebuild.MIN_AREA and angle.min() >= facebuild.MIN_ANGLE
    used = set(np.unique(tris).tolist())
    assert set(range(rig.N_FACE, len(verts))) <= used                     # the ring holds the mesh
    assert set(range(rig.N_FACE)) - used <= rig.MOUTH_SLIT                # only inner lips may float


def test_lip_regions_find_the_mid_rings_on_the_right_side():
    lm, _ = landmarks()
    lips = rig.lip_regions(lm)
    assert {42, 74} <= set(lips["upper_lip"])                # rings the named contours miss
    assert not set(lips["upper_lip"]) & set(lips["lower_lip"])
    added = (set(lips["upper_lip"]) | set(lips["lower_lip"])) - set(
        rig.REGIONS["upper_lip"] + rig.REGIONS["lower_lip"])
    assert added and all(rig._inside(lm[[i]], lm[list(rig._LIPS_UPPER_OUTER)
                                                + list(reversed(rig._LIPS_LOWER_OUTER))])[0] for i in added)
    mid = lm[13, 1]
    assert all(lm[i, 1] < mid + 1 for i in lips["upper_lip"] if i in added)
    assert all(lm[i, 1] > mid - 1 for i in lips["lower_lip"] if i in added)


def test_inside_polygon():
    square = np.array([[0.0, 0], [10, 0], [10, 10], [0, 10]])
    pts = np.array([[5.0, 5], [11, 5], [-1, -1], [9.9, 0.1]])
    assert rig._inside(pts, square).tolist() == [True, False, False, True]


def test_mouth_opens_with_no_triangle_stretched_across_it(face):
    """With slit triangles kept, jaw_open would drag points into the open mouth."""
    verts, tris, _, fields = face
    opened = rig.deform(verts, fields, {"jaw_open": 1.0})
    mouth = rig.MOUTH_SLIT | set(rig.face_regions(verts)["upper_lip"]) | set(rig.face_regions(verts)["lower_lip"])
    for t in tris:
        if any(int(i) in mouth for i in t):
            ys = opened[t, 1]
            if set(t.tolist()) & rig.MOUTH_CORNERS:
                continue                                          # corners stretch, as lips do
            assert ys.max() - ys.min() < 0.12 * rig.face_height(verts), f"triangle {t.tolist()}"


def test_barycentric_is_exact_inside_and_outside():
    a, b, c = np.array([0.0, 0]), np.array([4.0, 0]), np.array([0.0, 4])
    for p in ([1.0, 1], [5.0, 5], [-2.0, 1]):
        w = facebuild.barycentric(np.array(p), a, b, c)
        assert w.sum() == pytest.approx(1)
        np.testing.assert_allclose(w[0] * a + w[1] * b + w[2] * c, p)
    assert facebuild.barycentric(np.array([1.0, 1]), a, b, c).min() > 0


def test_bind_reconstructs_rest_positions_and_prefers_containing_triangles(face):
    verts, tris, side, _ = face
    rng = np.random.default_rng(3)
    pts = np.vstack([rng.uniform(2, side - 2, (3000, 2)),
                     verts[[13, 14]].mean(0, keepdims=True)])  # in the dropped mouth slit
    tri, bary = facebuild.bind(pts, verts, tris)
    np.testing.assert_allclose(rig.positions(verts.astype(np.float64), tris, tri, bary), pts, atol=0.05)
    assert (bary.min(axis=1) >= -1e-4).mean() > 0.99          # nearly all bound inside a triangle


def test_square_crop_centres_and_clamps():
    assert facebuild.square_crop(1408, 768, 704) == (320, 0, 768)
    assert facebuild.square_crop(1408, 768, 10) == (0, 0, 768)
    assert facebuild.square_crop(1408, 768, 1400) == (640, 0, 768)
    assert facebuild.square_crop(500, 800, 250) == (0, 150, 500)


def test_boundary_ring_is_inside_the_crop_and_outside_the_face():
    lm, side = landmarks()
    ring = facebuild.boundary_ring(lm, side)
    assert ring.min() >= 0 and ring.max() <= side
    centre = lm.mean(0)
    reach = np.linalg.norm(lm - centre, axis=1).max()
    ellipse = ring[:facebuild.RING_POINTS]
    assert (np.linalg.norm(ellipse - centre, axis=1) > 0.6 * reach).all()


def test_apply_overrides_moves_only_the_named_landmarks_and_rejects_bad_ones():
    lm, _ = landmarks()
    out = facebuild.apply_overrides(lm, {"landmarks": {"13": [1.0, 2.0]}})
    assert out[13].tolist() == [1.0, 2.0]
    np.testing.assert_array_equal(np.delete(out, 13, 0), np.delete(lm, 13, 0))
    for bad in ({"999": [1, 2]}, {"-1": [1, 2]}, {"3": [1]}):
        with pytest.raises(facebuild.BuildError):
            facebuild.apply_overrides(lm, {"landmarks": bad})
