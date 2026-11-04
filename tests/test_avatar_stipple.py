"""Weighted Voronoi stippling: deterministic, follows brightness (AVATAR_PLAN §5 step 5)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from homeai.avatar import stipple  # noqa: E402


def _half_bright(size=64):
    img = np.zeros((size, size, 3), np.uint8)
    img[:, size // 2:] = 255
    return img


def test_same_seed_same_points_other_seed_differs():
    d = stipple.density_map(_half_bright(), blur_sigma=0)
    a = stipple.stipple(d, 200, seed=5, iterations=5)
    b = stipple.stipple(d, 200, seed=5, iterations=5)
    c = stipple.stipple(d, 200, seed=6, iterations=5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (200, 2)


def test_points_land_only_where_the_art_is_bright():
    d = stipple.density_map(_half_bright(), blur_sigma=0)
    pts = stipple.stipple(d, 300, seed=1, iterations=10)
    assert (pts[:, 0] >= 31.5).all()


def test_relax_spreads_points_evenly():
    d = np.ones((64, 64), np.float32)
    rng = np.random.default_rng(0)
    seeded = stipple.seed_points(d, 256, rng)
    relaxed = stipple.relax(seeded, d, 20)
    from scipy.spatial import cKDTree

    def spread(p):
        nn = cKDTree(p).query(p, k=2)[0][:, 1]
        return nn.std() / nn.mean()
    assert spread(relaxed) < 0.5 * spread(seeded)


def test_density_map_floor_gamma_and_black_image():
    img = np.zeros((8, 8, 3), np.uint8)
    img[0, 0] = 10                                          # 10/255 of the top -> under the floor
    img[1:, :] = 255
    d = stipple.density_map(img, blur_sigma=0, floor=0.06)
    assert d[1:].min() == pytest.approx(1) and d[0, 0] == 0
    img[0, 0] = 64
    lo = stipple.density_map(img, blur_sigma=0, gamma=1.0)[0, 0]
    hi = stipple.density_map(img, blur_sigma=0, gamma=0.5)[0, 0]
    assert hi == pytest.approx(lo ** 0.5, rel=1e-5) and hi > lo
    assert not stipple.density_map(np.zeros((8, 8, 3), np.uint8)).any()


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        stipple.luminance(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        stipple.seed_points(np.zeros((4, 4)), 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        stipple.seed_points(np.ones((4, 4)), 0, np.random.default_rng(0))


def test_sample_is_bilinear_at_pixel_centres():
    img = np.array([[0.0, 1.0], [2.0, 3.0]])
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [1.0, 0.5], [1.0, 1.0], [-5, -5]])
    np.testing.assert_allclose(stipple.sample(img, pts), [0, 1, 0.5, 1.5, 0])
    rgb = np.dstack([img, img * 2])
    assert stipple.sample(rgb, pts).shape == (5, 2)


def test_sizes_scale_with_brightness():
    s = stipple.sizes(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(s, [stipple.SMALL, (stipple.SMALL + stipple.LARGE) / 2, stipple.LARGE])
    assert (stipple.sizes(np.zeros(3)) == stipple.SMALL).all()
