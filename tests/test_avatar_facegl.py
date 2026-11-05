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
    a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=0, points=400, seed=2, ears=0)
    assert a.shape[1] == 11 and len(a) >= 400 * 0.9
    np.testing.assert_array_equal(a, facegl.hair_layout(xy, verts, spacing, strands=20, crown=0, points=400, seed=2, ears=0))
    h = rig.face_height(verts)
    top, jaw = min(verts[105, 1], verts[334, 1]), max(verts[172, 1], verts[397, 1])
    assert (a[:, 1] >= top - 1e-3).all()                    # crown=0: nothing above the brows
    assert a[:, 1].max() > jaw + 0.6 * h                    # all the way down the neck
    assert set(np.unique(a[:, 2])) == {-1.0, 1.0}
    np.testing.assert_array_equal(a[:, 7], a[:, 2])         # side strands aim straight out
    assert (a[:, 8] <= 0).all() and (a[a[:, 1] > jaw, 8] == 0).all()   # lift fades out down the head
    order = np.argsort(a[:, 1])                            # less gravity up high: frames the face
    assert (np.diff(a[order, 9]) >= -1e-6).all() and a[order[0], 9] < 0.7
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
    assert by_y[-third:, 1].mean() > 1.25 * by_y[:third, 1].mean()   # shorter above the ears, longer below
    near_jaw = strands[np.abs(strands[:, 0] - jaw) < 0.15 * h, 1].mean()
    neck_end = strands[strands[:, 0] > jaw + 0.6 * h, 1].mean()
    assert neck_end < 0.7 * near_jaw                       # shorter low on the neck: the rest is off screen
    assert ((a[:, 4] >= 0) & (a[:, 4] < 1)).all()


def test_crown_hair_is_long_falls_beside_the_face_and_meets_the_sides(verts):
    xy = head_with_skull(verts)                            # the head alone, without the mesh's frame ring
    spacing = facegl.point_spacing(xy)
    h = rig.face_height(verts)
    a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=40, seed=2, ears=0)
    top = min(verts[105, 1], verts[334, 1])
    origin = np.array([verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean(), top])
    is_crown = a[:, 1] < top - 1e-3
    crown = np.unique(a[is_crown][:, [0, 1, 3]], axis=0)
    assert len(crown) > 30
    assert crown[:, 2].mean() > 2.4 * h                    # longer than the side hair, not tufts
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
    middle = np.abs(rest[:, 0] - origin[0]) < 0.1 * h       # a dome over the part, not a V (Adam)
    assert rest[middle, 1].min() - rest[:, 1].min() < 0.05 * h


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


def test_hair_wave_never_jumps_when_listening_starts_or_wraps():
    """Ripple easing 0 -> 1 -> 0 (a state change) late in a session: each frame the
    wave moves no more than the hair's own speed allows (Adam: wild wiggle at colour
    changes), runs backwards while listening, and crosses the phase wrap smoothly."""
    from homeai.avatar.face import HAIR_PHASE_WRAP

    w = facegl.HairWave()
    phase, speed, dt = HAIR_PHASE_WRAP - 0.3, 1.6, 1 / 60   # wraps mid-transition (ripple ~0.37)
    prev = w.advance(phase, 0.0)
    moves = []
    for i in range(120):
        phase = (phase + speed * dt) % HAIR_PHASE_WRAP
        ripple = min(i / 30, 1.0) if i < 60 else max(1.0 - (i - 60) / 30, 0.0)
        cur = w.advance(phase, ripple)
        d = (cur - prev + np.pi) % (2 * np.pi) - np.pi
        moves.append(d)
        prev = cur
    assert max(abs(d) for d in moves) <= speed * dt + 1e-9
    assert moves[0] > 0 and moves[45] < 0                   # outward, then toward the head


def test_hair_pulse_keeps_speech_in_time_order():
    p = facegl.HairPulse()
    assert p.advance(10.0, 0.2, 0.3, 0.4)
    first = p.head
    assert p.advance(10.0 + 1 / 60, 0.5, 0.6, 0.7)
    assert p.head == first + 1
    np.testing.assert_allclose(p.ring[first, :3], (0.2, 0.3, 0.4))
    np.testing.assert_allclose(p.ring[p.head, :3], (0.5, 0.6, 0.7))


def test_hair_pulse_runs_on_time_not_frames():
    """The window drops to low fps in some states: a slow frame still moves speech
    the right distance, and a fast one adds nothing."""
    p = facegl.HairPulse()
    p.advance(10.0, 0.0, 0.0, 0.0)
    start = p.head
    assert p.advance(10.1, 1.0, 1.0, 1.0)
    assert p.head - start == 6
    np.testing.assert_allclose(p.ring[start + 1:p.head + 1, 0], 1.0)
    assert not p.advance(10.1001, 0.0, 0.0, 0.0)
    assert p.ring[p.head, 0] == 1.0


def test_hair_pulse_clears_when_time_goes_backwards():
    p = facegl.HairPulse()
    p.advance(10.0, 1.0, 1.0, 1.0)
    p.advance(5.0, 0.0, 0.0, 0.0)
    assert not p.ring.any()


def test_hair_pulse_swell_rises_slowly():
    """Sway follows speech gently: Adam dislikes fast or jumpy hair."""
    p = facegl.HairPulse()
    p.advance(10.0, 1.0, 0.0, 0.0)
    assert 0.0 < p.ring[p.head, 3] < 0.1
    for i in range(1, 121):
        p.advance(10.0 + i / 60, 1.0, 0.0, 0.0)
    assert p.ring[p.head, 3] > 0.99


def test_hair_pulse_ring_outlasts_the_longest_strand():
    delay = facegl.HAIR_LENGTH * facegl.HAIR_CROWN_LEN[1] / facegl.HAIR_PULSE_SPEED * facegl.HAIR_PULSE_HZ
    assert delay < facegl.HAIR_PULSE_SAMPLES - 2


def test_view_sits_the_image_on_the_window_bottom():
    side = 1000.0
    px = facegl.to_pixels(np.array([[0, 0], [side, side]]), side, (800, 800), zoom=0.8)
    np.testing.assert_allclose(px[:, 1], (160, 800))       # no black strip under the shoulders
    np.testing.assert_allclose(px[:, 0], (80, 720))


def test_point_spacing_of_a_grid():
    g = np.stack(np.meshgrid(np.arange(30) * 4.0, np.arange(30) * 4.0), -1).reshape(-1, 2)
    assert facegl.point_spacing(g) == pytest.approx(4.0)


def test_breeze_is_connected_but_not_mirrored():
    """Adam: a gentle breeze, connected, not a uniform mirror left and right."""
    from homeai.avatar import tour

    rng = np.random.default_rng(3)
    rest = rng.uniform((-1.5, -0.5), (1.5, 2.0), (500, 2))
    result = tour.wind_coherence(rest, 1.0, 0.0, 36.0)
    assert result["neighbours"] > 0.9
    assert result["mirror"] < 0.3


def test_breeze_is_gentle():
    q = np.random.default_rng(4).uniform(-2, 2, (400, 2))
    step = 0.01
    speed = np.abs(facegl.hair_wind(q, 50.0 + step) - facegl.hair_wind(q, 50.0)).max() / step
    assert speed < 0.8                                      # per rad of hair_phase
    assert np.abs(facegl.hair_wind(q, 50.0)).max() < 1.2


def test_breeze_wraps_seamlessly():
    from homeai.avatar.face import HAIR_PHASE_WRAP

    q = np.random.default_rng(5).uniform(-2, 2, (50, 2))
    np.testing.assert_allclose(facegl.hair_wind(q, 3.0), facegl.hair_wind(q, 3.0 + HAIR_PHASE_WRAP), atol=1e-6)


def test_shader_uses_the_wind_table():
    for term in facegl.HAIR_WIND_X + facegl.HAIR_WIND_Y:
        assert f"{term[3]:.3f} * t" in facegl.HAIR_VERTEX


def test_sides_curl_differently():
    assert facegl.HAIR_SIDE_SHIFT > 0.5


def test_crown_rise_peaks_where_asked():
    s = np.linspace(0, 1, 513)
    for peak in (0.05, 0.1, 0.2):
        rise = facegl.crown_rise(peak)
        y = rise * (1 - (1 - s) ** 2) - facegl.HAIR_DROP * facegl.HAIR_HIGH_DROOP * (
            (1 - facegl.HAIR_GRAVITY) * s + facegl.HAIR_GRAVITY * s * s)
        assert y.max() == pytest.approx(peak, abs=0.005)


def _crown(verts, **kwargs):
    xy = head_with_skull(verts)
    spacing = facegl.point_spacing(xy)
    a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=120, seed=2, ears=0, **kwargs)
    top = min(verts[105, 1], verts[334, 1])
    origin_x = verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean()
    return a[a[:, 1] < top - 1e-3], origin_x, rig.face_height(verts)


def test_crown_hair_grows_up_then_falls(verts):
    """Adam: the hair should grow up, but fall due to gravity."""
    a, origin_x, h = _crown(verts)
    near_top = np.abs(a[:, 0] - origin_x) < 0.15 * h
    rest = facegl.hair_rest(a[near_top])
    rise = a[near_top, 1] - rest[:, 1]
    assert rise.max() > 0.15 * h                            # it goes up
    tips = a[near_top, 4] > 0.95
    assert (rise[tips] < 0).all()                           # and ends below its root
    early = (a[near_top, 4] > 0.15) & (a[near_top, 4] < 0.25)
    up = (a[near_top, 1] - rest[:, 1])[early]
    across = np.abs(rest[:, 0] - a[near_top, 0])[early]
    assert np.median(up) > np.median(across)                # up first, sideways later


def test_crown_has_no_hard_part_line(verts):
    a, origin_x, h = _crown(verts)
    for sign in (-1, 1):                                    # just off centre, some strands cross over
        here = a[(a[:, 0] - origin_x) * sign > 0]
        here = here[np.abs(here[:, 0] - origin_x) < 0.12 * h]
        assert set(np.unique(here[:, 2])) == {-1.0, 1.0}


def test_more_hair_around_the_ears(verts):
    xy = head_with_skull(verts)
    spacing = facegl.point_spacing(xy)
    top, jaw = min(verts[105, 1], verts[334, 1]), max(verts[172, 1], verts[397, 1])
    h = rig.face_height(verts)

    def strands(ears):
        a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=0, seed=2, ears=ears)
        return np.unique(a[:, [0, 1, 3]], axis=0)

    base, more = strands(0), strands(facegl.HAIR_EAR)
    added = len(more) - len(base)
    assert added > 0.8 * facegl.HAIR_EAR
    new = more[~np.isin(more[:, 2], base[:, 2])]
    assert (new[:, 1] >= top).all() and (new[:, 1] <= top + 0.75 * (jaw - top)).all()
    assert new[:, 2].min() >= 0.55 * h


def test_scalp_spots_fill_their_quads_by_area():
    quads = (((0, 0), (10, 0), (10, 10), (0, 10)), ((100, 0), (130, 0), (130, 10), (100, 10)))
    p = facegl.scalp_spots(quads, 2000)
    first = p[:, 0] < 50
    assert p[first, 0].min() >= 0 and p[first, 0].max() <= 10 and p[first, 1].max() <= 10
    assert p[~first, 0].min() >= 100 and p[~first, 0].max() <= 130
    assert p[first, 0].std() > 2.5 and p[first, 1].std() > 2.5     # spread out, not at a corner
    assert 0.2 < first.mean() < 0.3                                # 1 part in 4 by area
    assert len(facegl.scalp_spots((), 10)) == 0


def test_every_marked_artwork_has_five_scalp_areas():
    assert all(len(quads) == 5 for quads in facegl.SCALP_CHANNELS.values())


def _scalp_layout(verts):
    xy = head_with_skull(verts)
    spacing = facegl.point_spacing(xy)
    h = rig.face_height(verts)
    cx, top = verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean(), verts[10, 1]
    quad = ((cx - 0.1 * h, top - 0.1 * h), (cx + 0.1 * h, top - 0.1 * h), (cx + 0.1 * h, top), (cx - 0.1 * h, top))
    a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=0, ears=0, seed=2, scalp=(quad,), scalp_strands=40)
    return a[a[:, 1] <= top + 1e-3], cx, top, h


def test_scalp_hair_roots_in_the_channel_and_grows_up(verts):
    a, cx, top, h = _scalp_layout(verts)
    assert len(np.unique(a[:, [0, 1]], axis=0)) >= 39
    assert (np.abs(a[:, 0] - cx) <= 0.1 * h + 1e-3).all() and (a[:, 1] >= top - 0.1 * h - 1e-3).all()
    rise = a[:, 1] - facegl.hair_rest(a)[:, 1]
    assert rise.max() > 0.15 * h and rise[a[:, 4] > 0.95].max() < 0


def test_asset_hair_uses_the_arts_marked_channels(verts):
    from types import SimpleNamespace

    xy = head_with_skull(verts)
    art = next(iter(facegl.SCALP_CHANNELS))

    def strands(sha):
        asset = SimpleNamespace(xy=xy, colour=np.ones((len(xy), 3)), verts=verts,
                                build={"inputs": {"image_sha256": sha}})
        a = facegl.asset_hair(asset, facegl.point_spacing(xy))
        return len(np.unique(a[:, [0, 1]], axis=0))

    assert strands(art) >= strands("other art") + facegl.HAIR_SCALP - 1


def test_hair_is_long_everywhere(verts):
    xy = head_with_skull(verts)
    spacing = facegl.point_spacing(xy)
    a = facegl.hair_layout(xy, verts, spacing, strands=20, crown=40, seed=2)
    assert a[:, 3].max() > 3.5 * rig.face_height(verts)


def test_hair_changes_smoothly_round_the_head(verts):
    """Adam: no visible step between regions. Length and lift used to jump 3x at the brows."""
    xy = head_with_skull(verts)
    a = facegl.hair_layout(xy, verts, facegl.point_spacing(xy), seed=2)
    top = min(verts[105, 1], verts[334, 1])
    cx = verts[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean()
    strands = np.unique(a[:, [0, 1, 3, 8]], axis=0)
    turn = np.degrees(np.arctan2(np.abs(strands[:, 0] - cx), top - strands[:, 1]))
    bands = [strands[(turn >= lo) & (turn < lo + 20)] for lo in range(40, 140, 20)]
    length = [np.median(b[:, 2]) for b in bands if len(b) > 5]
    lift = [np.median(b[:, 3]) for b in bands if len(b) > 5]
    assert len(length) >= 4
    assert max(max(p, q) / min(p, q) for p, q in zip(length, length[1:])) < 1.6
    assert max(abs(p - q) for p, q in zip(lift, lift[1:])) < 0.45


def test_lift_fades_smoothly_down_the_head():
    h, length = 100.0, 200.0
    lifts = [-facegl.lift_at(float(x), h, length) for x in range(0, 135, 5)]
    assert all(p >= q - 1e-6 for p, q in zip(lifts, lifts[1:]))
    assert lifts[18] > 0.1 and lifts[-1] == 0.0             # still lifting at the brows, none by the cheek
    assert max(p - q for p, q in zip(lifts, lifts[1:])) < 0.25


def test_crown_strands_land_across_a_width(verts):
    """All aimed at one line, they piled into a bright ridge: Adam's "cat ears"."""
    a, origin_x, h = _crown(verts)
    tips = facegl.hair_rest(a)[a[:, 4] > 0.97]
    for side in (-1, 1):
        x = tips[(tips[:, 0] - origin_x) * side > 0, 0]
        assert x.std() > 0.25 * h
    xy = head_with_skull(verts)
    rows = np.linspace(min(verts[105, 1], verts[334, 1]), max(verts[172, 1], verts[397, 1]), 12)
    widest = max(abs(facegl.silhouette_edge(xy, y, s, 2 * facegl.point_spacing(xy), centre=origin_x,
                                            gap=facegl.HAIR_GAP * h) - origin_x) for y in rows for s in (-1, 1))
    side = np.sign(a[:, 7])
    reach = np.abs(a[:, 7]) * facegl.HAIR_SPREAD * a[:, 3] / facegl.HAIR_LENGTH
    landing = reach + (a[:, 0] - origin_x) * side           # clears the head's widest point, every strand
    assert landing[reach > 0].min() > 0.9 * widest + facegl.HAIR_CROWN_CLEAR * h
