"""Avatar captions: wrap, layout, rasterise (AVATAR_PLAN §10 Phase 2)."""
from __future__ import annotations

import pytest

from homeai.avatar import captions as cap


def chars(s):
    return len(s)


def test_wrap_fits_and_breaks_on_words():
    assert cap.wrap("one two three", chars, 20, 3) == ["one two three"]
    assert cap.wrap("one two three", chars, 7, 3) == ["one two", "three"]
    assert cap.wrap("", chars, 7, 3) == []


def test_wrap_overflow_ends_with_ellipsis_within_width():
    lines = cap.wrap("aa bb cc dd ee ff gg", chars, 5, 2)
    assert len(lines) == 2 and lines[-1].endswith(cap.ELLIPSIS)
    assert all(len(line) <= 5 for line in lines)


def test_wrap_long_word_and_no_lines():
    assert cap.wrap("supercalifragilistic ok", chars, 5, 3) == ["supercalifragilistic", "ok"]
    assert cap.wrap("anything", chars, 5, 0) == []


def test_layout_puts_jarvis_at_bottom_user_above_status_centre():
    lines = cap.layout([("user", "question", 1.0), ("jarvis", "answer", 0.5),
                        ("status", "coding mode", 1.0)], (400, 400), lambda who, s: len(s) * 8)
    y = {who: (line, alpha, base) for who, line, alpha, base in lines}
    assert y["jarvis"][1] == 0.5
    assert y["user"][2] < y["jarvis"][2] <= 400
    assert y["status"][2] < y["user"][2]


def test_layout_caps_lines_per_speaker():
    long = " ".join(["word"] * 200)
    lines = cap.layout([("jarvis", long, 1.0), ("user", long, 1.0)], (300, 300),
                       lambda who, s: len(s) * 8)
    assert sum(1 for w, *_ in lines if w == "jarvis") == cap.MAX_LINES["jarvis"]
    assert sum(1 for w, *_ in lines if w == "user") == cap.MAX_LINES["user"]


def test_rasteriser_draws_caches_and_skips_invisible():
    pytest.importorskip("pygame")
    r = cap.Rasteriser()
    assert r.image([], (200, 200)) is None
    assert r.image([("user", "hi", 0.0)], (200, 200)) is None
    img = r.image([("jarvis", "Hello there.", 1.0)], (200, 200))
    assert img.shape == (200, 200, 4) and img[..., 3].max() > 200
    assert img[:100, :, 3].max() > 0            # flipped for GL: bottom caption in the top rows
    assert r.image([("jarvis", "Hello there.", 1.0)], (200, 200)) is img
    assert r.image([("jarvis", "Hello there.", 1.0 - 0.001)], (200, 200)) is img   # quantised
    faded = r.image([("jarvis", "Hello there.", 0.3)], (200, 200))
    assert faded is not img and faded[..., 3].max() < img[..., 3].max()


def test_script_is_drawn_from_the_top_with_room_for_a_paragraph():
    text = " ".join(["word"] * 60)
    lines = cap.layout([("script", text, 1.0), ("jarvis", "answer", 1.0)], (400, 400), lambda who, s: len(s) * 8)
    script = [base for who, line, alpha, base in lines if who == "script"]
    jarvis = [base for who, line, alpha, base in lines if who == "jarvis"]
    assert 1 < len(script) <= cap.MAX_LINES["script"]
    assert script == sorted(script) and max(script) < min(jarvis)


def test_bus_keeps_the_latest_script_for_a_restarted_avatar():
    from homeai.avatar.bus import COALESCED
    assert "script" in COALESCED
