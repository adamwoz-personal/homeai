# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""The typing box in the art features window (homeai/art/viewer.py, ART_FEATURES_PLAN F6).

Why: Adam asked to type a description there as well as say it (2026-11-12). A box that sends half a line,
keeps the text after Enter, lets a pasted newline split a line, or shows the start of a long line instead of
where he's typing would make typing worse than speaking. Pure parts only; main() just feeds them keys.
"""
from __future__ import annotations

from types import SimpleNamespace

from homeai.art import viewer as V
from homeai.avatar.presence import Presence


def test_typing_then_enter_sends_the_tidy_line_once():
    e = V.Entry()
    e.type("a red  ")
    e.type("dragon ")
    assert e.key("return") == "a red dragon" and e.text == ""
    assert e.key("return") is None                               # nothing typed: nothing sent


def test_backspace_and_escape():
    e = V.Entry()
    e.type("abc")
    e.key("backspace")
    assert e.text == "ab"
    e.key("escape")
    assert e.text == "" and e.key("backspace") is None and e.text == ""


def test_pasted_newlines_stay_one_line_and_length_is_capped():
    e = V.Entry()
    e.type("a castle\non a hill\r")
    assert e.text == "a castle on a hill "
    e.type("x" * 1000)
    assert len(e.text) == V.TYPED_MAX


def test_the_box_shows_the_end_of_a_long_line_with_a_cursor():
    e = V.Entry()
    e.type("0123456789")
    assert e.visible(5) == "6789▏"
    assert V.Entry().visible(5) == "▏"


def test_typed_reply_events():
    assert V.typed_reply({"type": "typed_reply", "data": {"text": "Drawing it now."}}) == "Drawing it now."
    assert V.typed_reply({"type": "art", "data": {"text": "x"}}) is None
    assert V.typed_reply({"type": "typed_reply", "data": {"text": 3}}) is None


def test_key_names():
    pg = SimpleNamespace(K_BACKSPACE=8, K_ESCAPE=27, K_RETURN=13, K_KP_ENTER=1073741912)
    assert [V.key_name(pg, k) for k in (8, 27, 13, 1073741912, 113)] == [
        "backspace", "escape", "return", "return", None]


def test_paste_without_a_clipboard_is_empty():
    def broken():
        raise RuntimeError("no clipboard")
    assert V.paste_text(SimpleNamespace(scrap=SimpleNamespace(get_text=broken))) == ""
    assert V.paste_text(SimpleNamespace(scrap=SimpleNamespace(get_text=lambda: "a moon"))) == "a moon"


def test_presence_gives_the_bus_the_message_handler(tmp_path):
    handler = lambda msg: None  # noqa: E731
    p = Presence(True, str(tmp_path / "ev.sock"), on_message=handler)
    assert p._bus.on_message is handler and "typed" in p._bus.capabilities
