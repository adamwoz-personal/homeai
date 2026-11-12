"""Spoken art commands (homeai/art/grammar.py): each row of AVATAR_PLAN §7.5, and near-misses that must not match."""
from __future__ import annotations

import pytest

from homeai.art.grammar import Intent, parse


@pytest.mark.parametrize("said, intent", [
    ("draw a red fox in the snow", Intent("new", text="draw a red fox in the snow")),
    ("Hey Jarvis, paint me a lighthouse at dusk.", Intent("new", text="paint me a lighthouse at dusk")),
    ("make a castle on a hill", Intent("new", text="make a castle on a hill")),
    ("make it bluer", Intent("edit", text="make it bluer")),
    ("add some fog", Intent("edit", text="add some fog")),
    ("now make it night time", Intent("edit", text="make it night time")),
    ("get rid of the blue", Intent("edit", text="get rid of the blue")),
    ("use Mira", Intent("edit", text="use mira")),
    ("again", Intent("again")),
    ("try again please", Intent("again")),
    ("another one", Intent("again")),
    ("give me four", Intent("variations", n=4)),
    ("give me 3 options", Intent("variations", n=3)),
    ("a few options", Intent("variations", n=4)),
    ("show me ten versions", Intent("variations", n=4)),
    ("go back", Intent("back", n=1)),
    ("go back three", Intent("back", n=3)),
    ("back 2 steps", Intent("back", n=2)),
    ("undo that", Intent("back", n=1)),
    ("the second one", Intent("select", n=2)),
    ("I like the last one", Intent("select", n=-1)),
    ("keep that one", Intent("keep")),
    ("save that as my wizard", Intent("keep", name="wizard")),
    ("what prompt made this?", Intent("what")),
    ("show them side by side", Intent("compare")),
    ("upscale it", Intent("upscale")),
    ("stop", Intent("cancel")),
    ("never mind", Intent("cancel")),
    ("leave art mode", Intent("leave")),
    ("I'm done drawing", Intent("leave")),
    ("show me the last four", Intent("show_last", n=4)),
])
def test_each_phrase(said, intent):
    assert parse(said) == intent


@pytest.mark.parametrize("said", [
    "good night", "what time is it", "turn off the lights", "make the lights brighter", "set a timer for ten minutes",
    "play some music", "turn the volume down", "what's the weather", "", "   ", "thanks", "tell me a joke",
    "draw", "go backwards in time", "stop the music", "add milk to the shopping list",
])
def test_ordinary_talk_is_not_art(said):
    result = parse(said)
    assert result is None or result.kind not in ("new", "edit", "again", "variations"), result


def test_make_it_is_never_a_new_picture():
    for said in ("make it bigger", "make them smile", "make her taller", "make this red"):
        assert parse(said).kind == "edit"


# -- Adam's battleship session (2026-11-07) ------------------------------------------------------

@pytest.mark.parametrize("said, n", [("what is image three?", 3), ("image three.", 3), ("option three.", 3),
                                     ("number 2", 2), ("show me option four", 4), ("the third picture", 3),
                                     ("let's see the first one", 1), ("go with option one", 1)])
def test_picking_an_option_by_number(said, n):
    assert parse(said) == Intent("select", n=n)


def test_chatter_after_again_is_not_a_change():
    assert parse("again. Oh gosh, it's fake again. Oh my gosh, what's that?") == Intent("again")
    complaint = parse("again. It has two, a little head on this side and a big head on this side.")
    assert complaint.kind == "edit" and complaint.text.startswith("it has two")


def test_saying_the_wake_word_again_restarts_the_request():
    said = "Draw me a dragon. Hey Jarvis, let's draw a dragon flying over the ocean."
    from homeai.art.grammar import _tidy
    from homeai.art.session import drawing_sentence
    assert _tidy(said) == "let's draw a dragon flying over the ocean"
    assert drawing_sentence(_tidy(said)) == "Drawing a dragon flying over the ocean."
    assert _tidy("hey jarvis") == ""                             # only the wake word: nothing asked
    assert parse("hey jarvis, draw a fox") == Intent("new", text="draw a fox")
    assert parse("Jarvis, draw a fox") == Intent("new", text="draw a fox")
    assert parse("Hey Jarvis, stop.") == Intent("cancel")


# Adam's tries to leave on 2026-11-12, all of which missed; and "drawing mode" is what he calls it.
@pytest.mark.parametrize("said", ["switch to voice mode", "Hey Jarvis, cancel art mode", "Hey Jarvis, cancel art",
                                  "Hey Jarvis, we're done drawing", "we're done", "exit drawing mode",
                                  "leave drawing mode", "back to voice mode", "voice mode", "turn off art mode",
                                  "end art mode", "I'm finished with the drawing", "no more drawing",
                                  "stop drawing mode", "go back to normal mode"])
def test_more_ways_out_of_art_mode(said):
    assert parse(said) == Intent("leave")


@pytest.mark.parametrize("said", ["we're done to register as gassed if everybody can interrupt you",
                                  "cancel the art of the dragon", "draw a voice mode diagram", "stop",
                                  "stop the drawing", "cancel the picture", "end the picture"])
def test_not_a_way_out(said):
    assert (parse(said) or Intent("none")).kind != "leave"
