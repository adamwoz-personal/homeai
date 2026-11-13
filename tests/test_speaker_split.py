"""Splitting one recording by speaker (homeai/speaker_split.py, ART_FEATURES_PLAN S1), and the daemon using it."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from homeai import speaker_split as S
from tests.test_daemon import assistant, quick, voiced  # noqa: F401 - fixtures

SR = S.SR


def test_windows_cover_the_recording():
    sims = S.window_sims(lambda chunk: 0.5, np.zeros(3 * SR, dtype=np.float32))
    assert sims[0] == (0.0, 1.5, 0.5) and sims[-1][1] == 3.0 and len(sims) == 7
    assert S.window_sims(lambda chunk: 0.5, np.zeros(SR, dtype=np.float32)) == []      # shorter than a window


def test_one_low_window_is_never_enough():
    sims = [(0.0, 1.5, 0.7), (0.25, 1.75, 0.1), (0.5, 2.0, 0.7)]
    assert S.other_spans(sims, 0.25) == []
    sims = [(0.0, 1.5, 0.7), (0.25, 1.75, 0.1), (0.5, 2.0, 0.1), (0.75, 2.25, None), (1.0, 2.5, 0.1),
            (1.25, 2.75, 0.05)]
    assert S.other_spans(sims, 0.25) == [(0.25, 2.75)]          # overlapping runs merge; None breaks a run


def test_mixed_needs_the_owner_too():
    spans = [(0.0, 2.0)]
    assert not S.is_mixed([(0.0, 1.5, 0.1), (0.25, 1.75, 0.1)], spans, 0.45)
    assert S.is_mixed([(0.0, 1.5, 0.1), (2.0, 3.5, 0.6)], spans, 0.45)
    assert not S.is_mixed([(0.0, 1.5, 0.6)], [], 0.45)


def test_words_and_dropping():
    out = "[00:00:00.000 --> 00:00:00.400]  again.\n[00:00:01.000 --> 00:00:01.300]   It\n" \
          "[00:00:01.300 --> 00:00:01.600]   has\nnoise line\n[00:00:01.600 --> 00:00:01.700]  \n"
    words = S.parse_words(out)
    assert words == [(0.0, 0.4, "again."), (1.0, 1.3, "It"), (1.3, 1.6, "has")]
    assert S.keep_owner_words(words, [(0.9, 3.0)]) == ("again.", "It has")
    assert S.keep_owner_words(words, []) == ("again. It has", "")


# -- the daemon in art mode -------------------------------------------------------------------

class Art:
    active, owner = True, "Adam"

    def __init__(self):
        self.seen = []

    def handle(self, text, speaker, other=False):
        self.seen.append((text, other))
        return "" if other else "Another one."


@pytest.fixture
def mixed(voiced, monkeypatch):
    """Two seconds of Adam (audio 0.1), then two of someone else (0.5); windows straddling both score in between."""
    from tests.test_daemon import ADAM, STRANGER, _unit
    import dataclasses
    voiced.cfg = dataclasses.replace(voiced.cfg, art=dataclasses.replace(voiced.cfg.art, split=True))
    voiced.art = Art()
    voiced._speak = lambda text: 1.0
    voiced.transcript = MagicMock()

    def embed(chunk, trim=True):
        level = float(np.mean(chunk))
        return ADAM if level < 0.2 else STRANGER if level > 0.4 else _unit(1, 0, 1)
    monkeypatch.setattr(voiced.voice_id.embedder, "embed", embed)
    voiced.stt.words = MagicMock(return_value=[(0.2, 0.6, "again."), (2.6, 3.0, "it"), (3.0, 3.4, "has"),
                                                (3.4, 3.8, "two")])
    audio = np.concatenate([np.full(2 * SR, 0.1), np.full(2 * SR, 0.5)]).astype(np.float32)
    return voiced, audio


def _art_turn(q, audio, text):
    from homeai.daemon import Stopwatch
    ident = type("I", (), {"name": "Adam", "score": 0.5, "best": "Adam", "margin": 0.5, "reason": ""})()
    from tests.test_daemon import ADAM
    return q._handle_art(MagicMock(), text, ident, Stopwatch(), ADAM, audio)


def test_another_voice_inside_the_owners_recording_is_dropped(mixed):
    q, audio = mixed
    assert _art_turn(q, audio, "again. It has two")
    assert q.art.seen == [("again.", False)]


def test_a_recording_of_only_the_owner_is_left_alone(mixed):
    q, _ = mixed
    _art_turn(q, np.full(4 * SR, 0.1, dtype=np.float32), "again")
    assert q.art.seen == [("again", False)] and not q.stt.words.called


def test_nothing_of_the_owner_left_means_ignored(mixed):
    q, audio = mixed
    q.stt.words.return_value = [(2.6, 3.0, "it"), (3.0, 3.4, "has")]
    _art_turn(q, audio, "it has")
    assert q.art.seen == [("it has", True)]


def test_a_split_failure_keeps_the_sentence(mixed):
    q, audio = mixed
    q.stt.words.side_effect = RuntimeError("whisper fell over")
    _art_turn(q, audio, "again. It has two")
    assert q.art.seen == [("again. It has two", False)]


def test_leaving_is_never_split(mixed):
    """2026-11-12: "cancel art mode" kept "cancel art", dropped "mode.", and art mode stayed on."""
    q, audio = mixed
    q.stt.words.return_value = [(0.2, 0.6, "cancel"), (0.6, 1.0, "art"), (2.6, 3.0, "mode.")]
    _art_turn(q, audio, "cancel art mode.")
    assert q.art.seen == [("cancel art mode.", False)] and not q.stt.words.called



def test_the_split_is_off_by_default(mixed):
    """Adam 2026-11-13: off until measured; it ate his last word on most sentences it touched."""
    import dataclasses
    from homeai.config import ArtConfig
    q, audio = mixed
    assert ArtConfig().split is False
    q.cfg = dataclasses.replace(q.cfg, art=dataclasses.replace(q.cfg.art, split=False))
    _art_turn(q, audio, "again. It has two")
    assert q.art.seen == [("again. It has two", False)] and not q.stt.words.called
