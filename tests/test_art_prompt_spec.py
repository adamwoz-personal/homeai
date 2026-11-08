"""PromptSpec (homeai/art/prompt_spec.py) and the patcher (homeai/art/patcher.py), with a fake model."""
from __future__ import annotations

import json
import urllib.error

import pytest

from homeai.art import patcher as P
from homeai.art import prompt_spec as PS

FOX = {"subject": "a red fox", "setting": "in snow"}


def test_validate_fills_defaults_and_tidies_spaces():
    spec = PS.validate({"subject": "  a   red fox "})
    assert spec["subject"] == "a red fox" and spec["aspect"] == "square" and spec["references"] == []


@pytest.mark.parametrize("bad, words", [
    ({"subject": "fox", "colour": "red"}, "unknown field colour"),
    ({"subject": "x" * 201}, "over 200"),
    ({"subject": 7}, "must be text"),
    ({"subject": "fox", "aspect": "round"}, "aspect"),
    ({"subject": "fox", "references": ["a"] * 5}, "references"),
    ({"subject": "fox", "references": [""]}, "references"),
    ({"subject": "fox", "seed": -1}, "seed"),
    ({"subject": "fox", "seed": True}, "seed"),
    ({"subject": "fox", "strength": 1.5}, "strength"),
    ({"setting": "snow"}, "needs a subject"),
    ("fox", "JSON object"),
])
def test_validate_refuses(bad, words):
    with pytest.raises(PS.SpecError, match=words):
        PS.validate(bad)


def test_patch_changes_only_what_it_names():
    spec = PS.validate(FOX)
    new = PS.apply_patch(spec, {"palette": "blue", "setting": None})
    assert new["palette"] == "blue" and new["subject"] == "a red fox" and not new.get("setting")
    assert PS.diff(spec, new) == {"setting": ("in snow", None), "palette": (None, "blue")}


def test_bad_patch_changes_nothing():
    spec = PS.validate(FOX)
    for patch in ({"colour": "blue"}, {"subject": None}, ["palette"]):
        with pytest.raises(PS.SpecError):
            PS.apply_patch(spec, patch)
    assert spec == PS.validate(FOX)


def test_describe():
    changes = {"palette": (None, "blue"), "setting": ("snow", None), "seed": (1, 2), "references": ([], ["Mira"])}
    assert PS.describe(changes) == "palette to blue, no setting and references to Mira"
    long = {"details": (None, "being ridden by a cyberpunk samurai with a skeletal look")}
    assert PS.describe(long) == "the details", "long values are named, not read out"
    assert PS.describe({}) == "nothing"


def test_render_reads_as_sentences():
    out = PS.render({"subject": "a red fox", "setting": "in deep snow", "details": "ears up", "style": "watercolour",
                     "palette": "blue", "lighting": "moonlit", "camera": "low angle", "negative": "people",
                     "aspect": "landscape"})
    assert out.text == ("A red fox, in deep snow. Ears up. Watercolour style, blue colours, moonlit lighting. "
                        "Low angle.")
    assert "people" not in out.text, "negatives aren't rendered: naming them adds them (FLUX has no negative prompt)"
    assert (out.width, out.height) == (1216, 832)


def test_render_does_not_double_words():
    out = PS.render({"subject": "a dragon", "style": "cyberpunk style", "palette": "neon colours",
                     "lighting": "high contrast lighting"})
    assert out.text == "A dragon. Cyberpunk style, neon colours, high contrast lighting."
    assert PS.render({"subject": "a fox", "lighting": "soft light"}).text == "A fox. Soft light."
    assert "soft light lighting" not in PS.render({"subject": "a fox", "lighting": "soft light"}).text


def test_render_keeps_names_as_written():
    assert PS.render({"subject": "a castle", "style": "Studio Ghibli"}).text == "A castle. Studio Ghibli style."


def test_render_minimal():
    assert PS.render({"subject": "a fox"}).text == "A fox."


def test_schema_lists_only_fields_the_model_may_set():
    props = set(PS.PATCH_SCHEMA["properties"])
    assert props <= set(PS.FIELDS) and "seed" not in props and "strength" not in props
    assert PS.PATCH_SCHEMA["additionalProperties"] is False


# -- patcher --------------------------------------------------------------------------------------


def answering(reply):
    seen = {}

    def chat(messages, schema):
        seen["messages"], seen["schema"] = messages, schema
        if isinstance(reply, Exception):
            raise reply
        return reply
    return chat, seen


def test_patch_applied():
    chat, seen = answering(json.dumps({"palette": "blue"}))
    result = P.patch_for(PS.validate(FOX), "make it bluer", chat=chat)
    assert result.ok and result.spec["palette"] == "blue" and result.patch == {"palette": "blue"}
    assert seen["schema"] is PS.PATCH_SCHEMA
    assert "make it bluer" in seen["messages"][-1]["content"]
    assert '"subject": "a red fox"' in seen["messages"][-1]["content"]


def test_new_picture_from_empty():
    chat, _ = answering(json.dumps({"subject": "a lighthouse", "lighting": "dusk"}))
    result = P.patch_for({}, "draw a lighthouse at dusk", chat=chat)
    assert result.ok and result.spec["subject"] == "a lighthouse"


@pytest.mark.parametrize("reply", [
    "not json", json.dumps({}), json.dumps({"colour": "blue"}), json.dumps({"seed": 4}),
    json.dumps({"strength": 0.2}), json.dumps(["palette"]), urllib.error.URLError("down"), TimeoutError(),
])
def test_anything_unusable_keeps_the_picture(reply):
    spec = PS.validate(FOX)
    chat, _ = answering(reply)
    result = P.patch_for(spec, "make it bluer", chat=chat)
    assert not result.ok and result.spec == spec and result.message == P.MSG_NOT_CAUGHT and result.detail


def test_seed_and_strength_are_not_shown_to_the_model():
    chat, seen = answering(json.dumps({"palette": "blue"}))
    P.patch_for(PS.validate({**FOX, "seed": 7, "strength": 0.5}), "bluer", chat=chat)
    assert "seed" not in seen["messages"][-1]["content"] and "strength" not in seen["messages"][-1]["content"]


def test_voice_model_stays_pinned(monkeypatch):
    sent = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"message": {"content": "{}"}}).encode()

    def urlopen(req, timeout):
        sent.update(json.loads(req.data))
        return Reply()
    monkeypatch.setattr(P.urllib.request, "urlopen", urlopen)
    P.ollama_chat([{"role": "user", "content": "x"}], PS.PATCH_SCHEMA)
    assert sent["keep_alive"] == -1, "anything else unloads Jarvis"
    assert sent["format"] is not None and "tools" not in sent and sent["model"] == "llama31-voice"


def test_model_nulling_the_subject_in_passing_is_ignored():
    """llama31-voice answered {"subject": null, "references": ["Mira"]} to "use Mira" in 2 of 3 runs (2026-11-07)."""
    chat, _ = answering(json.dumps({"subject": None, "references": ["Mira"]}))
    result = P.patch_for(PS.validate(FOX), "use Mira", chat=chat)
    assert result.ok and result.spec["subject"] == "a red fox" and result.spec["references"] == ["Mira"]


def test_a_new_picture_still_needs_a_subject():
    chat, _ = answering(json.dumps({"subject": None, "palette": "blue"}))
    result = P.patch_for({}, "blue", chat=chat)
    assert not result.ok and "model said" in result.detail


def test_clearing_an_unknown_field_is_refused_too():
    with pytest.raises(PS.SpecError, match="unknown field colour"):
        PS.apply_patch(PS.validate(FOX), {"colour": None})
