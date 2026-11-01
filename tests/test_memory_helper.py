# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""The helper model wrapper (memory_store/helper.py): validation, retry, Ollama."""

from __future__ import annotations

import json

import pytest

from homeai.memory_store.extract import extract_schema, reconcile_schema, reflect_schema
from homeai.memory_store.helper import (HelperInvalid, HelperUnavailable, OllamaHelper,
                                        ScriptedHelper, validate)

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["a", "b"],
          "properties": {"a": {"type": "string", "enum": ["x", "y"]},
                         "b": {"type": "array", "minItems": 1, "maxItems": 2,
                               "items": {"type": "integer", "minimum": 0, "maximum": 9}},
                         "c": {"type": ["string", "null"], "minLength": 2, "maxLength": 3}}}


@pytest.mark.parametrize("value", [
    {"a": "x", "b": [1]}, {"a": "y", "b": [0, 9], "c": None}, {"a": "x", "b": [3], "c": "abc"},
])
def test_valid_values(value):
    assert validate(SCHEMA, value) == []


@pytest.mark.parametrize("value,needle", [
    ([], "expected object"),
    ({"a": "x"}, "missing b"),
    ({"a": "z", "b": [1]}, "not in"),
    ({"a": "x", "b": [1], "d": 1}, "unexpected d"),
    ({"a": "x", "b": []}, "fewer than"),
    ({"a": "x", "b": [1, 2, 3]}, "more than"),
    ({"a": "x", "b": [10]}, "above"),
    ({"a": "x", "b": [-1]}, "below"),
    ({"a": "x", "b": ["1"]}, "expected integer"),
    ({"a": "x", "b": [True]}, "expected integer"),       # bool is not a number here
    ({"a": "x", "b": [1], "c": "a"}, "shorter"),
    ({"a": "x", "b": [1], "c": "abcd"}, "longer"),
    ({"a": "x", "b": [1], "c": 5}, "expected string"),
])
def test_invalid_values(value, needle):
    errs = validate(SCHEMA, value)
    assert errs and any(needle in e for e in errs), errs


def test_number_accepts_int_and_float():
    assert validate({"type": "number"}, 1) == []
    assert validate({"type": "number"}, 0.5) == []
    assert validate({"type": "number"}, "1") != []


def test_real_schemas_accept_typical_answers():
    assert validate(extract_schema(), {"notes": []}) == []
    assert validate(extract_schema(), {"notes": [{
        "kind": "fact", "summary": "Adam keeps a corn snake", "topics": ["pets"],
        "thread": None, "importance": 0.8, "sources": [1]}]}) == []
    assert validate(reconcile_schema(), {"action": "add", "target": None, "summary": "x"}) == []
    assert validate(reflect_schema(), {"lines": [{"summary": "Adam likes snakes",
                                                  "sources": [1, 2]}]}) == []


def test_extract_schema_never_offers_instruction_or_profile():
    kinds = extract_schema()["properties"]["notes"]["items"]["properties"]["kind"]["enum"]
    assert "instruction" not in kinds and "profile" not in kinds


# -- structured(): retry once, then give up ------------------------------------------

def test_structured_returns_valid_json():
    h = ScriptedHelper([{"a": "x", "b": [1]}])
    assert h.structured("task", SCHEMA, {}) == {"a": "x", "b": [1]}
    assert len(h.calls) == 1


def test_structured_retries_once_after_bad_output():
    h = ScriptedHelper(["not json", {"a": "x", "b": [1]}])
    assert h.structured("task", SCHEMA, {})["a"] == "x"
    assert len(h.calls) == 2


def test_structured_gives_up_after_two_bad_answers():
    h = ScriptedHelper([{"a": "z", "b": [1]}, "{oops", {"a": "x", "b": [1]}])
    with pytest.raises(HelperInvalid):
        h.structured("task", SCHEMA, {})
    assert len(h.calls) == 2


def test_unavailable_is_not_retried():
    h = ScriptedHelper([HelperUnavailable("down"), {"a": "x", "b": [1]}])
    with pytest.raises(HelperUnavailable):
        h.structured("task", SCHEMA, {})
    assert len(h.calls) == 1


# -- Ollama -------------------------------------------------------------------------------

def test_ollama_request_runs_on_cpu_with_the_schema():
    seen = {}

    def post(url, body, timeout):
        seen.update(url=url, body=body, timeout=timeout)
        return {"message": {"content": json.dumps({"a": "x", "b": [2]})}}

    h = OllamaHelper("http://h:1/", "m:4b", threads=5, timeout_s=9, post=post)
    assert h.structured("do it", SCHEMA, {"k": "é"}) == {"a": "x", "b": [2]}
    body = seen["body"]
    assert seen["url"] == "http://h:1/api/chat" and seen["timeout"] == 9
    assert body["model"] == "m:4b" and body["format"] == SCHEMA and body["stream"] is False
    assert body["options"]["num_gpu"] == 0          # the GPU belongs to the voice model
    assert body["options"]["num_thread"] == 5 and body["options"]["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": "do it"}
    assert json.loads(body["messages"][1]["content"]) == {"k": "é"}
    assert "think" not in body
    assert h.name == "m:4b"


@pytest.mark.parametrize("post", [
    lambda u, b, t: (_ for _ in ()).throw(OSError("refused")),
    lambda u, b, t: {"error": "model not found"},
])
def test_ollama_failures_are_unavailable(post):
    with pytest.raises(HelperUnavailable):
        OllamaHelper("http://h:1", "m", post=post).structured("t", SCHEMA, {})


def test_ollama_unreachable_for_real_fails_fast():
    with pytest.raises(HelperUnavailable):
        OllamaHelper("http://127.0.0.1:9", "m", timeout_s=2).structured("t", SCHEMA, {})
