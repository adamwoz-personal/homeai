# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""HA client: settings loading and failure mapping. Every failure must be
an HAError with a speakable sentence, never a raw requests exception."""

from __future__ import annotations

import socket

import pytest
import requests

from homeai import ha_client as hc


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for key in ("HA_TOKEN", "HA_URL", "HOMEAI_HA_ENV"):
        monkeypatch.delenv(key, raising=False)


def test_load_settings_from_file(tmp_path):
    f = tmp_path / "ha.env"
    f.write_text("# comment\nHA_TOKEN=abc.def.ghi\nHA_URL=http://ha.local:8123/\n")
    s = hc.load_settings(f)
    assert s == hc.HASettings(url="http://ha.local:8123", token="abc.def.ghi")


def test_load_settings_default_url_and_quotes(tmp_path):
    f = tmp_path / "ha.env"
    f.write_text('HA_TOKEN="tok"\n')
    assert hc.load_settings(f) == hc.HASettings(url=hc.DEFAULT_URL, token="tok")


def test_env_overrides_file(tmp_path, monkeypatch):
    f = tmp_path / "ha.env"
    f.write_text("HA_TOKEN=file\n")
    monkeypatch.setenv("HA_TOKEN", "env")
    assert hc.load_settings(f).token == "env"


def test_missing_file_or_token(tmp_path):
    with pytest.raises(hc.HANotConfigured, match="isn't set up"):
        hc.load_settings(tmp_path / "nope.env")
    f = tmp_path / "ha.env"
    f.write_text("HA_URL=http://x\n")
    with pytest.raises(hc.HANotConfigured):
        hc.load_settings(f)
    assert not hc.is_configured(f)


def test_bare_token_line_is_not_a_token(tmp_path):
    # What Adam's file first looked like: just the token, no HA_TOKEN=.
    f = tmp_path / "ha.env"
    f.write_text("eyJhbGciOi.payload.sig\n")
    assert not hc.is_configured(f)


class FakeResponse:
    def __init__(self, status=200, body=b"[]", text=None):
        self.status_code = status
        self.content = body
        self.text = text if text is not None else body.decode()

    def json(self):
        import json
        return json.loads(self.content)


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.headers = {}
        self.response = response or FakeResponse()
        self.exc = exc
        self.requests = []

    def request(self, method, url, json=None, timeout=None):
        self.requests.append((method, url, json, timeout))
        if self.exc:
            raise self.exc
        return self.response


def _client(**kw):
    session = FakeSession(**kw)
    return hc.HAClient(hc.HASettings("http://ha:8123", "tok"), session=session), session


def test_auth_header_and_service_call():
    client, session = _client(response=FakeResponse(body=b"[]"))
    client.call_service("light", "turn_off", {"entity_id": ["light.foyer"]})
    assert session.headers["Authorization"] == "Bearer tok"
    assert session.requests[0][:3] == ("POST", "http://ha:8123/api/services/light/turn_off",
                                       {"entity_id": ["light.foyer"]})


def test_states_non_list_is_empty():
    client, _ = _client(response=FakeResponse(body=b'{"message": "odd"}'))
    assert client.states() == []


@pytest.mark.parametrize("kw,exc,msg", [
    ({"exc": requests.ConnectionError("refused")}, hc.HAUnavailable, "can't reach"),
    ({"exc": requests.Timeout("slow")}, hc.HAUnavailable, "in time"),
    ({"response": FakeResponse(401, b"")}, hc.HAAuthError, "rejected"),
    ({"response": FakeResponse(404, b"")}, hc.HARequestError, "doesn't know"),
    ({"response": FakeResponse(400, b"bad entity")}, hc.HARequestError, "refused that"),
    ({"response": FakeResponse(200, b"<html>")}, hc.HARequestError, "couldn't read"),
])
def test_failures_are_speakable(kw, exc, msg):
    client, _ = _client(**kw)
    with pytest.raises(exc, match=msg):
        client.states()


def test_empty_body_is_none():
    client, _ = _client(response=FakeResponse(200, b""))
    assert client.call_service("light", "turn_on") is None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_websocket_unreachable_is_unavailable():
    client = hc.HAClient(hc.HASettings(f"http://127.0.0.1:{_free_port()}", "tok"), timeout=1)
    with pytest.raises(hc.HAUnavailable):
        client.devices()


# -- installer check -----------------------------------------------------------

def test_check_not_set_up(tmp_path):
    code, msg = hc.check(tmp_path / "missing.env")
    assert code == 1 and "not set up" in msg and "ha_container.sh" in msg


def test_check_working(tmp_path):
    from tests.fake_ha import FakeHA

    f = tmp_path / "ha.env"
    f.write_text("HA_TOKEN=t\n")
    code, msg = hc.check(f, client_factory=lambda settings: FakeHA())
    assert code == 0
    assert "lights" in msg and "Alexa devices" in msg and " 0 Alexa" not in msg


def test_check_broken(tmp_path):
    from tests.fake_ha import FakeHA

    f = tmp_path / "ha.env"
    f.write_text("HA_TOKEN=t\n")
    broken = FakeHA(fail=hc.HAAuthError("the house controller rejected my access token"))
    code, msg = hc.check(f, client_factory=lambda settings: broken)
    assert code == 2 and "rejected my access token" in msg


def test_check_cli_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOMEAI_HA_ENV", str(tmp_path / "missing.env"))
    assert hc.main(["--check"]) == 1
    assert "not set up" in capsys.readouterr().out
