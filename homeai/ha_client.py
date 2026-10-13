# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Minimal Home Assistant client: REST for states and services, websocket
for the device registry.

Deliberately small. Jarvis uses a *non-admin* HA user, so several convenient
endpoints are off limits (``/api/template`` returns 401) and are not wrapped
here. What a non-admin may do -- read states, call services, list the device
and entity registries over the websocket -- is all the voice tools need.

Every failure surfaces as an ``HAError`` subclass with a sentence that can be
spoken or logged as-is: Jarvis must say "I can't reach the house controller",
never crash or go silent because HA was restarting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("homeai.ha")

DEFAULT_ENV = Path(os.path.expanduser("~/.config/homeai/ha.env"))
DEFAULT_URL = "http://127.0.0.1:8123"


class HAError(RuntimeError):
    """Base class. ``str(exc)`` is a speakable explanation."""


class HANotConfigured(HAError):
    pass


class HAUnavailable(HAError):
    pass


class HAAuthError(HAError):
    pass


class HARequestError(HAError):
    pass


class HAMissingDependency(HARequestError):
    pass


@dataclass(frozen=True)
class HASettings:
    url: str
    token: str


def load_settings(path: Path | None = None) -> HASettings:
    """Read HA_URL / HA_TOKEN from an env-style file (see RUNBOOK).

    Environment variables HA_URL / HA_TOKEN win, so tests and other machines
    need no file.
    """
    path = Path(path or os.environ.get("HOMEAI_HA_ENV") or DEFAULT_ENV)
    values: dict[str, str] = {}
    try:
        text = path.read_text()
    except FileNotFoundError:
        text = ""
    except OSError as exc:
        raise HANotConfigured(f"can't read the house controller settings in {path}: {exc}") from exc
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    token = os.environ.get("HA_TOKEN") or values.get("HA_TOKEN", "")
    url = os.environ.get("HA_URL") or values.get("HA_URL") or DEFAULT_URL
    if not token:
        raise HANotConfigured(f"the house controller isn't set up: no HA_TOKEN in {path}")
    return HASettings(url=url.rstrip("/"), token=token)


def is_configured(path: Path | None = None) -> bool:
    try:
        load_settings(path)
    except HAError:
        return False
    return True


class HAClient:
    """Synchronous facade. Thread-safe enough for one caller at a time."""

    def __init__(self, settings: HASettings, timeout: float = 8.0,
                 session: requests.Session | None = None) -> None:
        self.settings = settings
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {settings.token}",
                                      "Content-Type": "application/json"})
        self._devices: list[dict] | None = None
        self._devices_at = 0.0

    # -- REST --------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        url = self.settings.url + path
        try:
            resp = self._session.request(method, url, json=body, timeout=self.timeout)
        except requests.Timeout as exc:
            raise HAUnavailable("the house controller didn't answer in time") from exc
        except requests.RequestException as exc:
            raise HAUnavailable("I can't reach the house controller") from exc
        if resp.status_code == 401:
            raise HAAuthError("the house controller rejected my access token")
        if resp.status_code == 404:
            raise HARequestError(f"the house controller doesn't know {path.rsplit('/', 1)[-1]}")
        if resp.status_code >= 400:
            detail = resp.text.strip()[:200]
            raise HARequestError(f"the house controller refused that ({resp.status_code}: {detail})")
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError as exc:
            raise HARequestError("the house controller sent back something I couldn't read") from exc

    def states(self) -> list[dict]:
        result = self._request("GET", "/api/states")
        return result if isinstance(result, list) else []

    def services(self) -> list[dict]:
        result = self._request("GET", "/api/services")
        return result if isinstance(result, list) else []

    def state(self, entity_id: str) -> dict:
        return self._request("GET", f"/api/states/{entity_id}")

    def call_service(self, domain: str, service: str, data: dict | None = None) -> Any:
        log.info("HA call %s.%s %s", domain, service, data or {})
        return self._request("POST", f"/api/services/{domain}/{service}", data or {})

    # -- websocket registry ------------------------------------------------

    def devices(self, max_age_s: float = 600.0) -> list[dict]:
        """Device registry (cached). Needed for device_id-based services."""
        now = time.monotonic()
        if self._devices is None or now - self._devices_at > max_age_s:
            self._devices = self._ws_command("config/device_registry/list")
            self._devices_at = now
        return self._devices

    def _ws_command(self, command: str) -> list[dict]:
        try:
            import websockets  # noqa: PLC0415 - optional until a device lookup is needed
        except ImportError as exc:
            raise HAMissingDependency("the websockets package is missing; reinstall homeai") from exc

        url = self.settings.url.replace("http", "ws", 1) + "/api/websocket"

        async def run() -> list[dict]:
            async with asyncio.timeout(self.timeout * 3):
                async with websockets.connect(url, max_size=2**24,
                                              open_timeout=self.timeout) as ws:
                    await ws.recv()
                    await ws.send(json.dumps({"type": "auth",
                                              "access_token": self.settings.token}))
                    auth = json.loads(await ws.recv())
                    if auth.get("type") != "auth_ok":
                        raise HAAuthError("the house controller rejected my access token")
                    await ws.send(json.dumps({"id": 1, "type": command}))
                    msg = json.loads(await ws.recv())
                    if not msg.get("success"):
                        raise HARequestError(f"the house controller refused {command}: "
                                             f"{msg.get('error')}")
                    return msg.get("result") or []

        try:
            return asyncio.run(run())
        except HAError:
            raise
        except (OSError, TimeoutError, asyncio.TimeoutError) as exc:
            raise HAUnavailable("I can't reach the house controller") from exc
        except Exception as exc:  # noqa: BLE001 - websocket library errors vary by version
            raise HAUnavailable(f"I can't reach the house controller ({type(exc).__name__})") from exc


def hint_for(exc: HAError, path: Path | None = None) -> str:
    """What to do about a failure, for the installer and --check."""
    env = path or DEFAULT_ENV
    if isinstance(exc, HAAuthError):
        return (f"Home Assistant rejected the token: in HA open your profile > Security, create a "
                f"new long-lived token for the jarvis user, and replace HA_TOKEN in {env}.")
    if isinstance(exc, HAUnavailable):
        return ("Home Assistant isn't answering: run tools/ha/ha_container.sh status, and "
                "tools/ha/ha_container.sh start if it is stopped. If HA runs elsewhere, "
                f"fix HA_URL in {env}.")
    if isinstance(exc, HAMissingDependency):
        return "A Python package is missing: re-run ./install.sh, or .venv/bin/pip install -e ."
    if isinstance(exc, HANotConfigured):
        return f"Put HA_URL= and HA_TOKEN= in {env} (chmod 600)."
    return "See docs/RUNBOOK.md \"Home control\" and the Home Assistant log: tools/ha/ha_container.sh logs."


def check(path: Path | None = None, client_factory=None) -> tuple[int, str]:
    """(exit code, message) for the installer: 0 working, 1 not set up, 2 broken
    or only partly set up. Failures say what to do about them."""
    if not is_configured(path):
        return 1, ("home control not set up (optional). To add it: ./install.sh --home-assistant, "
                   "or start Home Assistant (tools/ha/ha_container.sh start), create a non-admin "
                   f"user with a long-lived token, and put HA_URL= and HA_TOKEN= in {DEFAULT_ENV} "
                   "(chmod 600). See docs/RUNBOOK.md \"Home control\".")
    try:
        settings = load_settings(path)
        client = (client_factory or HAClient)(settings)
        states = client.states()
        devices = client.devices()
    except HAError as exc:
        return 2, f"home control is configured but not working: {exc}. {hint_for(exc, path)}"
    echos = sum(1 for d in devices if any(i and i[0] == "alexa_devices"
                                          for i in d.get("identifiers") or []))
    lights = sum(1 for s in states if s.get("entity_id", "").startswith("light."))
    summary = (f"Home Assistant at {settings.url}: {len(states)} entities, "
               f"{lights} lights, {echos} Alexa devices")
    missing = []
    if not lights:
        missing.append("no lights: add the Philips Hue integration in HA "
                       "(Settings > Devices & services)")
    if not echos:
        missing.append("no Alexa devices: add the Alexa Devices integration in HA "
                       "(needs your Amazon login and an authenticator code)")
    if missing:
        return 2, f"{summary}. Partly set up: " + "; ".join(missing) + "."
    return 0, summary


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="Home Assistant connection for homeai")
    ap.add_argument("--check", action="store_true", help="test the connection and exit "
                    "0 working, 1 not set up, 2 configured but broken")
    args = ap.parse_args(argv)
    if not args.check:
        ap.print_help()
        return 0
    code, message = check()
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
