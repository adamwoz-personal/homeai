# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Fake Home Assistant for tests, shaped like Adam's real one (2026-10-09):
Hue rooms as ``is_hue_group`` lights, member bulbs, Echos from alexa_devices
with notify/media_player/next_timer entities, and an Everywhere group."""

from __future__ import annotations

from homeai.ha_client import HAError


def _light(eid, name, state="on", group=False, members=()):
    attrs = {"friendly_name": name}
    if group:
        attrs.update(is_hue_group=True, entity_id=list(members))
    return {"entity_id": eid, "state": state, "attributes": attrs}


def _echo_entities(slug, name, state="idle"):
    return [
        {"entity_id": f"media_player.{slug}", "state": state, "attributes": {"friendly_name": name}},
        {"entity_id": f"notify.{slug}_announce", "state": "unknown",
         "attributes": {"friendly_name": f"{name} Announce"}},
        {"entity_id": f"notify.{slug}_speak", "state": "unknown",
         "attributes": {"friendly_name": f"{name} Speak"}},
        {"entity_id": f"sensor.{slug}_next_timer", "state": "unavailable",
         "attributes": {"friendly_name": f"{name} Next timer", "device_class": "timestamp"}},
    ]


ECHOS = [
    ("kitchen_alexa", "Kitchen alexa", "Echo Show 8", "idle"),
    ("adam_s_bedroom", "Adam's bedroom", "Echo Dot with Clock", "idle"),
    ("farm_bedroom_alexa", "Farm Bedroom alexa", "Echo Spot", "unavailable"),
    ("basement_echo", "basement echo", "Echo", "idle"),
    ("everywhere", "Everywhere", "Speaker Group", "idle"),
]


def default_states():
    states = [
        _light("light.foyer_foyer", "foyer", group=True, members=["light.foyer_hue_color_lamp_13"]),
        _light("light.foyer_hue_color_lamp_13", "Hue color lamp 13"),
        _light("light.den_fan_den_fan", "den fan", group=True),
        _light("light.den_lamp_den_lamp", "den lamp", "off", group=True),
        _light("light.den_lamp_2_den_lamp_2", "Den lamp 2", group=True),
        _light("light.kitchen_cabinets_kitchen_cabinets", "kitchen cabinets", group=True),
        _light("light.fireplace_spots_fireplace_spots", "fireplace spots", group=True),
        _light("light.fireplace_strips_fireplace_strips", "fireplace strips", group=True),
        _light("light.utility_room_utility_room", "utility room", "unavailable", group=True),
        _light("light.hue_white_lamp_1_in_kitchen", "Hue white lamp 1 in Kitchen", "unavailable"),
        {"entity_id": "sun.sun", "state": "above_horizon", "attributes": {"friendly_name": "Sun"}},
    ]
    for slug, name, _model, state in ECHOS:
        states += _echo_entities(slug, name, state)
    return states


def default_devices():
    devs = [{"id": f"dev-{slug}", "name": name, "name_by_user": None, "model": model,
             "identifiers": [["alexa_devices", f"serial-{slug}"]], "disabled_by": None}
            for slug, name, model, _ in ECHOS]
    devs.append({"id": "dev-hue", "name": "Hue Bridge", "model": "BSB002",
                 "identifiers": [["hue", "001788fffe4debc1"]], "disabled_by": None})
    return devs


class FakeHA:
    """Stands in for HAClient. Records every service call."""

    def __init__(self, states=None, devices=None, fail: HAError | None = None):
        self._states = default_states() if states is None else states
        self._devices = default_devices() if devices is None else devices
        self.calls: list[tuple[str, str, dict]] = []
        self.fail = fail
        self.slept: list[float] = []  # Home(sleep=ha.slept.append) in tests

    def states(self):
        if self.fail:
            raise self.fail
        return self._states

    def devices(self, max_age_s: float = 600.0):
        if self.fail:
            raise self.fail
        return self._devices

    def call_service(self, domain, service, data=None):
        if self.fail:
            raise self.fail
        self.calls.append((domain, service, data or {}))

    def set_state(self, entity_id, state):
        for s in self._states:
            if s["entity_id"] == entity_id:
                s["state"] = state
                return
        raise KeyError(entity_id)

    def alexa_commands(self):
        return [(d["device_id"], d["text_command"]) for dom, svc, d in self.calls
                if (dom, svc) == ("alexa_devices", "send_text_command")]
