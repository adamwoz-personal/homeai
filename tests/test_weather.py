# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for weather retrieval.

Network calls are stubbed throughout. A test suite that depends on a live
weather API would fail for reasons unrelated to this code -- which is the
exact failure mode (wttr.in's expired certificate) that made this module
necessary.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from homeai import weather
from homeai.weather import (
    Conditions,
    Place,
    WeatherError,
    c_to_f,
    describe_code,
    fetch_conditions,
    geocode,
    kmh_to_mph,
    weather_summary,
)

LILBURN = Place("Lilburn", 33.89, -84.14, admin="Georgia", country="US")

GEOCODE_OK = {
    "results": [
        {
            "name": "Lilburn",
            "latitude": 33.8901,
            "longitude": -84.14297,
            "admin1": "Georgia",
            "country": "United States",
            "timezone": "America/New_York",
        }
    ]
}

FORECAST_OK = {
    "current": {
        "temperature_2m": 77.3,
        "apparent_temperature": 82.1,
        "relative_humidity_2m": 61,
        "weather_code": 0,
        "wind_speed_10m": 7.7,
    },
    "daily": {
        "temperature_2m_max": [91.4],
        "temperature_2m_min": [72.0],
        "precipitation_probability_max": [15],
    },
}


@pytest.fixture
def stub_http(monkeypatch):
    """Route _get_json through a table of canned responses."""
    calls: list[str] = []
    table: dict[str, object] = {}

    def fake(url, params, timeout):
        calls.append(url)
        for key, value in table.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(weather, "_get_json", fake)
    return table, calls


class TestConversions:
    def test_celsius_to_fahrenheit(self):
        assert c_to_f(0) == 32.0
        assert c_to_f(100) == 212.0
        assert round(c_to_f(28), 1) == 82.4

    def test_kmh_to_mph(self):
        assert round(kmh_to_mph(6), 1) == 3.7

    def test_known_weather_codes(self):
        assert describe_code(0) == "clear"
        assert describe_code(95) == "thunderstorms"

    def test_unknown_code_degrades_gracefully(self):
        """An unrecognised code must not crash or claim false certainty."""
        assert describe_code(1234) == "unsettled"
        assert describe_code(None) == "unclear"


class TestGeocode:
    def test_resolves_place(self, stub_http):
        table, _ = stub_http
        table["geocoding-api"] = GEOCODE_OK
        place = geocode("Lilburn")
        assert place.name == "Lilburn"
        assert place.admin == "Georgia"
        assert round(place.latitude, 2) == 33.89

    def test_unknown_place_has_distinguishable_error(self, stub_http):
        """'Unknown place' is user-correctable; 'outage' is not."""
        table, _ = stub_http
        table["geocoding-api"] = {"results": []}
        with pytest.raises(WeatherError, match="unknown location"):
            geocode("Xyzzyville")

    def test_blank_place_rejected_without_network(self, stub_http):
        _table, calls = stub_http
        with pytest.raises(WeatherError):
            geocode("   ")
        assert calls == []


class TestFetchConditions:
    def test_parses_current_and_daily(self, stub_http):
        table, _ = stub_http
        table["forecast"] = FORECAST_OK
        conditions = fetch_conditions(LILBURN)
        assert round(conditions.temperature_f) == 77
        assert conditions.description == "clear"
        assert round(conditions.high_f) == 91
        assert conditions.precipitation_chance == 15

    def test_missing_current_block_raises(self, stub_http):
        table, _ = stub_http
        table["forecast"] = {"current": {}}
        with pytest.raises(WeatherError, match="no current conditions"):
            fetch_conditions(LILBURN)

    def test_missing_daily_is_tolerated(self, stub_http):
        """Current conditions alone are still a useful answer."""
        table, _ = stub_http
        table["forecast"] = {"current": FORECAST_OK["current"]}
        conditions = fetch_conditions(LILBURN)
        assert conditions.high_f is None
        assert "77 degrees" in conditions.spoken()


class TestSpokenOutput:
    def _conditions(self, **kwargs):
        base = dict(
            place=LILBURN,
            temperature_f=77.0,
            feels_like_f=77.0,
            description="clear",
            humidity=60,
            wind_mph=8.0,
        )
        base.update(kwargs)
        return Conditions(**base)

    def test_mentions_temperature_and_place(self):
        spoken = self._conditions().spoken()
        assert "77 degrees" in spoken
        assert "Lilburn, Georgia" in spoken

    def test_feels_like_omitted_when_close(self):
        """Saying 'feels like 77' when it is 77 is noise."""
        assert "feels" not in self._conditions(feels_like_f=78.0).spoken()

    def test_feels_like_included_when_different(self):
        assert "feels more like 85" in self._conditions(feels_like_f=85.0).spoken()

    def test_low_precipitation_chance_omitted(self):
        assert "percent" not in self._conditions(precipitation_chance=5).spoken()

    def test_high_precipitation_chance_mentioned(self):
        assert "70 percent" in self._conditions(precipitation_chance=70).spoken()

    def test_no_symbols_reach_speech(self):
        """Piper reads '%' and '°' literally or drops them."""
        spoken = self._conditions(precipitation_chance=70, high_f=91, low_f=72).spoken()
        assert "%" not in spoken and "°" not in spoken


class TestSummaryNeverRaises:
    def test_unknown_location_message_is_actionable(self, stub_http):
        table, _ = stub_http
        table["geocoding-api"] = {"results": []}
        assert "couldn't find" in weather_summary("Xyzzyville")

    def test_outage_refuses_to_guess(self, stub_http):
        """The cardinal rule: never invent a forecast."""
        table, _ = stub_http
        table["geocoding-api"] = WeatherError("cannot reach weather service")
        message = weather_summary("Lilburn")
        assert "don't want to guess" in message
        assert "degrees" not in message

    def test_success_path(self, stub_http):
        table, _ = stub_http
        table["geocoding-api"] = GEOCODE_OK
        table["forecast"] = FORECAST_OK
        assert "77 degrees" in weather_summary("Lilburn")


class TestTransportErrors:
    """_get_json must translate every transport failure into WeatherError."""

    def test_expired_certificate_becomes_weather_error(self, monkeypatch):
        """The exact wttr.in failure of 2026-09-15."""

        def boom(*_args, **_kwargs):
            raise urllib.error.URLError("certificate has expired")

        monkeypatch.setattr(weather.urllib.request, "urlopen", boom)
        with pytest.raises(WeatherError, match="cannot reach"):
            geocode("Lilburn")

    def test_malformed_json_becomes_weather_error(self, monkeypatch):
        class FakeResponse:
            status = 200

            def read(self):
                return b"<html>not json</html>"

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        monkeypatch.setattr(
            weather.urllib.request, "urlopen", lambda *a, **k: FakeResponse()
        )
        with pytest.raises(WeatherError, match="malformed"):
            geocode("Lilburn")

    def test_json_is_parsed_on_success(self, monkeypatch):
        class FakeResponse:
            status = 200

            def read(self):
                return json.dumps(GEOCODE_OK).encode()

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        monkeypatch.setattr(
            weather.urllib.request, "urlopen", lambda *a, **k: FakeResponse()
        )
        assert geocode("Lilburn").name == "Lilburn"
