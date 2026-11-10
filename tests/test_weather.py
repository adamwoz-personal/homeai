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


# 2026-11-10 is a Tuesday.
WEEK = ["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"]
DAILY_OK = {"daily": {
    "time": WEEK,
    "weather_code": [0, 61, 3, 95, 0, 0, 71],
    "temperature_2m_max": [70.2, 64.6, 60, 58, 66, 68, 50],
    "temperature_2m_min": [50, 48.4, 45, 44, 47, 49, 35],
    "precipitation_probability_max": [5, 70, 10, 90, 0, 0, 40],
}}


class TestForecast:
    @pytest.mark.parametrize("day, index", [
        ("today", 0), ("", 0), ("tomorrow", 1), ("Tomorrow ", 1), ("tuesday", 0), ("wednesday", 1),
        ("saturday", 4), ("monday", 6),
    ])
    def test_days_come_from_the_service_s_own_calendar(self, day, index):
        assert weather.day_index(day, WEEK) == index

    def test_an_unknown_day_is_an_error_not_a_guess(self):
        with pytest.raises(WeatherError, match="unknown day"):
            weather.day_index("next month", WEEK)
        with pytest.raises(WeatherError, match="no forecast for friday"):
            weather.day_index("friday", WEEK[:2])

    def test_tomorrow(self, stub_http):
        table, _ = stub_http
        table["forecast"] = DAILY_OK
        f = weather.fetch_forecast(LILBURN, "tomorrow")
        assert (f.label, f.date, f.description, round(f.high_f), round(f.low_f), f.precipitation_chance) \
            == ("tomorrow", "2026-11-11", "light rain", 65, 48, 70)
        assert f.spoken() == ("Tomorrow in Lilburn, Georgia, the forecast is light rain, with a high of 65 and a low of 48. "
                              "There's a 70 percent chance of precipitation.")

    def test_a_weekday_is_named_as_such(self, stub_http):
        table, _ = stub_http
        table["forecast"] = DAILY_OK
        f = weather.fetch_forecast(LILBURN, "Saturday")
        assert f.label == "saturday" and f.date == "2026-11-14"
        assert f.spoken(include_place=False) == "Saturday, the forecast is clear, with a high of 66 and a low of 47."

    def test_the_week_is_asked_for_in_fahrenheit_and_local_time(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(weather, "_get_json", lambda url, params, timeout: seen.update(params) or DAILY_OK)
        weather.fetch_forecast(Place("X", 1, 2, timezone="America/New_York"), "tomorrow")
        assert seen["forecast_days"] == 10 and seen["temperature_unit"] == "fahrenheit"
        assert seen["timezone"] == "America/New_York" and "weather_code" in seen["daily"]

    def test_no_forecast_raises(self, stub_http):
        table, _ = stub_http
        table["forecast"] = {"daily": {}}
        with pytest.raises(WeatherError, match="no forecast"):
            weather.fetch_forecast(LILBURN, "tomorrow")

    def test_a_missing_value_is_left_out_not_invented(self, stub_http):
        table, _ = stub_http
        daily = {k: list(v) for k, v in DAILY_OK["daily"].items()}
        daily["temperature_2m_max"][1] = None
        daily["precipitation_probability_max"] = []
        table["forecast"] = {"daily": daily}
        f = weather.fetch_forecast(LILBURN, "tomorrow")
        assert f.high_f is None and f.precipitation_chance is None
        assert f.spoken() == "Tomorrow in Lilburn, Georgia, the forecast is light rain."

    def test_tomorrow_beyond_what_the_service_sent_is_an_error(self, stub_http):
        table, _ = stub_http
        table["forecast"] = {"daily": {k: v[:1] for k, v in DAILY_OK["daily"].items()}}
        with pytest.raises(WeatherError, match="no forecast for tomorrow"):
            weather.fetch_forecast(LILBURN, "tomorrow")


TEN = WEEK + ["2026-11-17", "2026-11-18", "2026-11-19"]
TEN_OK = {"daily": {
    "time": TEN,
    "weather_code": [0, 61, 3, 95, 0, 0, 71, 2, 63, 0],
    "temperature_2m_max": [70, 65, 60, 58, 66, 68, 50, 61.4, 57, 59],
    "temperature_2m_min": [50, 48, 45, 44, 47, 49, 35, 42.2, 40, 41],
    "precipitation_probability_max": [5, 70, 10, 90, 0, 0, 40, 10, 80, 0],
}}


class TestTenDays:
    """2026-11-10: "seven days from now" (the eighth day) asked as "tuesday" on a Tuesday came back as today."""

    def test_a_week_from_today_is_next_tuesday_with_its_date(self, stub_http):
        table, _ = stub_http
        table["forecast"] = TEN_OK
        f = weather.fetch_forecast(LILBURN, "today", days_ahead=7)
        assert (f.date, f.label, f.description, round(f.high_f)) == ("2026-11-17", "tuesday the 17th", "partly cloudy", 61)
        assert f.spoken().startswith("Tuesday the 17th in Lilburn, Georgia, the forecast is partly cloudy")

    def test_days_ahead_overrides_day(self, stub_http):
        table, _ = stub_http
        table["forecast"] = TEN_OK
        assert weather.fetch_forecast(LILBURN, "tomorrow", days_ahead=9).date == "2026-11-19"

    @pytest.mark.parametrize("n", [-1, 10, 16])
    def test_beyond_nine_days_is_refused_without_asking(self, stub_http, n):
        table, calls = stub_http
        table["forecast"] = TEN_OK
        with pytest.raises(WeatherError, match="9 days ahead"):
            weather.fetch_forecast(LILBURN, "today", days_ahead=n)
        assert calls == []

    @pytest.mark.parametrize("i, iso, label", [
        (0, "2026-11-10", "today"), (1, "2026-11-11", "tomorrow"), (6, "2026-11-16", "monday"),
        (7, "2026-11-17", "tuesday the 17th"), (8, "2026-11-18", "wednesday the 18th"),
        (9, "2026-12-01", "tuesday the 1st"), (9, "2026-12-02", "wednesday the 2nd"), (9, "2026-12-03", "thursday the 3rd"),
        (9, "2026-12-11", "friday the 11th"), (9, "2026-12-12", "saturday the 12th"), (9, "2026-12-22", "tuesday the 22nd"),
    ])
    def test_labels(self, i, iso, label):
        assert weather.label_for(i, iso) == label
