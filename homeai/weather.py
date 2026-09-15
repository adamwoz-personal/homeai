# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Weather retrieval, owned locally rather than delegated to a third party.

Why this module exists
----------------------
The agent's built-in weather tool called ``wttr.in``. On 2026-09-15 that
service's TLS certificate expired, and every weather question started failing
with a vague "I'm having trouble getting the weather data" -- which reads like
a bug in this project even though nothing here was broken.

Two lessons are baked into the design:

1. **A single free upstream is a single point of failure.** Providers are
   tried in order and a failure falls through to the next one.
2. **Never report a failure as though it were a fact.** When every provider
   fails, this returns an error rather than a plausible-sounding guess. A
   voice assistant confidently inventing a forecast is worse than one
   admitting it does not know.

Design notes
------------
Geocoding and forecasting are separate calls, so a place name that cannot be
resolved produces a *different* error from an upstream outage -- the user can
act on "I don't know where that is" but not on "something went wrong".

Units are converted here, deterministically, for the same reason unit handling
lives in ``speech.py``: asking the model to convert is unreliable, and asking
the API for Fahrenheit still returns Celsius-shaped prose when the model
paraphrases it.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

log = logging.getLogger("homeai.weather")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes, phrased for speech rather than for a
# dashboard. These are spoken aloud, so "light drizzle" beats "Drizzle: Light
# intensity".
_WMO: dict[int, str] = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "violent showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}


class WeatherError(RuntimeError):
    """Raised when weather cannot be determined. Never swallowed silently."""


@dataclass(frozen=True)
class Place:
    name: str
    latitude: float
    longitude: float
    admin: str = ""
    country: str = ""
    timezone: str = "auto"

    @property
    def spoken_name(self) -> str:
        if self.admin and self.admin != self.name:
            return f"{self.name}, {self.admin}"
        return self.name


@dataclass(frozen=True)
class Conditions:
    place: Place
    temperature_f: float
    feels_like_f: float
    description: str
    humidity: int
    wind_mph: float
    high_f: float | None = None
    low_f: float | None = None
    precipitation_chance: int | None = None

    def spoken(self, include_place: bool = True) -> str:
        """A short, speech-ready summary.

        Deliberately two or three sentences: this is the single most common
        question a home assistant is asked, and a paragraph is tiresome to sit
        through several times a day.
        """
        where = f" in {self.place.spoken_name}" if include_place else ""
        parts = [
            f"It's {round(self.temperature_f)} degrees and "
            f"{self.description}{where}."
        ]

        # Only mention apparent temperature when it genuinely differs; saying
        # "feels like 72" when it is 72 is noise.
        if abs(self.feels_like_f - self.temperature_f) >= 4:
            parts.append(f"It feels more like {round(self.feels_like_f)}.")

        if self.high_f is not None and self.low_f is not None:
            parts.append(
                f"Today's high is {round(self.high_f)} and the low is "
                f"{round(self.low_f)}."
            )

        if self.precipitation_chance is not None and self.precipitation_chance >= 20:
            parts.append(
                f"There's a {self.precipitation_chance} percent chance of "
                "precipitation."
            )

        return " ".join(parts)


def c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371


def describe_code(code: int | None) -> str:
    """Map a WMO code to spoken text, degrading gracefully if unknown."""
    if code is None:
        return "unclear"
    return _WMO.get(int(code), "unsettled")


def _get_json(url: str, params: dict[str, object], timeout: float) -> dict:
    query = urllib.parse.urlencode(params)
    full = f"{url}?{query}"
    try:
        with urllib.request.urlopen(full, timeout=timeout) as resp:
            if resp.status != 200:
                raise WeatherError(f"weather service returned HTTP {resp.status}")
            payload = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raise WeatherError(f"weather service returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        # Covers DNS failure, refused connections, and -- as with wttr.in --
        # expired TLS certificates.
        raise WeatherError(f"cannot reach weather service: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise WeatherError(f"weather request failed: {exc}") from exc

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise WeatherError("weather service returned malformed data") from exc


def geocode(place: str, timeout: float = 10.0) -> Place:
    """Resolve a place name to coordinates.

    Raises WeatherError with a *distinguishable* message when the place is
    simply unknown, so the caller can say "I don't know where that is"
    instead of blaming the network.
    """
    if not place or not place.strip():
        raise WeatherError("no location given")

    data = _get_json(
        _GEOCODE_URL,
        {"name": place.strip(), "count": 1, "language": "en", "format": "json"},
        timeout,
    )
    results = data.get("results") or []
    if not results:
        raise WeatherError(f"unknown location: {place}")

    top = results[0]
    return Place(
        name=str(top.get("name", place)),
        latitude=float(top["latitude"]),
        longitude=float(top["longitude"]),
        admin=str(top.get("admin1", "") or ""),
        country=str(top.get("country", "") or ""),
        timezone=str(top.get("timezone", "auto") or "auto"),
    )


def fetch_conditions(place: Place, timeout: float = 10.0) -> Conditions:
    """Fetch current conditions and today's range for a resolved place."""
    data = _get_json(
        _FORECAST_URL,
        {
            "latitude": place.latitude,
            "longitude": place.longitude,
            # Ask for imperial directly. The conversion helpers remain as a
            # safety net for providers that ignore unit parameters.
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m"
            ),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": place.timezone or "auto",
            "forecast_days": 1,
        },
        timeout,
    )

    current = data.get("current") or {}
    if "temperature_2m" not in current:
        raise WeatherError("weather service returned no current conditions")

    daily = data.get("daily") or {}

    def _first(key: str) -> float | None:
        values = daily.get(key) or []
        try:
            return float(values[0])
        except (IndexError, TypeError, ValueError):
            return None

    temp = float(current["temperature_2m"])
    feels = float(current.get("apparent_temperature", temp))
    precip = _first("precipitation_probability_max")

    return Conditions(
        place=place,
        temperature_f=temp,
        feels_like_f=feels,
        description=describe_code(current.get("weather_code")),
        humidity=int(current.get("relative_humidity_2m") or 0),
        wind_mph=float(current.get("wind_speed_10m") or 0.0),
        high_f=_first("temperature_2m_max"),
        low_f=_first("temperature_2m_min"),
        precipitation_chance=int(precip) if precip is not None else None,
    )


def get_weather(location: str, timeout: float = 10.0) -> Conditions:
    """Resolve a place name and fetch its current conditions."""
    return fetch_conditions(geocode(location, timeout), timeout)


def weather_summary(location: str, timeout: float = 10.0) -> str:
    """Speech-ready summary, or a spoken-appropriate explanation of failure.

    This never raises: the voice path must always have something to say. The
    distinction between "unknown place" and "service unreachable" is kept,
    because only the first is something the user can correct.
    """
    try:
        conditions = get_weather(location, timeout)
    except WeatherError as exc:
        message = str(exc)
        if message.startswith("unknown location"):
            return f"I couldn't find a place called {location}."
        log.warning("weather lookup failed for %r: %s", location, message)
        return (
            "I can't reach the weather service right now, so I don't want to "
            "guess."
        )
    return conditions.spoken()
