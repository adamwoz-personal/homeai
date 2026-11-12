"""'Status update' (homeai/status_update.py) and the daemon hook."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from homeai import status_update as S

NOW = 1_800_000_000.0
DAY = 24 * 3600


def profile(name, created):
    return SimpleNamespace(name=name, created_at=created)


class Doors:
    def __init__(self, alerts=None, fail=False):
        self.alerts, self.fail, self.asked = alerts or {}, fail, []

    def __call__(self, since, until):
        self.asked.append((since, until))
        if self.fail:
            raise RuntimeError("HA down")
        return self.alerts


@pytest.fixture
def make(tmp_path):
    def make(profiles=(), doors=None, weather="It's 61 degrees and clear.", clock=None):
        return S.StatusUpdate(lambda: list(profiles), doors if doors is not None else Doors(),
                              (lambda: weather) if isinstance(weather, str) else weather,
                              ledger=S.Ledger(tmp_path / "ledger.json"), clock=clock or (lambda: NOW))
    return make


@pytest.mark.parametrize("said", ["status update", "Hey Jarvis, status update.", "give me a status update",
                                  "can I get a status report", "what did I miss", "catch me up please",
                                  "what's the status"])
def test_requests(said):
    assert S.is_request(said)


@pytest.mark.parametrize("said", ["update the shopping list", "what's the status of the dishwasher",
                                  "status of the front door", "turn off the lights"])
def test_not_requests(said):
    assert not S.is_request(said)


def test_not_a_request_is_passed_on(make):
    assert make().handle("what's the weather", "Adam") is None


def test_first_update_lists_the_last_week_of_voices_and_day_of_alerts(make):
    doors = Doors({("front door", "ring"): [NOW - 600], ("front door", "motion"): [NOW - 7200, NOW - 3600]})
    st = make([profile("Adam", NOW - 100), profile("Lily", NOW - 3 * DAY), profile("Sam", NOW - 30 * DAY)], doors)
    reply = st.handle("status update", "Adam")
    latest = datetime.fromtimestamp(NOW - 600).strftime("%I:%M %p").lstrip("0").lower()
    assert reply == ("New voice since your last update: Lily. "
                     f"At the front door: one ring and two motion alerts, the latest at {latest}. "
                     "It's 61 degrees and clear.")
    assert doors.asked == [(NOW - DAY, NOW)]


def test_second_update_only_says_whats_new(make):
    clock = SimpleNamespace(now=NOW)
    doors = Doors({("front door", "ring"): [NOW - 600]})
    st = make([profile("Lily", NOW - 3 * DAY)], doors, clock=lambda: clock.now)
    st.handle("status update", "Adam")
    clock.now = NOW + 3600
    doors.alerts = {}
    assert st.handle("status update", "Adam") == f"{S.MSG_NO_NEWS} It's 61 degrees and clear."
    assert doors.asked[-1] == (NOW, NOW + 3600)


def test_alerts_never_reach_back_more_than_a_day(make, tmp_path):
    S.Ledger(tmp_path / "ledger.json").mark("adam", NOW - 5 * DAY)
    doors = Doors()
    make(doors=doors).handle("status update", "Adam")
    assert doors.asked == [(NOW - DAY, NOW)]


def test_a_failed_doorbell_check_is_said_and_reported_again(make, tmp_path):
    st = make(doors=Doors(fail=True))
    assert S.MSG_DOOR_FAILED in st.handle("status update", "Adam")
    assert S.Ledger(tmp_path / "ledger.json").last("Adam") is None


def test_others_get_the_weather_only(make, tmp_path):
    doors = Doors({("front door", "ring"): [NOW - 600]})
    st = make([profile("Lily", NOW - DAY)], doors)
    assert st.handle("status update", "Lily") == "It's 61 degrees and clear."
    assert st.handle("status update", None) == "It's 61 degrees and clear."
    assert doors.asked == [] and S.Ledger(tmp_path / "ledger.json").last("Lily") is None


def test_weather_failure_is_said(make):
    def broken():
        raise RuntimeError("no network")
    assert make(weather=broken).handle("status update", "Adam").endswith(S.MSG_WEATHER_FAILED)


def test_no_doorbell_configured(tmp_path):
    st = S.StatusUpdate(lambda: [], None, lambda: "Clear.", ledger=S.Ledger(tmp_path / "l.json"), clock=lambda: NOW)
    assert st.handle("status update", "Adam") == f"{S.MSG_NO_NEWS} Clear."


def test_doors_sentence_two_doors_and_plurals():
    t = NOW
    text = S.doors_sentence({("front door", "ring"): [t, t - 1], ("back door", "motion"): [t - 5]})
    assert text.startswith("At the back door: one motion alert, the latest at ")
    assert "At the front door: two rings, the latest at" in text


def test_voices_sentence():
    assert S.voices_sentence(["Lily", "Sam"]) == "New voices since your last update: Lily and Sam."


# -- failed remote-app logins (Adam, 2026-11-12) --------------------------------------------------
def alerts(failed=None, locked=None, unknown=0):
    from homeai.remote.accounts import LoginAlerts
    return LoginAlerts(failed or {}, locked or {}, unknown)


class Logins:
    def __init__(self, result=None, fail=False):
        self.result, self.fail, self.asked = result if result is not None else alerts(), fail, []

    def __call__(self, since, until):
        self.asked.append((since, until))
        if self.fail:
            raise RuntimeError("database down")
        return self.result


def with_logins(tmp_path, logins):
    return S.StatusUpdate(lambda: [], None, lambda: "Clear.", ledger=S.Ledger(tmp_path / "l.json"),
                          clock=lambda: NOW, login_alerts=logins)


def test_failed_logins_are_in_an_admins_update(tmp_path):
    until = NOW + 3000
    logins = Logins(alerts({"ani": 4, "adam": 1}, {"ani": until}, unknown=2))
    reply = with_logins(tmp_path, logins).handle("status update", "Adam")
    t = datetime.fromtimestamp(until).strftime("%I:%M %p").lstrip("0").lower()
    assert reply == ("One failed app login for Adam. Four failed app logins for Ani; that login is locked "
                     f"until {t}. Two failed app logins with a name that has no login. Clear.")
    assert logins.asked == [(NOW - S.FIRST_LOOK_S, NOW)]


def test_no_failed_logins_is_no_news(tmp_path):
    assert with_logins(tmp_path, Logins()).handle("status update", "Adam") == f"{S.MSG_NO_NEWS} Clear."


def test_a_failed_login_check_is_said_and_reported_again(tmp_path):
    reply = with_logins(tmp_path, Logins(fail=True)).handle("status update", "Adam")
    assert S.MSG_LOGINS_FAILED in reply and S.Ledger(tmp_path / "l.json").last("Adam") is None


def test_others_never_hear_about_logins(tmp_path):
    logins = Logins(alerts({"ani": 4}))
    assert with_logins(tmp_path, logins).handle("status update", "Ani") == "Clear."
    assert logins.asked == []


def test_a_lock_without_new_failures_is_still_said():
    assert S.logins_sentence(alerts(locked={"ani": NOW})).startswith("Ani's app login is locked until ")


def test_the_real_check_reads_the_auth_audit(clean_db):
    import psycopg
    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute("TRUNCATE auth.audit, auth.devices, auth.accounts CASCADE")
        conn.execute("INSERT INTO auth.audit (at, username, event) VALUES (to_timestamp(%s), 'ani', 'login_failed')",
                     (NOW - 60,))
    assert S.db_login_alerts(clean_db)(NOW - 3600, NOW).failed == {"ani": 1}


def test_status_update_built_with_memory_checks_logins():
    st = S.build(None, ("Adam",), lambda: [], "Boston", "dbname=nowhere")
    assert st._logins is not None and S.build(None, ("Adam",), lambda: [], "Boston")._logins is None


def test_broken_ledger_starts_fresh(tmp_path):
    path = tmp_path / "l.json"
    path.write_text("{not json")
    ledger = S.Ledger(path)
    assert ledger.last("Adam") is None
    ledger.mark("Adam", 5.0)
    assert ledger.last("adam") == 5.0


# -- daemon hook ----------------------------------------------------------------------



class FakeStatus:
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def handle(self, text, speaker):
        self.seen.append((text, speaker))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply
