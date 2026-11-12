# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Remote app logins (homeai/remote/accounts.py, REMOTE_APP_PLAN §4).

Why: these rules stand between the internet and the house. A lockout that
resets, an unknown name that answers differently, a token kept in clear, a
password change that leaves old phones logged in, or memory roles able to read
password hashes would each be a hole. Runs on the throwaway test database
(tests/memory_db.py), never the real one.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("argon2", reason="pip install .[remote]")

from homeai.remote import accounts as A  # noqa: E402

PW = "correct horse battery"


class Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 11, 3, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


@pytest.fixture
def auth_db(clean_db):
    import psycopg

    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute("TRUNCATE auth.audit, auth.devices, auth.accounts CASCADE")
        for user, child in (("adam", False), ("ani", True), ("ryan", False)):
            conn.execute("INSERT INTO memory.people (username, display_name, is_child) VALUES (%s, %s, %s)",
                         (user, user.title(), child))
        conn.execute("INSERT INTO memory.identities (channel, handle, username, verified) VALUES "
                     "('voice', 'Adam', 'adam', true), ('voice', 'Ani', 'ani', true), ('voice', 'Ryan', 'ryan', false)")
    return clean_db


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def acc(auth_db, clock):
    import psycopg
    from argon2 import PasswordHasher

    return A.Accounts(lambda: psycopg.connect(auth_db), now=clock,
                      hasher=PasswordHasher(time_cost=1, memory_cost=8, parallelism=1))


def rows(dsn, sql, *args):
    import psycopg

    with psycopg.connect(dsn) as conn:
        return conn.execute(sql, args).fetchall()


# -- setting passwords ---------------------------------------------------------------------------
@pytest.mark.parametrize("user, pw, why", [
    ("adam", "short pw", "at least 12"),
    ("nobody", PW, "no person"),
    ("ryan", PW, "no verified voice"),
    ("household", PW, "not a person"),
])
def test_set_password_refusals(acc, user, pw, why):
    with pytest.raises(A.AuthError, match=why):
        acc.set_password(user, pw)


def test_the_hash_is_argon2id_and_the_password_is_nowhere(acc, auth_db):
    acc.set_password("adam", PW)
    (h,), = rows(auth_db, "SELECT password_hash FROM auth.accounts")
    assert h.startswith("$argon2id$") and PW not in h
    assert not any(PW in str(r) for r in rows(auth_db, "SELECT detail FROM auth.audit"))


def test_guardian_access_defaults_to_child_and_stays_unless_told(acc, auth_db):
    acc.set_password("ani", PW)
    acc.set_password("adam", PW)
    got = dict(rows(auth_db, "SELECT username, guardian_access FROM auth.accounts"))
    assert got == {"ani": True, "adam": False}
    acc.set_password("ani", PW + "!", guardian_access=False)
    acc.set_password("ani", PW + "?")
    assert dict(rows(auth_db, "SELECT username, guardian_access FROM auth.accounts"))["ani"] is False


# -- logging in ----------------------------------------------------------------------------------
def test_login_gives_a_device_that_resumes(acc, auth_db):
    acc.set_password("adam", PW)
    r = acc.login("Adam ", PW, "  Pixel   tablet ")
    assert r.ok and r.username == "adam" and len(r.device_id) == 32 and r.token
    assert acc.resume(r.device_id, r.token) == "adam"
    assert [d["name"] for d in acc.devices("adam")] == ["Pixel tablet"]
    assert rows(auth_db, "SELECT channel, username, verified FROM memory.identities WHERE handle = %s",
                r.device_id) == [("app", "adam", True)]


def test_the_token_is_stored_only_as_its_hash(acc, auth_db):
    import hashlib

    acc.set_password("adam", PW)
    r = acc.login("adam", PW, "p")
    (stored,), = rows(auth_db, "SELECT token_sha256 FROM auth.devices")
    assert bytes(stored) == hashlib.sha256(A.token_raw(r.token)).digest()
    dump = str(rows(auth_db, "SELECT * FROM auth.devices")) + str(rows(auth_db, "SELECT * FROM auth.audit"))
    assert r.token not in dump


@pytest.mark.parametrize("token", ["", "x", A.token_text(b"\0" * 32), "!!!not base64!!!"])
def test_a_wrong_token_does_not_resume(acc, token):
    acc.set_password("adam", PW)
    r = acc.login("adam", PW, "p")
    assert acc.resume(r.device_id, token) is None
    assert acc.resume("0" * 32, r.token) is None


def test_the_fourth_failure_locks_for_an_hour(acc, clock):
    acc.set_password("adam", PW)
    waits = [acc.login("adam", "wrong password!", "p").retry_after_s for _ in range(3)]
    assert waits == [0, 0, 0]
    assert acc.login("adam", "wrong password!", "p").retry_after_s == 3600       # 4th: locked
    refused = acc.login("adam", PW, "p")                                         # right, but locked
    assert not refused.ok and refused.code == A.BAD and refused.retry_after_s == 3600
    clock.advance(minutes=59)
    assert not acc.login("adam", PW, "p").ok
    clock.advance(minutes=1)
    assert acc.login("adam", PW, "p").ok
    assert acc.login("adam", "wrong password!", "p").retry_after_s == 0         # success reset the count


def test_login_alerts_for_the_status_update(acc, auth_db, clock):
    import psycopg

    acc.set_password("adam", PW)
    acc.set_password("ani", PW)
    start = clock()
    clock.advance(seconds=1)
    acc.login("adam", "wrong password!", "p")
    clock.advance(seconds=59)
    for _ in range(4):
        acc.login("ani", "wrong password!", "p")
    acc.login("ani", PW, "p")                       # refused while locked: counted too
    acc.login("mallory", "wrong password!", "p")
    acc.login("adam", PW, "p")                      # a good login is not an alert
    got = A.login_alerts(lambda: psycopg.connect(auth_db), start, clock())
    assert got.failed == {"adam": 1, "ani": 5} and got.unknown == 1
    assert got.locked == {"ani": start + timedelta(minutes=1, hours=1)}
    later = A.login_alerts(lambda: psycopg.connect(auth_db), clock(), clock() + timedelta(hours=2))
    assert not later                               # nothing new, and the lock has run out


def test_an_unknown_name_answers_exactly_like_a_wrong_password(acc, clock):
    acc.set_password("adam", PW)

    def run(user):
        out = []
        for _ in range(5):
            r = acc.login(user, "wrong password!", "p")
            out.append((r.ok, r.code, r.retry_after_s, r.token))
            clock.advance(seconds=r.retry_after_s)
        out.append(tuple(vars(acc.login(user, "wrong password!", "p")).values())[:3])
        return out

    assert run("adam") == run("mallory")


def test_unknown_name_counts_are_in_the_audit_and_survive_a_restart(acc, auth_db, clock):
    import psycopg
    from argon2 import PasswordHasher

    for _ in range(3):
        assert acc.login("mallory", "wrong password!", "p").retry_after_s == 0
    restarted = A.Accounts(lambda: psycopg.connect(auth_db), now=clock,
                           hasher=PasswordHasher(time_cost=1, memory_cost=8, parallelism=1))
    assert restarted.login("mallory", "wrong password!", "p").retry_after_s == 3600    # 4th, after the restart
    clock.advance(minutes=30)
    assert restarted.login("mallory", "wrong password!", "p").retry_after_s == 1800    # still locked
    assert restarted.login("eve", "wrong password!", "p").retry_after_s == 0          # names counted apart
    got = rows(auth_db, "SELECT event, detail->>'name' FROM auth.audit ORDER BY id")
    assert got == [("login_unknown", "mallory")] * 4 + [("login_unknown_refused", "mallory"),
                                                        ("login_unknown", "eve")]
    alerts = A.login_alerts(lambda: psycopg.connect(auth_db), clock() - timedelta(days=1), clock())
    assert alerts.unknown == 6                                                       # tries while locked count
    clock.advance(minutes=30)
    assert restarted.login("mallory", "wrong password!", "p").retry_after_s == 3600    # one try per hour after


def test_a_password_change_logs_every_device_out(acc, auth_db):
    acc.set_password("adam", PW)
    r1, r2 = acc.login("adam", PW, "a"), acc.login("adam", PW, "b")
    assert acc.set_password("adam", PW + "2") == 2
    assert acc.resume(r1.device_id, r1.token) is None and acc.resume(r2.device_id, r2.token) is None
    assert rows(auth_db, "SELECT count(*) FROM memory.identities WHERE channel = 'app'") == [(0,)]


def test_revoke_and_logout_end_one_device(acc):
    acc.set_password("adam", PW)
    r1, r2 = acc.login("adam", PW, "a"), acc.login("adam", PW, "b")
    assert acc.revoke(r1.device_id) and not acc.revoke(r1.device_id)
    assert acc.resume(r1.device_id, r1.token) is None and acc.resume(r2.device_id, r2.token) == "adam"
    assert acc.logout(r2.device_id) and acc.resume(r2.device_id, r2.token) is None


def test_a_device_idle_for_90_days_expires(acc, clock):
    acc.set_password("adam", PW)
    r = acc.login("adam", PW, "p")
    clock.advance(days=89)
    assert acc.resume(r.device_id, r.token) == "adam"          # use resets the idle clock
    clock.advance(days=90, seconds=1)
    assert acc.resume(r.device_id, r.token) is None
    clock.advance(seconds=1)
    assert acc.resume(r.device_id, r.token) is None


def test_a_disabled_login_fails_and_loses_its_devices(acc):
    acc.set_password("adam", PW)
    r = acc.login("adam", PW, "p")
    acc.set_disabled("adam", True)
    assert acc.resume(r.device_id, r.token) is None
    assert not acc.login("adam", PW, "p").ok
    acc.set_disabled("adam", False)
    assert acc.login("adam", PW, "p").ok


def test_everything_is_audited(acc, auth_db, clock):
    acc.set_password("adam", PW)
    r = acc.login("adam", PW, "p")
    acc.login("adam", "wrong password!", "p")
    acc.login("nobody", "x" * 12, "p")
    acc.resume(r.device_id, r.token)
    acc.resume(r.device_id, "bad")
    acc.revoke(r.device_id)
    events = [e for (e,) in rows(auth_db, "SELECT event FROM auth.audit ORDER BY id")]
    assert events == ["password_set", "login_ok", "login_failed", "login_unknown", "resume_ok", "resume_failed",
                      "device_revoked"]


# -- the schema ----------------------------------------------------------------------------------
@pytest.mark.parametrize("role", ["homeai_app", "homeai_mcp", "homeai_ro"])
def test_memory_roles_cannot_read_logins(auth_db, role):
    import psycopg

    with psycopg.connect(auth_db) as conn:
        conn.execute(f"SET ROLE {role}")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM auth.accounts")


def test_app_is_a_memory_channel(auth_db):
    import psycopg

    with psycopg.connect(auth_db, autocommit=True) as conn:
        conn.execute("INSERT INTO memory.identities (channel, handle, username) VALUES ('app', 'x', 'adam')")
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("INSERT INTO memory.identities (channel, handle, username) VALUES ('fax', 'y', 'adam')")


def test_wait_schedule():
    assert [A.wait_after(n).total_seconds() for n in range(1, 7)] == [0, 0, 0, 3600, 3600, 3600]


# -- bin/homeai-user -----------------------------------------------------------------------------
def cli(acc, *argv, answers=(PW, PW)):
    import io

    from homeai.remote import user_cli

    out, it = io.StringIO(), iter(answers)
    code = user_cli.main(list(argv), accounts=acc, out=out, prompt=lambda _: next(it))
    return code, out.getvalue()


def test_cli_sets_a_password_from_the_prompt_and_lists_it(acc):
    assert cli(acc, "set-password", "ani") == (0, "password set for ani; 0 device(s) logged out\n")
    code, text = cli(acc, "list")
    assert code == 0 and text.startswith("ani ") and "guardian yes" in text and "devices 0" in text


def test_cli_refuses_two_different_passwords(acc, capsys):
    assert cli(acc, "set-password", "adam", answers=(PW, PW + "x"))[0] == 1
    assert "differ" in capsys.readouterr().err and acc.accounts() == []


def test_cli_revokes_a_device_and_says_when_there_is_none(acc, capsys):
    cli(acc, "set-password", "adam")
    r = acc.login("adam", PW, "tablet")
    code, text = cli(acc, "devices", "adam")
    assert code == 0 and r.device_id in text and "tablet" in text
    assert cli(acc, "revoke", r.device_id) == (0, f"revoked {r.device_id}\n")
    assert cli(acc, "revoke", r.device_id)[0] == 1 and "no live device" in capsys.readouterr().err


def test_cli_disable_enable(acc):
    cli(acc, "set-password", "adam")
    assert cli(acc, "disable", "adam") == (0, "adam disabled\n")
    assert "disabled" in cli(acc, "list")[1]
    assert cli(acc, "enable", "adam") == (0, "adam enabled\n")


def test_cli_pair_is_not_built_yet(acc):
    assert cli(acc, "pair", "adam")[0] == 2
