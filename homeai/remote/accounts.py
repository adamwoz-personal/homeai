# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Remote app logins and devices (REMOTE_APP_PLAN §4, Phase 1).

Rules, each from the plan:

* Passwords are Argon2id hashes, at least 12 characters, set only on this
  machine (``bin/homeai-user``). Only people with a verified voice identity
  may have one (D5): the voice decides who is eligible, it is never a factor.
* Failed logins: the 4th wrong one in a row locks the login for an hour (Adam,
  2026-11-12), and the status update reports failures to admins (``login_alerts``).
  Unknown usernames get the same answers, lock and (one hash) time as wrong
  passwords, so a login attempt can't tell whether a name exists. Their counts
  are kept in the audit (the name tried, in ``detail``), so a restart forgets nothing.
* A login creates a device: 256 random bits shown to the app once, stored as
  sha256. Tokens expire after 90 days unused. A password change revokes every
  device. Each device is a memory identity (channel 'app', handle device_id),
  so memory attributes app turns to the person.
* Everything is audited, never with message content or secrets.

The tables live in the ``auth`` schema (migrations/0004_auth.sql), readable
only as role ``homeai_auth``. ``connect`` returns a psycopg connection; times
come from ``now`` so tests can move the clock.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

MIN_PASSWORD = 12
LOCK_AT = 4
LOCK_FOR = timedelta(hours=1)
TOKEN_IDLE = timedelta(days=90)
ROLE = "homeai_auth"
NOT_ELIGIBLE = ("household", "guest")
BAD = "bad_credentials"


class AuthError(ValueError):
    """A refused admin action (bin/homeai-user prints the message)."""


@dataclass(frozen=True)
class LoginResult:
    ok: bool
    code: str = "ok"
    retry_after_s: int = 0
    username: str | None = None
    device_id: str | None = None
    token: str | None = None          # shown to the app once; never stored or logged


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def token_text(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def token_raw(text: str) -> bytes | None:
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == 32 else None


def wait_after(failures: int) -> timedelta:
    """How long the next attempt is refused after ``failures`` wrong ones in a row."""
    return LOCK_FOR if failures >= LOCK_AT else timedelta(0)


class Accounts:
    def __init__(self, connect: Callable, *, hasher=None, now: Callable[[], datetime] = _utc_now,
                 random_bytes: Callable[[int], bytes] = secrets.token_bytes) -> None:
        from argon2 import PasswordHasher

        self.connect = connect
        self.hasher = hasher or PasswordHasher()       # argon2-cffi's default type is Argon2id
        self.now = now
        self.random_bytes = random_bytes
        self._dummy_hash = self.hasher.hash(secrets.token_hex(16))

    # -- plumbing ---------------------------------------------------------------------------------
    def _conn(self):
        conn = self.connect()
        conn.execute(f"SET ROLE {ROLE}")
        return conn

    def _audit(self, conn, event: str, username: str | None = None, device_id: str | None = None,
               **detail) -> None:
        from psycopg.types.json import Jsonb

        conn.execute("INSERT INTO auth.audit (at, username, device_id, event, detail) VALUES (%s, %s, %s, %s, %s)",
                     (self.now(), username, device_id, event, Jsonb(detail)))

    def _verify(self, password_hash: str, password: str) -> bool:
        from argon2.exceptions import InvalidHashError, VerificationError

        try:
            return self.hasher.verify(password_hash, password)
        except (VerificationError, InvalidHashError):
            return False

    def _revoke_all(self, conn, username: str) -> int:
        rows = conn.execute("UPDATE auth.devices SET revoked_at = %s WHERE username = %s AND revoked_at IS NULL "
                            "RETURNING device_id", (self.now(), username)).fetchall()
        for (device_id,) in rows:
            conn.execute("DELETE FROM memory.identities WHERE channel = 'app' AND handle = %s", (device_id,))
        return len(rows)

    # -- admin (bin/homeai-user) -----------------------------------------------------------------
    def set_password(self, username: str, password: str, *, guardian_access: bool | None = None) -> int:
        """Create or change a login; returns how many devices were revoked. Raises AuthError if refused."""
        username = username.strip().lower()
        if len(password) < MIN_PASSWORD:
            raise AuthError(f"a password needs at least {MIN_PASSWORD} characters")
        if username in NOT_ELIGIBLE:
            raise AuthError(f"{username} is not a person")
        with self._conn() as conn:
            person = conn.execute("SELECT is_child FROM memory.people WHERE username = %s", (username,)).fetchone()
            if person is None:
                raise AuthError(f"no person {username!r} in memory (homeai-memory people)")
            voiced = conn.execute("SELECT 1 FROM memory.identities WHERE username = %s AND channel = 'voice' "
                                  "AND verified", (username,)).fetchone()
            if voiced is None:
                raise AuthError(f"{username} has no verified voice: enrol first (\"remember my voice as ...\")")
            exists = conn.execute("SELECT guardian_access FROM auth.accounts WHERE username = %s",
                                  (username,)).fetchone()
            guardian = (person[0] if exists is None else exists[0]) if guardian_access is None else guardian_access
            conn.execute(
                "INSERT INTO auth.accounts (username, password_hash, password_changed_at, guardian_access) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash, "
                "failed_logins = 0, locked_until = NULL, password_changed_at = EXCLUDED.password_changed_at, "
                "guardian_access = EXCLUDED.guardian_access",
                (username, self.hasher.hash(password), self.now(), guardian))
            revoked = self._revoke_all(conn, username)
            self._audit(conn, "password_set", username, created=exists is None, revoked=revoked)
        return revoked

    def set_disabled(self, username: str, disabled: bool) -> None:
        with self._conn() as conn:
            row = conn.execute("UPDATE auth.accounts SET disabled = %s WHERE username = %s RETURNING username",
                               (disabled, username)).fetchone()
            if row is None:
                raise AuthError(f"no login for {username!r}")
            revoked = self._revoke_all(conn, username) if disabled else 0
            self._audit(conn, "disabled" if disabled else "enabled", username, revoked=revoked)

    def accounts(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT a.username, a.disabled, a.locked_until, a.guardian_access, a.password_changed_at, "
                "count(d.device_id) FILTER (WHERE d.revoked_at IS NULL) FROM auth.accounts a "
                "LEFT JOIN auth.devices d USING (username) GROUP BY a.username ORDER BY a.username").fetchall()
        keys = ("username", "disabled", "locked_until", "guardian_access", "password_changed_at", "devices")
        return [dict(zip(keys, r)) for r in rows]

    def devices(self, username: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT device_id, name, created_at, last_seen, revoked_at FROM auth.devices "
                                "WHERE username = %s ORDER BY created_at", (username,)).fetchall()
        keys = ("device_id", "name", "created_at", "last_seen", "revoked_at")
        return [dict(zip(keys, r)) for r in rows]

    def revoke(self, device_id: str, *, event: str = "device_revoked") -> bool:
        with self._conn() as conn:
            row = conn.execute("UPDATE auth.devices SET revoked_at = %s WHERE device_id = %s AND revoked_at IS NULL "
                               "RETURNING username", (self.now(), device_id)).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM memory.identities WHERE channel = 'app' AND handle = %s", (device_id,))
            self._audit(conn, event, row[0], device_id)
        return True

    # -- the app (homeai-remote) -----------------------------------------------------------------
    def login(self, username: str, password: str, device_name: str) -> LoginResult:
        username = username.strip().lower()[:64]
        now = self.now()
        with self._conn() as conn:
            row = conn.execute("SELECT password_hash, failed_logins, locked_until, disabled FROM auth.accounts "
                               "WHERE username = %s FOR UPDATE", (username,)).fetchone()
            if row is None:
                return self._unknown_login(conn, username, password, now)
            password_hash, failures, locked_until, disabled = row
            if locked_until and locked_until > now:
                self._verify(self._dummy_hash, password)   # same time as a real check
                self._audit(conn, "login_refused", username, retry_after_s=self._seconds(locked_until - now))
                return LoginResult(False, BAD, self._seconds(locked_until - now))
            if disabled or not self._verify(password_hash, password):
                failures += 1
                wait = wait_after(failures)
                conn.execute("UPDATE auth.accounts SET failed_logins = %s, locked_until = %s WHERE username = %s",
                             (failures, now + wait if wait else None, username))
                self._audit(conn, "login_failed", username, failures=failures, disabled=disabled)
                return LoginResult(False, BAD, self._seconds(wait))
            if self.hasher.check_needs_rehash(password_hash):
                conn.execute("UPDATE auth.accounts SET password_hash = %s WHERE username = %s",
                             (self.hasher.hash(password), username))
            conn.execute("UPDATE auth.accounts SET failed_logins = 0, locked_until = NULL WHERE username = %s",
                         (username,))
            device_id, token = self._new_device(conn, username, device_name, now)
            self._audit(conn, "login_ok", username, device_id)
        return LoginResult(True, "ok", 0, username, device_id, token)

    def _unknown_login(self, conn, username: str, password: str, now: datetime) -> LoginResult:
        """No such login: spend one hash and answer exactly like a wrong password, with the same delays."""
        self._verify(self._dummy_hash, password)
        # The count lives in the audit (Adam 2026-11-12), so it survives a restart: every failed try with this
        # name, as an account's count runs until a success, which an unknown name never has.
        failures, last = conn.execute("SELECT count(*), max(at) FROM auth.audit WHERE event = 'login_unknown' "
                                      "AND detail->>'name' = %s", (username,)).fetchone()
        locked_until = last + wait_after(failures) if failures and wait_after(failures) else None
        if locked_until and locked_until > now:
            self._audit(conn, "login_unknown_refused", None, name=username,
                        retry_after_s=self._seconds(locked_until - now))
            return LoginResult(False, BAD, self._seconds(locked_until - now))
        failures += 1
        wait = wait_after(failures)
        self._audit(conn, "login_unknown", None, name=username, failures=failures)
        return LoginResult(False, BAD, self._seconds(wait))

    def _new_device(self, conn, username: str, device_name: str, now: datetime) -> tuple[str, str]:
        device_id = self.random_bytes(16).hex()
        raw = self.random_bytes(32)
        name = " ".join(device_name.split())[:64] or "phone"
        conn.execute("INSERT INTO auth.devices (device_id, username, name, token_sha256, created_at, last_seen) "
                     "VALUES (%s, %s, %s, %s, %s, %s)", (device_id, username, name, hashlib.sha256(raw).digest(), now, now))
        conn.execute("INSERT INTO memory.identities (channel, handle, username, verified) VALUES ('app', %s, %s, true)",
                     (device_id, username))
        return device_id, token_text(raw)

    def resume(self, device_id: str, token: str) -> str | None:
        """The username for a live device and its token, else None. Expires devices idle for 90 days."""
        now = self.now()
        raw = token_raw(token)
        with self._conn() as conn:
            row = conn.execute("SELECT d.username, d.token_sha256, d.last_seen, d.revoked_at, a.disabled "
                               "FROM auth.devices d JOIN auth.accounts a USING (username) WHERE d.device_id = %s",
                               (device_id,)).fetchone()
            expected = bytes(row[1]) if row else b"\0" * 32
            given = hashlib.sha256(raw).digest() if raw else b""
            if row is None or not hmac.compare_digest(expected, given) or row[3] is not None or row[4]:
                self._audit(conn, "resume_failed", row[0] if row else None, device_id if row else None)
                return None
            username, _, last_seen, _, _ = row
            if now - last_seen > TOKEN_IDLE:
                conn.execute("UPDATE auth.devices SET revoked_at = %s WHERE device_id = %s", (now, device_id))
                conn.execute("DELETE FROM memory.identities WHERE channel = 'app' AND handle = %s", (device_id,))
                self._audit(conn, "device_expired", username, device_id)
                return None
            conn.execute("UPDATE auth.devices SET last_seen = %s WHERE device_id = %s", (now, device_id))
            self._audit(conn, "resume_ok", username, device_id)
        return username

    def logout(self, device_id: str) -> bool:
        return self.revoke(device_id, event="logout")

    @staticmethod
    def _seconds(delta: timedelta) -> int:
        return max(0, int(-(-delta.total_seconds() // 1)))


@dataclass(frozen=True)
class LoginAlerts:
    """Failed app logins in a time window, for the status update. Names, counts and times only."""
    failed: dict[str, int]                 # username -> wrong passwords + tries while locked
    locked: dict[str, datetime]            # username -> locked until (still locked at the window's end)
    unknown: int = 0                       # tries with a name that has no login

    def __bool__(self) -> bool:
        return bool(self.failed or self.locked or self.unknown)


def login_alerts(connect: Callable, since: datetime, until: datetime) -> LoginAlerts:
    """What the audit says went wrong with logins between ``since`` and ``until``."""
    with connect() as conn:
        conn.execute(f"SET ROLE {ROLE}")
        failed = dict(conn.execute(
            "SELECT username, count(*) FROM auth.audit WHERE at > %s AND at <= %s "
            "AND event IN ('login_failed', 'login_refused') GROUP BY username", (since, until)).fetchall())
        unknown = conn.execute("SELECT count(*) FROM auth.audit WHERE at > %s AND at <= %s "
                               "AND event IN ('login_unknown', 'login_unknown_refused')", (since, until)).fetchone()[0]
        locked = dict(conn.execute("SELECT username, locked_until FROM auth.accounts WHERE locked_until > %s",
                                   (until,)).fetchall())
    return LoginAlerts(failed, locked, unknown)
