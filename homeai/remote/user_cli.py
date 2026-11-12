# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""bin/homeai-user: remote app logins, run only at this machine (REMOTE_APP_PLAN §4).

    homeai-user list                         logins, devices in use, locks
    homeai-user set-password ani [--guardian | --no-guardian]
                                             create or change a login (typed at a prompt, never an argument);
                                             logs every device of that person out
    homeai-user devices adam                 that person's devices
    homeai-user revoke <device_id>           a lost phone is one command
    homeai-user disable ani / enable ani

Needs migration 0004 (``homeai-memory migrate``). Pairing (``pair``, the QR
code) comes with the gateway in Phase 2.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from typing import Callable

from .accounts import Accounts, AuthError


def _when(t) -> str:
    return t.strftime("%Y-%m-%d %H:%M") if t else "-"


def ask_password(prompt: Callable[[str], str] = getpass.getpass) -> str:
    first = prompt("New password: ")
    if prompt("Again: ") != first:
        raise AuthError("the two passwords differ")
    return first


def main(argv: list[str] | None = None, *, accounts: Accounts | None = None, out=None,
         prompt: Callable[[str], str] = getpass.getpass) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(prog="homeai-user", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sp = sub.add_parser("set-password")
    sp.add_argument("username")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--guardian", dest="guardian", action="store_true", default=None,
                   help="an admin may read this person's chats and images (default: on for children)")
    g.add_argument("--no-guardian", dest="guardian", action="store_false")
    for name in ("devices", "disable", "enable"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("revoke").add_argument("device_id")
    sub.add_parser("pair").add_argument("username")
    a = ap.parse_args(argv)
    if a.cmd == "pair":
        print("pairing needs the gateway's Noise key (REMOTE_APP_PLAN Phase 2); not built yet", file=out)
        return 2
    if accounts is None:
        import psycopg

        from ..config import Config

        dsn = Config().memory.dsn
        accounts = Accounts(lambda: psycopg.connect(dsn))
    try:
        if a.cmd == "list":
            for r in accounts.accounts():
                state = "disabled" if r["disabled"] else f"locked to {_when(r['locked_until'])}" \
                    if r["locked_until"] else "ok"
                print(f"{r['username']:<12} {state:<24} devices {r['devices']}  guardian "
                      f"{'yes' if r['guardian_access'] else 'no'}  password set {_when(r['password_changed_at'])}",
                      file=out)
        elif a.cmd == "set-password":
            revoked = accounts.set_password(a.username, ask_password(prompt), guardian_access=a.guardian)
            print(f"password set for {a.username}; {revoked} device(s) logged out", file=out)
        elif a.cmd == "devices":
            for d in accounts.devices(a.username):
                state = f"revoked {_when(d['revoked_at'])}" if d["revoked_at"] else f"last seen {_when(d['last_seen'])}"
                print(f"{d['device_id']}  {d['name']:<20} added {_when(d['created_at'])}  {state}", file=out)
        elif a.cmd == "revoke":
            if not accounts.revoke(a.device_id):
                raise AuthError(f"no live device {a.device_id}")
            print(f"revoked {a.device_id}", file=out)
        else:
            accounts.set_disabled(a.username, a.cmd == "disable")
            print(f"{a.username} {a.cmd}d", file=out)
    except AuthError as exc:
        print(f"homeai-user: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
