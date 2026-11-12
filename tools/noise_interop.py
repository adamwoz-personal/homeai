#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Remote app Phase 0: prove Python and Kotlin speak the same Noise (REMOTE_APP_PLAN §5.3, §10).

Why: the app (Kotlin, noise-java) and homeai-remote (Python, noiseprotocol)
must agree byte for byte on Noise_NK_25519_ChaChaPoly_SHA256, or the phone
can't talk to home. Neither library's interop was verified. With fixed test
keys the handshake and every transport message are deterministic, so each
side writes the bytes it produces and the other must (1) produce the same
bytes and (2) decrypt them, and both must reject a flipped byte.

The keys below are public test values, never used for a real pairing.

    .venv/bin/python tools/noise_interop.py write     # Python's vectors -> tests/fixtures/noise/nk-python.txt
    .venv/bin/python tools/noise_interop.py check FILE  # Python checks a vector file (e.g. Kotlin's)
    .venv/bin/python tools/noise_interop.py kotlin    # build the Kotlin side, cross-check both ways
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from noise.connection import Keypair, NoiseConnection

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "noise"
KOTLIN = REPO / "tools" / "noise_interop_kotlin"
PROTOCOL = b"Noise_NK_25519_ChaChaPoly_SHA256"
MAX_FRAME = 65535          # one Noise message per relay frame (§5.3)
TAG = 16

# Public test values: sha256 of a label, so anyone can recompute them. Never a real pairing key.
TEST_KEYS = {name: hashlib.sha256(f"homeai noise test {name}".encode()).hexdigest()
             for name in ("home_static_priv", "app_ephemeral_priv", "home_ephemeral_priv")}
PROLOGUE = b"homeai-remote/1"
HANDSHAKE_PAYLOADS = (b'{"v":1,"type":"hello"}', b"")
APP_TO_HOME = (b"", b'{"v":1,"id":"1","type":"auth.login","body":{}}', bytes(range(256)) * 255 + bytes(range(239)))
HOME_TO_APP = (b'{"v":1,"type":"auth.ok"}', b"x" * 1000)


def public_of(priv_hex: str) -> bytes:
    return X25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex)).public_key().public_bytes_raw()


def _pair() -> tuple[NoiseConnection, NoiseConnection]:
    """App (initiator) and home (responder) with the fixed test keys; the library warns that fixed ephemerals are
    for testing only, which is the point here."""
    warnings.filterwarnings("ignore", message="One of ephemeral keypairs is already set")
    app = NoiseConnection.from_name(PROTOCOL)
    app.set_as_initiator()
    app.set_keypair_from_public_bytes(Keypair.REMOTE_STATIC, public_of(TEST_KEYS["home_static_priv"]))
    app.set_keypair_from_private_bytes(Keypair.EPHEMERAL, bytes.fromhex(TEST_KEYS["app_ephemeral_priv"]))
    home = NoiseConnection.from_name(PROTOCOL)
    home.set_as_responder()
    home.set_keypair_from_private_bytes(Keypair.STATIC, bytes.fromhex(TEST_KEYS["home_static_priv"]))
    home.set_keypair_from_private_bytes(Keypair.EPHEMERAL, bytes.fromhex(TEST_KEYS["home_ephemeral_priv"]))
    for side in (app, home):
        side.set_prologue(PROLOGUE)
        side.start_handshake()
    return app, home


def vectors() -> dict[str, bytes]:
    """Every byte Python produces for the fixed session, keyed as in the vector file."""
    app, home = _pair()
    out: dict[str, bytes] = {"prologue": PROLOGUE, "home_static_pub": public_of(TEST_KEYS["home_static_priv"])}
    out["msg1_payload"], out["msg2_payload"] = HANDSHAKE_PAYLOADS
    out["msg1"] = bytes(app.write_message(HANDSHAKE_PAYLOADS[0]))
    assert bytes(home.read_message(out["msg1"])) == HANDSHAKE_PAYLOADS[0]
    out["msg2"] = bytes(home.write_message(HANDSHAKE_PAYLOADS[1]))
    assert bytes(app.read_message(out["msg2"])) == HANDSHAKE_PAYLOADS[1]
    out["handshake_hash"] = app.get_handshake_hash()
    for n, plain in enumerate(APP_TO_HOME):
        out[f"a2h_{n}_plain"], out[f"a2h_{n}_cipher"] = plain, app.encrypt(plain)
    for n, plain in enumerate(HOME_TO_APP):
        out[f"h2a_{n}_plain"], out[f"h2a_{n}_cipher"] = plain, home.encrypt(plain)
    return out


def write(path: Path, vec: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k} {v.hex()}\n" for k, v in vec.items()), encoding="ascii")


def read(path: Path) -> dict[str, bytes]:
    vec = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, _, value = line.partition(" ")
        vec[key] = bytes.fromhex(value)
    return vec


def _flip(data: bytes) -> bytes:
    return data[:-1] + bytes([data[-1] ^ 1])


def check(theirs: dict[str, bytes]) -> list[str]:
    """Problems with another implementation's vectors; [] means Python agrees with every byte and decrypts them."""
    problems = []
    ours = vectors()
    for key in ours:
        if key not in theirs:
            problems.append(f"missing {key}")
        elif theirs[key] != ours[key]:
            problems.append(f"{key} differs")
    if problems:
        return problems
    # Python as home reads their app's bytes, Python as app reads their home's.
    app, home = _pair()
    app.write_message(HANDSHAKE_PAYLOADS[0])          # moves the app's state on; its bytes were compared above
    if bytes(home.read_message(theirs["msg1"])) != theirs["msg1_payload"]:
        problems.append("home could not read msg1")
    home.write_message(HANDSHAKE_PAYLOADS[1])
    if bytes(app.read_message(theirs["msg2"])) != theirs["msg2_payload"]:
        problems.append("app could not read msg2")
    for n in range(len(APP_TO_HOME)):
        if home.decrypt(theirs[f"a2h_{n}_cipher"]) != theirs[f"a2h_{n}_plain"]:
            problems.append(f"home could not decrypt a2h_{n}")
    for n in range(len(HOME_TO_APP)):
        if app.decrypt(theirs[f"h2a_{n}_cipher"]) != theirs[f"h2a_{n}_plain"]:
            problems.append(f"app could not decrypt h2a_{n}")
    return problems + tamper_problems(theirs)


def tamper_problems(vec: dict[str, bytes]) -> list[str]:
    """A flipped byte in the handshake or in a transport message must be refused."""
    problems = []
    _, home = _pair()
    try:
        home.read_message(_flip(vec["msg1"]))
        problems.append("tampered msg1 accepted")
    except Exception:  # noqa: BLE001 - the library raises several types for a bad tag
        pass
    app, home = _pair()
    app.write_message(HANDSHAKE_PAYLOADS[0])
    home.read_message(vec["msg1"])
    home.write_message(HANDSHAKE_PAYLOADS[1])
    app.read_message(vec["msg2"])
    try:
        home.decrypt(_flip(vec["a2h_1_cipher"]))
        problems.append("tampered a2h_1 accepted")
    except Exception:  # noqa: BLE001
        pass
    return problems


def frame_problems(vec: dict[str, bytes]) -> list[str]:
    """Every message must fit one relay frame."""
    return [f"{k} is {len(v)} bytes" for k, v in vec.items() if k.endswith(("_cipher", "msg1", "msg2"))
            and len(v) > MAX_FRAME]


def kotlin(out: Path, vectors_file: Path = FIXTURES / "nk-python.txt") -> subprocess.CompletedProcess:
    """Build and run the Kotlin side: it checks ``vectors_file`` (Python's) and writes its own to ``out``."""
    env = {**os.environ, "GRADLE_USER_HOME": os.environ.get("GRADLE_USER_HOME", str(Path.home() / ".gradle"))}
    return subprocess.run(["./gradlew", "-q", "--console=plain", "run", f"--args={vectors_file} {out}"],
                          cwd=KOTLIN, env=env, capture_output=True, text=True, timeout=900)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write")
    c = sub.add_parser("check")
    c.add_argument("file", type=Path)
    k = sub.add_parser("kotlin")
    k.add_argument("--out", type=Path, default=Path.home() / "homeai-bench" / "noise-interop-kotlin.txt")
    args = ap.parse_args(argv)
    if args.cmd == "write":
        vec = vectors()
        problems = frame_problems(vec) + tamper_problems(vec)
        write(FIXTURES / "nk-python.txt", vec)
    elif args.cmd == "check":
        problems = check(read(args.file))
    else:
        proc = kotlin(args.out)
        print(proc.stdout.strip())
        if proc.returncode:
            print(proc.stderr.strip()[-3000:], file=sys.stderr)
            return 1
        problems = check(read(args.out))
    print("\n".join(problems) if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
