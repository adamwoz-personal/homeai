#!/usr/bin/env python3
"""Patch firebase-messaging inside the Home Assistant container so Ring's doorbell push alerts survive bad messages.

Why: on 2026-10-16 10:28 Ring's push listener (firebase_messaging FcmPushClient 0.4.5, the newest release) got a
message whose crypto-key was base64 without padding. ``_decrypt_raw_data`` decoded it strictly ("Incorrect
padding"), the error escaped the listen loop, and the client shut itself down for good: Jarvis missed every ring
and motion for a month (docs/RUNBOOK.md "Ring events stuck"). Two fixes, both idempotent:

  1. pad the crypto-key and salt before decoding (the library already pads the private key and secret this way);
  2. an error while handling one message is logged and skipped instead of ending the listener.

Runs inside the container with the container's python (stdlib only), from tools/ha/ha_container.sh after start and
update; HA must restart to load it. Lines that no longer match (a new library version) are reported, never forced.

Usage: docker exec -i homeassistant python3 - [--check] < tools/ha/patch_firebase_messaging.py
Exit: 0 patched now, 3 already patched, 2 didn't match (left alone), 1 file missing.
"""
from __future__ import annotations

import glob
import sys

MARK = "homeai patch"
PATCHES = [
    ('        crypto_key = urlsafe_b64decode(crypto_key_str.encode("ascii"))\n',
     '        crypto_key = urlsafe_b64decode(crypto_key_str.encode("ascii") + b"========")  # homeai patch\n'),
    ('        salt = urlsafe_b64decode(salt_str.encode("ascii"))\n',
     '        salt = urlsafe_b64decode(salt_str.encode("ascii") + b"========")  # homeai patch\n'),
    ('                    elif msg := await self._receive_msg():\n'
     '                        await self._handle_message(msg)\n',
     '                    elif msg := await self._receive_msg():\n'
     '                        try:\n'
     '                            await self._handle_message(msg)\n'
     '                        except Exception:  # homeai patch: one bad message must not end the listener\n'
     '                            _logger.exception("homeai patch: skipped a message that failed to handle")\n'),
]


def patch(text: str) -> tuple[str, str]:
    """Returns (new text, status): "patched", "already" or "mismatch" (then the text is unchanged)."""
    if all(new in text for _, new in PATCHES):
        return text, "already"
    if not all(old in text or new in text for old, new in PATCHES):
        return text, "mismatch"
    for old, new in PATCHES:
        if new not in text:
            text = text.replace(old, new, 1)
    return text, "patched"


def main(argv: list[str]) -> int:
    paths = glob.glob("/usr/local/lib/python3*/site-packages/firebase_messaging/fcmpushclient.py")
    if not paths:
        print("firebase_messaging not found")
        return 1
    path = paths[0]
    with open(path, encoding="utf-8") as f:
        text, status = patch(f.read())
    if status == "mismatch":
        print(f"{path}: lines changed (new library version?), left alone; re-check the bug upstream")
        return 2
    if status == "already":
        print(f"{path}: already patched")
        return 3
    if "--check" in argv:
        print(f"{path}: would patch")
        return 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{path}: patched; restart Home Assistant to load it")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
