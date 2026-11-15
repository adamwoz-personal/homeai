"""tools/ha/patch_firebase_messaging.py on the real library file (firebase-messaging 0.4.5, MIT, copied from the
Home Assistant container 2026-11-15 as tests/fixtures/fcmpushclient-0.4.5.py.txt)."""
from __future__ import annotations

import base64
from pathlib import Path

from tools.ha import patch_firebase_messaging as P

REAL = (Path(__file__).parent / "fixtures" / "fcmpushclient-0.4.5.py.txt").read_text()


def test_patches_the_real_file_once_and_it_still_compiles():
    text, status = P.patch(REAL)
    assert status == "patched" and text.count("homeai patch") == 4
    compile(text, "fcmpushclient.py", "exec")
    again, status = P.patch(text)
    assert status == "already" and again == text


def test_a_changed_library_is_left_alone():
    changed = REAL.replace('crypto_key = urlsafe_b64decode(crypto_key_str.encode("ascii"))', "crypto_key = new_code()")
    assert P.patch(changed) == (changed, "mismatch")


def test_one_bad_message_is_skipped_inside_the_listen_loop():
    text, _ = P.patch(REAL)
    loop = text[text.index("while self.do_listen:"):text.index("except (OSError, EOFError) as osex:")]
    assert "except Exception:  # homeai patch" in loop and "await self._handle_message(msg)" in loop


def test_padding_added_the_library_way_decodes_the_crash_input():
    key = base64.urlsafe_b64encode(b"sixteen byte key!").rstrip(b"=")       # what Ring sent: no padding
    try:
        base64.urlsafe_b64decode(key)
        raise AssertionError("expected the unpatched decode to fail")
    except ValueError:
        pass
    assert base64.urlsafe_b64decode(key + b"========") == b"sixteen byte key!"


def test_patched_key_and_salt_lines_decode_unpadded_base64():
    text, _ = P.patch(REAL)
    unpadded = base64.urlsafe_b64encode(b"sixteen byte key!").decode().rstrip("=")
    for name in ("crypto_key", "salt"):
        line = next(l.strip() for l in text.splitlines() if l.strip().startswith(f"{name} = urlsafe_b64decode"))
        scope = {"urlsafe_b64decode": base64.urlsafe_b64decode, f"{name}_str": unpadded}
        exec(line, scope)
        assert scope[name] == b"sixteen byte key!"
