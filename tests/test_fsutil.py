"""homeai/fsutil.py: every save is all-or-nothing, flushed, cleaned up on failure (CODE_REVIEW_PLAN item 2)."""
from __future__ import annotations

import json
import os
import stat

import pytest

from homeai import fsutil as FS


def leftovers(folder):
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(".tmp"))


def test_json_replaces_the_old_file_and_leaves_no_temp(tmp_path):
    path = tmp_path / "deep" / "state.json"
    FS.atomic_write_json(path, {"a": 1})
    FS.atomic_write_json(path, {"a": 2}, indent=1)
    assert json.loads(path.read_text()) == {"a": 2} and path.read_text() == json.dumps({"a": 2}, indent=1)
    assert leftovers(path.parent) == []


def test_a_failed_save_keeps_the_old_file_and_removes_the_temp(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    FS.atomic_write_text(path, "old")

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(FS.os, "replace", boom)
    with pytest.raises(OSError):
        FS.atomic_write_text(path, "new")
    with pytest.raises(OSError):
        FS.atomic_copy(path, tmp_path / "copy.png")
    assert path.read_text() == "old" and leftovers(tmp_path) == []


def test_every_save_is_flushed_to_disk(tmp_path, monkeypatch):
    synced = []
    real = os.fsync
    monkeypatch.setattr(FS.os, "fsync", lambda fd: (synced.append(fd), real(fd)))
    FS.atomic_write_text(tmp_path / "a", "x")
    assert len(synced) == 1
    FS.atomic_write_text(tmp_path / "b", "x", fsync=False)
    assert len(synced) == 1


def test_a_secret_is_private_from_its_first_byte(tmp_path, monkeypatch):
    seen = []
    real = os.fsync
    monkeypatch.setattr(FS.os, "fsync", lambda fd: (seen.append(stat.S_IMODE(os.fstat(fd).st_mode)), real(fd)))
    old = os.umask(0)
    try:
        FS.atomic_write_text(tmp_path / "env", "TOKEN=x", mode=0o600)
        FS.atomic_write_text(tmp_path / "plain", "x")
    finally:
        os.umask(old)
    assert seen == [0o600, 0o666]                    # while being written, not only after
    assert stat.S_IMODE((tmp_path / "env").stat().st_mode) == 0o600


def test_a_strict_umask_cannot_lock_the_owner_out(tmp_path):
    old = os.umask(0o277)
    try:
        FS.atomic_write_text(tmp_path / "profiles", "x", mode=0o600)
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "profiles").stat().st_mode) == 0o600


def test_copy(tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(b"png")
    FS.atomic_copy(src, tmp_path / "out" / "b.png")
    assert (tmp_path / "out" / "b.png").read_bytes() == b"png" and leftovers(tmp_path / "out") == []
