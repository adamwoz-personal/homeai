"""Saving files so a crash leaves the old file or the new one, never half of either (CODE_REVIEW_PLAN item 2).

Nine places did "write a temp file, then os.replace" six different ways; only
the art store flushed to disk first and cleaned up its temp file on failure.
Here every save does both. ``mode`` sets the file's permissions from its first
byte (0o600 for secrets and voice profiles); without it the usual umask applies.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path


def _temp_beside(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None, fsync: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_beside(path)
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)                 # in case the umask took bits away
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, *, mode: int | None = None, fsync: bool = True) -> None:
    atomic_write_bytes(path, text.encode(), mode=mode, fsync=fsync)


def atomic_write_json(path: Path, data, *, indent: int | None = None, mode: int | None = None,
                      fsync: bool = True) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent), mode=mode, fsync=fsync)


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` over ``dst`` in one step (pictures into the store and into ~/Pictures)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_beside(dst)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
