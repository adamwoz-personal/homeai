#!/usr/bin/env python3
"""Report whether homeai's copy of the pose spec example still matches Ani's repo (contract R8).

Why: the pose repo owns the example. Homeai keeps a copy so an edit there can't break
its gate, and this says when the two have drifted. Report only, and not part of the
gate: it exits 0 when her repo is missing, 1 when they differ.

Usage:
    .venv/bin/python tools/art/check_pose_fixture.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OURS = REPO / "tests/fixtures/pose-spec-example.json"
THEIRS = Path.home() / "ani/pose-app/tests/fixtures/pose-spec-example.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(ours: Path, theirs: Path) -> tuple[int, str]:
    """Exit code and a line for the person reading."""
    if not theirs.exists():
        return 0, f"skipped: {theirs} not found"
    if sha256(ours) == sha256(theirs):
        return 0, f"same: {sha256(ours)[:16]}"
    return 1, f"differ: ours {sha256(ours)[:16]}, theirs {sha256(theirs)[:16]}; review, then copy it over"


def main() -> int:
    code, line = compare(OURS, THEIRS)
    print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
