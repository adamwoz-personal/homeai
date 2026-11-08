#!/usr/bin/env python3
"""Do these tools still import? Without running them. (Lesson 2026-11-08, ROUTING_GUIDE.)

Why: after a refactor touched 19 tools I checked them with ``--help``. Four live
tools take no arguments, so ``--help`` ran them for real: one switched the card
to art mode and opened a picture window. This loads each file as a module under
another name, so only its top level runs (sys.path setup and imports), never
its ``if __name__ == "__main__"`` block.

Exit 1 if any file fails to import.

    .venv/bin/python tools/inspect_tool_imports.py tools/art/bakeoff.py [...]
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


def try_import(path: Path) -> str | None:
    """None if the file imports, else the last line of the error."""
    spec = importlib.util.spec_from_file_location(f"_tool_check_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:  # noqa: BLE001 - report every kind of failure, SystemExit included
        return traceback.format_exception_only(type(exc), exc)[-1].strip()
    finally:
        sys.path[:] = saved
    return None


def main(argv: list[str]) -> int:
    failed = 0
    for name in argv:
        error = try_import(Path(name).resolve())
        print(("ok   " if error is None else "FAIL ") + name + ("" if error is None else f": {error}"))
        failed += error is not None
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
