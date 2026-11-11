#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Show which of a coder's own tests reject the correct reference (eval_local_coder.py "on reference=False").

Why: that verdict says the coder's tests demand something the spec never
asked for, but not what. This runs the coder's tests in a kept worktree
(eval_local_coder.py --keep) with the reference module swapped in, prints the
failures, and puts the coder's module back.

    .venv/bin/python tools/inspect_coder_eval.py /tmp/coder-eval-plan-follow-1794374767
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_local_coder", REPO / "tools" / "eval_local_coder.py")
E = importlib.util.module_from_spec(spec)
sys.modules["eval_local_coder"] = E
spec.loader.exec_module(E)

# task -> (module the coder wrote, its test file, reference)
TARGETS = {
    "plan-follow": ("homeai/spoken_time.py", "tests/test_spoken_time.py", E._PLAN_REFERENCE),
    "viseme-module": ("homeai/avatar/visemes.py", "tests/test_visemes.py", E._VISEME_REFERENCE),
}


def task_of(wt: Path) -> str:
    for name in TARGETS:
        if wt.name.startswith(f"coder-eval-{name}-"):
            return name
    raise SystemExit(f"{wt.name}: not a kept worktree of {', '.join(TARGETS)}")


def failures_on_reference(wt: Path) -> str:
    module_rel, tests, reference = TARGETS[task_of(wt)]
    module = wt / module_rel
    original = module.read_text(encoding="utf-8")
    try:
        module.write_text(reference, encoding="utf-8")
        proc = E.run(f"{E.VENV_PY} -m pytest -q -p no:cacheprovider -o addopts= --tb=line -rf {tests}", wt,
                     timeout=300)
    finally:
        module.write_text(original, encoding="utf-8")
    return "\n".join(ln for ln in proc.stdout.splitlines() if ln.startswith(("FAILED", "E ", "/")) or "passed" in ln
                     or "failed" in ln)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("worktree", type=Path)
    print(failures_on_reference(ap.parse_args().worktree))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
