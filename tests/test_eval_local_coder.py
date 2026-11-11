# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""The plan-follow task in tools/eval_local_coder.py scores a coder fairly.

Why: this task decides which local coder Adam uses for plan-then-code work. A
scorer that passes weak tests, ignores files edited outside the plan, or whose
hidden tests miss a planted bug would pick the wrong model. Runs on a throwaway
git repo; no agent is called.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_local_coder", REPO / "tools" / "eval_local_coder.py")
E = importlib.util.module_from_spec(spec)
sys.modules["eval_local_coder"] = E
spec.loader.exec_module(E)

WEAK_TESTS = """\
from homeai.spoken_time import say_duration


def test_zero():
    assert say_duration(0) == "0 seconds"
"""


def _repo(tmp_path: Path, own_tests: str) -> Path:
    wt = tmp_path / "wt"
    (wt / "homeai").mkdir(parents=True)
    (wt / "tools").mkdir()
    (wt / "tests").mkdir()
    (wt / "homeai" / "__init__.py").write_text("")
    (wt / "conftest.py").write_text("")
    (wt / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    for argv in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"]):
        subprocess.run(["git", *argv], cwd=wt, check=True, capture_output=True)
    (wt / "homeai" / "spoken_time.py").write_text(E._PLAN_REFERENCE)
    (wt / "tools" / "say_duration.py").write_text(E._PLAN_REFERENCE_CLI)
    (wt / "tests" / "test_spoken_time.py").write_text(own_tests)
    return wt


def test_the_reference_passes_and_every_planted_bug_fails_the_hidden_tests(tmp_path):
    assert E.plan_self_check(tmp_path) == 0


def test_a_bug_the_hidden_tests_miss_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "_PLAN_BUGS", {"harmless": ('"cannot read', '"can not read')})
    assert E.plan_self_check(tmp_path) == 1


def test_a_correct_job_passes(tmp_path):
    ok, detail = E._accept_plan(_repo(tmp_path, E._PLAN_HIDDEN_TESTS))
    assert ok, detail
    assert "bugs caught 6/6" in detail


def test_a_file_outside_the_plan_fails(tmp_path):
    wt = _repo(tmp_path, E._PLAN_HIDDEN_TESTS)
    (wt / "homeai" / "__init__.py").write_text("# touched\n")
    ok, detail = E._accept_plan(wt)
    assert not ok and "OUTSIDE THE PLAN: homeai/__init__.py" in detail


def test_tests_that_catch_no_bugs_fail(tmp_path):
    ok, detail = E._accept_plan(_repo(tmp_path, WEAK_TESTS))
    assert not ok and "bugs caught 0/6" in detail and "on reference=True" in detail


def test_the_coder_s_module_is_put_back_after_judging(tmp_path):
    wt = _repo(tmp_path, E._PLAN_HIDDEN_TESTS)
    mine = E._PLAN_REFERENCE + "\n# mine\n"
    (wt / "homeai" / "spoken_time.py").write_text(mine)
    E._accept_plan(wt)
    assert (wt / "homeai" / "spoken_time.py").read_text() == mine


def test_the_greeting_fixture_is_the_scorer_s_reference():
    fixture = REPO / "tests" / "fixtures" / "android-hello" / "app" / "src" / "main" / "java" / "com" / "homeai"
    assert (fixture / "hello" / "Greeting.kt").read_text() == E._GREETING_REFERENCE


class _Task:
    def __init__(self, results, nudges):
        self.results, self.nudges, self.name = list(results), nudges, "t"

    def accept(self, wt):
        return self.results.pop(0)


def test_a_failed_check_is_handed_back_until_it_passes(tmp_path):
    asked = []
    task = _Task([(False, "e: Greeting.kt:3 unresolved"), (True, "built")], nudges=2)
    fixed, detail, reply, rounds = E.attempt(task, tmp_path, "PLAN", lambda t: asked.append(t) or "ok")
    assert fixed and detail == "built" and rounds == ["e: Greeting.kt:3 unresolved", "built"]
    assert asked[0] == "PLAN" and "e: Greeting.kt:3 unresolved" in asked[1] and "PLAN" in asked[1]


def test_nudges_stop_at_the_limit(tmp_path):
    task = _Task([(False, "a"), (False, "b"), (False, "c"), (True, "never")], nudges=2)
    fixed, detail, _, rounds = E.attempt(task, tmp_path, "P", lambda t: "")
    assert not fixed and rounds == ["a", "b", "c"]


def test_a_task_without_nudges_is_asked_once(tmp_path):
    task = _Task([(False, "a"), (True, "b")], nudges=0)
    assert E.attempt(task, tmp_path, "P", lambda t: "")[3] == ["a"]


def test_gradle_errors_keep_what_a_developer_reads():
    proc = subprocess.CompletedProcess([], 1, "> Task :app:compileDebugKotlin\nnoise\n"
                                       "e: file:///x/Greeting.kt:3:5 Unresolved reference 'foo'.\n", "")
    text = E._gradle_errors(proc)
    assert "Unresolved reference" in text and "noise" not in text


def test_the_reference_android_project_passes_and_a_wrong_package_fails(tmp_path):
    """Builds the fixture with Gradle offline (~2 min): the plan is solvable with this machine's toolchain."""
    if not E.ANDROID_HOME.exists():
        pytest.skip("no Android SDK")
    shutil.copytree(REPO / "tests" / "fixtures" / "android-hello", tmp_path / E.ANDROID_DIR)
    ok, detail = E._accept_android_plan(tmp_path)
    assert ok, detail
    assert "bugs caught 4/4" in detail
    assert E._accept_android_bare(tmp_path)[0]
    gradle = tmp_path / E.ANDROID_DIR / "app" / "build.gradle.kts"
    gradle.write_text(gradle.read_text().replace('applicationId = "com.homeai.hello"', 'applicationId = "com.x"'))
    ok, detail = E._accept_android_plan(tmp_path)
    assert not ok and "package is com.x" in detail
    gradle.write_text(gradle.read_text().replace('applicationId = "com.x"', 'applicationId = "com.homeai.hello"'))
    test = tmp_path / E.ANDROID_DIR / "app" / "src" / "test" / "java" / "com" / "homeai" / "hello" / "GreetingTest.kt"
    test.write_text("package com.homeai.hello\n\nimport org.junit.Assert.assertEquals\nimport org.junit.Test\n\n"
                    "class GreetingTest {\n    @Test fun morning() { assertEquals(\"Good morning\", greeting(8)) }\n}\n")
    ok, detail = E._accept_android_plan(tmp_path)
    assert not ok and "bugs caught 0/4" in detail and "hidden greeting tests pass=True" in detail


def _apk(tmp_path, text):
    import zipfile
    apk = tmp_path / "a.apk"
    with zipfile.ZipFile(apk, "w") as z:
        z.writestr("classes.dex", b"\x00dex" + text.encode())
        z.writestr("res/x.png", b"\x89PNG")
    return apk


def test_the_apk_must_contain_hello_world(tmp_path):
    assert E._apk_shows_hello(_apk(tmp_path, "Hello, World!"))
    assert not E._apk_shows_hello(_apk(tmp_path, "Good morning"))


def test_a_venv_repointed_at_a_worktree_is_reported_and_put_back():
    fixed = []
    out = E.venv_guard(where=lambda: "/tmp/coder-eval-x/homeai", reinstall=lambda: fixed.append(1))
    assert fixed == [1] and "/tmp/coder-eval-x/homeai" in out[0]
    assert E.venv_guard(where=lambda: str(E.REPO / "homeai"), reinstall=lambda: fixed.append(2)) == []
    assert fixed == [1]


def test_the_real_venv_imports_this_repo():
    assert E.venv_imports_homeai_from() == str(E.REPO / "homeai")


def test_inspect_names_the_tests_that_reject_the_reference(tmp_path):
    spec2 = importlib.util.spec_from_file_location("inspect_coder_eval", REPO / "tools" / "inspect_coder_eval.py")
    inspect = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(inspect)
    wt = _repo(tmp_path, E._PLAN_HIDDEN_TESTS.replace('(1, "1 second")', '(1, "one second")'))
    wt = wt.rename(tmp_path / "coder-eval-plan-follow-1")
    mine = E._PLAN_REFERENCE + "\n# the coder's own\n"
    (wt / "homeai" / "spoken_time.py").write_text(mine)
    out = inspect.failures_on_reference(wt)
    assert "FAILED" in out and "test_say" in out and "1 failed" in out
    assert (wt / "homeai" / "spoken_time.py").read_text() == mine


def test_a_missing_file_fails_without_running_anything(tmp_path):
    wt = _repo(tmp_path, E._PLAN_HIDDEN_TESTS)
    (wt / "tools" / "say_duration.py").unlink()
    assert E._accept_plan(wt) == (False, "missing: tools/say_duration.py")
