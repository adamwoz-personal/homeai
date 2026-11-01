# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""homeai-memory CLI and the transcript back-fill (memory_store/cli.py)."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import pytest

from homeai.config import Config
from homeai.memory_store import cli


def line(ts="2026-10-09T12:00:00.000+00:00", **kw):
    base = {"ts": ts, "heard": "hey jarvis what's a good snake", "reply": "A corn snake.",
            "verdict": "allow", "ok": True, "speaker": "Adam", "speaker_score": 0.7}
    return json.dumps({**base, **kw})


def run_backfill(lines, include_unknown=False):
    return list(cli.backfill_episodes(lines, enrolled={"adam"}, ignore=("ryan",),
                                      include_unknown=include_unknown))


def test_backfill_imports_enrolled_speakers_only():
    [(ep, display)] = run_backfill([line()])
    assert (ep.username, display, ep.user) == ("adam", "Adam", "what's a good snake")
    assert ep.source_ref == "transcript:2026-10-09T12:00:00.000+00:00"
    assert ep.occurred_at.year == 2026 and ep.confidence == pytest.approx(0.7)
    assert ep.extra == {"verdict": "allow", "backfill": True}


@pytest.mark.parametrize("change", [
    {"speaker": "Ryan"}, {"speaker": "Joe"}, {"speaker": None}, {"verdict": "unverified"},
    {"reply": ""}, {"ok": False}, {"verdict": "home-lights"},
])
def test_backfill_skips(change):
    assert run_backfill([line(**change)]) == []


def test_backfill_unknown_goes_to_household_only_when_asked():
    lines = [line(speaker=None), line(speaker="Joe"), line(speaker="Ryan")]
    out = run_backfill(lines, include_unknown=True)
    assert [(e.username, e.confidence, e.trust) for e, _ in out] == [
        ("household", 0.0, "household"), ("household", 0.0, "household")]


def test_backfill_ignores_garbage_lines():
    assert run_backfill(["not json", "{}", json.dumps({"ts": "nope"}), ""]) == []


def test_enrolled_speakers(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"version": 1, "speakers": {"Adam": {}, "Mary Jane": {}}}))
    assert cli.enrolled_speakers(str(p)) == {"adam", "mary_jane"}
    assert cli.enrolled_speakers(str(tmp_path / "missing.json")) == set()
    p.write_text("[]")
    assert cli.enrolled_speakers(str(p)) == set()


def test_dry_run_needs_no_database(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join([line(), line(ts="2026-10-09T12:01:00+00:00", speaker=None)]))
    reg = tmp_path / "s.json"
    reg.write_text(json.dumps({"speakers": {"Adam": {}}}))
    cfg = Config()
    cfg = replace(cfg, speaker=replace(cfg.speaker, registry_path=reg))
    out = io.StringIO()
    assert cli.main(["backfill", "--dry-run", "--transcript", str(t)], cfg=cfg, out=out) == 0
    assert out.getvalue().strip() == "would import 1 episodes: {'adam': 1}"


# -- with a throwaway database ----------------------------------------------------------

def make_cli(dsn, tmp_path):
    from homeai.memory_store.embed import HashEmbedder

    reg = tmp_path / "s.json"
    reg.write_text(json.dumps({"speakers": {"Adam": {}}}))
    cfg = Config()
    cfg = replace(cfg, memory=replace(cfg.memory, dsn=dsn, min_similarity=0.2),
                  speaker=replace(cfg.speaker, registry_path=reg))

    def go(*argv):
        out = io.StringIO()
        code = cli.main(list(argv), cfg=cfg, embedder=HashEmbedder(), out=out)
        return code, out.getvalue()

    return go


@pytest.fixture
def run(clean_db, tmp_path):
    go = make_cli(clean_db, tmp_path)
    go("add-person", "adam", "Adam")
    return go


def test_cli_bootstraps_configured_admin_on_fresh_db(clean_db, tmp_path, capsys):
    go = make_cli(clean_db, tmp_path)
    assert go("stats")[0] == 0
    code, out = go("people")
    assert "adam         Adam             is_admin" in out
    assert go("--as", "ryan", "stats")[0] == 2      # non-admins are never auto-created
    assert "no such person" in capsys.readouterr().err


def test_cli_migrate_people_and_add(run):
    assert run("migrate") == (0, "schema up to date\n")
    assert run("add-person", "ryan", "Ryan", "--child") == (0, "added\n")
    assert run("add-person", "ryan", "Ryan") == (0, "already exists\n")
    code, out = run("people")
    assert "adam         Adam             is_admin" in out
    assert "ryan         Ryan             is_child" in out
    assert "guest" in out and "memory off" in out


def test_cli_note_search_forget(run):
    code, fid = run("note", "--profile", "Adam keeps a corn snake called Noodle")
    assert code == 0
    code, out = run("search", "corn snake Noodle")
    assert fid.strip() in out and "fact" in out
    assert run("forget", fid.strip()) == (0, "forgot fact\n")
    assert run("forget", fid.strip())[0] == 1


def test_cli_notes_for_others_need_admin(run, capsys):
    run("add-person", "ryan", "Ryan")
    assert run("--as", "ryan", "note", "--for", "adam", "planted")[0] == 2
    assert "only an admin" in capsys.readouterr().err
    assert run("note", "--for", "ryan", "Ryan likes dragons")[0] == 0


def test_cli_unknown_person(run, capsys):
    assert run("--as", "nobody", "list")[0] == 2
    assert "no such person" in capsys.readouterr().err


def test_cli_backfill_is_idempotent_then_list_export_stats(run, tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join([line(), line(ts="2026-10-10T08:00:00+00:00", heard="hey jarvis dragons?",
                                         reply="Dragons are myths.")]))
    assert run("backfill", "--transcript", str(t)) == (0, "imported 2 new episodes\n")
    assert run("backfill", "--transcript", str(t)) == (0, "imported 0 new episodes\n")
    code, out = run("list")
    assert "dragons?" in out and "what's a good snake" in out
    code, out = run("export")
    data = json.loads(out)
    assert len(data["episodes"]) == 2 and data["facts"] == []
    code, out = run("stats")
    assert json.loads(out)["episodes"] == {"adam": 2}
