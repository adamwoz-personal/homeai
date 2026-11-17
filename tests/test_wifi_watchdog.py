"""tools/net/wifi_watchdog.py: when to reconnect, when to reload the driver, and what it records for Jarvis."""
from __future__ import annotations

import json

from tools.net import wifi_watchdog as W


def run(times_up: list[tuple[float, bool]]) -> list[tuple[float, str]]:
    state: dict = {}
    acted = []
    for now, up in times_up:
        action, state = W.decide(state, up, now)
        if action:
            acted.append((now, action))
    return acted


def test_up_does_nothing():
    assert run([(t, True) for t in range(0, 3600, 60)]) == []


def test_a_short_blip_does_nothing_and_isnt_recorded():
    """A router reboot takes ~2 min: no action inside DOWN_S, and back again isn't an outage to report."""
    assert run([(0, False), (60, False), (120, False), (170, True)]) == []


def test_escalates_reconnect_then_reload_then_paces_itself():
    acted = run([(t, False) for t in range(0, 7300, 60)])
    assert acted[0] == (180, "reconnect")                 # down DOWN_S: reconnect
    assert acted[1] == (360, "reload")                    # still down DOWN_S later: reload the driver
    later = acted[2:]
    gaps = [b[0] - a[0] for a, b in zip(acted[1:], later)]
    assert all(g >= W.RETRY_S for g in gaps)              # then no more often than RETRY_S
    reloads = [t for t, a in acted if a == "reload"]
    assert reloads == [360, 3960]                         # and the driver at most once an hour


def test_recovery_after_acting_is_reported_once_and_state_clears():
    state: dict = {}
    for now in (0, 180):
        _, state = W.decide(state, False, now)
    assert state["acted"] == 1
    action, state = W.decide(state, True, 240)
    assert (action, state) == ("recovered", {})
    assert W.decide(state, True, 300) == (None, {})


def test_reload_commands_reload_the_driver_then_reconnect():
    cmds = W.commands("reload", "bednarz", "mt7925e")
    assert cmds[:2] == [["modprobe", "-r", "mt7925e"], ["modprobe", "mt7925e"]]
    assert cmds[-1] == ["nmcli", "connection", "up", "bednarz"]
    assert W.commands("reconnect", "bednarz", "mt7925e") == [["nmcli", "device", "wifi", "rescan"],
                                                             ["nmcli", "connection", "up", "bednarz"]]


def test_main_runs_the_commands_and_records_the_outage_for_jarvis(tmp_path, monkeypatch):
    state, events = tmp_path / "state.json", tmp_path / "events.jsonl"
    clock = iter([0.0, 200.0, 260.0])
    up = iter([False, False, True])
    ran = []
    monkeypatch.setattr(W.time, "time", lambda: next(clock))
    monkeypatch.setattr(W, "gateway", lambda: "192.168.1.254")
    monkeypatch.setattr(W, "is_up", lambda gw: next(up))

    class Done:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr(W.subprocess, "run", lambda cmd, **kw: ran.append(cmd) or Done())
    for _ in range(3):
        W.main(["--state", str(state), "--events", str(events)])
    assert ["nmcli", "connection", "up", "bednarz"] in ran
    [line] = events.read_text().splitlines()
    event = json.loads(line)
    assert event["down_since"] == 0.0 and event["back_at"] == 260.0
    assert event["actions"] == [{"at": 200.0, "action": "reconnect", "ok": True}]


def test_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(W, "gateway", lambda: "192.168.1.254")
    monkeypatch.setattr(W, "is_up", lambda gw: False)
    monkeypatch.setattr(W.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"down_since": 0}))
    W.main(["--state", str(state), "--events", str(tmp_path / "e"), "--dry-run"])
    assert "would run: nmcli connection up bednarz" in capsys.readouterr().out
    assert json.loads(state.read_text()) == {"down_since": 0}
