#!/usr/bin/env python3
"""Speak text through the real Speaker and voice, publishing avatar events, without the daemon.

Why: the functional check for the speech timeline and in-process TTS. It runs
an EventBus on its own socket (default ``$XDG_RUNTIME_DIR/homeai/demo.sock``) so
a renderer started with ``--socket`` can follow along, and it reports what was
published: one utterance per Piper sentence, chained starts, viseme counts and
first-audio latency. Plays on the real speaker; the daemon may keep running.

Usage:
    .venv/bin/python tools/avatar/avatar_demo.py --text "Hello. How are you?" \
        [--engine inprocess|subprocess] [--socket PATH] [--wait-client 0] [--repeat 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from homeai.avatar.bus import EventBus, default_socket_path  # noqa: E402
from homeai.config import load  # noqa: E402
from homeai.tts import Speaker  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"
DEFAULT_TEXT = ("Good evening. The weather today is cloudy, with a high of twelve degrees. "
                "Should I turn on the living room lights?")


def summarise(events: list[tuple[str, dict]]) -> dict:
    """Pure: check utterances from the same say() call chain back to back.

    Ids are ``<call>-<index>``; separate calls start a new aplay, so only chunks
    within one call are expected to abut.
    """
    utts = [d for t, d in events if t == "utterance"]
    gaps_ms = []
    for a, b in zip(utts, utts[1:]):
        if a["id"].rsplit("-", 1)[0] != b["id"].rsplit("-", 1)[0]:
            continue
        gaps_ms.append(round((b["start_mono"] - a["start_mono"]) * 1000 - a["duration_ms"], 1))
    return {"utterances": len(utts),
            "visemes": [len(u.get("visemes", [])) for u in utts],
            "durations_ms": [u["duration_ms"] for u in utts],
            "chain_gaps_ms": gaps_ms,
            "chained": all(abs(g) <= 1.0 for g in gaps_ms),
            "stops": sum(1 for t, _ in events if t == "speech_stop")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--engine", choices=("inprocess", "subprocess"), default="inprocess")
    parser.add_argument("--socket", type=Path, default=default_socket_path().with_name("demo.sock"))
    parser.add_argument("--wait-client", type=float, default=0.0,
                        help="seconds to wait for a renderer to connect before speaking")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    cfg = load()
    tts_cfg = replace(cfg.tts, engine=args.engine)
    bus = EventBus(args.socket, source="demo")
    try:
        bus.start()
    except (RuntimeError, OSError) as exc:
        print(f"cannot start event bus: {exc}")
        return 2
    events: list[tuple[str, dict]] = []

    def publish(etype: str, data: dict) -> None:
        events.append((etype, data))
        bus.publish(etype, data)

    speaker = Speaker(tts_cfg, cfg.audio.output_device, publish=publish)
    try:
        ok, why = speaker.available()
        if not ok:
            print(f"speaker unavailable: {why}")
            return 2
        t0 = time.monotonic()
        speaker.warm()
        load_ms = (time.monotonic() - t0) * 1000
        end = time.monotonic() + args.wait_client
        while bus.client_count == 0 and time.monotonic() < end:
            time.sleep(0.1)
        runs = []
        for _ in range(max(1, args.repeat)):
            bus.publish("state", {"state": "speaking"})
            res = speaker.say_chunked(args.text)
            bus.publish("state", {"state": "idle"})
            runs.append({"ok": res.ok, "error": res.error,
                         "first_audio_ms": round(res.first_audio_ms, 1),
                         "total_ms": round(res.total_ms)})
        summary = summarise(events)
    finally:
        bus.stop()
    report = {"when": datetime.now().isoformat(timespec="seconds"), "engine": args.engine,
              "load_ms": round(load_ms), "runs": runs, **summary,
              "clients": "renderer connected" if args.wait_client else "none awaited"}
    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"avatar_demo-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({**report, "events": events}, indent=1))
    print(json.dumps(report))
    print(f"-> {out}")
    passed = all(r["ok"] for r in runs) and summary["chained"] and (
        args.engine == "subprocess" or summary["utterances"] > 0)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
