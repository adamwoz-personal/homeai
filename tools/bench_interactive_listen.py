"""Interactive mode step 0: transcribe the listen probe's speech segments as they arrive, and time it.

Why: can Jarvis keep up transcribing a whole room beside voice turns and art jobs (plans/INTERACTIVE_MODE_PLAN.md
step 0)? Jarvis (homeai/listen_probe.py) saves each stretch of speech to ~/.local/share/homeai/listen-probe; this
worker runs Parakeet TDT (whisper.cpp parakeet-cli; `--engine whisper` for small.en) on each new one, at lower
priority so a real voice turn keeps the CPU first, and records how far behind real time it fell and what else the box was doing:

  lag_s     seconds from the end of the speech to its transcript (the go/no-go number: under 2 s with art running)
  wall_s    whisper time for the segment, rtf = wall_s / duration
  cpu_busy  share of CPU busy over the transcription (from /proc/stat), load1
  vram_used GiB on the biggest card, comfy_busy (ComfyUI has a job running or queued)

Each result goes into the segment's json ("stt") and a line of ~/homeai-bench/interactive-listen-<date>.jsonl.
tools/report_interactive_listen.py summarises them.

Usage:  nice -n 10 .venv/bin/python tools/bench_interactive_listen.py [--once] [--folder DIR]
        .venv/bin/python tools/bench_interactive_listen.py --smoke some.wav   (real whisper on one clip, scratch dir)
(run as a transient user unit; see docs/RUNBOOK.md "Listen probe").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from homeai import art_clips, gpu_info, listen_probe
from homeai.config import load
from homeai.fsutil import atomic_write_json
from homeai.stt import Transcriber, make_transcriber

BENCH = Path.home() / "homeai-bench"
COMFY = "http://127.0.0.1:8188"
ENGINE = "parakeet"           # Adam 2026-11-13: fewest mistakes on room clips (tools/bench_stt_room.py)


def cpu_times() -> tuple[int, int]:
    """(busy, total) jiffies from /proc/stat."""
    fields = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    idle = fields[3] + fields[4]
    return sum(fields) - idle, sum(fields)


def comfy_busy(url: str = COMFY) -> bool | None:
    try:
        with urllib.request.urlopen(f"{url}/queue", timeout=1) as r:
            q = json.load(r)
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except (OSError, ValueError):
        return None          # ComfyUI not running


def waiting(folder: Path) -> list[Path]:
    """Segment json files not yet transcribed, oldest first."""
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        try:
            if "stt" not in json.loads(path.read_text()) and path.with_suffix(".wav").exists():
                out.append(path)
        except (OSError, ValueError):
            continue
    return out


def transcribe(path: Path, stt: Transcriber, clock=time.time) -> dict:
    meta = json.loads(path.read_text())
    busy0, total0 = cpu_times()
    t0 = clock()
    result = stt.transcribe_file(path.with_suffix(".wav"))
    t1 = clock()
    busy1, total1 = cpu_times()
    card = gpu_info.biggest()
    duration = float(meta.get("duration") or 0) or 1e-6
    out = {"ok": result.ok, "text": result.text, "error": result.error, "wall_s": round(t1 - t0, 3),
           "engine": stt.engine, "rtf": round((t1 - t0) / duration, 3), "lag_s": round(t1 - float(meta.get("end", t1)), 3),
           "cpu_busy": round((busy1 - busy0) / max(1, total1 - total0), 3), "load1": os.getloadavg()[0],
           "vram_used": round(card.used / 2**30, 2) if card else None, "comfy_busy": comfy_busy()}
    meta["stt"] = out
    atomic_write_json(path, meta)
    return {"segment": path.stem, "duration": meta.get("duration"), **{k: meta.get(k) for k in
            ("jarvis_speaking", "mic_paused", "art_active")}, **out}


def mark_failed(path: Path, exc: Exception) -> None:
    """Record the failure in the clip's json so ``waiting`` skips it from now on."""
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError):
        meta = {}
    meta["stt"] = {"ok": False, "text": "", "error": f"worker: {exc}"[:200]}
    try:
        atomic_write_json(path, meta)
    except OSError:
        path.with_suffix(".wav").unlink(missing_ok=True)   # waiting() needs the wav, so this also stops retries


def smoke(wav: Path, engine: str = ENGINE) -> dict:
    """One clip through the real whisper, as if the probe had saved it now: proves the binary, model and timing."""
    import tempfile
    import wave

    import numpy as np

    from homeai import art_clips
    with wave.open(str(wav)) as f:
        rate, pcm = f.getframerate(), np.frombuffer(f.readframes(f.getnframes()), "<i2") / 32768.0
    audio = np.interp(np.arange(0, len(pcm), rate / 16000), np.arange(len(pcm)), pcm).astype(np.float32)
    folder = Path(tempfile.mkdtemp(prefix="listen-smoke-"))
    end = time.time()
    art_clips.save(folder, audio, 16000, {"end": end, "duration": round(len(audio) / 16000, 2)}, now=end)
    return transcribe(waiting(folder)[0], make_transcriber(load().stt, engine))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--once", action="store_true", help="transcribe what is waiting, then exit")
    ap.add_argument("--folder", type=Path, default=listen_probe.DIR)
    ap.add_argument("--smoke", type=Path, help="transcribe one wav file through the real whisper and exit")
    ap.add_argument("--engine", choices=("parakeet", "whisper"), default=ENGINE)
    ap.add_argument("--until", default="", help="YYYY-MM-DD: exit once past it and nothing is waiting")
    args = ap.parse_args(argv)
    if args.smoke:
        print(json.dumps(smoke(args.smoke, args.engine)))
        return 0
    stt = make_transcriber(load().stt, args.engine)
    BENCH.mkdir(exist_ok=True)
    while True:
        for path in waiting(args.folder):
            try:
                row = transcribe(path, stt)
            except (OSError, ValueError) as exc:
                print(f"skip {path.name}: {exc}", file=sys.stderr)
                mark_failed(path, exc)       # never retry it: one bad clip stalled the queue 13 s a pass
                continue
            log = BENCH / f"interactive-listen-{time.strftime('%Y%m%d')}.jsonl"
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{row['segment']} {row['duration']}s lag {row['lag_s']}s: {row['text'][:80]}", flush=True)
        if args.once or (args.until and not art_clips.recording_on(args.until) and not waiting(args.folder)):
            return 0
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
