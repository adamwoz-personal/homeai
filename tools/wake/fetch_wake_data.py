#!/usr/bin/env python3
"""Fetch and convert the wake-word training's room echoes and background noise (plans/WAKE_WORD_PLAN.md).

Why: openWakeWord's training mixes each synthetic "hey jeeves" with a room's echo (MIT impulse responses, 270
rooms) and background sound (AudioSet clips: speech, music, TV, household noise) so the model hears it like a
real room. The upstream notebook's AudioSet .tar files and its FMA loader no longer exist (2026-11-23: AudioSet is
parquet now; FMA needs a 7 GB zip), so this takes AudioSet parquet parts instead and writes 16 kHz mono WAVs.

Run with the training venv (tools/wake/install_wake_train.sh calls it):
    ~/.local/share/homeai/wake-train/venv/bin/python -I tools/wake/fetch_wake_data.py ~/.cache/homeai/wake-train [--parts 3]
Output: DATA/mit_rirs/*.wav, DATA/audioset_16k/*.wav
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
AGENT = {"User-Agent": "homeai-wake-train/1"}
RIRS = "datasets/davidscripka/MIT_environmental_impulse_responses"
AUDIOSET = "datasets/agkphysics/AudioSet"
RATE = 16000


def _get(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as reply, tmp.open("wb") as out:
        shutil.copyfileobj(reply, out, 1 << 20)
    tmp.replace(dest)
    return dest


def listing(repo: str, folder: str) -> list[dict]:
    url = f"{HF}/api/{repo}/tree/main/{folder}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=60) as reply:
        return json.load(reply)


def to_16k_mono(samples, rate: int):
    import numpy as np
    from scipy.signal import resample_poly
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != RATE:
        audio = resample_poly(audio, RATE, rate).astype(np.float32)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def fetch_rirs(data: Path) -> int:
    out = data / "mit_rirs"
    for row in listing(RIRS, "16khz"):
        if row["path"].endswith(".wav"):
            _get(f"{HF}/{RIRS}/resolve/main/{row['path']}", out / Path(row["path"]).name)
    return len(list(out.glob("*.wav")))


def fetch_audioset(data: Path, parts: int) -> int:
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.io import wavfile
    out = data / "audioset_16k"
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(r["path"] for r in listing(AUDIOSET, "data/bal_train") if r["path"].endswith(".parquet"))[:parts]
    for name in names:
        part = _get(f"{HF}/{AUDIOSET}/resolve/main/{name}", data / "audioset" / Path(name).name)
        table = pq.read_table(part)
        audio_col = "audio" if "audio" in table.column_names else table.column_names[-1]
        ids = table.column("video_id").to_pylist() if "video_id" in table.column_names else range(table.num_rows)
        for vid, cell in zip(ids, table.column(audio_col).to_pylist()):
            dest = out / f"{Path(name).stem}_{vid}.wav"
            if dest.exists():
                continue
            raw = cell.get("bytes") if isinstance(cell, dict) else cell
            if not raw:
                continue
            try:
                samples, rate = sf.read(io.BytesIO(raw), always_2d=False)
            except Exception as exc:                      # a clip libsndfile can't read: skip it, say so
                print(f"skip {vid}: {exc}", file=sys.stderr)
                continue
            wavfile.write(dest, RATE, to_16k_mono(samples, rate))
    return len(list(out.glob("*.wav")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--parts", type=int, default=3, help="AudioSet bal_train parquet parts (~0.7 GB, ~1500 clips each)")
    args = ap.parse_args(argv)
    print("room echoes:", fetch_rirs(args.data), flush=True)
    print("background clips:", fetch_audioset(args.data, args.parts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
