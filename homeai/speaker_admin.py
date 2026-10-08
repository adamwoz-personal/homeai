# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""``homeai-mode speaker``: turn speaker recognition on/off, manage voices.

    homeai-mode speaker [status]         on/off, model, who is enrolled
    homeai-mode speaker on|off           sets HOMEAI_SPEAKER_ID in .env, restarts homeai
    homeai-mode speaker fetch            download the model (SHA256-checked)
    homeai-mode speaker remove NAME
    homeai-mode speaker enrol NAME WAV...    from recordings (16 kHz WAV)
    homeai-mode speaker identify WAV...

Enrolling by voice ("Hey Jarvis, remember my voice as Adam") needs no files
and uses the same microphone and distance as everyday use, so prefer it.
The daemon holds the microphone, so there is no record-from-mic command.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import os
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT, SpeakerConfig

ENV_FILE = PROJECT_ROOT / ".env"
ENV_KEY = "HOMEAI_SPEAKER_ID"

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx")
# From the release's checksum.txt; checked 2026-10-07.
MODEL_SHA256 = "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"


class SpeakerAdminError(Exception):
    pass


def set_env_line(text: str, key: str, value: str) -> str:
    """``text`` (a .env file) with ``key`` set to ``value``; other lines kept."""
    lines = text.splitlines()
    out, done = [], False
    for line in lines:
        if line.split("=", 1)[0].strip() == key and not line.lstrip().startswith("#"):
            if not done:
                out.append(f"{key}={value}")
                done = True
            continue
        out.append(line)
    if not done:
        out.append(f"{key}={value}")
    return "\n".join(out) + "\n"


def write_env(path: Path, key: str, value: str) -> None:
    text = path.read_text() if path.exists() else ""
    path.write_text(set_env_line(text, key, value))
    os.chmod(path, 0o600)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_model(dest: Path, url: str = MODEL_URL, expected: str = MODEL_SHA256,
                opener=urllib.request.urlopen) -> str:
    """Download the model to ``dest`` unless it is already there and intact."""
    if dest.is_file() and sha256(dest) == expected:
        return f"model already present: {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as fh, opener(url, timeout=120) as resp:
            while block := resp.read(1 << 20):
                fh.write(block)
        got = sha256(Path(tmp))
        if got != expected:
            raise SpeakerAdminError(f"checksum mismatch for {url}: {got}")
        os.replace(tmp, dest)
    except OSError as exc:
        raise SpeakerAdminError(f"download failed: {exc}") from exc
    finally:
        Path(tmp).unlink(missing_ok=True)
    return f"downloaded {dest}"


def read_wav(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path)) as w:
            if w.getframerate() != 16000 or w.getsampwidth() != 2:
                raise SpeakerAdminError(f"{path}: need 16 kHz 16-bit WAV "
                                        f"(ffmpeg -i in -ar 16000 -ac 1 out.wav)")
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() > 1:
                raw = raw.reshape(-1, w.getnchannels())[:, 0]
    except (OSError, wave.Error, EOFError) as exc:
        raise SpeakerAdminError(f"{path}: {exc}") from exc
    return raw.astype(np.float32) / 32768.0


def restart_homeai(run=subprocess.run) -> str:
    proc = run(["systemctl", "--user", "restart", "homeai"], capture_output=True, text=True)
    return "homeai restarted" if proc.returncode == 0 else \
        f"restart homeai yourself: {proc.stderr.strip() or 'systemctl failed'}"


def status_lines(cfg: SpeakerConfig) -> list[str]:
    from .voice_id import VoiceId

    v = VoiceId(cfg)
    lines = [
        f"speaker recognition: {'ON' if cfg.enabled else 'off'}  ({ENV_KEY} in {ENV_FILE})",
        f"model: {cfg.model_path}" + ("" if cfg.model_path.is_file() else
                                      "  MISSING - run: homeai-mode speaker fetch"),
        f"threshold {cfg.threshold}, margin {cfg.margin}; follow-ups from other voices "
        f"{'ignored' if cfg.followup_same_speaker else 'answered'} "
        f"(below {cfg.different_below})",
        f"voices ({v.registry.path}):",
    ]
    for p in v.registry.profiles():
        when = datetime.datetime.fromtimestamp(p.enrolled_at).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  {p.name:12} enrolled {when}, {p.sample_count} clip(s)")
    if not v.registry.names():
        lines.append('  nobody yet - say "Hey Jarvis, remember my voice as <name>"')
    if v.registry.problem:
        lines.append(f"  warning: {v.registry.problem}")
    return lines


def _loaded(cfg: SpeakerConfig):
    from .voice_id import VoiceId

    v = VoiceId(cfg)
    ok, problem = v.load()
    if not ok:
        raise SpeakerAdminError(problem)
    return v


def main(args, cfg: SpeakerConfig | None = None, env_file: Path = ENV_FILE,
         run=subprocess.run, log=print) -> int:
    from .voice_id import Enrolment, describe

    cfg = cfg or SpeakerConfig()
    cmd = args.speaker_cmd or "status"
    try:
        if cmd in ("on", "off"):
            if cmd == "on" and not cfg.model_path.is_file():
                log(fetch_model(cfg.model_path))
            write_env(env_file, ENV_KEY, "1" if cmd == "on" else "0")
            log(restart_homeai(run))
            cfg = dataclasses.replace(cfg, enabled=cmd == "on")
        elif cmd == "fetch":
            log(fetch_model(cfg.model_path))
        elif cmd == "remove":
            from .voice_id import VoiceId
            if not VoiceId(cfg).forget(args.name):
                raise SpeakerAdminError(f"no voice called {args.name}")
            log(f"removed {args.name}")
        elif cmd == "enrol":
            v = _loaded(cfg)
            enrolment = Enrolment(args.name.strip().capitalize())
            for path in args.wav:
                enrolment.add(read_wav(path))
            if enrolment.speech_s < 3.0:
                raise SpeakerAdminError(f"only {enrolment.speech_s:.1f} s of speech; need 3+")
            profile = v.finish_enrolment(enrolment)
            if profile is None:
                raise SpeakerAdminError("no usable speech in those files")
            log(f"enrolled {profile.name} from {enrolment.speech_s:.1f} s of speech")
        elif cmd == "identify":
            v = _loaded(cfg)
            for path in args.wav:
                ident, _ = v.identify(read_wav(path))
                log(f"{path}: {describe(ident)}")
            return 0
        log("\n".join(status_lines(cfg)))
        return 0
    except SpeakerAdminError as exc:
        log(f"homeai-mode speaker: {exc}")
        return 1


def add_parser(sub) -> None:
    s = sub.add_parser("speaker", help="speaker recognition: on/off and enrolled voices")
    ssub = s.add_subparsers(dest="speaker_cmd")
    ssub.add_parser("status", help="on/off, model, enrolled voices")
    ssub.add_parser("on", help="turn on (downloads the model if needed) and restart homeai")
    ssub.add_parser("off", help="turn off and restart homeai")
    ssub.add_parser("fetch", help="download the speaker model")
    r = ssub.add_parser("remove", help="forget a voice")
    r.add_argument("name")
    e = ssub.add_parser("enrol", help="enrol a voice from 16 kHz WAV recordings")
    e.add_argument("name")
    e.add_argument("wav", type=Path, nargs="+")
    i = ssub.add_parser("identify", help="who is speaking in these WAV files")
    i.add_argument("wav", type=Path, nargs="+")
