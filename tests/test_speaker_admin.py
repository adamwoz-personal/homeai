"""Tests for homeai.speaker_admin (homeai-mode speaker)."""
from __future__ import annotations

import argparse
import hashlib
import io
import wave
from pathlib import Path

import numpy as np
import pytest

from homeai import speaker_admin as sa
from homeai.config import SpeakerConfig


def test_set_env_line_replaces_adds_and_keeps_comments():
    text = "A=1\n# HOMEAI_SPEAKER_ID=1\nHOMEAI_SPEAKER_ID=0\nB=2\nHOMEAI_SPEAKER_ID=0\n"
    out = sa.set_env_line(text, "HOMEAI_SPEAKER_ID", "1")
    assert out == "A=1\n# HOMEAI_SPEAKER_ID=1\nHOMEAI_SPEAKER_ID=1\nB=2\n"
    assert sa.set_env_line("", "K", "v") == "K=v\n"


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_fetch_model_checks_the_checksum(tmp_path):
    body = b"model bytes"
    good = hashlib.sha256(body).hexdigest()
    dest = tmp_path / "m.onnx"
    opener = lambda url, timeout: FakeResp(body)  # noqa: E731
    with pytest.raises(sa.SpeakerAdminError, match="checksum"):
        sa.fetch_model(dest, "u", "0" * 64, opener)
    assert not dest.exists() and not list(tmp_path.glob(".download-*"))
    assert sa.fetch_model(dest, "u", good, opener).startswith("downloaded")
    assert sa.fetch_model(dest, "u", good, opener).startswith("model already present")


def test_fetch_model_reports_network_failure(tmp_path):
    def opener(url, timeout):
        raise OSError("no route to host")
    with pytest.raises(sa.SpeakerAdminError, match="download failed"):
        sa.fetch_model(tmp_path / "m.onnx", "u", "0" * 64, opener)


def args(cmd, **kw):
    return argparse.Namespace(speaker_cmd=cmd, **kw)


class Ran:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        return argparse.Namespace(returncode=0, stderr="")


def test_on_and_off_edit_env_and_restart(tmp_path):
    env, run, out = tmp_path / ".env", Ran(), []
    env.write_text("HOMEAI_PIPER_VOICE=x\n")
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    cfg = SpeakerConfig(model_path=model, registry_path=tmp_path / "s.json")
    assert sa.main(args("on"), cfg, env, run, out.append) == 0
    assert env.read_text() == "HOMEAI_PIPER_VOICE=x\nHOMEAI_SPEAKER_ID=1\n"
    assert run.calls == [["systemctl", "--user", "restart", "homeai"]]
    assert any("ON" in line for line in "\n".join(out).splitlines())
    sa.main(args("off"), cfg, env, run, out.append)
    assert "HOMEAI_SPEAKER_ID=0" in env.read_text()


def test_remove_unknown_voice_fails(tmp_path):
    out = []
    cfg = SpeakerConfig(registry_path=tmp_path / "s.json")
    assert sa.main(args("remove", name="Zed"), cfg, log=out.append) == 1
    assert "no voice called Zed" in out[0]


def test_read_wav_rejects_wrong_rate(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(44100)
        w.writeframes(np.zeros(100, dtype=np.int16).tobytes())
    with pytest.raises(sa.SpeakerAdminError, match="16 kHz"):
        sa.read_wav(path)
    with pytest.raises(sa.SpeakerAdminError):
        sa.read_wav(tmp_path / "missing.wav")


SAMPLES = SpeakerConfig().model_path.parent / "samples"


@pytest.mark.skipif(not SpeakerConfig().model_path.is_file() or not SAMPLES.is_dir(),
                    reason="speaker model not downloaded")
def test_enrol_then_identify_from_files(tmp_path):
    out = []
    cfg = SpeakerConfig(registry_path=tmp_path / "s.json")
    enrol = [SAMPLES / f"leijun-sr-{i}.wav" for i in (1, 2)]
    assert sa.main(args("enrol", name="lei", wav=enrol), cfg, log=out.append) == 0
    assert out[0].startswith("enrolled Lei")
    out.clear()
    tests = [SAMPLES / "leijun-test-sr-1.wav", SAMPLES / "fangjun-test-sr-1.wav"]
    assert sa.main(args("identify", wav=tests), cfg, log=out.append) == 0
    assert ": Lei (" in out[0] and ": unknown" in out[1]


@pytest.mark.skipif(not SpeakerConfig().model_path.is_file(), reason="speaker model not downloaded")
def test_enrol_from_too_little_speech_is_refused(tmp_path):
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes((np.sin(np.arange(16000) / 5) * 9000).astype(np.int16).tobytes())
    out = []
    cfg = SpeakerConfig(registry_path=tmp_path / "s.json")
    assert sa.main(args("enrol", name="a", wav=[path]), cfg, log=out.append) == 1
    assert "need 3+" in out[0]
