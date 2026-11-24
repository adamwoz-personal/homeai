"""torchaudio.load / torchaudio.info through soundfile, for openWakeWord's trainer (tools/wake).

Why: torchaudio >= 2.9 dropped its own audio backends; load() needs the separate TorchCodec + FFmpeg and info()
is gone, but openWakeWord's data.py calls both on plain WAVs. install_wake_train.sh puts this file in the training
venv's site-packages with a .pth line that imports it, so the trainer runs unpatched. Only WAV/FLAC (libsndfile).
"""
from __future__ import annotations

from types import SimpleNamespace

import soundfile as sf
import torch
import torchaudio


def load(path, *args, **kwargs):
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)    # frames x channels
    return torch.from_numpy(data.T.copy()), rate


def info(path, *args, **kwargs):
    i = sf.info(str(path))
    return SimpleNamespace(sample_rate=i.samplerate, num_frames=i.frames, num_channels=i.channels,
                           bits_per_sample=16, encoding="PCM_S")


torchaudio.load = load
torchaudio.info = info
