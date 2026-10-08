# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Speaker recognition: who is talking, not what they said.

Each utterance becomes a fixed-length voice embedding (WeSpeaker ResNet34-LM,
ONNX, CPU) and is compared with enrolled profiles by cosine similarity.
See plans/VOICE_ID_PLAN.md.

This is courtesy, not security: a recording of a voice will match it.
Unknown speakers get full normal service.

Independent of audio I/O and the daemon so it can be tested with synthetic
arrays. The CLI (``python -m homeai.speaker``) enrols from the microphone.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Kaldi fbank as used to train the WeSpeaker models (wespeaker/dataset/
# processor.py: 80 mel bins, 25 ms frames every 10 ms, hamming window, no
# dither), followed by per-utterance mean normalisation.
FRAME_LEN = 400
FRAME_SHIFT = 160
N_FFT = 512
N_MELS = 80
PREEMPH = 0.97
LOW_HZ = 20.0
_EPS = float(np.finfo(np.float32).eps)

# Energy-based speech trimming. Frames quieter than this far below the
# loudest frame are treated as silence and left out of the embedding.
SPEECH_DB_BELOW_PEAK = 35.0
# An absolute floor too, so a silent clip is not "all speech" relative to itself.
SPEECH_MIN_RMS = 1e-3

# Below this much speech an embedding is too noisy to be worth comparing.
MIN_IDENTIFY_SPEECH_S = 1.0
# Enrolment needs more: a weak profile poisons every later comparison.
MIN_ENROL_SPEECH_S = 3.0


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _mel(hz):
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_banks(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Kaldi's triangular mel filterbank, shape (N_MELS, N_FFT // 2 + 1)."""
    n_bins = N_FFT // 2
    fft_mel = _mel(np.arange(n_bins) * sample_rate / N_FFT)
    mel_low, mel_high = _mel(LOW_HZ), _mel(sample_rate / 2)
    delta = (mel_high - mel_low) / (N_MELS + 1)
    banks = np.zeros((N_MELS, n_bins + 1), dtype=np.float64)
    for m in range(N_MELS):
        left, centre, right = (mel_low + (m + k) * delta for k in range(3))
        up = (fft_mel - left) / (centre - left)
        down = (right - fft_mel) / (right - centre)
        w = np.where((fft_mel > left) & (fft_mel < right), np.minimum(up, down), 0.0)
        banks[m, :n_bins] = w
    return banks.astype(np.float32)


_BANKS = _mel_banks()
_WINDOW = np.hamming(FRAME_LEN).astype(np.float32)


def _frames(samples: np.ndarray) -> np.ndarray:
    n = 1 + (len(samples) - FRAME_LEN) // FRAME_SHIFT
    if n <= 0:
        return np.zeros((0, FRAME_LEN), dtype=np.float32)
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(n)[:, None]
    return samples[idx]


def fbank(samples: np.ndarray) -> np.ndarray:
    """80-dim log-mel features, shape (frames, 80), not yet mean-normalised.

    ``samples`` is mono float audio in [-1, 1] at 16 kHz. The models were
    trained on int16-scale input, so it is scaled up first.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1) * 32768.0
    frames = _frames(x).astype(np.float32, copy=True)
    if not len(frames):
        return np.zeros((0, N_MELS), dtype=np.float32)
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= PREEMPH * frames[:, :-1].copy()
    frames[:, 0] -= PREEMPH * frames[:, 0]
    frames *= _WINDOW
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    return np.log(np.maximum(power.astype(np.float32) @ _BANKS.T, _EPS))


def speech_mask(samples: np.ndarray) -> np.ndarray:
    """One bool per fbank frame: True where the frame looks like speech."""
    frames = _frames(np.asarray(samples, dtype=np.float32).reshape(-1))
    if not len(frames):
        return np.zeros(0, dtype=bool)
    level = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(level.max())
    if peak < SPEECH_MIN_RMS:
        return np.zeros(len(frames), dtype=bool)
    floor = max(SPEECH_MIN_RMS, peak * 10 ** (-SPEECH_DB_BELOW_PEAK / 20))
    return level >= floor


def speech_seconds(samples: np.ndarray) -> float:
    return float(speech_mask(samples).sum()) * FRAME_SHIFT / SAMPLE_RATE


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


class SpeakerEmbedder:
    """Turns audio into a unit-length voice embedding.

    ``load()`` must be called and must succeed before ``embed()``; an
    unloaded embedder raises rather than returning something neutral (a
    detector that silently stayed unloaded once produced a confident, false
    "no detections" verdict -- see tools/probe_bargein.py).
    """

    def __init__(self, model_path: str | Path, threads: int = 2) -> None:
        self.model_path = Path(model_path)
        self._threads = threads
        self._session = None
        self._input = ""
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def model_name(self) -> str:
        return self.model_path.stem

    def load(self) -> tuple[bool, str]:
        if self._session is not None:
            return True, ""
        if not self.model_path.is_file():
            return False, f"speaker model missing: {self.model_path}"
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self._threads
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            inputs = session.get_inputs()
            if len(inputs) != 1 or inputs[0].shape[-1] != N_MELS:
                return False, f"unexpected model input {[(i.name, i.shape) for i in inputs]}"
        except Exception as exc:  # noqa: BLE001 - any load failure is reported, not raised
            return False, f"could not load speaker model {self.model_path}: {exc}"
        self._session, self._input = session, inputs[0].name
        return True, ""

    def embed(self, samples: np.ndarray, trim: bool = True) -> np.ndarray | None:
        """Unit-length embedding, or None if there is too little audio.

        ``trim`` drops silent frames first, so the long silent tail of a
        capture does not dilute the voice.
        """
        if self._session is None:
            raise RuntimeError("SpeakerEmbedder.embed() called before a successful load()")
        feats = fbank(samples)
        if trim and len(feats):
            mask = speech_mask(samples)[: len(feats)]
            feats = feats[mask]
        # Under ~0.5 s the models produce near-random vectors.
        if len(feats) < 50:
            return None
        feats = feats - feats.mean(axis=0, keepdims=True)
        with self._lock:
            out = self._session.run(None, {self._input: feats[None, :, :].astype(np.float32)})
        return _unit(out[0][0])


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass
class SpeakerProfile:
    name: str
    embedding: np.ndarray
    model: str
    enrolled_at: float = field(default_factory=time.time)
    sample_count: int = 1

    def merged(self, embedding: np.ndarray, count: int = 1) -> SpeakerProfile:
        """This profile with more samples averaged in (weighted by count)."""
        total = self.sample_count + count
        mean = (self.embedding * self.sample_count + _unit(embedding) * count) / total
        return SpeakerProfile(self.name, _unit(mean), self.model, time.time(), total)


class SpeakerRegistry:
    """Enrolled voices, persisted as JSON.

    A missing file is an empty registry. A corrupt one is also treated as
    empty, with the reason in ``problem``, and is moved aside on the next
    save rather than silently overwritten.
    """

    VERSION = 1

    def __init__(self, path: str | Path, model: str) -> None:
        self.path = Path(path).expanduser()
        self.model = model
        self.problem = ""
        self._profiles: dict[str, SpeakerProfile] = {}
        self._corrupt = False
        self._lock = threading.Lock()
        self._stamp = None
        self._load()

    def _file_stamp(self):
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def reload_if_changed(self) -> bool:
        """Pick up edits made by another process (``homeai-mode speaker
        remove`` while the daemon runs). Returns True if it reloaded."""
        stamp = self._file_stamp()
        if stamp == self._stamp:
            return False
        with self._lock:
            self._profiles, self.problem, self._corrupt = {}, "", False
            self._load()
        return True

    def _load(self) -> None:
        self._stamp = self._file_stamp()
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            speakers = data["speakers"]
            if not isinstance(speakers, dict):
                raise ValueError("'speakers' is not an object")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.problem = f"speaker registry {self.path} unreadable ({exc}); starting empty"
            self._corrupt = True
            return
        skipped = []
        for name, raw in speakers.items():
            try:
                emb = np.asarray(raw["embedding"], dtype=np.float32)
                model = str(raw["model"])
            except (KeyError, TypeError, ValueError):
                skipped.append(f"{name} (malformed)")
                continue
            # Embeddings from different models are not comparable.
            if model != self.model:
                skipped.append(f"{name} (enrolled with {model}; re-enrol)")
                continue
            self._profiles[name] = SpeakerProfile(
                name, _unit(emb), model,
                float(raw.get("enrolled_at", 0.0)), int(raw.get("sample_count", 1)),
            )
        if skipped:
            self.problem = "ignored speaker profiles: " + ", ".join(skipped)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._profiles)

    def profiles(self) -> list[SpeakerProfile]:
        with self._lock:
            return list(self._profiles.values())

    def get(self, name: str) -> SpeakerProfile | None:
        with self._lock:
            return self._profiles.get(_find(self._profiles, name))

    def add(self, name: str, embedding: np.ndarray, count: int = 1, replace: bool = False) -> SpeakerProfile:
        name = name.strip()
        if not name:
            raise ValueError("speaker name is empty")
        with self._lock:
            key = _find(self._profiles, name) or name
            old = self._profiles.get(key)
            if old is None or replace:
                profile = SpeakerProfile(key, _unit(embedding), self.model, time.time(), count)
            else:
                profile = old.merged(embedding, count)
            self._profiles[key] = profile
            return profile

    def remove(self, name: str) -> bool:
        with self._lock:
            key = _find(self._profiles, name)
            return self._profiles.pop(key, None) is not None if key else False

    def save(self) -> None:
        with self._lock:
            data = {
                "version": self.VERSION,
                "speakers": {
                    p.name: {
                        "embedding": [round(float(x), 6) for x in p.embedding],
                        "model": p.model,
                        "enrolled_at": p.enrolled_at,
                        "sample_count": p.sample_count,
                    }
                    for p in self._profiles.values()
                },
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._corrupt and self.path.exists():
                self.path.replace(self.path.with_suffix(f".corrupt-{int(time.time())}"))
                self._corrupt = False
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".speakers-")
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(data, fh, indent=1)
                os.chmod(tmp, 0o600)
                os.replace(tmp, self.path)
                self._stamp = self._file_stamp()
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise


def _find(profiles: dict, name: str) -> str | None:
    """Case-insensitive lookup: "adam" and "Adam" are one person."""
    want = name.strip().casefold()
    return next((k for k in profiles if k.casefold() == want), None)


# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Identification:
    """The outcome for one utterance. ``name`` is None for "don't know"."""

    name: str | None
    score: float = 0.0          # similarity to the best-matching profile
    best: str | None = None     # best-matching profile, even if not accepted
    margin: float = 0.0         # best minus second-best score
    reason: str = ""            # why name is None


class SpeakerIdentifier:
    """Best match wins only if it is both close enough and clearly ahead.

    The margin rule stops confident misattribution between two similar
    voices: being told "you must be Adam" when you are not is worse than
    not being named at all.
    """

    def __init__(self, registry: SpeakerRegistry, threshold: float = 0.45, margin: float = 0.1) -> None:
        self.registry = registry
        self.threshold = threshold
        self.margin = margin

    def identify(self, embedding: np.ndarray | None) -> Identification:
        if embedding is None:
            return Identification(None, reason="too little speech")
        profiles = self.registry.profiles()
        if not profiles:
            return Identification(None, reason="nobody enrolled")
        scored = sorted(((cosine(embedding, p.embedding), p.name) for p in profiles), reverse=True)
        score, best = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        margin = score - second
        if score < self.threshold:
            return Identification(None, score, best, margin, "below threshold")
        if margin < self.margin:
            return Identification(None, score, best, margin, "ambiguous")
        return Identification(best, score, best, margin)
