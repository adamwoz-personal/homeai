"""Speech timeline: turn a synthesised chunk into an ``utterance`` event for the avatar.

Synthesis runs ~45x faster than playback, so each chunk's whole mouth timeline is known
before it is heard. The renderer plays it against its own monotonic clock; nothing
streams at 60 Hz over the socket.

- ``visemes_from_alignments``: Piper phoneme alignments (IPA symbol, sample count) ->
  ``[t_ms, viseme, weight]`` keyframes. Stress marks give their time to the next phoneme
  and raise the next vowel's weight; length/diacritic marks extend the previous phoneme;
  short word gaps don't close the mouth, real pauses do.
- ``level_envelope``: int16 PCM -> 60 Hz loudness, 0-255 (also the fallback lip sync
  when alignments are unavailable).
- ``chain_starts``: start times of consecutive chunks from sample counts, not wall clock.
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger("homeai.avatar.timeline")

LEVEL_HZ = 60
LEVEL_FLOOR_DB = -60.0
PAUSE_MIN_MS = 80          # a word gap shorter than this keeps the mouth moving
VOWEL_WEIGHT = 0.8
STRESSED_WEIGHT = 1.0
CONSONANT_WEIGHT = 1.0

REST = "rest"
VISEME_CLASSES: dict[str, str] = {
    "PP": "pbmɓʙɸβ",                                  # lips closed
    "FF": "fvɱʋⱱ",                                    # lip on teeth
    "TH": "θð",                                       # tongue between teeth
    "DD": "tdnszlɾɫɬɮɭɖɗɳʈɽɺcɟɲʎçʝɕʑʂʐʄ",             # alveolar, palatal, retroflex
    "SH": "ʃʒ",                                       # post-alveolar, lips forward
    "RR": "rɹɻʀʁɚ",                                   # r and r-coloured vowel
    "KK": "kgɡŋqxɣχħhɦʔʕʡʢɢɠʛɴʜɧʟɰ",                  # back of the mouth, jaw half open
    "OU": "uowʊɔɒʍʉɵøœɞyʏɥ",                          # rounded
    "AA": "aɑæɐʌɶ",                                   # open
    "EH": "eɛəɜɘεɤɯ",                                 # mid
    "IH": "iɪjɨᵻ",                                    # close front, lips wide
}
VOWEL_VISEMES = frozenset({"RR", "OU", "AA", "EH", "IH"})
# Pauses: sentence/phrase punctuation, Piper's ^ $ _ markers, clicks, digits.
PAUSE_SYMBOLS = frozenset("!\"#$'(),-.:;?^_0123456789ǀǁǂǃʘ")
STRESS_MARKS = frozenset("ˈˌ")
# Modify the previous phoneme: length, aspiration, palatalisation, rhotic/pharyngeal,
# combining diacritics, intonation arrows.
EXTENDERS = frozenset("ːˑʰʲ˞ˤ\u0303\u0327\u0329\u032a\u032f\u033a\u033b↑↓")
WORD_GAP = " "

VISEME_OF: dict[str, str] = {ch: v for v, chars in VISEME_CLASSES.items() for ch in chars}
VISEMES = (REST, *VISEME_CLASSES)


def classify(phoneme: str) -> str:
    """'stress', 'extend', 'gap', REST, or a viseme name. Unknown symbols -> 'EH' (logged)."""
    if phoneme in STRESS_MARKS:
        return "stress"
    if phoneme in EXTENDERS:
        return "extend"
    if phoneme == WORD_GAP:
        return "gap"
    if phoneme in PAUSE_SYMBOLS:
        return REST
    if phoneme in VISEME_OF:
        return VISEME_OF[phoneme]
    log.debug("unknown phoneme %r, using EH", phoneme)
    return "EH"


def visemes_from_alignments(alignments: list[tuple[str, int]], sample_rate: int) -> list[list]:
    """Phoneme alignments -> ``[t_ms, viseme, weight]`` keyframes, ending in rest."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frames: list[list] = []
    cursor = 0
    lead_in: int | None = None          # where pending stress marks started
    stressed = False

    def add(at: int, viseme: str, weight: float) -> None:
        t_ms = int(round(at * 1000 / sample_rate))
        if frames and frames[-1][1] == viseme:
            frames[-1][2] = max(frames[-1][2], weight)
            return
        if frames and frames[-1][0] == t_ms:
            frames.pop()
        frames.append([t_ms, viseme, weight])

    for phoneme, n in alignments:
        n = max(int(n), 0)
        kind = classify(phoneme)
        if kind == "stress":
            stressed = True
            lead_in = cursor if lead_in is None else lead_in
        elif kind == "extend":
            pass
        elif kind == "gap":
            if n * 1000 >= PAUSE_MIN_MS * sample_rate:
                add(cursor, REST, 0.0)
        else:
            at = cursor if lead_in is None else lead_in
            lead_in = None
            if kind == REST:
                weight = 0.0
            elif kind in VOWEL_VISEMES:
                weight = STRESSED_WEIGHT if stressed else VOWEL_WEIGHT
                stressed = False
            else:
                weight = CONSONANT_WEIGHT
            add(at, kind, weight)
        cursor += n
    if not frames or frames[-1][1] != REST:
        add(cursor, REST, 0.0)
    return frames


def level_envelope(pcm: np.ndarray | bytes, sample_rate: int, hz: int = LEVEL_HZ) -> list[int]:
    """RMS loudness per 1/hz s window, mapped linearly in dB from LEVEL_FLOOR_DB..0 to 0..255."""
    if sample_rate <= 0 or hz <= 0:
        raise ValueError("sample_rate and hz must be positive")
    samples = np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) else pcm
    samples = np.asarray(samples, dtype=np.float64) / 32768.0
    if samples.size == 0:
        return []
    window = sample_rate / hz
    count = math.ceil(samples.size / window)
    out = []
    for i in range(count):
        seg = samples[int(round(i * window)):int(round((i + 1) * window))]
        rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
        db = 20 * math.log10(rms) if rms > 0 else LEVEL_FLOOR_DB
        frac = (db - LEVEL_FLOOR_DB) / -LEVEL_FLOOR_DB
        out.append(int(round(255 * min(1.0, max(0.0, frac)))))
    return out


def chain_starts(first_start: float, sample_counts: list[int], sample_rate: int) -> list[float]:
    """Start time of each chunk when they play back to back."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    starts, elapsed = [], 0
    for n in sample_counts:
        starts.append(first_start + elapsed / sample_rate)
        elapsed += n
    return starts


def utterance_data(uid: str, start_mono: float, sample_rate: int, num_samples: int,
                   level: list[int], visemes: list[list] | None = None, text: str = "") -> dict:
    """Payload of an ``utterance`` event; ``visemes`` omitted when alignments are unavailable."""
    data = {"id": uid, "start_mono": start_mono, "sample_rate": sample_rate,
            "duration_ms": int(round(num_samples * 1000 / sample_rate)),
            "level_hz": LEVEL_HZ, "level": level, "text": text}
    if visemes:
        data["visemes"] = visemes
    return data
