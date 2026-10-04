#!/usr/bin/env python3
"""Print how close words Whisper might produce are to the wake word.

Used to choose WAKE_SIMILARITY in homeai/wake_verify.py:
    .venv/bin/python tools/wake_word_similarity.py [word ...]
"""
import sys
from difflib import SequenceMatcher

DEFAULT = ("jarvis jarvis's jervis javis jarvus garvis travis charvis harvis darvis "
           "service david valor nervous jargon harvest carvings marvin jarvi drivers").split()
for w in sys.argv[1:] or DEFAULT:
    ratio = SequenceMatcher(None, "jarvis", w.lower().removesuffix("'s")).ratio()
    print(f"{w:10s} {ratio:.2f}")
