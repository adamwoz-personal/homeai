# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Jarvis's voice-ID turns: who's speaking, "who am I", forgetting, and enrolment (CODE_REVIEW_PLAN item 6).

Moved out of daemon.py unchanged (about 215 lines). ``VoiceIdTurns`` is a mixin of
``VoiceAssistant``: it uses the assistant's speaker, transcript, voice-ID store
and presence, and the daemon calls ``_identify``, ``_handle_voice_id``,
``_pool_stranger`` and ``_with_offer`` from its turn loop.
"""
from __future__ import annotations

import logging

import numpy as np

from . import voice_id as vid
from .bargein import is_dismissal
from .transcript import Stopwatch, Turn

log = logging.getLogger("homeai")


class VoiceIdTurns:
    def _identify(self, audio: np.ndarray, turn: Turn):
        """(Identification, embedding), or (None, None) when it is off or fails."""
        if self.voice_id is None:
            return None, None
        stage = Stopwatch()
        try:
            ident, emb = self.voice_id.identify(audio)
        except Exception as exc:  # noqa: BLE001 - never lose a turn to speaker-id
            log.warning("speaker identification failed: %s", exc)
            return None, None
        turn.speaker_ms = stage.ms()
        turn.speaker = ident.name
        turn.speaker_score = round(float(ident.score), 3) if ident.best else None
        log.info("speaker: %s  [%.0fms]", vid.describe(ident), turn.speaker_ms)
        return ident, emb

    def _handle_voice_id(self, turn: Turn, text: str, source: str, audio: np.ndarray,
                         ident, emb, total: Stopwatch) -> bool:
        """Enrolment answers, follow-ups from other voices, and voice-ID
        commands. Returns True if the turn was dealt with here."""
        if self._enrolment is not None:
            enrolment = self._enrolment
            # An offered enrolment only takes its answer from the voice it was offered to.
            same_voice = not enrolment.offered or self._strangers.matches(self._offer_voice, emb)
            if source == "followup" and not enrolment.expired() and same_voice \
                    and not is_dismissal(text, self.cfg.wake.model):
                if enrolment.stage == "name":
                    self._name_enrolment(turn, text, audio, ident, total)
                elif enrolment.stage == "how":
                    self._choose_enrolment(turn, text, audio, total)
                else:
                    self._continue_enrolment(turn, audio, total)
                return True
            log.info("enrolment for %s abandoned", enrolment.name)
            self._end_enrolment()

        # Family chatter during a follow-up window is not an answer.
        if source == "followup" and self.voice_id is not None and ident is not None:
            owner = self.memory.speaker
            other, sim = self.voice_id.is_someone_else(owner, ident, emb)
            if other:
                log.info("follow-up from another voice (%s; %s to %s) - ignoring",
                         vid.describe(ident), "?" if sim is None else f"{sim:.2f}", owner)
                turn.verdict = "other-speaker"
                self._end_turn(turn, total)
                return True

        # After an interruption the mic opens with no wake word, so whoever
        # talks next in the room is captured: background conversation was
        # answered three times in a row (scores 0.01, -0.03, 0.31). The first
        # fix demanded a positive match to the owner and then ignored Adam
        # himself four times in his first art session (barge-in clips scored
        # 0.29-0.45: short, with Jarvis's own voice fading out underneath).
        # So barge-ins get the follow-up rule: ignored only when the voice is
        # confidently someone else. A short "stop" can't be judged, so it reaches the dismissal.
        owner = self.memory.speaker
        if source == "bargein" and self.voice_id is not None and ident is not None:
            other, sim = self.voice_id.is_someone_else(owner, ident, emb)
            if other:
                log.info("barge-in follow-up from another voice (%s; %s to %s) - staying quiet",
                         vid.describe(ident), "?" if sim is None else f"{sim:.2f}", owner)
                turn.verdict = "other-speaker"
                self._end_turn(turn, total)
                return True

        cmd = vid.parse_command(text)
        if cmd is None or (self.voice_id is None and cmd.kind != "enrol"):
            return False
        listen = False
        if self.voice_id is None:
            reply = vid.MSG_OFF
        elif cmd.kind == "enrol" and not cmd.name:
            # Ask, and take the name from the answer.
            self._enrolment = vid.Enrolment(None, stage="name")
            self._enrolment.add(audio)
            reply, listen = vid.MSG_ENROL_NO_NAME, True
            log.info("enrolling: no name heard in %r; asking", text)
        elif self._name_taken(cmd.name, ident):
            reply = vid.MSG_ENROL_TAKEN.format(name=cmd.name)
            log.info("enrolling: %s is taken and this isn't their voice", cmd.name)
        elif cmd.kind == "enrol":
            self._enrolment = vid.Enrolment(cmd.name, stage="how")
            self._enrolment.add(audio)
            reply, listen = vid.MSG_ENROL_HOW.format(name=cmd.name), True
            log.info("enrolling %s", cmd.name)
        elif not self.voice_id.registry.names():
            reply = vid.MSG_WHO_NOBODY
        elif ident is None or ident.name is None:
            reply = vid.MSG_WHO_UNSURE if cmd.kind == "who" else vid.MSG_FORGET_UNKNOWN
        elif cmd.kind == "who":
            reply = vid.MSG_WHO_KNOWN.format(name=ident.name)
        else:
            self.voice_id.forget(ident.name)
            self.memory.clear()
            reply = vid.MSG_FORGOTTEN.format(name=ident.name)
            log.info("forgot the voice of %s", ident.name)
        turn.verdict = f"voice-{cmd.kind}"
        self._say_turn(turn, reply, total, listen)
        return True

    def _pool_stranger(self, source: str, audio: np.ndarray, ident, emb):
        """Pool a clearly unknown voice; returns it when it's time to ask its name.

        "Clearly": no known profile is even close (below the follow-up gate's
        "different" line), so a known person who sounds odd today isn't pooled.
        """
        if self.voice_id is None or ident is None or ident.name is not None or emb is None \
                or source not in ("wake", "followup") or self._enrolment is not None:
            return None
        if ident.best is not None and ident.score >= self.cfg.speaker.different_below:
            return None
        return self._strangers.add(emb, audio)

    def _with_offer(self, spoken: str, held: bool) -> str:
        """After answering an unknown voice that keeps talking to Jarvis, ask its name (once)."""
        voice, self._pending_offer = self._pending_offer, None
        if voice is None or held or spoken.rstrip().endswith("?") or self._enrolment is not None:
            return spoken
        self._enrolment = self._strangers.take(voice)
        self._offer_voice = voice
        log.info("offering a voice profile to an unknown voice (%d clips, %.1f s)",
                 len(voice.clips), voice.speech_s)
        return f"{spoken} {vid.MSG_OFFER}"

    def _name_taken(self, name: str | None, ident) -> bool:
        """Someone else's profile already has this name: never overwrite it from another voice."""
        if not name or self.voice_id is None:
            return False
        existing = self.voice_id.registry.get(name)
        if existing is None or (ident is not None and ident.name == existing.name):
            return False
        # A weak profile is why people re-enrol: Adam's own voice can score below the threshold. Closest to that
        # profile and not confidently someone else lets them refresh it; a different voice still can't.
        close = (ident is not None and ident.best == existing.name
                 and ident.score >= self.voice_id.cfg.different_below)
        return not close

    def _end_enrolment(self) -> None:
        if self._enrolment is not None and self._enrolment.method == "read":
            self.presence.publish("script", {"text": ""})
        self._enrolment = None
        self._offer_voice = None

    def _name_enrolment(self, turn: Turn, text: str, audio: np.ndarray, ident, total: Stopwatch) -> None:
        """The answer to "what name should I remember you by?" (or, offered, "what's your name?")."""
        enrolment = self._enrolment
        name = vid.extract_name(text)
        turn.verdict = "voice-enrol"
        if not name:
            log.info("enrolling: no name in %r; %s", text, "declined" if enrolment.offered else "giving up")
            self._end_enrolment()
            self._say_turn(turn, vid.MSG_OFFER_DECLINED if enrolment.offered else vid.MSG_ENROL_NAME_AGAIN, total)
            return
        if self._name_taken(name, ident):
            log.info("enrolling: %s is taken and this isn't their voice", name)
            self._end_enrolment()
            self._say_turn(turn, vid.MSG_ENROL_TAKEN.format(name=name), total)
            return
        enrolment.name = name
        enrolment.add(audio)
        if enrolment.offered:
            # The speech was gathered in passing; the name was all that was missing.
            self._finish_enrolment(turn, total, vid.MSG_OFFER_DONE)
            return
        enrolment.stage = "how"
        log.info("enrolling %s", name)
        self._say_turn(turn, vid.MSG_ENROL_HOW.format(name=name), total, listen=True)

    def _choose_enrolment(self, turn: Turn, text: str, audio: np.ndarray, total: Stopwatch) -> None:
        """The answer to "talk for a bit, or read something?"."""
        enrolment = self._enrolment
        enrolment.method, enrolment.stage = vid.enrol_method(text), "speech"
        enrolment.add(audio)
        turn.verdict = "enrolling"
        log.info("enrolling %s: %s", enrolment.name, enrolment.method)
        if enrolment.method == "read":
            self.presence.publish("script", {"text": vid.ENROL_SCRIPT})
            self._say_turn(turn, vid.MSG_ENROL_READ, total, listen=True)
        else:
            self._say_turn(turn, vid.MSG_ENROL_TALK, total, listen=True)

    def _continue_enrolment(self, turn: Turn, audio: np.ndarray, total: Stopwatch) -> None:
        enrolment = self._enrolment
        enrolment.add(audio)
        log.info("enrolling %s: %d answer(s), %.1f s of speech",
                 enrolment.name, enrolment.rounds, enrolment.speech_s)
        turn.verdict = "enrolling"
        if not (enrolment.done or enrolment.gave_up):
            more = vid.MSG_ENROL_READ_MORE if enrolment.method == "read" else vid.MSG_ENROL_MORE
            self._say_turn(turn, more, total, listen=True)
            return
        self._finish_enrolment(turn, total, vid.MSG_ENROL_DONE)

    def _finish_enrolment(self, turn: Turn, total: Stopwatch, done_msg: str) -> None:
        enrolment = self._enrolment
        self._end_enrolment()
        profile = None
        if enrolment.done:
            try:
                profile = self.voice_id.finish_enrolment(enrolment)
            except OSError as exc:
                log.error("could not save voice profile: %s", exc)
        if profile is None:
            log.warning("enrolment of %s failed (%.1f s of speech)",
                        enrolment.name, enrolment.speech_s)
            self._say_turn(turn, vid.MSG_ENROL_FAILED, total)
            return
        log.info("enrolled %s from %.1f s of speech%s", enrolment.name, enrolment.speech_s,
                 " (offered)" if enrolment.offered else "")
        self._say_turn(turn, done_msg.format(name=enrolment.name), total)

    # -- art mode ------------------------------------------------------------
