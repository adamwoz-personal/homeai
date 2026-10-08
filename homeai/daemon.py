# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Voice assistant orchestrator.

Wires the components together and enforces the one rule that matters for an
always-on service: **it must never die**. Every failure path logs, degrades,
and returns to idle.

Threading model:
  * PortAudio callback thread  - fills the ring buffer (in ``audio``).
  * Wake thread                - polls frames, runs the state machine.
  * Worker                     - one utterance at a time, strictly serialised.

The worker is deliberately serial. The LLM server has a single request slot, so
concurrent requests only produce confusing stalls.
"""

from __future__ import annotations

import collections
import concurrent.futures
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import replace

import numpy as np

from .agent_cli import build_agent_client
from .memory import ConversationMemory
from .audio import Microphone
from .config import Config, load
from .safety import Verdict, check_utterance, sanitise_for_speech
from .speech import flatten_markdown, normalise_for_speech
from .stt import Transcriber
from .transcript import Stopwatch, TranscriptLog, Turn
from .tts import Speaker
from .bargein import InterruptListener, contains_wake_fragments, is_dismissal
from .dialogue import (
    CONTINUE_PROMPT,
    FollowupChain,
    HeldRemainder,
    invites_reply,
    is_continue_request,
    is_decline,
    is_trailing_fragment,
    one_sentence,
    split_for_budget,
)
from .wake import CaptureMachine, State, build_detector
from . import voice_id as vid
from .wake_verify import (
    closing_reply,
    is_hallucination_only,
    is_pleasantry_only,
    mentions_wake_word,
    strip_wake_phrase,
)

log = logging.getLogger("homeai")

# Spoken responses for failure paths. Short by design: a long apology is worse
# than a brief one.
MSG_AGENT_DOWN = "Sorry, my brain is offline right now."
MSG_AGENT_ERROR = "Sorry, something went wrong."
MSG_REFUSED = "I can't do that by voice."
MSG_NOT_UNDERSTOOD = "Sorry, I didn't catch that."
# Not a question: asking one would open a follow-up window, and the people
# talking to each other would "answer" it.
# Replaces the style hint for thanks and goodbyes mid-conversation.
CLOSING_HINT = ("(The user is thanking you, Jarvis, for your help, or saying goodbye. "
                "Reply warmly and naturally in ONE short sentence. Do not "
                "talk about gratitude, do not continue the earlier topic and do not ask "
                "a question.)")
MSG_FRAGMENT = "Sorry, I only caught part of that. Say hey Jarvis again if you meant me."

# Spoken when the agent is taking long enough that silence reads as failure.
# The threshold sits above a normal conversational turn (~0.5-1.5s) so simple
# questions are never padded, but below the point where a user assumes the
# assistant did not hear them.
MSG_WORKING = "Let me look that up."
PROGRESS_AFTER_S = 2.5

# Pause between the end of a reply and opening a follow-up window, so the
# capture does not start on the acoustic tail of Jarvis's own voice.
FOLLOWUP_SETTLE_S = 0.25


class VoiceAssistant:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load()
        self.mic = Microphone(self.cfg.audio)
        self.stt = Transcriber(self.cfg.stt)
        self.tts = Speaker(self.cfg.tts, self.cfg.audio.output_device)
        # Continuity between turns. Bounded and self-expiring; see memory.py
        # for why this does not reintroduce session poisoning.
        self.memory = ConversationMemory()
        self.agent = build_agent_client(self.cfg.agent, memory=self.memory)
        # Who is speaking; None when turned off or the model failed to load.
        self.voice_id = vid.VoiceId(self.cfg.speaker) if self.cfg.speaker.enabled else None
        self._enrolment: vid.Enrolment | None = None
        self.transcript = TranscriptLog(
            self.cfg.transcript.path, enabled=self.cfg.transcript.enabled
        )
        self.machine = CaptureMachine(self.cfg.wake)

        self._detector = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        # Set when a reply was cut short by the wake word. The wake loop is
        # deaf while speaking, so it never saw that wake word; this hands the
        # capture over to it explicitly.
        self._bargein_armed = threading.Event()
        # Set when a reply ended with a question: capture the answer without
        # the wake word. Bounded by _followup_chain; see homeai/dialogue.py.
        self._followup_armed = threading.Event()
        self._last_wake_score: float | None = None
        self._followup_chain = FollowupChain(self.cfg.wake.followup_max_chain)
        # The unspoken tail of a reply that ran over the spoken budget.
        self._held = HeldRemainder()
        self._last_interrupted = False
        self._threads: list[threading.Thread] = []

    # -- startup -----------------------------------------------------------

    def preflight(self) -> list[str]:
        """Check everything before claiming to be ready. Returns problems."""
        problems = list(self.cfg.validation_errors())

        ok, problem = self.stt.available()
        if not ok:
            problems.append(problem)

        ok, problem = self.tts.available()
        if not ok:
            problems.append(problem)

        if not self.agent.health():
            problems.append(
                f"ZeroClaw agent unreachable ({self.cfg.agent.transport} transport: "
                f"{self.cfg.agent.cli_binary if self.cfg.agent.transport == 'cli' else self.cfg.agent.health_url})"
            )

        return problems

    # -- threads -----------------------------------------------------------

    def _wake_loop(self) -> None:
        """Poll the ring buffer, detect wake, and delimit utterances.

        Uses an absolute cursor so each sample is examined exactly once.
        Reading 'the last N samples' instead causes the detector to see the
        same audio repeatedly and fire multiple times per utterance.
        """
        frame_size = self.cfg.audio.block_size
        poll_interval = frame_size / self.cfg.audio.sample_rate / 2
        cursor = self.mic.buffer.total_written
        utterance: list[np.ndarray] = []
        source = "wake"
        # The frames just before a trigger, which contain the wake word itself.
        preroll_frames = max(1, round(self.cfg.wake.preroll_s * self.cfg.audio.sample_rate
                                      / frame_size))
        recent: collections.deque[np.ndarray] = collections.deque(maxlen=preroll_frames)
        preroll: np.ndarray | None = None

        while not self._stop.is_set():
            time.sleep(poll_interval)

            # Never listen to ourselves.
            if self.mic.paused or self.tts.speaking:
                cursor = self.mic.buffer.total_written
                continue

            # A reply was just interrupted. This loop was deaf for the whole
            # of that reply, so it never detected the wake word that stopped
            # it -- only the InterruptListener did, and that listener captures
            # nothing. Without this handoff the user's follow-up question is
            # never recorded at all: the field symptom was an interrupt that
            # visibly worked, followed by total silence in the log.
            if self._bargein_armed.is_set():
                self._bargein_armed.clear()
                self._followup_armed.clear()
                utterance = []
                cursor = self.mic.buffer.total_written
                self.machine.on_wake(time.monotonic())
                source = "bargein"
                log.info("barge-in: capturing follow-up")
            elif self._followup_armed.is_set():
                # The reply ended with a question. Listen for the answer
                # without the wake word, for a longer lead-in than usual.
                self._followup_armed.clear()
                utterance = []
                cursor = self.mic.buffer.total_written
                self.machine.on_wake(
                    time.monotonic(), lead_in_s=self.cfg.wake.followup_lead_in_s
                )
                source = "followup"
                log.info(
                    "listening for a reply (%.1fs, no wake word needed)",
                    self.cfg.wake.followup_lead_in_s,
                )

            try:
                chunk, cursor = self.mic.buffer.read_new(cursor)
                if chunk.size < frame_size:
                    continue

                # Process in whole frames; keep any remainder for next time.
                usable = (chunk.size // frame_size) * frame_size
                cursor -= chunk.size - usable
                now = time.monotonic()

                for start in range(0, usable, frame_size):
                    frame = chunk[start:start + frame_size]

                    if self.machine.state is State.IDLE:
                        recent.append(frame)
                        if not self.machine.accepts_wake(now):
                            continue
                        if self._detector and self._detector.detect(frame):
                            wake_score = getattr(self._detector, "last_score", None)
                            if wake_score is None:
                                log.info("wake word detected")
                            else:
                                log.info("wake word detected (score %.3f)", wake_score)
                            self.machine.on_wake(now)
                            utterance = []
                            source = "wake"
                            self._last_wake_score = wake_score
                            preroll = np.concatenate(recent)
                            recent.clear()
                        continue

                    utterance.append(frame)
                    if self.machine.feed(frame, now):
                        audio = np.concatenate(utterance) if utterance else np.zeros(0, np.float32)
                        heard = self.machine.heard_speech
                        utterance = []
                        self.machine.reset(now)
                        if self._detector:
                            self._detector.reset()
                        # Drop anything captured during handoff so the next
                        # turn cannot re-detect this utterance's wake word.
                        cursor = self.mic.buffer.total_written
                        if source == "followup" and not heard:
                            # Nobody answered. The common case, and not worth
                            # a Whisper run that could hallucinate words out
                            # of room noise and send them to the agent.
                            log.info("no reply to follow-up; back to wake word")
                        else:
                            self._enqueue(audio, source,
                                          preroll if source == "wake" else None)
                        source = "wake"
                        preroll = None
                        break
            except Exception:  # noqa: BLE001 - this thread must never die
                log.exception("wake loop error; resetting")
                self.machine.reset()
                utterance = []
                source = "wake"
                cursor = self.mic.buffer.total_written

    def _enqueue(self, audio: np.ndarray, source: str = "wake",
                 preroll: np.ndarray | None = None) -> None:
        try:
            self._queue.put_nowait((audio, source, preroll))
        except queue.Full:
            log.warning("worker busy; dropping utterance")

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio, source, preroll = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle(audio, source, preroll)
            except Exception:  # noqa: BLE001 - never let one turn kill the service
                log.exception("unhandled error processing utterance")
                # The apology goes through the same speaker that may have just
                # failed. If it raises too, the worker thread would die and
                # Jarvis would go deaf until restarted.
                try:
                    self._speak(MSG_AGENT_ERROR)
                except Exception:  # noqa: BLE001
                    log.exception("could not speak the error message either")

    # -- one turn ----------------------------------------------------------

    def _ask_with_progress(self, utterance: str):
        """Ask the agent, announcing a wait if the answer is slow to arrive.

        A research call reads several web pages and can take fifteen seconds.
        Without a cue, that silence is indistinguishable from the assistant
        having failed to hear -- the user repeats themselves, which re-triggers
        the wake word and makes things worse.

        The agent runs on a worker thread so the announcement can be spoken
        while it is still working. Speaking is deliberately done on *this*
        thread: ``_speak`` mutes the microphone, and two threads racing on mute
        state would leave the mic muted after one of them finished.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.agent.ask, utterance)
            try:
                return future.result(timeout=PROGRESS_AFTER_S)
            except concurrent.futures.TimeoutError:
                pass

            log.info("agent slow (>%.1fs), announcing wait", PROGRESS_AFTER_S)
            try:
                self._speak(MSG_WORKING)
            except Exception:  # noqa: BLE001 - a cue must never break the turn
                log.warning("progress announcement failed", exc_info=True)

            return future.result()

    def _speak(self, text: str, chunked: bool = False) -> float:
        """Speak with the mic muted, so we never transcribe our own voice.

        Returns milliseconds to first audible sound, which is the latency a
        listener perceives. Total elapsed time is dominated by the length of
        the sentence being spoken and is not a delay.
        """
        listener = self._start_interrupt_listener(text) if chunked else None

        # With a listener running the mic must stay open, or there is nothing
        # to interrupt with. Without one, keep the mic muted as before.
        if listener is None:
            self.mic.pause()
        try:
            if listener is not None:
                result = self.tts.say_chunked(text, should_stop=listener.should_stop)
                if result.interrupted:
                    log.info(
                        "reply interrupted after %d chunk(s)", result.chunks_spoken
                    )
            elif chunked:
                result = self.tts.say_chunked(text)
            else:
                result = self.tts.say_safe(text)
            return result.first_audio_ms
        finally:
            interrupted = listener is not None and listener.triggered
            self._last_interrupted = interrupted
            if listener is not None:
                listener.stop()
            else:
                self.mic.resume()
            # Our own speech is still inside openWakeWord's ~1s of internal
            # context even though the mic was muted, so it can re-trigger the
            # detector the moment we stop talking. Observed live: a wake fired
            # 0.487s after a reply finished. Muting the mic is not enough;
            # the detector must also be cleared and made briefly deaf.
            #
            if self._detector:
                self._detector.reset()
            self.machine.reset(time.monotonic())

            # The person said the wake word deliberately, so a question is
            # almost certainly coming. Arm the wake loop to start capturing it
            # without requiring a second wake word.
            if interrupted:
                self._bargein_armed.set()

    def _start_interrupt_listener(self, text: str):
        """Return a running InterruptListener, or None if barge-in is unsafe.

        Barge-in is skipped when the reply itself contains the wake word,
        because measurement showed Jarvis reliably triggers its own detector in
        that case (peak score 0.9953). Skipping costs one missed chance to
        interrupt; not skipping makes Jarvis cut itself off mid-reply.
        """
        if not self.cfg.wake.barge_in:
            return None
        if contains_wake_fragments(text, self.cfg.wake.model):
            log.info("barge-in skipped: reply contains wake-word fragments")
            return None

        listener = InterruptListener(
            self.mic,
            lambda: build_detector(self.cfg.wake),
            self.cfg.audio.block_size,
        )
        return listener if listener.start() else None

    def _handle(self, audio: np.ndarray, source: str = "wake",
                preroll: np.ndarray | None = None) -> None:
        turn = Turn(source=source,
                    wake_score=self._last_wake_score if source == "wake" else None)
        total = Stopwatch()
        stage = Stopwatch()
        self._followup_chain.record_turn(source == "followup")

        verify = source == "wake" and self.cfg.wake.verify and preroll is not None
        if verify:
            audio = np.concatenate([preroll, audio])
        transcript = self.stt.transcribe_audio(audio, self.cfg.audio.sample_rate)
        turn.stt_ms = stage.ms()
        if not transcript.ok:
            log.info("discarding utterance: %s", transcript.error)
            turn.ok = False
            turn.error = transcript.error
            turn.verdict = "discarded"
            turn.total_ms = total.ms()
            self.transcript.write(turn)
            return

        turn.heard = transcript.text
        if verify:
            if not mentions_wake_word(transcript.text, self.cfg.wake.model):
                log.info("unverified wake (score %.3f): Whisper heard %r, no wake word - "
                         "ignoring", turn.wake_score or 0.0, transcript.text)
                turn.verdict = "unverified"
                turn.total_ms = total.ms()
                self.transcript.write(turn)
                return
            transcript = replace(transcript,
                                 text=strip_wake_phrase(transcript.text, self.cfg.wake.model))
            if not transcript.text:
                log.info("wake word only (%r); nothing to answer", turn.heard)
                turn.verdict = "wake-only"
                turn.total_ms = total.ms()
                self.transcript.write(turn)
                return
        log.info("heard: %s  [stt %.0fms]", transcript.text, turn.stt_ms)

        ident, emb = self._identify(audio, turn)
        if self._handle_voice_id(turn, transcript.text, source, audio, ident, emb, total):
            return
        if ident is not None:
            self.memory.switch_speaker(ident.name)

        # "Stop", "quiet", "I'm not talking to you". After a barge-in the mic
        # stays open for a follow-up question, but the interruption is often
        # the whole point. Querying the agent here would answer a question
        # nobody asked and produce more speech to interrupt.
        if is_dismissal(transcript.text, self.cfg.wake.model):
            log.info("dismissal: %r - staying quiet", transcript.text)
            self._held.clear()
            turn.verdict = "dismissed"
            turn.reply = ""
            turn.total_ms = total.ms()
            self.transcript.write(turn)
            return

        # Answering "want me to keep going?". Handled here, without the agent:
        # the rest of the reply already exists, and asking the model again
        # would produce a different answer rather than the remainder.
        now = time.monotonic()
        if self._held.pending(now):
            if is_continue_request(transcript.text):
                log.info("continuing held reply")
                turn.verdict = "continued"
                self._deliver(turn, self._held.take(now), total)
                return
            if is_decline(transcript.text):
                log.info("declined the rest of the reply")
                self._held.clear()
                turn.verdict = "declined"
                turn.reply = ""
                turn.total_ms = total.ms()
                self.transcript.write(turn)
                return
            # Anything else is a new question; the old tail is stale.
            self._held.clear()

        # "Thank you" ends a conversation; it doesn't start one. Out of nowhere
        # it is room audio or a hallucination: stay quiet. Mid-conversation,
        # thanks and goodbyes are good manners and get a reply from the model,
        # held to one sentence: without that, "thank you" was taken as a cue
        # to keep talking (23 s, live 2026-10-03).
        if is_pleasantry_only(transcript.text):
            in_conversation = bool(self.memory.recent())
            fallback = "" if is_hallucination_only(transcript.text) or not in_conversation \
                else closing_reply(transcript.text)
            self._held.clear()
            if not fallback:
                log.info("pleasantry with no conversation in progress: %r - staying quiet",
                         transcript.text)
                turn.verdict = "pleasantry"
                turn.total_ms = total.ms()
                self.transcript.write(turn)
                return
            turn.verdict = "closing"
            stage.reset()
            reply = self.agent.ask(transcript.text, hint=CLOSING_HINT)
            turn.agent_ms = stage.ms()
            turn.attempts = reply.attempts
            ack = one_sentence(normalise_for_speech(sanitise_for_speech(reply.text))) \
                if reply.ok else ""
            if not ack:
                log.warning("closing reply unusable (%s); using %r",
                            reply.error or "empty", fallback)
                ack = fallback
            log.info("closing %r - %s", transcript.text, ack)
            turn.reply = ack
            stage.reset()
            turn.tts_first_audio_ms = self._speak(ack)
            turn.tts_ms = stage.ms()
            turn.total_ms = total.ms()
            self.transcript.write(turn)
            return

        if is_trailing_fragment(transcript.text):
            log.info("fragment: %r - probably not addressed to me", transcript.text)
            turn.verdict = "fragment"
            stage.reset()
            self._speak(MSG_FRAGMENT)
            turn.tts_ms = stage.ms()
            turn.reply = MSG_FRAGMENT
            turn.total_ms = total.ms()
            self.transcript.write(turn)
            return

        verdict = check_utterance(transcript.text)
        if verdict.verdict in (Verdict.REFUSE, Verdict.CONFIRM):
            # Confirmation dialogue is Phase 4. Until then, refuse rather than
            # silently performing a state-changing action.
            log.warning("refused (%s): %s", verdict.reason, transcript.text)
            turn.ok = False
            turn.verdict = verdict.verdict.value
            turn.error = verdict.reason
            stage.reset()
            self._speak(MSG_REFUSED)
            turn.tts_ms = stage.ms()
            turn.reply = MSG_REFUSED
            turn.total_ms = total.ms()
            self.transcript.write(turn)
            return

        turn.verdict = Verdict.ALLOW.value
        stage.reset()
        reply = self._ask_with_progress(transcript.text)
        turn.agent_ms = stage.ms()
        turn.attempts = reply.attempts
        if not reply.ok:
            log.error("agent failed after %.0fms: %s", turn.agent_ms, reply.error)
            turn.ok = False
            turn.error = reply.error
            spoken = MSG_AGENT_DOWN if "unreachable" in reply.error else MSG_AGENT_ERROR
            turn.reply = spoken
            self._speak(spoken)
            turn.total_ms = total.ms()
            self.transcript.write(turn)
            return

        # Order matters: flatten_markdown needs line breaks, which
        # sanitise_for_speech collapses; normalise_for_speech needs the
        # security/runaway guard to have already run.
        spoken = normalise_for_speech(
            sanitise_for_speech(flatten_markdown(reply.text))
        )
        self._deliver(turn, spoken, total)

    # -- speaker recognition -------------------------------------------------

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
            if source == "followup" and not self._enrolment.expired() \
                    and not is_dismissal(text, self.cfg.wake.model):
                self._continue_enrolment(turn, audio, total)
                return True
            log.info("enrolment for %s abandoned", self._enrolment.name)
            self._enrolment = None

        # Family chatter during a follow-up window is not an answer.
        if source == "followup" and self.voice_id is not None and ident is not None:
            owner = self.memory.speaker
            other, sim = self.voice_id.is_someone_else(owner, ident, emb)
            if other:
                log.info("follow-up from another voice (%s; %s to %s) - ignoring",
                         vid.describe(ident), "?" if sim is None else f"{sim:.2f}", owner)
                turn.verdict = "other-speaker"
                turn.total_ms = total.ms()
                self.transcript.write(turn)
                return True

        cmd = vid.parse_command(text)
        if cmd is None or (self.voice_id is None and cmd.kind != "enrol"):
            return False
        listen = False
        if self.voice_id is None:
            reply = vid.MSG_OFF
        elif cmd.kind == "enrol" and not cmd.name:
            reply = vid.MSG_ENROL_NO_NAME
        elif cmd.kind == "enrol":
            self._enrolment = vid.Enrolment(cmd.name)
            self._enrolment.add(audio)
            reply, listen = vid.MSG_ENROL_START.format(name=cmd.name), True
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

    def _continue_enrolment(self, turn: Turn, audio: np.ndarray, total: Stopwatch) -> None:
        enrolment = self._enrolment
        enrolment.add(audio)
        log.info("enrolling %s: %d answer(s), %.1f s of speech",
                 enrolment.name, enrolment.rounds, enrolment.speech_s)
        turn.verdict = "enrolling"
        if not (enrolment.done or enrolment.gave_up):
            self._say_turn(turn, vid.MSG_ENROL_MORE, total, listen=True)
            return
        self._enrolment = None
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
        log.info("enrolled %s from %.1f s of speech", enrolment.name, enrolment.speech_s)
        self._say_turn(turn, vid.MSG_ENROL_DONE.format(name=enrolment.name), total)

    def _say_turn(self, turn: Turn, text: str, total: Stopwatch, listen: bool = False) -> None:
        """Speak a fixed reply, log the turn, and optionally listen without
        the wake word for the answer."""
        turn.reply = text
        stage = Stopwatch()
        turn.tts_first_audio_ms = self._speak(text)
        turn.tts_ms = stage.ms()
        turn.total_ms = total.ms()
        self.transcript.write(turn)
        if listen:
            time.sleep(FOLLOWUP_SETTLE_S)
            self._followup_armed.set()

    def _deliver(self, turn: Turn, spoken: str, total: Stopwatch) -> None:
        """Speak a finished reply within the spoken budget, then log it."""
        spoken, rest = split_for_budget(spoken, self.cfg.tts.spoken_budget_words)
        if rest:
            self._held.hold(rest, time.monotonic())
            spoken = f"{spoken} {CONTINUE_PROMPT}"
            log.info("reply over budget: holding %d words", len(rest.split()))
        turn.reply = spoken
        stage = Stopwatch()
        turn.tts_first_audio_ms = self._speak(spoken, chunked=True)
        turn.tts_ms = stage.ms()
        if self._last_interrupted:
            # Cut off mid-reply: they do not want the rest either.
            self._held.clear()
        turn.total_ms = total.ms()
        # Perceived lag ends when sound starts; the rest is Jarvis talking.
        turn.perceived_ms = turn.stt_ms + turn.agent_ms + turn.tts_first_audio_ms

        log.info(
            "perceived lag %.0fms (stt %.0f, agent %.0f, tts-first-audio %.0f); "
            "spoke for %.0fms; attempts %d",
            turn.perceived_ms, turn.stt_ms, turn.agent_ms,
            turn.tts_first_audio_ms, turn.tts_ms, turn.attempts,
        )
        self.transcript.write(turn)
        self._maybe_listen_for_reply(spoken)

    def _maybe_listen_for_reply(self, spoken: str) -> None:
        """Open a wake-word-free window if the reply just asked a question.

        Skipped after an interruption: the barge-in path is already capturing,
        and the question that was cut off was never fully heard anyway.
        """
        if not self.cfg.wake.followup or self._last_interrupted:
            return
        if not invites_reply(spoken):
            return
        if not self._followup_chain.may_open():
            log.info(
                "reply asked a question, but %d follow-ups in a row; "
                "wake word required", self._followup_chain.count,
            )
            return
        # Let the room settle. The tail of our own voice can still be in the
        # air when aplay returns, and a capture that opens on it would close
        # after silence_s on a fragment of Jarvis, losing the real answer.
        time.sleep(FOLLOWUP_SETTLE_S)
        self._followup_armed.set()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        problems = self.preflight()
        if problems:
            for problem in problems:
                log.error("preflight: %s", problem)
            return False

        ok, problem = self.mic.start()
        if not ok:
            log.error("microphone: %s", problem)
            return False

        self._detector, description = build_detector(self.cfg.wake)
        log.info("wake detector: %s", description)
        self._start_voice_id()

        # Warm the agent loop. The first request against a cold session was
        # measured at ~29s versus ~4s steady state; without this the user's
        # first real question pays that cost.
        threading.Thread(target=self._warm_up, name="warmup", daemon=True).start()

        for target, name in ((self._wake_loop, "wake"), (self._worker_loop, "worker")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

        log.info("voice assistant ready")
        return True

    def _start_voice_id(self) -> None:
        """Load speaker recognition. A failure turns it off; it never stops Jarvis."""
        if self.voice_id is None:
            return
        ok, problem = self.voice_id.load()
        if not ok:
            log.error("speaker recognition off: %s", problem)
            self.voice_id = None
            return
        if self.voice_id.registry.problem:
            log.warning("%s", self.voice_id.registry.problem)
        names = self.voice_id.registry.names()
        log.info("speaker recognition: %s (%s enrolled)", self.voice_id.embedder.model_name,
                 ", ".join(names) or "nobody")

    def _warm_up(self) -> None:
        """Prime the agent loop in the background so the first real question
        does not pay cold-start cost. Failure here is not fatal."""
        started = time.monotonic()
        reply = self.agent.ask("Reply with the single word: ready")
        elapsed = time.monotonic() - started
        if reply.ok:
            log.info("agent warm-up completed in %.1fs", elapsed)
        else:
            log.warning("agent warm-up failed after %.1fs: %s", elapsed, reply.error)

    def stop(self) -> None:
        log.info("shutting down")
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3)
        self.mic.stop()

    def run_forever(self) -> int:
        if not self.start():
            return 1

        def handle_signal(signum, frame):  # noqa: ANN001, ARG001
            self.stop()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        try:
            while not self._stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Home AI voice assistant")
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the startup checks (speech, voice, agent, config), print any "
        "problems and exit 1 if there are some; does not open the microphone",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.check:
        problems = VoiceAssistant().preflight()
        for problem in problems:
            print(f"problem: {problem}")
        print("check: OK" if not problems else f"check: {len(problems)} problem(s)")
        return 1 if problems else 0
    return VoiceAssistant().run_forever()


if __name__ == "__main__":
    sys.exit(main())
