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
import logging
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import replace

import numpy as np

from .agent_cli import build_agent_client
from .memory import ConversationMemory
from .audio import Microphone, MicWatchdog
from .avatar.presence import Presence
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
    asks_to_repeat,
    drop_repeated_sentences,
    invites_reply,
    is_continue_request,
    is_decline,
    is_trailing_fragment,
    one_sentence,
    split_for_budget,
    unwrap_quotes,
)
from .wake import CaptureMachine, State, build_detector
from . import home_intents
from .ha_client import HAError
from .doorbell import DoorbellWatcher, parse_quiet_hours
from .home import EchoQuiet, HomeError, build_home, run_intent
from . import voice_id as vid
from .voice_id_turns import VoiceIdTurns
from .wake_verify import (
    _name_words,
    closing_reply,
    is_goodnight,
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

# No spoken cue while the agent works: the avatar's thinking colours in hair and eyes
# say it (Adam, 2026-11-07; was "Let me look that up.", then "One moment.").

# How long an unanswered "Which lights?" stays open.
HOME_PENDING_S = 30.0

# Text typed in the art features window (ART_FEATURES_PLAN F6, Adam 2026-11-12): art mode only, shown not spoken.
TYPED_MAX = 500
TYPED_MIN_GAP_S = 0.5
MSG_TYPED_ART_OFF = "Typing works in art mode. Say “hey Jarvis, art mode” first."
MSG_TYPED_NOT_ART = "Typed text only changes the picture."

# Pause between the end of a reply and opening a follow-up window, so the
# capture does not start on the acoustic tail of Jarvis's own voice.
FOLLOWUP_SETTLE_S = 0.25
FORCED_EXIT_S = 15.0    # a wedged device can hang shutdown; systemd must still get its restart



def _is_leave(text: str) -> bool:
    """The sentence asks to leave art mode (homeai/art/grammar.py "leave")."""
    from homeai.art import grammar
    try:
        intent = grammar.parse(text)
    except Exception:  # noqa: BLE001 - not sure: split as usual
        return False
    return intent is not None and intent.kind == "leave"

class VoiceAssistant(VoiceIdTurns):
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load()
        self.mic = Microphone(self.cfg.audio)
        self.mic_watchdog = MicWatchdog(self.mic, self._mic_gave_up)
        self._exit_code = 0
        self.stt = Transcriber(self.cfg.stt)
        self.tts = Speaker(self.cfg.tts, self.cfg.audio.output_device)
        # Avatar event feed; a no-op (no socket) when HOMEAI_AVATAR=0. It also brings text typed in the
        # art features window (F6) to the "typed" worker.
        self._typed: queue.Queue[str] = queue.Queue(maxsize=8)
        self._typed_last = float("-inf")
        self.presence = Presence(self.cfg.avatar.enabled, self.cfg.avatar.socket, on_message=self._on_bus_message)
        if self.presence.enabled:
            self.tts.publish = self.presence.publish
        # Long-term memory (plans/MEMORY_PLAN.md). Optional: if Postgres or the
        # extra packages are missing, Jarvis works exactly as before.
        self.ltm = self._build_ltm()
        # Continuity between turns. Bounded and self-expiring; see memory.py
        # for why this does not reintroduce session poisoning.
        self.memory = ConversationMemory(long_term=self.ltm.recall if self.ltm else None)
        self.agent = build_agent_client(self.cfg.agent, memory=self.memory)
        # Who is speaking; None when turned off or the model failed to load.
        self.voice_id = vid.VoiceId(self.cfg.speaker) if self.cfg.speaker.enabled else None
        # Art mode by voice; None when HOMEAI_ART_VOICE=0 (homeai/art/voice.py).
        self.art = self._build_art()
        # "Status update"; built in start() once the house controller is known.
        self.status = None
        self._enrolment: vid.Enrolment | None = None
        # Unknown voices heard in passing, in memory only, so Jarvis can offer to learn one.
        self._strangers = vid.StrangerPool()
        self._offer_voice = None        # the pooled voice an offered enrolment belongs to
        self._pending_offer = None      # a pooled voice ready to be asked its name after this reply
        # Home control; built in start() when HA credentials exist.
        self.home = None
        # Set while an Echo announcement plays, so it cannot wake us.
        self._echo_quiet = EchoQuiet()
        # A light command waiting for "which lights?" to be answered.
        self._pending_home: tuple[home_intents.HomeIntent, float] | None = None
        self.transcript = TranscriptLog(
            self.cfg.transcript.path, enabled=self.cfg.transcript.enabled
        )
        if self.ltm is not None:
            self.transcript.add_listener(self.ltm.on_turn)
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
                self.machine.on_wake(time.monotonic(), long=self._long_capture())
                source = "bargein"
                self.presence.state("listening")
                log.info("barge-in: capturing follow-up")
            elif self._followup_armed.is_set():
                # The reply ended with a question. Listen for the answer
                # without the wake word, for a longer lead-in than usual.
                self._followup_armed.clear()
                utterance = []
                cursor = self.mic.buffer.total_written
                self.machine.on_wake(
                    time.monotonic(), lead_in_s=self.cfg.wake.followup_lead_in_s,
                    long=self._long_capture(),
                )
                source = "followup"
                self.presence.state("followup")
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
                            quiet = self._echo_quiet.remaining()
                            if quiet > 0:
                                # An Echo is reading out our announcement.
                                log.info("wake word ignored (score %.3f): Echo announcement "
                                         "playing for %.0fs more", wake_score or 0.0, quiet)
                                self._detector.reset()
                                continue
                            if wake_score is None:
                                log.info("wake word detected")
                            else:
                                log.info("wake word detected (score %.3f)", wake_score)
                            self.machine.on_wake(now, long=self._long_capture())
                            self.presence.state("listening")
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
                            self.presence.state("idle")
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
            return
        self.presence.state("thinking")

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
            finally:
                self._end_turn_state()

    def _end_turn_state(self) -> None:
        """Say what happens next: listen again (follow-up or barge-in) or idle."""
        if self._bargein_armed.is_set():
            self.presence.state("listening")
        elif self._followup_armed.is_set():
            self.presence.state("followup")
        else:
            self.presence.state("idle")

    # -- one turn ----------------------------------------------------------

    def _speak(self, text: str, chunked: bool = False) -> float:
        """Speak with the mic muted, so we never transcribe our own voice.

        Returns milliseconds to first audible sound, which is the latency a
        listener perceives. Total elapsed time is dominated by the length of
        the sentence being spoken and is not a delay.
        """
        listener = self._start_interrupt_listener(text) if chunked else None
        self.presence.state("speaking")
        self.presence.caption("jarvis", text)

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
            # Still mid-turn until the worker says otherwise (_end_turn_state).
            self.presence.state("thinking")

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
            self._end_turn(turn, total)
            return

        turn.heard = transcript.text
        if verify:
            if not mentions_wake_word(transcript.text, self.cfg.wake.model):
                log.info("unverified wake (score %.3f): Whisper heard %r, no wake word - "
                         "ignoring", turn.wake_score or 0.0, transcript.text)
                turn.verdict = "unverified"
                self._end_turn(turn, total)
                return
            transcript = replace(transcript,
                                 text=strip_wake_phrase(transcript.text, self.cfg.wake.model))
            if not transcript.text:
                log.info("wake word only (%r); nothing to answer", turn.heard)
                turn.verdict = "wake-only"
                self._end_turn(turn, total)
                return
        log.info("heard: %s  [stt %.0fms]", transcript.text, turn.stt_ms)
        self.presence.caption("user", transcript.text)

        ident, emb = self._identify(audio, turn)
        if self._handle_voice_id(turn, transcript.text, source, audio, ident, emb, total):
            return
        if ident is not None:
            self.memory.switch_speaker(ident.name)
        self._pending_offer = self._pool_stranger(source, audio, ident, emb)

        if self._handle_memory(turn, transcript.text, total):
            return

        # "Stop", "quiet", "I'm not talking to you". After a barge-in the mic
        # stays open for a follow-up question, but the interruption is often
        # the whole point. Querying the agent here would answer a question
        # nobody asked and produce more speech to interrupt.
        if is_dismissal(transcript.text, self.cfg.wake.model):
            log.info("dismissal: %r - staying quiet", transcript.text)
            self._held.clear()
            turn.verdict = "dismissed"
            turn.reply = ""
            self._end_turn(turn, total)
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
                self._end_turn(turn, total)
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
            fallback = "" if is_hallucination_only(transcript.text) or not (
                in_conversation or is_goodnight(transcript.text)) else closing_reply(transcript.text)
            self._held.clear()
            if not fallback:
                log.info("pleasantry with no conversation in progress: %r - staying quiet",
                         transcript.text)
                turn.verdict = "pleasantry"
                self._end_turn(turn, total)
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
            self._end_turn(turn, total)
            return

        if self._handle_status(turn, transcript.text, ident, total):
            return

        if self._handle_art(turn, transcript.text, ident, total, emb, audio):
            return

        if self._handle_home(turn, transcript.text, source, total):
            return

        if is_trailing_fragment(transcript.text):
            log.info("fragment: %r - probably not addressed to me", transcript.text)
            turn.verdict = "fragment"
            stage.reset()
            self._speak(MSG_FRAGMENT)
            turn.tts_ms = stage.ms()
            turn.reply = MSG_FRAGMENT
            self._end_turn(turn, total)
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
            self._end_turn(turn, total)
            return

        turn.verdict = Verdict.ALLOW.value
        stage.reset()
        # Snapshot before asking: the agent client adds this reply to memory.
        earlier = [] if asks_to_repeat(transcript.text) else \
            [e.assistant for e in self.memory.recent()]
        reply = self.agent.ask(transcript.text)
        turn.agent_ms = stage.ms()
        turn.attempts = reply.attempts
        if not reply.ok:
            log.error("agent failed after %.0fms: %s", turn.agent_ms, reply.error)
            turn.ok = False
            turn.error = reply.error
            spoken = MSG_AGENT_DOWN if "unreachable" in reply.error else MSG_AGENT_ERROR
            turn.reply = spoken
            self._speak(spoken)
            self._end_turn(turn, total)
            return

        # Order matters: flatten_markdown needs line breaks, which
        # sanitise_for_speech collapses; normalise_for_speech needs the
        # security/runaway guard to have already run.
        spoken = normalise_for_speech(
            sanitise_for_speech(flatten_markdown(unwrap_quotes(reply.text)))
        )
        spoken, dropped = drop_repeated_sentences(spoken, earlier)
        if dropped:
            log.info("dropped %d sentence(s) repeating an earlier reply", dropped)
        self._deliver(turn, spoken, total)

    # -- home control ----------------------------------------------------------

    def _home_intent(self, text: str, source: str) -> home_intents.HomeIntent | None:
        intent = home_intents.parse(text) if self.cfg.home.fast_path else None
        pending, self._pending_home = self._pending_home, None
        if intent is not None or pending is None or source != "followup":
            return intent
        waiting, asked_at = pending
        if time.monotonic() - asked_at > HOME_PENDING_S:
            return None
        # The answer to "Which lights?": "the foyer", "den lamp please".
        target = home_intents._light_target(home_intents._clean(text))
        if not target or len(target.split()) > 4:
            return None
        return home_intents.HomeIntent(waiting.kind, {**waiting.args, "target": target})

    def _handle_memory(self, turn: Turn, text: str, total: Stopwatch) -> bool:
        """"Remember that...", "forget...", "what do you remember...". True if handled."""
        if self.ltm is None:
            return False
        stage = Stopwatch()
        result = self.ltm.handle_command(turn.speaker, text)
        if result is None:
            return False
        reply, listen = result
        turn.agent_ms = stage.ms()
        turn.verdict = "memory-command"
        log.info("memory command %r -> %s  [%.0fms]", text, reply, turn.agent_ms)
        self._held.clear()
        self._say_turn(turn, reply, total, listen)
        return True

    def _handle_home(self, turn: Turn, text: str, source: str, total: Stopwatch) -> bool:
        """Everyday home commands, without the model. True if handled."""
        if self.home is None:
            return False
        intent = self._home_intent(text, source)
        if intent is None:
            return False
        stage = Stopwatch()
        listen = False
        try:
            reply = run_intent(self.home, intent.kind, intent.args)
        except HomeError as exc:
            reply = str(exc)
            if intent.kind == "lights" and reply == "Which lights?":
                self._pending_home = (intent, time.monotonic())
                listen = True
        except HAError as exc:
            log.error("home %s failed: %s", intent.kind, exc)
            reply = f"Sorry, {exc}."
        except Exception:  # noqa: BLE001 - a broken house must not break Jarvis
            log.exception("home %s crashed", intent.kind)
            reply = MSG_AGENT_ERROR
        turn.agent_ms = stage.ms()
        turn.verdict = f"home-{intent.kind}"
        log.info("home %s %s -> %s  [%.0fms]", intent.kind, intent.args, reply, turn.agent_ms)
        self.memory.add(text, reply)
        self._say_turn(turn, reply, total, listen)
        return True

    def _start_home(self) -> None:
        if not self.cfg.home.enabled:
            log.info("home control: off (HOMEAI_HOME=0)")
            return
        self.home = build_home(self.cfg.home.env_path, default_echo=self.cfg.home.default_echo,
                               wake_words=_name_words(self.cfg.wake.model))
        if self.home is not None:
            log.info("home control: Home Assistant at %s (fast path %s)",
                     self.home.client.settings.url, "on" if self.cfg.home.fast_path else "off")
            self._start_doorbell()

    def _start_status(self) -> None:
        if not self.cfg.status.enabled:
            return
        from . import status_update
        from .mcp_server import DEFAULT_LOCATION
        profiles = (lambda: self.voice_id.registry.profiles()) if self.voice_id is not None else (lambda: [])
        dsn = self.cfg.memory.dsn if self.cfg.memory.enabled else None
        self.status = status_update.build(self.home, self.cfg.status.admins, profiles, DEFAULT_LOCATION, dsn)

    def _start_doorbell(self) -> None:
        cfg = self.cfg.home
        if not cfg.doorbell:
            log.info("doorbell watch: off (HOMEAI_DOORBELL=0)")
            return
        watcher = DoorbellWatcher(self.home, announce_motion=cfg.doorbell_motion,
                                  where=cfg.doorbell_where, phone=cfg.doorbell_phone,
                                  quiet_hours=parse_quiet_hours(cfg.doorbell_quiet_hours))
        thread = threading.Thread(target=watcher.run, args=(self._stop,), name="doorbell",
                                  daemon=True)
        thread.start()
        self._threads.append(thread)

    # -- speaker recognition -------------------------------------------------

    def _long_capture(self) -> bool:
        """In art mode, people describe scenes: allow pauses and long descriptions."""
        return self.art is not None and self.art.active

    def _build_art(self):
        if not self.cfg.art.voice:
            return None
        from homeai import gpu_mode
        from homeai.art.voice import ArtVoice, build_studio
        switcher = gpu_mode.ModeSwitcher(gpu_mode.System(), log=lambda line: log.info("mode: %s", line),
                                         lock_path=gpu_mode.mode_lock_path(),
                                         lock_wait_s=gpu_mode.VOICE_LOCK_WAIT_S)
        checker = None
        if self.cfg.art.check:
            from homeai.art import checker as art_checker
            checker = art_checker.ollama_ask
        return ArtVoice(switcher, build_studio, self.presence.publish, adults=self.cfg.art.adults,
                        on_state=self.presence.state, checker=checker)

    def _handle_status(self, turn: Turn, text: str, ident, total: Stopwatch) -> bool:
        """'Status update': built from the voice registry, doorbell history and weather, no agent."""
        if self.status is None:
            return False
        try:
            reply = self.status.handle(text, ident.name if ident is not None else None)
        except Exception:  # noqa: BLE001 - never lose a turn to the status update
            log.exception("status update failed")
            reply = "Sorry, I couldn't put the status update together."
        if reply is None:
            return False
        log.info("status update: %s", reply)
        turn.verdict = "status"
        self._say_turn(turn, reply, total)
        return True

    def _handle_art(self, turn: Turn, text: str, ident, total: Stopwatch, emb=None, audio=None) -> bool:
        """Drawing sentences, and entering or leaving art mode. False: not about art."""
        if self.art is None:
            return False
        speaker = ident.name if ident is not None else None
        other = False
        owner = getattr(self.art, "owner", None)
        if getattr(self.art, "active", False) and owner and self.voice_id is not None and ident is not None:
            other, _ = self.voice_id.is_someone_else(owner, ident, emb)
            # "Cancel art mode" lost "mode." to the split on 2026-11-12 and art mode stayed on: a way out is
            # never split (leaving is harmless; whoever says it, they can say "let's draw" again).
            if not other and audio is not None and self.cfg.art.split and not _is_leave(text):
                text, other = self._owner_words(owner, audio, text)
            self._keep_art_clip(owner, audio, text, ident, other)
        try:
            reply = self.art.handle(text, speaker, other=other)
        except Exception:  # noqa: BLE001 - a broken art mode must not take Jarvis down
            log.exception("art mode failed on %r", text)
            reply = "Something went wrong in art mode."
        if reply is None:
            return False
        if reply == "":
            turn.verdict = "other-speaker"
            self._end_turn(turn, total)
            return True
        log.info("art: %r -> %r", text, reply)
        turn.verdict = "art"
        self._say_turn(turn, reply, total, listen=reply.endswith("?"))
        return True

    # -- typed text (art features window, ART_FEATURES_PLAN F6) ----------------

    def _on_bus_message(self, msg: dict) -> None:
        """On the event bus's I/O thread: check and queue, never block."""
        if msg.get("type") != "typed":
            return
        text = (msg.get("data") or {}).get("text")
        if not isinstance(text, str):
            return
        text = " ".join(text.split())
        now = time.monotonic()
        if not text or len(text) > TYPED_MAX or now - self._typed_last < TYPED_MIN_GAP_S:
            return
        self._typed_last = now
        try:
            self._typed.put_nowait(text)
        except queue.Full:
            log.warning("typed text dropped: %d already waiting", self._typed.qsize())

    def _typed_loop(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._typed.get(timeout=0.5)
            except queue.Empty:
                continue
            self._handle_typed(text)

    def _handle_typed(self, text: str) -> None:
        """Typed text changes the picture exactly like a heard sentence from the session's owner (their rules),
        and only that: never the chat model or the house. The reply is shown, not spoken (Adam 2026-11-12)."""
        art = self.art
        if art is None or not art.active:
            reply = MSG_TYPED_ART_OFF
        else:
            log.info("typed: %r", text)
            try:
                reply = art.handle(text, art.owner, other=False)
            except Exception:  # noqa: BLE001 - a broken art mode must not take Jarvis down
                log.exception("art mode failed on typed %r", text)
                reply = "Something went wrong in art mode."
            if not reply:
                reply = MSG_TYPED_NOT_ART
            log.info("typed art: %r -> %r", text, reply)
        self.presence.publish("typed_reply", {"text": reply})

    def _keep_art_clip(self, owner: str, audio, text: str, ident, other: bool) -> None:
        """An adult owner's art-mode recording, kept for a week to measure the split (homeai/art_clips.py)."""
        from . import art_clips
        if audio is None or not art_clips.recording_on(self.cfg.art.keep_clips_until):
            return
        if owner.lower() not in {a.lower() for a in self.cfg.art.adults}:
            return                                    # never a child's session
        try:
            meta = {"owner": owner, "heard": text, "speaker": getattr(ident, "name", None),
                    "score": getattr(ident, "score", None), "best": getattr(ident, "best", None),
                    "margin": getattr(ident, "margin", None), "someone_else": bool(other),
                    "split_on": bool(self.cfg.art.split)}
            art_clips.save(art_clips.DIR / owner.lower(), audio, self.cfg.audio.sample_rate, meta)
            art_clips.prune(art_clips.DIR / owner.lower())
        except Exception:  # noqa: BLE001 - a clip is a nicety; the turn goes on
            log.warning("keeping the art clip failed", exc_info=True)

    def _prune_art_clips(self) -> None:
        """Delete week-old art clips at start and every few hours, even if art mode is never used again."""
        from . import art_clips
        while True:
            try:
                for folder in (art_clips.DIR.iterdir() if art_clips.DIR.is_dir() else ()):
                    art_clips.prune(folder)
            except OSError:
                log.warning("pruning art clips failed", exc_info=True)
            if self._stop.wait(art_clips.PRUNE_EVERY_S):
                return

    def _owner_words(self, owner: str, audio: np.ndarray, text: str) -> tuple[str, bool]:
        """Art mode: drop the words another voice said into the owner's recording (ART_FEATURES_PLAN S1).

        Returns (text to use, whether nothing of the owner's is left). Any failure keeps the text as heard.
        """
        from . import speaker_split as split
        try:
            vid_ = self.voice_id
            profile = vid_.registry.get(owner)
            if profile is None:
                return text, False

            def similarity(chunk):
                emb = vid_.embedder.embed(chunk)
                return None if emb is None else vid_.similarity_to(owner, emb)

            sims = split.window_sims(similarity, audio)
            spans = split.other_spans(sims, vid_.cfg.different_below)
            if not split.is_mixed(sims, spans, vid_.cfg.threshold):
                return text, False
            words = self.stt.words(audio, self.cfg.audio.sample_rate)
            if not words:
                return text, False
            kept, dropped = split.keep_owner_words(words, spans)
            log.info("art: another voice in %s's recording: kept %r, dropped %r", owner, kept, dropped)
            return (kept, False) if kept else (text, True)
        except Exception:  # noqa: BLE001 - never lose the owner's sentence to a split failure
            log.exception("splitting the recording failed")
            return text, False

    def _end_turn(self, turn: Turn, total: Stopwatch) -> None:
        """Every turn ends here: its total time, then its transcript line."""
        turn.total_ms = total.ms()
        self.transcript.write(turn)

    def _say_turn(self, turn: Turn, text: str, total: Stopwatch, listen: bool = False) -> None:
        """Speak a fixed reply, log the turn, and optionally listen without
        the wake word for the answer."""
        turn.reply = text
        stage = Stopwatch()
        turn.tts_first_audio_ms = self._speak(text)
        turn.tts_ms = stage.ms()
        self._end_turn(turn, total)
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
        spoken = self._with_offer(spoken, held=bool(rest))
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

    def _start_services(self) -> None:
        self._start_voice_id()
        self._start_home()
        self._start_status()
        if self.art is not None:
            self.art.adopt()        # art mode left on from before a restart still idles back to voice
        threading.Thread(target=self._prune_art_clips, name="art-clips", daemon=True).start()

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
        self.mic_watchdog.start()

        self._detector, description = build_detector(self.cfg.wake)
        log.info("wake detector: %s", description)
        self._start_services()
        self.presence.start()
        # Load the in-process voice now, not on the first reply (~0.5 s).
        threading.Thread(target=self.tts.warm, name="tts-warm", daemon=True).start()

        # Warm the agent loop. The first request against a cold session was
        # measured at ~29s versus ~4s steady state; without this the user's
        # first real question pays that cost.
        threading.Thread(target=self._warm_up, name="warmup", daemon=True).start()

        for target, name in ((self._wake_loop, "wake"), (self._worker_loop, "worker"), (self._typed_loop, "typed")):
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

    def _build_ltm(self):
        if not self.cfg.memory.enabled:
            return None
        try:
            from .memory_store.service import MemoryService

            return MemoryService(self.cfg.memory).start()
        except Exception as exc:  # noqa: BLE001 - optional feature
            log.warning("long-term memory disabled: %s", exc)
            return None

    def _mic_gave_up(self) -> None:
        """Exit non-zero so systemd restarts the service; a new process can reach a device this one can't."""
        self._exit_code = 1
        self._stop.set()
        forced = threading.Timer(FORCED_EXIT_S, os._exit, (1,))
        forced.daemon = True
        forced.start()

    def stop(self) -> None:
        log.info("shutting down")
        self._stop.set()
        self.mic_watchdog.stop()        # first: a closing stream is not a stalled one
        for thread in self._threads:
            thread.join(timeout=3)
        self.mic.stop()
        if self.art is not None:
            self.art.close()
        self.presence.stop()
        if self.ltm is not None:
            self.ltm.close()

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
        return self._exit_code


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
