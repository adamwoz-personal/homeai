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

import concurrent.futures
import logging
import queue
import signal
import sys
import threading
import time

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
from .bargein import InterruptListener, contains_wake_fragments
from .wake import CaptureMachine, State, build_detector

log = logging.getLogger("homeai")

# Spoken responses for failure paths. Short by design: a long apology is worse
# than a brief one.
MSG_AGENT_DOWN = "Sorry, my brain is offline right now."
MSG_AGENT_ERROR = "Sorry, something went wrong."
MSG_REFUSED = "I can't do that by voice."
MSG_NOT_UNDERSTOOD = "Sorry, I didn't catch that."

# Spoken when the agent is taking long enough that silence reads as failure.
# The threshold sits above a normal conversational turn (~0.5-1.5s) so simple
# questions are never padded, but below the point where a user assumes the
# assistant did not hear them.
MSG_WORKING = "Let me look that up."
PROGRESS_AFTER_S = 2.5


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
        self.transcript = TranscriptLog(
            self.cfg.transcript.path, enabled=self.cfg.transcript.enabled
        )
        self.machine = CaptureMachine(self.cfg.wake)

        self._detector = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
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
            problems.append(f"ZeroClaw gateway unreachable at {self.cfg.agent.health_url}")

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

        while not self._stop.is_set():
            time.sleep(poll_interval)

            # Never listen to ourselves.
            if self.mic.paused or self.tts.speaking:
                cursor = self.mic.buffer.total_written
                continue

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
                        if not self.machine.accepts_wake(now):
                            continue
                        if self._detector and self._detector.detect(frame):
                            log.info("wake word detected")
                            self.machine.on_wake(now)
                            utterance = []
                        continue

                    utterance.append(frame)
                    if self.machine.feed(frame, now):
                        audio = np.concatenate(utterance) if utterance else np.zeros(0, np.float32)
                        utterance = []
                        self.machine.reset(now)
                        if self._detector:
                            self._detector.reset()
                        # Drop anything captured during handoff so the next
                        # turn cannot re-detect this utterance's wake word.
                        cursor = self.mic.buffer.total_written
                        self._enqueue(audio)
                        break
            except Exception:  # noqa: BLE001 - this thread must never die
                log.exception("wake loop error; resetting")
                self.machine.reset()
                utterance = []
                cursor = self.mic.buffer.total_written

    def _enqueue(self, audio: np.ndarray) -> None:
        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            log.warning("worker busy; dropping utterance")

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle(audio)
            except Exception:  # noqa: BLE001 - never let one turn kill the service
                log.exception("unhandled error processing utterance")
                self._speak(MSG_AGENT_ERROR)

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
            # The one case where we must NOT reset: the listener was triggered,
            # which means the person said the wake word deliberately. The main
            # wake loop heard the same words and has already begun capturing
            # their follow-up, so resetting here would discard the very
            # utterance they interrupted us to say.
            if not interrupted:
                if self._detector:
                    self._detector.reset()
                self.machine.reset(time.monotonic())

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

    def _handle(self, audio: np.ndarray) -> None:
        turn = Turn()
        total = Stopwatch()
        stage = Stopwatch()

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
        log.info("heard: %s  [stt %.0fms]", transcript.text, turn.stt_ms)

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
        turn.reply = spoken
        stage.reset()
        turn.tts_first_audio_ms = self._speak(spoken, chunked=True)
        turn.tts_ms = stage.ms()
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


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return VoiceAssistant().run_forever()


if __name__ == "__main__":
    sys.exit(main())
