# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for daemon orchestration that needs no audio hardware.

The daemon is the least-covered module despite being the orchestrator, and two
field bugs in a row lived here rather than in the components it wires together.
These tests construct a VoiceAssistant with its heavyweight collaborators
replaced, so no microphone, model, or subprocess is touched.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from homeai.config import Config
from homeai.wake import State


@pytest.fixture
def assistant():
    """A VoiceAssistant whose expensive collaborators are inert fakes."""
    with patch("homeai.daemon.Microphone"), \
         patch("homeai.daemon.Transcriber"), \
         patch("homeai.daemon.Speaker"), \
         patch("homeai.daemon.build_agent_client"), \
         patch("homeai.daemon.TranscriptLog"):
        from homeai.daemon import VoiceAssistant

        va = VoiceAssistant(Config())
    return va


class FakeBuffer:
    def __init__(self) -> None:
        self.total_written = 0
        self._pending = np.zeros(0, dtype=np.float32)

    def read_new(self, cursor, max_samples=None):
        data, self._pending = self._pending, np.zeros(0, dtype=np.float32)
        return data, cursor + len(data)

    def provide(self, frames: np.ndarray) -> None:
        self._pending = frames
        self.total_written += len(frames)


def _drive_wake_loop(assistant, iterations: float = 0.25) -> None:
    """Run the real wake loop briefly in a thread, then stop it."""
    t = threading.Thread(target=assistant._wake_loop, daemon=True)
    t.start()
    time.sleep(iterations)
    assistant._stop.set()
    t.join(timeout=2)
    assistant._stop.clear()


# ---------------------------------------------------------------------------
# Barge-in handoff
#
# Field failure: the interrupt worked (10ms cut) but the follow-up question was
# never captured, and the log showed nothing at all afterwards. The wake loop
# skips every frame while tts.speaking is true, so it never observed the wake
# word that caused the interrupt -- only InterruptListener did, and that
# listener captures no audio.
# ---------------------------------------------------------------------------


def test_interrupt_arms_the_wake_loop(assistant) -> None:
    assistant.mic.buffer = FakeBuffer()
    assistant.mic.paused = False
    assistant.tts.speaking = False
    assistant._bargein_armed.set()

    assert assistant.machine.state is State.IDLE
    _drive_wake_loop(assistant)

    assert assistant.machine.state is State.LISTENING, (
        "after an interrupt the daemon must capture the follow-up without "
        "requiring a second wake word"
    )
    assert not assistant._bargein_armed.is_set(), "flag must be consumed once"


def test_no_arming_means_no_capture(assistant) -> None:
    """Without the flag the loop stays idle, so this cannot fire on its own."""
    assistant.mic.buffer = FakeBuffer()
    assistant.mic.paused = False
    assistant.tts.speaking = False

    _drive_wake_loop(assistant)
    assert assistant.machine.state is State.IDLE


def test_arming_bypasses_the_refractory_window(assistant) -> None:
    """_speak resets the machine, opening a refractory window.

    The handoff must not be blocked by it: the person has already spoken the
    wake word deliberately, so refusing to listen would drop their question.
    """
    assistant.mic.buffer = FakeBuffer()
    assistant.mic.paused = False
    assistant.tts.speaking = False

    assistant.machine.reset(time.monotonic())
    assert assistant.machine.accepts_wake(time.monotonic()) is False

    assistant._bargein_armed.set()
    _drive_wake_loop(assistant)
    assert assistant.machine.state is State.LISTENING


def test_loop_ignores_arming_while_still_speaking(assistant) -> None:
    """Arming must not take effect until our own audio has actually stopped."""
    assistant.mic.buffer = FakeBuffer()
    assistant.mic.paused = False
    assistant.tts.speaking = True
    assistant._bargein_armed.set()

    _drive_wake_loop(assistant)

    assert assistant.machine.state is State.IDLE
    assert assistant._bargein_armed.is_set(), "flag must survive until audible"


def _heard(assistant, text: str) -> None:
    """Make the fake transcriber return ``text`` for the next utterance."""
    result = assistant.stt.transcribe_audio.return_value
    result.ok = True
    result.text = text
    result.error = ""


@pytest.mark.parametrize("text", ["Stop.", "Jarvis, be quiet.", "I'm not talking to you."])
def test_dismissal_never_reaches_the_agent(assistant, text) -> None:
    # "Stop" after a barge-in previously went to the agent as a question,
    # costing a lookup and producing more speech to interrupt.
    _heard(assistant, text)
    assistant._handle(np.zeros(1600, dtype=np.float32))

    assistant.agent.ask.assert_not_called()
    assistant.tts.say_chunked.assert_not_called()
    assistant.tts.say_safe.assert_not_called()
    turn = assistant.transcript.write.call_args.args[0]
    assert turn.verdict == "dismissed"


def test_real_question_still_reaches_the_agent(assistant) -> None:
    # Guards against the dismissal check being too eager.
    _heard(assistant, "stop and tell me about the moon")
    reply = assistant.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "The moon is far.", 1, ""

    assistant._handle(np.zeros(1600, dtype=np.float32))

    assistant.agent.ask.assert_called_once()
    assert assistant.agent.ask.call_args.args[0] == "stop and tell me about the moon"


# ---------------------------------------------------------------------------
# Follow-up listening
#
# When a reply ends with a question, the answer should be captured without
# "Hey Jarvis". Bounded so background audio cannot hold a conversation with
# Jarvis indefinitely.
# ---------------------------------------------------------------------------


from dataclasses import replace  # noqa: E402


def _with_wake(assistant, **changes) -> None:
    """Swap in a modified WakeConfig everywhere the daemon reads it."""
    wake = replace(assistant.cfg.wake, **changes)
    assistant.cfg = replace(assistant.cfg, wake=wake)
    assistant.machine.cfg = wake
    from homeai.dialogue import FollowupChain
    assistant._followup_chain = FollowupChain(wake.followup_max_chain)


def _answer(assistant, heard: str, reply_text: str, source: str = "wake") -> None:
    """Run one full turn through _handle with the given transcript and reply."""
    _heard(assistant, heard)
    reply = assistant.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, reply_text, 1, ""
    spoken = assistant.tts.say_chunked.return_value
    spoken.first_audio_ms, spoken.interrupted, spoken.chunks_spoken = 100.0, False, 1
    assistant._handle(np.zeros(1600, dtype=np.float32), source)


@pytest.fixture
def quick(assistant, monkeypatch):
    """No barge-in listener (it needs a real mic) and no settle delay."""
    monkeypatch.setattr("homeai.daemon.FOLLOWUP_SETTLE_S", 0.0)
    _with_wake(assistant, barge_in=False)
    return assistant


def test_question_at_end_of_reply_opens_a_followup_window(quick) -> None:
    _answer(quick, "is free will real", "I think it is. What do you think?")
    assert quick._followup_armed.is_set()


def test_statement_does_not_open_a_window(quick) -> None:
    _answer(quick, "capital of france", "The capital of France is Paris.")
    assert not quick._followup_armed.is_set()


def test_interrupted_reply_does_not_open_a_window(quick) -> None:
    # Barge-in already captures the follow-up; a second arm would race it.
    _heard(quick, "is free will real")
    reply = quick.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "What do you think?", 1, ""
    quick.tts.say_chunked.return_value.first_audio_ms = 100.0
    # _speak's finally block records the real interrupted state; emulate a
    # reply that was cut off by setting it after speaking.
    original = quick._speak

    def speak_then_mark(text, chunked=False):
        ms = original(text, chunked)
        quick._last_interrupted = True
        return ms

    quick._speak = speak_then_mark
    quick._handle(np.zeros(1600, dtype=np.float32))
    assert not quick._followup_armed.is_set()


def test_followup_can_be_disabled(quick) -> None:
    _with_wake(quick, barge_in=False, followup=False)
    _answer(quick, "is free will real", "What do you think?")
    assert not quick._followup_armed.is_set()


def test_chain_of_followups_is_capped(quick) -> None:
    _with_wake(quick, barge_in=False, followup_max_chain=2)
    _answer(quick, "q", "What do you think?", source="wake")
    assert quick._followup_armed.is_set()
    quick._followup_armed.clear()

    _answer(quick, "a1", "Why?", source="followup")
    assert quick._followup_armed.is_set()
    quick._followup_armed.clear()

    _answer(quick, "a2", "And then?", source="followup")
    assert not quick._followup_armed.is_set(), (
        "after max_chain wake-word-free turns the wake word must be required"
    )

    # Saying the wake word starts a fresh chain.
    _answer(quick, "q2", "What do you think?", source="wake")
    assert quick._followup_armed.is_set()


def test_turn_records_how_it_started(quick) -> None:
    _answer(quick, "an answer", "Interesting.", source="followup")
    turn = quick.transcript.write.call_args.args[0]
    assert turn.source == "followup"


class StreamingBuffer:
    """A mic buffer that always has fresh audio, as a real microphone does."""

    def __init__(self, block: int, level: float = 0.0) -> None:
        self.total_written = 0
        self._block = block
        self.level = level

    def read_new(self, cursor, max_samples=None):
        frames = np.full(self._block * 2, self.level, dtype=np.float32)
        self.total_written += len(frames)
        return frames, cursor + len(frames)


def _streaming(assistant, level: float) -> StreamingBuffer:
    buf = StreamingBuffer(assistant.cfg.audio.block_size, level)
    assistant.mic.buffer = buf
    assistant.mic.paused = False
    assistant.tts.speaking = False
    return buf


def test_followup_arm_starts_capture_with_the_longer_lead_in(quick) -> None:
    _with_wake(quick, barge_in=False, followup_lead_in_s=30.0)
    _streaming(quick, level=0.0)
    quick._followup_armed.set()

    _drive_wake_loop(quick)

    assert quick.machine.state is State.LISTENING
    assert quick._queue.empty(), "30s lead-in must not have expired yet"


def test_unanswered_followup_is_not_sent_to_whisper(quick) -> None:
    # Silence for the whole window. Transcribing it risks Whisper inventing
    # words from room noise, which would then go to the agent unprompted.
    _with_wake(quick, barge_in=False, followup_lead_in_s=0.05)
    _streaming(quick, level=0.0)
    quick._followup_armed.set()

    _drive_wake_loop(quick, iterations=0.4)

    assert quick._queue.empty()
    assert quick.machine.state is State.IDLE


def test_answered_followup_is_queued_as_a_followup(quick) -> None:
    _with_wake(quick, barge_in=False, silence_s=0.05, followup_lead_in_s=5.0)
    buf = _streaming(quick, level=0.5)  # someone answering
    quick._followup_armed.set()

    t = threading.Thread(target=quick._wake_loop, daemon=True)
    t.start()
    time.sleep(0.15)
    buf.level = 0.0  # they stop talking
    time.sleep(0.3)
    quick._stop.set()
    t.join(timeout=2)
    quick._stop.clear()

    audio, source, preroll = quick._queue.get_nowait()
    assert preroll is None  # only wake-started turns are verified
    assert source == "followup"
    assert audio.size > 0


def test_barge_in_takes_precedence_over_a_pending_followup(quick) -> None:
    _streaming(quick, level=0.0)
    quick._followup_armed.set()
    quick._bargein_armed.set()

    _drive_wake_loop(quick)

    assert not quick._followup_armed.is_set(), "stale follow-up must be dropped"
    assert quick.machine.state is State.LISTENING


# -- spoken budget -------------------------------------------------------------

LONG_REPLY = " ".join(f"Point number {i} is worth making here." for i in range(40))


def _spoken(assistant) -> str:
    return assistant.tts.say_chunked.call_args.args[0]


def test_long_reply_is_cut_and_offers_to_continue(quick) -> None:
    _answer(quick, "explain free will", LONG_REPLY)
    spoken = _spoken(quick)
    assert len(spoken.split()) < 130
    assert spoken.endswith("Want me to keep going?")
    assert quick._followup_armed.is_set(), "the offer must open the reply window"


def test_yes_speaks_the_rest_without_asking_the_agent(quick) -> None:
    _answer(quick, "explain free will", LONG_REPLY)
    first = _spoken(quick)
    asked = quick.agent.ask.call_count

    _answer(quick, "yes please", "SHOULD NOT BE USED", source="followup")
    assert quick.agent.ask.call_count == asked
    second = _spoken(quick)
    assert "SHOULD NOT" not in second
    heard = (first.replace(" There's more to it. Want me to keep going?", "")
             + " " + second.replace(" There's more to it. Want me to keep going?", ""))
    assert heard.split()[: len(LONG_REPLY.split())] == LONG_REPLY.split()[: len(heard.split())]
    assert quick.transcript.write.call_args.args[0].verdict == "continued"


def test_whole_reply_is_eventually_spoken(quick) -> None:
    _answer(quick, "explain free will", LONG_REPLY)
    parts = [_spoken(quick)]
    for _ in range(10):
        if not quick._held.pending(time.monotonic()):
            break
        _answer(quick, "go on", "x", source="followup")
        parts.append(_spoken(quick))
    suffix = " There's more to it. Want me to keep going?"
    joined = " ".join(p.removesuffix(suffix) for p in parts)
    assert joined == LONG_REPLY


def test_no_declines_quietly(quick) -> None:
    _answer(quick, "explain free will", LONG_REPLY)
    calls = quick.tts.say_chunked.call_count
    _answer(quick, "no thanks", "x", source="followup")
    assert quick.tts.say_chunked.call_count == calls
    assert not quick._held.pending(time.monotonic())
    assert quick.transcript.write.call_args.args[0].verdict == "declined"


def test_new_question_drops_the_held_rest(quick) -> None:
    _answer(quick, "explain free will", LONG_REPLY)
    _answer(quick, "what's the weather", "Sunny.", source="followup")
    assert _spoken(quick) == "Sunny."
    assert not quick._held.pending(time.monotonic())


def test_yes_with_nothing_held_goes_to_the_agent(quick) -> None:
    _answer(quick, "yes", "Yes to what?")
    assert _spoken(quick) == "Yes to what?"


def test_interrupting_the_first_part_drops_the_rest(quick) -> None:
    _heard(quick, "explain free will")
    reply = quick.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, LONG_REPLY, 1, ""
    quick.tts.say_chunked.return_value.first_audio_ms = 100.0
    original = quick._speak

    def speak_then_mark(text, chunked=False):
        ms = original(text, chunked)
        quick._last_interrupted = True
        return ms

    quick._speak = speak_then_mark
    quick._handle(np.zeros(1600, dtype=np.float32))
    assert not quick._held.pending(time.monotonic())


def test_budget_zero_speaks_everything(quick) -> None:
    quick.cfg = replace(quick.cfg, tts=replace(quick.cfg.tts, spoken_budget_words=0))
    _answer(quick, "explain free will", LONG_REPLY)
    assert _spoken(quick) == LONG_REPLY


# -- _speak: mic handling ------------------------------------------------------


def test_plain_speech_mutes_and_restores_the_mic(quick) -> None:
    quick._speak("hello")
    quick.mic.pause.assert_called_once()
    quick.mic.resume.assert_called_once()


def test_mic_is_restored_even_if_tts_fails(quick) -> None:
    quick.tts.say_safe.side_effect = RuntimeError("piper died")
    with pytest.raises(RuntimeError):
        quick._speak("hello")
    quick.mic.resume.assert_called_once()


def test_with_listener_the_mic_stays_open(quick) -> None:
    listener = MagicMock(triggered=False)
    quick._start_interrupt_listener = lambda text: listener
    quick._speak("A long reply.", chunked=True)
    quick.mic.pause.assert_not_called()
    listener.stop.assert_called_once()
    assert quick.tts.say_chunked.call_args.kwargs["should_stop"] is listener.should_stop


def test_listener_is_stopped_even_if_tts_fails(quick) -> None:
    listener = MagicMock(triggered=False)
    quick._start_interrupt_listener = lambda text: listener
    quick.tts.say_chunked.side_effect = RuntimeError("piper died")
    with pytest.raises(RuntimeError):
        quick._speak("A long reply.", chunked=True)
    listener.stop.assert_called_once()


def test_triggered_listener_marks_the_reply_interrupted(quick) -> None:
    listener = MagicMock(triggered=True)
    quick._start_interrupt_listener = lambda text: listener
    quick._speak("A long reply.", chunked=True)
    assert quick._last_interrupted
    assert quick._bargein_armed.is_set()


def test_reply_containing_the_wake_word_skips_barge_in(assistant) -> None:
    # Measured: Jarvis saying "Jarvis" triggers its own detector at 0.995.
    _with_wake(assistant, barge_in=True)
    assert assistant._start_interrupt_listener("I'm Jarvis, nice to meet you.") is None


def test_detector_is_reset_after_speaking(quick) -> None:
    quick._detector = MagicMock()
    quick._speak("hello")
    quick._detector.reset.assert_called_once()


# -- _ask_with_progress ----------------------------------------------------------


def test_fast_answer_has_no_progress_announcement(quick, monkeypatch) -> None:
    monkeypatch.setattr("homeai.daemon.PROGRESS_AFTER_S", 0.5)
    quick.agent.ask.return_value = "answer"
    assert quick._ask_with_progress("q") == "answer"
    quick.tts.say_safe.assert_not_called()


def test_slow_answer_announces_the_wait_then_returns_it(quick, monkeypatch) -> None:
    from homeai.daemon import MSG_WORKING

    monkeypatch.setattr("homeai.daemon.PROGRESS_AFTER_S", 0.05)
    quick.agent.ask.side_effect = lambda q: (time.sleep(0.3), "answer")[1]
    assert quick._ask_with_progress("q") == "answer"
    quick.tts.say_safe.assert_called_once_with(MSG_WORKING)


def test_failed_announcement_does_not_lose_the_answer(quick, monkeypatch) -> None:
    monkeypatch.setattr("homeai.daemon.PROGRESS_AFTER_S", 0.05)
    quick.agent.ask.side_effect = lambda q: (time.sleep(0.3), "answer")[1]
    quick.tts.say_safe.side_effect = RuntimeError("speaker unplugged")
    assert quick._ask_with_progress("q") == "answer"


# -- worker loop survives failures ---------------------------------------------


def _run_worker(assistant, items, handle) -> list:
    """Run the worker loop over ``items`` and return what _handle saw."""
    seen = []

    def fake_handle(audio, source="wake", preroll=None):
        seen.append(source)
        handle(len(seen))
        if len(seen) == len(items):
            assistant._stop.set()

    assistant._handle = fake_handle
    for item in items:
        assistant._queue.put(item)
    thread = threading.Thread(target=assistant._worker_loop, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "worker loop hung"
    return seen


def test_one_failing_turn_does_not_stop_the_worker(quick) -> None:
    def handle(n):
        if n == 1:
            raise RuntimeError("whisper crashed")

    seen = _run_worker(quick, [(np.zeros(10), "wake", None), (np.zeros(10), "wake", None)], handle)
    assert len(seen) == 2


def test_worker_survives_when_the_error_message_also_fails(quick) -> None:
    # A broken speaker makes both the turn and the apology fail. The worker
    # must still be alive for the next utterance.
    quick.tts.say_safe.side_effect = RuntimeError("speaker unplugged")

    def handle(n):
        if n == 1:
            raise RuntimeError("whisper crashed")

    seen = _run_worker(quick, [(np.zeros(10), "wake", None), (np.zeros(10), "wake", None)], handle)
    assert len(seen) == 2


def test_trailing_fragment_is_not_sent_to_the_agent(quick) -> None:
    """Observed live: overheard 'I will run by the local AI, so there are...'
    produced a 28 s answer continuing the previous topic."""
    from homeai.daemon import MSG_FRAGMENT
    quick.agent.ask.reset_mock()
    _answer(quick, "I will run by the local AI, so there are...", "unused")
    quick.agent.ask.assert_not_called()
    spoken = [c.args[0] for c in quick.tts.say_safe.call_args_list]
    assert spoken == [MSG_FRAGMENT]
    assert not quick._followup_armed.is_set()


def test_trailing_question_still_reaches_the_agent(quick) -> None:
    quick.agent.ask.reset_mock()
    _answer(quick, "what's the weather in...?", "Sunny.")
    quick.agent.ask.assert_called_once()


def test_wake_score_is_recorded_in_the_transcript(quick) -> None:
    quick._last_wake_score = 0.734
    quick.transcript = MagicMock()
    _answer(quick, "capital of france", "Paris.")
    assert quick.transcript.write.call_args.args[0].wake_score == 0.734
    _answer(quick, "and germany", "Berlin.", source="followup")
    assert quick.transcript.write.call_args.args[0].wake_score is None


# -- second-stage wake verification --------------------------------------------
# Field failure 2026-10-03 20:26: openWakeWord fired at 0.924 on room audio,
# Whisper heard "Thank you. You're welcome.", and the agent narrated its own
# reasoning for 27 s. About a dozen false wakes that day, many above 0.85.


def _verified_turn(quick, heard: str, source: str = "wake", preroll=True):
    from homeai.stt import Transcript
    quick.stt.transcribe_audio.return_value = Transcript(ok=True, text=heard)
    quick.agent.ask.reset_mock()
    reply = quick.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "Six.", 1, ""
    quick.tts.say_chunked.return_value.interrupted = False
    quick.tts.say_chunked.return_value.first_audio_ms = 100.0
    quick.transcript = MagicMock()
    pre = np.ones(800, dtype=np.float32) if preroll else None
    quick._handle(np.zeros(1600, dtype=np.float32), source, pre)
    return quick.transcript.write.call_args.args[0]


def test_wake_without_the_wake_word_in_the_audio_is_ignored(quick) -> None:
    turn = _verified_turn(quick, "Thank you. You're welcome.")
    quick.agent.ask.assert_not_called()
    quick.tts.say_safe.assert_not_called()
    quick.tts.say_chunked.assert_not_called()
    assert turn.verdict == "unverified"


def test_verified_wake_sends_the_question_without_the_wake_word(quick) -> None:
    _verified_turn(quick, "Hey Jarvis, what is 3 plus 3?")
    assert quick.agent.ask.call_args.args[0] == "what is 3 plus 3?"
    audio = quick.stt.transcribe_audio.call_args.args[0]
    assert audio.size == 800 + 1600 and audio[:800].all(), "pre-roll must come first"


def test_wake_word_alone_is_silent(quick) -> None:
    turn = _verified_turn(quick, "Hey Jarvis.")
    quick.agent.ask.assert_not_called()
    assert turn.verdict == "wake-only"


def test_verification_can_be_disabled(quick) -> None:
    _with_wake(quick, barge_in=False, verify=False)
    _verified_turn(quick, "what is 3 plus 3?")
    quick.agent.ask.assert_called_once()
    assert quick.stt.transcribe_audio.call_args.args[0].size == 1600


def test_followups_are_not_verified(quick) -> None:
    _verified_turn(quick, "I think it is.", source="followup", preroll=False)
    quick.agent.ask.assert_called_once()


@pytest.mark.parametrize("source", ["wake", "followup"])
def test_pleasantry_only_never_reaches_the_agent(quick, source) -> None:
    heard = "Hey Jarvis, thank you." if source == "wake" else "Thank you. You're welcome."
    turn = _verified_turn(quick, heard, source=source, preroll=source == "wake")
    quick.agent.ask.assert_not_called()
    assert turn.verdict == "pleasantry"


@pytest.mark.parametrize("fire_on,expected_frames", [(4, 4), (40, 25)])
def test_wake_loop_queues_the_audio_before_the_trigger(quick, fire_on, expected_frames) -> None:
    """Up to preroll_s (2.0 s = 25 frames) of audio up to and including the trigger."""
    class FiresOnTenth:
        calls = 0

        def detect(self, frame):
            self.calls += 1
            return self.calls == fire_on

        def reset(self):
            pass

    _with_wake(quick, barge_in=False, silence_s=0.05, lead_in_s=5.0)
    quick._detector = FiresOnTenth()
    buf = _streaming(quick, level=0.5)
    t = threading.Thread(target=quick._wake_loop, daemon=True)
    t.start()
    # Two frames per 40 ms poll; keep talking for a while after the wake.
    time.sleep(fire_on * 0.02 + 0.4)
    buf.level = 0.0
    time.sleep(0.3)
    quick._stop.set()
    t.join(timeout=2)
    quick._stop.clear()

    audio, source, preroll = quick._queue.get_nowait()
    assert source == "wake"
    block = quick.cfg.audio.block_size
    assert preroll.size == expected_frames * block


def _closing(quick, model_reply, ok=True, heard="Hey Jarvis, thank you.", **kw):
    from homeai.stt import Transcript
    quick.memory.add("is the universe alive", "Some philosophers think so.")
    quick.stt.transcribe_audio.return_value = Transcript(ok=True, text=heard)
    quick.agent.ask.reset_mock()
    reply = quick.agent.ask.return_value
    reply.ok, reply.text, reply.attempts = ok, model_reply, 1
    reply.error = "" if ok else "agent leaked tool-call markup"
    quick.transcript = MagicMock()
    quick._handle(np.zeros(1600, dtype=np.float32), kw.get("source", "wake"),
                  np.ones(800, dtype=np.float32) if kw.get("source", "wake") == "wake" else None)
    spoken = [c.args[0] for c in quick.tts.say_safe.call_args_list]
    return quick.transcript.write.call_args.args[0], spoken


def test_thanks_mid_conversation_goes_to_the_model_with_the_one_sentence_hint(quick) -> None:
    from homeai.daemon import CLOSING_HINT
    turn, spoken = _closing(quick, "My pleasure, Adam.")
    quick.agent.ask.assert_called_once_with("thank you.", hint=CLOSING_HINT)
    assert spoken == ["My pleasure, Adam."]
    assert turn.verdict == "closing" and not quick._followup_armed.is_set()


def test_monologue_after_thanks_is_cut_to_one_sentence(quick) -> None:
    """Live 2026-10-03: "thank you" produced 23 s more on the previous topic."""
    _, spoken = _closing(quick, "You're welcome! Also, the universe may be alive because "
                                "of fine tuning. Many philosophers agree.")
    assert spoken == ["You're welcome!"]


@pytest.mark.parametrize("model_reply,ok", [
    ("Anything else I can help with?", True),  # would open a follow-up window
    ("", True),
    ("<tool_call>", False),
])
def test_unusable_closing_reply_falls_back(quick, model_reply, ok) -> None:
    _, spoken = _closing(quick, model_reply, ok=ok)
    assert spoken == ["You're welcome."]
    assert not quick._followup_armed.is_set()


def test_followup_goodbye_mid_conversation_goes_to_the_model(quick) -> None:
    _, spoken = _closing(quick, "Goodnight, sleep well.", heard="Bye bye.", source="followup")
    quick.agent.ask.assert_called_once()
    assert spoken == ["Goodnight, sleep well."]


def test_okay_mid_conversation_is_quiet(quick) -> None:
    quick.agent.ask.reset_mock()
    quick.memory.add("is the universe alive", "Some philosophers think so.")
    turn = _verified_turn(quick, "Okay.", source="followup", preroll=False)
    quick.agent.ask.assert_not_called()
    quick.tts.say_safe.assert_not_called()
    assert turn.verdict == "pleasantry"


def test_thanks_out_of_nowhere_is_quiet(quick) -> None:
    turn = _verified_turn(quick, "Hey Jarvis, thank you.")
    quick.tts.say_safe.assert_not_called()
    quick.agent.ask.assert_not_called()
    assert turn.verdict == "pleasantry"


def test_whisper_hallucination_is_dropped_even_mid_conversation(quick) -> None:
    quick.memory.add("is the universe alive", "Some philosophers think so.")
    turn = _verified_turn(quick, "Thanks for watching!", source="followup", preroll=False)
    quick.agent.ask.assert_not_called()
    assert turn.verdict == "pleasantry"


# -- speaker recognition ------------------------------------------------------

def _unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


ADAM, BETH, STRANGER = _unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1)


@pytest.fixture
def voiced(quick, tmp_path, monkeypatch):
    """quick, with speaker-id on: Adam and Beth enrolled, embeddings scripted."""
    from homeai.config import SpeakerConfig
    from homeai.voice_id import VoiceId
    v = VoiceId(SpeakerConfig(enabled=True, registry_path=tmp_path / "speakers.json"))
    v.registry.add("Adam", ADAM)
    v.registry.add("Beth", BETH)
    quick.voiced_next = ADAM
    monkeypatch.setattr(v.embedder, "embed", lambda audio, trim=True: quick.voiced_next)
    quick.voice_id = v
    return quick


def _speak_as(q, emb, heard, source="wake", seconds=1.0):
    from homeai.stt import Transcript
    q.voiced_next = emb
    q.stt.transcribe_audio.return_value = Transcript(ok=True, text=heard)
    q.agent.ask.reset_mock()
    q.tts.say_safe.reset_mock()
    reply = q.agent.ask.return_value
    reply.ok, reply.text, reply.attempts, reply.error = True, "Six.", 1, ""
    q.tts.say_chunked.return_value.interrupted = False
    q.tts.say_chunked.return_value.first_audio_ms = 100.0
    q.transcript = MagicMock()
    q._followup_armed.clear()
    t = np.arange(int(seconds * 16000)) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    q._handle(audio, source, np.zeros(800, dtype=np.float32) if source == "wake" else None)
    spoken = [c.args[0] for c in q.tts.say_safe.call_args_list]
    return q.transcript.write.call_args.args[0], spoken


def test_recognised_speaker_is_logged_and_owns_the_conversation(voiced) -> None:
    turn, _ = _speak_as(voiced, ADAM, "Hey Jarvis, what is two plus four?")
    voiced.agent.ask.assert_called_once()
    assert turn.speaker == "Adam" and turn.speaker_score == pytest.approx(1.0)
    assert voiced.memory.speaker == "Adam"


def test_unknown_speaker_still_gets_an_answer(voiced) -> None:
    turn, _ = _speak_as(voiced, STRANGER, "Hey Jarvis, what is two plus four?")
    voiced.agent.ask.assert_called_once()
    assert turn.speaker is None and turn.verdict == "allow"


def test_followup_from_another_voice_is_ignored(voiced) -> None:
    _speak_as(voiced, ADAM, "Hey Jarvis, what is two plus four?")
    voiced.memory.add("what is two plus four", "Six. Anything else?")
    turn, spoken = _speak_as(voiced, BETH, "Mum, where are my shoes?", source="followup")
    voiced.agent.ask.assert_not_called()
    assert turn.verdict == "other-speaker" and spoken == []


def test_followup_from_the_same_voice_is_answered(voiced) -> None:
    _speak_as(voiced, ADAM, "Hey Jarvis, what is two plus four?")
    _speak_as(voiced, _unit(0.9, 0.1, 0.1), "and times two?", source="followup")
    voiced.agent.ask.assert_called_once()


def test_who_am_i(voiced) -> None:
    turn, spoken = _speak_as(voiced, ADAM, "Hey Jarvis, who am I?")
    voiced.agent.ask.assert_not_called()
    assert spoken == ["You sound like Adam."] and turn.verdict == "voice-who"
    _, spoken = _speak_as(voiced, STRANGER, "Hey Jarvis, who am I?")
    assert spoken == ["I'm not sure who you are."]


def test_enrolment_by_voice(voiced, monkeypatch) -> None:
    import homeai.voice_id as vid
    monkeypatch.setattr(vid, "ENROL_TARGET_SPEECH_S", 6.0)
    carl = _unit(1, 1, 1)
    _, spoken = _speak_as(voiced, carl, "Hey Jarvis, remember my voice as Carl.", seconds=2.0)
    assert spoken[0].startswith("Okay, Carl.") and voiced._followup_armed.is_set()
    turn, spoken = _speak_as(voiced, carl, "I had a busy day.", source="followup", seconds=2.0)
    assert spoken == [vid.MSG_ENROL_MORE] and voiced._followup_armed.is_set()
    assert turn.verdict == "enrolling"
    _, spoken = _speak_as(voiced, carl, "Lots of meetings.", source="followup", seconds=3.0)
    assert spoken == ["Got it. I'll know your voice now, Carl."]
    assert not voiced._followup_armed.is_set()
    voiced.agent.ask.assert_not_called()
    assert voiced.voice_id.registry.get("Carl") is not None
    assert "Carl" in voiced.voice_id.registry.path.read_text()


def test_enrolment_is_abandoned_by_a_new_wake_turn(voiced) -> None:
    _speak_as(voiced, ADAM, "Hey Jarvis, remember my voice as Adam.")
    _speak_as(voiced, ADAM, "Hey Jarvis, what is two plus four?")
    voiced.agent.ask.assert_called_once()
    assert voiced._enrolment is None


def test_enrolment_without_a_name_asks_for_one(voiced) -> None:
    _, spoken = _speak_as(voiced, ADAM, "Hey Jarvis, remember my voice.")
    assert spoken == ["Say: remember my voice as, and then your name."]
    assert voiced._enrolment is None


def test_forget_my_voice(voiced) -> None:
    _, spoken = _speak_as(voiced, BETH, "Hey Jarvis, forget my voice.")
    assert spoken == ["Okay, Beth. I've forgotten your voice."]
    assert voiced.voice_id.registry.names() == ["Adam"]


def test_voice_commands_when_speaker_id_is_off(quick) -> None:
    assert quick.voice_id is None
    turn = _verified_turn(quick, "Hey Jarvis, remember my voice as Adam.")
    assert turn.reply == "Voice recognition is turned off."
    quick.agent.ask.assert_not_called()
    _verified_turn(quick, "Hey Jarvis, who am I?")
    quick.agent.ask.assert_called_once()


def test_identification_failure_never_loses_the_turn(voiced, monkeypatch) -> None:
    def boom(audio, trim=True):
        raise RuntimeError("onnx exploded")
    monkeypatch.setattr(voiced.voice_id.embedder, "embed", boom)
    turn, _ = _speak_as(voiced, None, "Hey Jarvis, what is two plus four?")
    voiced.agent.ask.assert_called_once()
    assert turn.speaker is None


def test_missing_speaker_model_turns_speaker_id_off(quick, tmp_path) -> None:
    from homeai.config import SpeakerConfig
    from homeai.voice_id import VoiceId
    quick.voice_id = VoiceId(SpeakerConfig(enabled=True, model_path=tmp_path / "x.onnx",
                                           registry_path=tmp_path / "s.json"))
    quick._start_voice_id()
    assert quick.voice_id is None
