# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for turn-taking rules (homeai/dialogue.py)."""

from __future__ import annotations

import pytest

from homeai.dialogue import (
    FollowupChain,
    HeldRemainder,
    invites_reply,
    is_continue_request,
    is_decline,
    split_for_budget,
)


class TestInvitesReply:
    @pytest.mark.parametrize("text", [
        "What do you think?",
        "I lean toward compatibilism. Where do you land on it?",
        "That's my view. Does that match your experience?\n",
        'Do you mean the "hard problem"?',
        "Want me to keep going?)",
    ])
    def test_final_question_invites_a_reply(self, text):
        assert invites_reply(text)

    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "The capital of France is Paris.",
        # Rhetorical question mid-answer; the speaker has moved on.
        "Is free will real? I think it is, in the sense that matters.",
        "Sorry, I didn't catch that.",
        "It's sixty degrees and overcast!",
    ])
    def test_statement_does_not(self, text):
        assert not invites_reply(text)

    def test_abbreviation_does_not_split_the_final_question(self):
        # "Dr." must not be treated as a sentence end, or the final sentence
        # would be mis-identified.
        assert invites_reply("Have you read anything by Dr. Dennett?")


class TestFollowupChain:
    def test_opens_until_the_cap_then_refuses(self):
        chain = FollowupChain(max_chain=2)
        assert chain.may_open()
        chain.record_turn(from_followup=True)
        assert chain.may_open()
        chain.record_turn(from_followup=True)
        assert not chain.may_open(), "a loop with background audio must end"

    def test_wake_word_turn_resets_the_count(self):
        chain = FollowupChain(max_chain=1)
        chain.record_turn(from_followup=True)
        assert not chain.may_open()
        chain.record_turn(from_followup=False)
        assert chain.may_open()

    def test_zero_disables_followups(self):
        assert not FollowupChain(max_chain=0).may_open()


# -- spoken budget -------------------------------------------------------------

def _sentences(n: int) -> str:
    # Eight words each.
    return " ".join(f"This is sentence number {i} of the reply." for i in range(n))


class TestSplitForBudget:
    def test_short_reply_is_untouched(self):
        assert split_for_budget("It is sunny.", 110) == ("It is sunny.", "")

    def test_zero_budget_disables(self):
        text = _sentences(40)
        assert split_for_budget(text, 0) == (text, "")

    def test_long_reply_splits_on_a_sentence_boundary(self):
        head, rest = split_for_budget(_sentences(30), 40)
        assert len(head.split()) <= 40
        assert head.endswith(".") and rest
        assert f"{head} {rest}" == _sentences(30)

    def test_nothing_is_lost(self):
        text = _sentences(25)
        head, rest = split_for_budget(text, 33)
        assert (head + " " + rest).split() == text.split()

    def test_small_remainder_is_spoken_not_offered(self):
        # 6 sentences = 48 words; budget 40 leaves 8 words, under a quarter.
        text = _sentences(6)
        assert split_for_budget(text, 40) == (text, "")

    def test_first_sentence_kept_whole_even_if_over_budget(self):
        long_first = " ".join(["word"] * 50) + "."
        text = f"{long_first} {_sentences(10)}"
        head, rest = split_for_budget(text, 20)
        assert head == long_first
        assert rest


class TestReplyToContinuePrompt:
    @pytest.mark.parametrize("text", [
        "yes", "Yes please.", "Sure, go on.", "yeah", "Go ahead, Jarvis.",
        "keep going", "tell me more", "Yes, thank you.", "okay",
    ])
    def test_continue(self, text):
        assert is_continue_request(text)
        assert not is_decline(text)

    @pytest.mark.parametrize("text", [
        "No.", "no thanks", "nah", "I'm good.", "thank you", "That's enough.",
    ])
    def test_decline(self, text):
        assert is_decline(text)
        assert not is_continue_request(text)

    @pytest.mark.parametrize("text", [
        "tell me more about mars", "what is the weather", "go on about the moon",
        "", "no, go on",
    ])
    def test_new_question_is_neither(self, text):
        assert not is_continue_request(text)
        assert not is_decline(text)


class TestHeldRemainder:
    def test_take_returns_and_clears(self):
        held = HeldRemainder(ttl_s=60)
        held.hold("the rest", now=0)
        assert held.pending(10)
        assert held.take(10) == "the rest"
        assert not held.pending(10)

    def test_expires(self):
        held = HeldRemainder(ttl_s=60)
        held.hold("the rest", now=0)
        assert not held.pending(61)
        assert held.take(61) == ""

    def test_empty_is_not_pending(self):
        assert not HeldRemainder().pending(0)
