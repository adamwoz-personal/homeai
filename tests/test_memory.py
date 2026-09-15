# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
import threading
import time
from unittest.mock import Mock

import pytest

from homeai.memory import ConversationMemory, Exchange
from homeai.safety import detect_leaked_markup


def fake_clock():
    return 1000.0


def test_add_returns_true_and_stores_normal_exchange():
    """Test that add() returns True and stores a normal exchange."""
    # max_turns must exceed the thread count, otherwise trimming - not a
    # race - explains any shortfall and the test proves nothing.
    memory = ConversationMemory(max_turns=50, max_chars=100_000, clock=fake_clock)
    assert memory.add("user message", "assistant response")
    assert len(memory) == 1
    entries = memory.recent()
    assert len(entries) == 1
    assert entries[0].user == "user message"
    assert entries[0].assistant == "assistant response"


def test_add_returns_false_and_stores_nothing_when_user_empty():
    """Test that add() returns False and stores nothing when user is empty."""
    memory = ConversationMemory(clock=fake_clock)
    assert not memory.add("", "assistant response")
    assert len(memory) == 0


def test_add_returns_false_and_stores_nothing_when_assistant_empty():
    """Test that add() returns False and stores nothing when assistant is empty."""
    memory = ConversationMemory(clock=fake_clock)
    assert not memory.add("user message", "")
    assert len(memory) == 0


def test_add_returns_false_and_stores_nothing_when_user_whitespace():
    """Test that add() returns False and stores nothing when user is whitespace."""
    memory = ConversationMemory(clock=fake_clock)
    assert not memory.add("   ", "assistant response")
    assert len(memory) == 0


def test_add_returns_false_and_stores_nothing_when_assistant_whitespace():
    """Test that add() returns False and stores nothing when assistant is whitespace."""
    memory = ConversationMemory(clock=fake_clock)
    assert not memory.add("user message", "   ")
    assert len(memory) == 0


def test_add_returns_false_when_leaked_markup_in_user():
    """Test that add() returns False when user contains leaked tool-call markup."""
    memory = ConversationMemory(clock=fake_clock)
    leaked = '{"name": "shell", "arguments": {"cmd": "ls"}}'
    assert not memory.add(leaked, "assistant response")
    assert len(memory) == 0
    # Verify that the leaked markup is detected
    assert detect_leaked_markup(leaked) == True


def test_add_returns_false_when_leaked_markup_in_assistant():
    """Test that add() returns False when assistant contains leaked tool-call markup."""
    memory = ConversationMemory(clock=fake_clock)
    leaked = '{"name": "shell", "arguments": {"cmd": "ls"}}'
    assert not memory.add("user message", leaked)
    assert len(memory) == 0
    # Verify that the leaked markup is detected
    assert detect_leaked_markup(leaked) == True


def test_max_turns_evicts_oldest_exchange_first():
    """Test that max_turns evicts the oldest exchange first."""
    memory = ConversationMemory(max_turns=2, clock=fake_clock)
    memory.add("1", "a")
    memory.add("2", "b")
    memory.add("3", "c")
    entries = memory.recent()
    assert len(entries) == 2
    assert entries[0].user == "2"
    assert entries[1].user == "3"


def test_max_chars_evicts_oldest_first_but_retains_most_recent():
    """Test that max_chars evicts oldest-first, but the most recent exchange is ALWAYS retained."""
    memory = ConversationMemory(max_turns=10, max_chars=20, clock=fake_clock)
    # Add exchanges that will exceed max_chars
    memory.add("x" * 10, "y" * 10)  # 20 chars
    memory.add("a" * 10, "b" * 10)  # 20 chars - total 40, exceeds max_chars
    entries = memory.recent()
    # Should keep the most recent one (even though it alone exceeds max_chars)
    assert len(entries) == 1
    assert entries[0].user == "a" * 10


def test_entry_chars_truncates_long_stored_entry():
    """Test that entry_chars truncates a long stored entry."""
    memory = ConversationMemory(entry_chars=5, clock=fake_clock)
    memory.add("hello world", "assistant")
    assert memory.recent()[-1].user == "hello..."

    # With room for the first sentence, truncation prefers a sentence
    # boundary so the stored context does not end mid-word.
    memory = ConversationMemory(entry_chars=30, clock=fake_clock)
    memory.add("This is a long sentence. And this is another.", "assistant")
    assert memory.recent()[-1].user == "This is a long sentence."


def test_idle_expiry_s_clears_entire_window():
    """Test that idle_expiry_s clears the ENTIRE window once the gap since the last entry exceeds it."""
    clock = Mock()
    clock.return_value = 1000.0  # Start at 1000
    memory = ConversationMemory(idle_expiry_s=10.0, clock=clock)
    memory.add("user", "assistant")
    assert len(memory.recent()) == 1
    # Advance clock by 11 seconds (exceeds expiry)
    clock.return_value = 1011.0
    entries = memory.recent()
    assert len(entries) == 0


def test_context_prompt_returns_empty_when_empty():
    """Test that context_prompt() returns "" when empty."""
    memory = ConversationMemory(clock=fake_clock)
    assert memory.context_prompt() == ""


def test_build_request_returns_utterance_when_empty():
    """Test that build_request() returns the bare utterance unchanged when memory is empty."""
    memory = ConversationMemory(clock=fake_clock)
    assert memory.build_request("test utterance") == "test utterance"


def test_build_request_includes_prior_exchanges_when_populated():
    """Test that build_request() includes both prior user and assistant text plus the new utterance when memory is populated."""
    memory = ConversationMemory(clock=fake_clock)
    memory.add("user1", "assistant1")
    memory.add("user2", "assistant2")
    request = memory.build_request("user3")
    expected = (
        "Earlier in this conversation (for context; the user cannot hear "
        "this and may refer back to it):\n"
        "\n"
        "User: user1\n"
        "You: assistant1\n"
        "\n"
        "User: user2\n"
        "You: assistant2\n"
        "\n"
        "Now answer this, resolving any references to the exchange above. "
        "If you search or look something up, expand those references into "
        "a self-contained query first.\n"
        "\n"
        "User: user3"
    )
    assert request == expected


def test_clear_empties_memory():
    """Test that clear() empties it."""
    memory = ConversationMemory(clock=fake_clock)
    memory.add("user", "assistant")
    assert len(memory) == 1
    memory.clear()
    assert len(memory) == 0


def test_max_turns_zero_stores_nothing():
    """Test that max_turns=0 stores nothing."""
    memory = ConversationMemory(max_turns=0, clock=fake_clock)
    # add() must report honestly: trimming discards the entry immediately,
    # so a True return would describe a window that does not exist.
    assert not memory.add("user", "assistant")
    assert len(memory) == 0

def test_thread_safety_concurrent_add_calls():
    """Concurrent add() must not raise and must leave a consistent length."""
    memory = ConversationMemory(max_turns=50, max_chars=100000, clock=fake_clock)
    results = []

    def add_exchange():
        result = memory.add("user", "assistant")
        results.append(result)

    threads = []
    for _ in range(10):
        t = threading.Thread(target=add_exchange)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # All calls should succeed
    assert all(results)
    # Should have 10 entries
    assert len(memory) == 10