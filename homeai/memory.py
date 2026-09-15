# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Conversation memory: continuity without reopening the poisoning hole.

The problem this solves
-----------------------
Asked "what do you think about modified gravity versus string theory", then
"specifically referring to MOND", the assistant searched the bare token cold
and returned a Rainmeter theme and two football recruits. It had no idea the
previous sentence existed.

That was a *deliberate* trade-off, not an oversight. Every request runs in a
fresh ZeroClaw session because the local model intermittently emits malformed
tool-call markup, and a session that has seen it once imitates it forever --
measured at 2/10 turns from a clean history versus 10/10 from a poisoned one.
Discarding history cured the poisoning and destroyed continuity with it.

Why this is safe
----------------
The poison was never "history" in the abstract; it was *raw tool-call markup*
sitting in the transcript for the model to imitate. This module keeps a
rolling window of **cleaned, plain-text** exchanges only:

* every entry passes through the same leaked-markup detection the safety layer
  uses, and anything matching is dropped entirely rather than sanitised -- a
  partially-scrubbed tool call is exactly the malformed shape that teaches the
  model to emit more of them;
* the window is bounded by turns *and* by characters, so a long answer cannot
  crowd out the recent exchange or blow the context budget;
* it expires after an idle gap, because a question asked at breakfast should
  not inherit the thread from midnight. Stale context is worse than none: it
  produces confident answers to a question nobody asked.

Sessions are still never reused. Context is passed as plain text in the
request, so there is no session state to corrupt.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .safety import detect_leaked_markup

# Defaults chosen for a voice assistant rather than a chat window.
#
# Four exchanges covers the natural span of a spoken follow-up chain
# ("what about X" -> "why" -> "is that settled") without dragging in an
# unrelated topic from ten minutes ago.
DEFAULT_MAX_TURNS = 4

# Roughly a page. Large enough to preserve a detailed answer's gist, small
# enough that it never dominates the prompt or slows the model measurably.
DEFAULT_MAX_CHARS = 3000

# Five minutes. Long enough to survive thinking about a follow-up, short
# enough that the next person to walk in starts clean.
DEFAULT_IDLE_EXPIRY_S = 300.0

# Per-entry cap. A 68-second answer is ~6000 characters; storing it whole
# would evict everything else. The opening of an answer carries its thesis.
DEFAULT_ENTRY_CHARS = 700


@dataclass(frozen=True)
class Exchange:
    """One completed user/assistant pair."""

    user: str
    assistant: str
    at: float


class ConversationMemory:
    """A bounded, self-expiring window of recent exchanges.

    Thread-safe: the daemon writes from its worker thread while diagnostics
    may read from another.
    """

    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_chars: int = DEFAULT_MAX_CHARS,
        idle_expiry_s: float = DEFAULT_IDLE_EXPIRY_S,
        entry_chars: int = DEFAULT_ENTRY_CHARS,
        clock=time.monotonic,
    ) -> None:
        self._max_turns = max(0, max_turns)
        self._max_chars = max(0, max_chars)
        self._idle_expiry_s = idle_expiry_s
        self._entry_chars = max(0, entry_chars)
        self._clock = clock
        self._entries: list[Exchange] = []
        self._lock = threading.Lock()

    # -- writing -----------------------------------------------------------

    def add(self, user: str, assistant: str) -> bool:
        """Record an exchange. Returns False if it was rejected.

        Rejection is not an error: refusing to remember a tainted turn is the
        entire point. The caller carries on normally.
        """
        user = (user or "").strip()
        assistant = (assistant or "").strip()
        if not user or not assistant:
            return False

        # Drop, never scrub. A half-removed tool call is the malformed shape
        # that teaches the model to emit more of them.
        if detect_leaked_markup(user) or detect_leaked_markup(assistant):
            return False

        entry = Exchange(
            user=_truncate(user, self._entry_chars),
            assistant=_truncate(assistant, self._entry_chars),
            at=self._clock(),
        )

        with self._lock:
            self._entries.append(entry)
            self._trim_locked()
            # Report what actually happened. Trimming can discard the entry we
            # just added (max_turns=0, or an entry over the character budget),
            # and an add() that returns True while storing nothing would make
            # callers trust a window that does not exist.
            return entry in self._entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    # -- reading -----------------------------------------------------------

    def recent(self) -> list[Exchange]:
        """Non-expired entries, oldest first."""
        with self._lock:
            self._expire_locked()
            return list(self._entries)

    def context_prompt(self) -> str:
        """Render the window as plain text to prepend to a request.

        Returns "" when there is nothing to say, so the caller can send the
        bare utterance unchanged and pay nothing for an empty window.
        """
        entries = self.recent()
        if not entries:
            return ""

        lines = [
            "Earlier in this conversation (for context; the user cannot hear "
            "this and may refer back to it):",
            "",
        ]
        for entry in entries:
            lines.append(f"User: {entry.user}")
            lines.append(f"You: {entry.assistant}")
            lines.append("")

        lines.append(
            "Now answer this, resolving any references to the exchange above. "
            "If you search or look something up, expand those references into "
            "a self-contained query first."
        )
        # Two blank lines: the separation makes it unambiguous to the model
        # where the recalled context ends and the live question begins.
        lines.extend(["", ""])
        return "\n".join(lines)

    def build_request(self, utterance: str) -> str:
        """Combine context with the current utterance."""
        context = self.context_prompt()
        if not context:
            return utterance
        return f"{context}User: {utterance}"

    # -- internals ---------------------------------------------------------

    def _expire_locked(self) -> None:
        """Drop the whole window once idle too long.

        All-or-nothing on purpose: keeping the oldest half of a stale
        conversation is the worst case, since it supplies context that no
        longer matches what the user is thinking about.
        """
        if not self._entries or self._idle_expiry_s <= 0:
            return
        if self._clock() - self._entries[-1].at > self._idle_expiry_s:
            self._entries.clear()

    def _trim_locked(self) -> None:
        self._expire_locked()

        if self._max_turns == 0:
            self._entries.clear()
            return

        del self._entries[: max(0, len(self._entries) - self._max_turns)]

        # Evict oldest-first until the window fits the character budget, but
        # always keep the most recent exchange: the immediately preceding turn
        # is what a follow-up almost always refers to.
        while len(self._entries) > 1 and self._total_chars() > self._max_chars:
            self._entries.pop(0)

    def _total_chars(self) -> int:
        return sum(len(e.user) + len(e.assistant) for e in self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _truncate(text: str, limit: int) -> str:
    """Cut at a sentence boundary where possible, so context reads cleanly."""
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary > limit // 2:
        return cut[: boundary + 1]
    return cut.rstrip() + "..."
