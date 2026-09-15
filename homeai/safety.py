# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Safety filters for the voice pipeline.

Two independent concerns live here, both pure functions so they can be tested
exhaustively without hardware:

1. ``check_utterance``  - refuse dangerous spoken commands *before* they ever
   reach the agent. This is defence in depth. The real control is the
   restricted ZeroClaw agent profile; this layer is cheap insurance.

2. ``detect_leaked_markup`` - catch the local model's habit of emitting raw
   tool-call markup as assistant text. Measured at roughly 10-20% of
   tool-calling turns, and a single occurrence poisons the session
   permanently, so detection must be reliable and the session then discarded.

Design note: everything is deliberately conservative. A false positive costs
the user one repeated sentence. A false negative could delete their files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------------------
# Leaked tool-call markup detection
# --------------------------------------------------------------------------

# Patterns the model has actually been observed emitting as plain text, plus
# the well-formed variants. Matching is case-insensitive because the model is
# inconsistent about casing when it malfunctions.
_MARKUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\s*function\s*=", re.IGNORECASE),
    re.compile(r"</?\s*tool_call\s*>", re.IGNORECASE),
    re.compile(r"<\s*parameter\s*=", re.IGNORECASE),
    re.compile(r"</?\s*function_calls\s*>", re.IGNORECASE),
    re.compile(r"</?\s*invoke\b", re.IGNORECASE),
    re.compile(r"<\|tool_call", re.IGNORECASE),
    # The model's *dominant* failure mode is omitting the opening <tool_call>
    # tag, which delivers the bare JSON payload as prose. Tag matching alone
    # misses this completely, so match the JSON shape directly.
    re.compile(
        r"""["']?(?:name|function|tool|tool_name)["']?\s*:\s*["'][^"']+["']"""
        r"""\s*,\s*["']?(?:arguments|parameters|args)["']?\s*:""",
        re.IGNORECASE,
    ),
    re.compile(
        r"""["']?(?:arguments|parameters|args)["']?\s*:\s*\{"""
        r"""[^{}]*["']?(?:command|cmd|path|file_path|content|script)["']?\s*:""",
        re.IGNORECASE,
    ),
    re.compile(r"""\{\s*["']?(?:tool_call|function_call|tool_name)["']?\s*:""", re.IGNORECASE),
    # Bare mentions of the agent's own tool identifiers inside a JSON-ish
    # fragment. A spoken answer has no legitimate reason to contain these.
    re.compile(
        r"""["'](?:shell_tool|file_write|file_read|file_download|browser_open)["']""",
        re.IGNORECASE,
    ),
)


def detect_leaked_markup(text: str) -> bool:
    """True if ``text`` contains tool-call markup that leaked into prose.

    Such a reply must never be spoken aloud and must never be retained in
    conversation history - see the poisoning behaviour documented in the
    project plan.

    Covers both the well-formed ``<tool_call>`` wrapper and the far more
    common *unwrapped* JSON payload, which llama.cpp passes through verbatim
    as assistant text when the model drops the opening tag.
    """
    if not text:
        return False
    return any(p.search(text) for p in _MARKUP_PATTERNS)


# --------------------------------------------------------------------------
# Spoken command screening
# --------------------------------------------------------------------------


class Verdict(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    REFUSE = "refuse"


@dataclass(frozen=True)
class SafetyResult:
    verdict: Verdict
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW


# Phrases that must never be actioned from a voice command. Word-boundary
# anchored so that "formatting" does not trigger "format", and "deleted" in
# casual conversation does not match the destructive "delete the".
_REFUSE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\brm\s+-[rf]", re.IGNORECASE), "recursive delete"),
    (re.compile(r"\bdelete\s+(all|every|the\s+\w+\s+director)", re.IGNORECASE), "bulk delete"),
    (re.compile(r"\bwipe\b|\berase\s+(all|everything|the\s+disk)", re.IGNORECASE), "wipe"),
    (re.compile(r"\bformat\s+(the\s+)?(disk|drive|partition|/dev)", re.IGNORECASE), "format disk"),
    (re.compile(r"\bmkfs\b|\bfdisk\b|\bdd\s+if=", re.IGNORECASE), "raw disk operation"),
    (re.compile(r"\b(shut\s*down|power\s*off|reboot|restart)\s+(the\s+)?(system|machine|computer|server|box)", re.IGNORECASE), "power state change"),
    (re.compile(r"\bssh\s+key|\bprivate\s+key|\bid_rsa\b|\.ssh\b", re.IGNORECASE), "ssh credentials"),
    (re.compile(r"\bpassword|\bcredential|\bapi[_\s-]?key|\btoken\b|\bsecret\b", re.IGNORECASE), "credential access"),
    (re.compile(r"\bsudo\b|\bchmod\s+777|\bchown\s+root", re.IGNORECASE), "privilege escalation"),
    (re.compile(r"\bcurl\b.*\|\s*(ba)?sh|\bwget\b.*\|\s*(ba)?sh", re.IGNORECASE), "pipe to shell"),
    (re.compile(r"\bdisable\s+(the\s+)?(firewall|ufw|selinux|apparmor)", re.IGNORECASE), "security downgrade"),
    (re.compile(r"\bgit\s+push\b|\bforce\s+push\b", re.IGNORECASE), "publishing code"),
    (re.compile(r"\btransfer\b.*\bmoney|\bbuy\b.*\border|\bpurchase\b", re.IGNORECASE), "financial action"),
)

# Actions that are reasonable but change state; confirm them out loud first.
_CONFIRM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(turn|switch)\s+(on|off)\b", re.IGNORECASE), "device state change"),
    (re.compile(r"\b(unlock|lock)\b", re.IGNORECASE), "lock state change"),
    (re.compile(r"\bopen\s+(the\s+)?(door|garage|gate)", re.IGNORECASE), "physical access"),
    (re.compile(r"\b(send|text|email|message)\s+", re.IGNORECASE), "outbound message"),
    (re.compile(r"\bdelete\b|\bremove\b", re.IGNORECASE), "deletion"),
    (re.compile(r"\bset\s+(the\s+)?(thermostat|temperature|heating)", re.IGNORECASE), "climate change"),
)

# An utterance longer than this is almost certainly the microphone picking up
# a broadcast, a conversation, or a podcast rather than a command.
MAX_UTTERANCE_CHARS = 500


def check_utterance(text: str) -> SafetyResult:
    """Screen a transcribed utterance before dispatching it to the agent."""
    if text is None:
        return SafetyResult(Verdict.REFUSE, "empty transcript")

    stripped = text.strip()
    if not stripped:
        return SafetyResult(Verdict.REFUSE, "empty transcript")

    if len(stripped) > MAX_UTTERANCE_CHARS:
        return SafetyResult(
            Verdict.REFUSE,
            f"utterance too long ({len(stripped)} chars) - likely ambient audio, not a command",
        )

    for pattern, reason in _REFUSE_PATTERNS:
        if pattern.search(stripped):
            return SafetyResult(Verdict.REFUSE, reason)

    for pattern, reason in _CONFIRM_PATTERNS:
        if pattern.search(stripped):
            return SafetyResult(Verdict.CONFIRM, reason)

    return SafetyResult(Verdict.ALLOW)


def sanitise_for_speech(text: str, limit: int = 4000) -> str:
    """Make agent output safe and sensible to read aloud.

    Strips code fences and collapses whitespace: reading a code block aloud is
    useless.

    ``limit`` is a runaway guard, not a style control. It was originally 600
    characters, which silently truncated any genuinely detailed answer at
    roughly half a page - the user asked for deeper discussions and would have
    got cut off mid-thought with no indication why. Answer *length* is now
    steered by the agent's prompt; this only stops a pathological reply from
    monopolising the speaker for many minutes.
    """
    if not text:
        return ""

    without_code = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.DOTALL)
    without_inline = re.sub(r"`([^`]*)`", r"\1", without_code)
    collapsed = re.sub(r"\s+", " ", without_inline).strip()

    if len(collapsed) <= limit:
        return collapsed

    # Prefer to cut at a sentence boundary so speech does not stop mid-word.
    cut = collapsed[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary > limit // 2:
        return cut[: boundary + 1]
    return cut.rstrip() + "..."
