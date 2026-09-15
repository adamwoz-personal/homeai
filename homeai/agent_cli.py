# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Subprocess transport to the ZeroClaw agent.

Why this exists instead of the gateway webhook
----------------------------------------------
The gateway's ``POST /webhook`` route was measured to **ignore the target
agent's risk profile**: with ``[agents.local]`` bound to a ``readonly`` profile
whose ``allowed_commands`` excluded ``date``, the webhook still executed
``date`` and returned its output. The same request through the CLI was refused.

That makes the webhook unusable for the voice path, because a voice waveform
carries no authentication - anyone in earshot, including a television, can
reach it. The CLI path enforces the profile, so the voice service uses it.

It is also far faster, because the restricted profile exposes ~3 tools instead
of 62:

    gateway webhook : ~5.2 s steady
    CLI, restricted : ~0.5 s steady

Every call uses a **fresh session-state file**, for the same reason the HTTP
client uses a fresh session id: the local model intermittently emits malformed
tool-call markup, and a session that has seen it once imitates it forever.
Nothing is ever carried between utterances.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
import uuid

from .agent_client import AgentReply
from .config import AgentConfig
from .safety import detect_leaked_markup

log = logging.getLogger(__name__)


class CliAgentClient:
    """Talk to ZeroClaw by spawning its CLI.

    Mirrors :class:`~homeai.agent_client.AgentClient`: same ``ask`` signature,
    same :class:`AgentReply` result, and it never raises into the caller.
    """

    def __init__(self, cfg: AgentConfig, runner=None, memory=None) -> None:
        self._cfg = cfg
        # Injectable so tests never spawn a real process.
        self._run = runner or subprocess.run
        # Optional ConversationMemory. Continuity is carried as plain text in
        # the request, never by reusing a session, so the poisoning fix that
        # made every call stateless stays intact.
        self._memory = memory

    def _build_command(self, utterance: str, state_file: str) -> list[str]:
        return [
            self._cfg.cli_binary,
            "agent",
            "-a",
            self._cfg.agent_name,
            "--session-state-file",
            state_file,
            "-m",
            utterance,
        ]

    def health(self) -> bool:
        """Cheap check that the ZeroClaw binary is present and runnable."""
        try:
            proc = self._run(
                [self._cfg.cli_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return proc.returncode == 0
        except Exception:  # noqa: BLE001 - health must never raise
            return False

    def ask(self, utterance: str) -> AgentReply:
        if not utterance or not utterance.strip():
            return AgentReply(ok=False, error="empty utterance")

        # Compose once, outside the retry loop: a retry must resend the same
        # context, and re-rendering could pick up an expiry mid-loop.
        request = (
            self._memory.build_request(utterance) if self._memory else utterance
        )

        last_error = "unknown error"
        attempts = 0

        for attempt in range(self._cfg.max_retries + 1):
            attempts = attempt + 1
            state_file = os.path.join(
                tempfile.gettempdir(), f"homeai-voice-{uuid.uuid4().hex[:12]}.json"
            )

            try:
                proc = self._run(
                    self._build_command(request, state_file),
                    capture_output=True,
                    text=True,
                    timeout=self._cfg.timeout_s,
                )
            except subprocess.TimeoutExpired:
                last_error = f"agent timed out after {self._cfg.timeout_s}s"
                log.warning("%s (attempt %d)", last_error, attempts)
                continue
            except FileNotFoundError:
                # Misconfiguration, not a transient fault. Retrying cannot help.
                return AgentReply(
                    ok=False,
                    error=f"zeroclaw binary not found at {self._cfg.cli_binary}",
                    attempts=attempts,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"agent invocation failed: {exc}"
                log.warning("%s (attempt %d)", last_error, attempts)
                time.sleep(min(2**attempt, 5))
                continue
            finally:
                # The state file is single-use; never let it leak or be reused.
                try:
                    os.unlink(state_file)
                except OSError:
                    pass

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                last_error = f"agent exited {proc.returncode}: {stderr[:200]}"
                log.warning("%s (attempt %d)", last_error, attempts)
                time.sleep(min(2**attempt, 5))
                continue

            text = (proc.stdout or "").strip()

            if not text:
                last_error = "agent returned an empty reply"
                log.warning("%s (attempt %d)", last_error, attempts)
                continue

            if detect_leaked_markup(text):
                # Discard entirely. Never speak it, never keep it.
                last_error = "agent leaked tool-call markup"
                log.warning("%s (attempt %d) - discarding and retrying", last_error, attempts)
                continue

            if self._memory is not None:
                # Store the raw utterance, not the context-laden request:
                # otherwise each turn would nest the previous window inside
                # itself and grow without bound.
                self._memory.add(utterance, text)

            return AgentReply(ok=True, text=text, attempts=attempts)

        return AgentReply(
            ok=False,
            error=last_error,
            leaked=last_error.endswith("markup"),
            attempts=attempts,
        )


def build_agent_client(cfg: AgentConfig, memory=None):
    """Return the transport named by ``cfg.transport``.

    Defaults to the CLI client, which is the only transport that enforces the
    agent's risk profile. ``http`` is retained for debugging and for future
    use if the gateway gains profile enforcement.
    """
    if cfg.transport == "http":
        from .agent_client import AgentClient

        log.warning(
            "using HTTP gateway transport: this does NOT enforce the agent "
            "risk profile and should not be used for the voice path"
        )
        return AgentClient(cfg)
    return CliAgentClient(cfg, memory=memory)
