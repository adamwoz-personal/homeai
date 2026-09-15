# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Client for the ZeroClaw gateway.

Responsibilities:

* Send a transcribed utterance and return the agent's reply.
* Use a **fresh session identifier for every request**. This is not a stylistic
  choice - the local model leaks malformed tool-call markup on a minority of
  turns, and once that markup is in conversation history the model imitates it
  on every subsequent turn. Reuse of a poisoned session is unrecoverable, so
  the pipeline never reuses one.
* Never raise into the caller's main loop. Every failure is reported as an
  ``AgentReply`` with ``ok=False`` so the voice service can apologise and stay
  alive.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

import requests

from .config import AgentConfig
from .safety import detect_leaked_markup

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentReply:
    ok: bool
    text: str = ""
    error: str = ""
    leaked: bool = False
    attempts: int = 0


class AgentClient:
    def __init__(self, cfg: AgentConfig, session: requests.Session | None = None) -> None:
        self._cfg = cfg
        self._session = session or requests.Session()

    # -- internals ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"
        return headers

    @staticmethod
    def _extract_text(payload: object) -> str:
        """Pull reply text out of the gateway response.

        The gateway's exact response shape is not contractually fixed, so try
        the plausible keys in order rather than assuming one.
        """
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return ""
        for key in ("reply", "response", "message", "text", "content", "output", "result"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
            # Some gateways nest one level, e.g. {"result": {"text": ...}}
            if isinstance(value, dict):
                nested = AgentClient._extract_text(value)
                if nested:
                    return nested
        return ""

    # -- public API --------------------------------------------------------

    def health(self) -> bool:
        try:
            resp = self._session.get(self._cfg.health_url, timeout=5)
            return resp.status_code == 200
        except requests.RequestException as exc:
            log.warning("gateway health check failed: %s", exc)
            return False

    def ask(self, utterance: str) -> AgentReply:
        """Send ``utterance`` to the agent and return its reply.

        Retries on transport errors and on leaked markup. Each attempt uses a
        brand-new session id, so a poisoned attempt cannot contaminate the
        retry.
        """
        if not utterance or not utterance.strip():
            return AgentReply(ok=False, error="empty utterance")

        last_error = "unknown error"
        attempts = 0

        for attempt in range(self._cfg.max_retries + 1):
            attempts = attempt + 1
            session_id = f"voice-{uuid.uuid4().hex[:12]}"
            body = {
                "message": utterance,
                "session_id": session_id,
                "source": "homeai-voice",
            }

            try:
                resp = self._session.post(
                    self._cfg.url,
                    json=body,
                    headers=self._headers(),
                    timeout=self._cfg.timeout_s,
                )
            except requests.Timeout:
                last_error = f"agent timed out after {self._cfg.timeout_s}s"
                log.warning("%s (attempt %d)", last_error, attempts)
                continue
            except requests.RequestException as exc:
                last_error = f"agent unreachable: {exc}"
                log.warning("%s (attempt %d)", last_error, attempts)
                time.sleep(min(2 ** attempt, 5))
                continue

            if resp.status_code == 401:
                # Retrying will not help; the token is wrong.
                return AgentReply(
                    ok=False,
                    error="gateway rejected credentials (401) - check HOMEAI_AGENT_TOKEN",
                    attempts=attempts,
                )
            if resp.status_code >= 500:
                last_error = f"gateway error {resp.status_code}"
                log.warning("%s (attempt %d)", last_error, attempts)
                time.sleep(min(2 ** attempt, 5))
                continue
            if resp.status_code != 200:
                return AgentReply(
                    ok=False,
                    error=f"gateway returned {resp.status_code}",
                    attempts=attempts,
                )

            try:
                payload = resp.json()
            except ValueError:
                payload = resp.text

            text = self._extract_text(payload).strip()

            if not text:
                last_error = "agent returned an empty reply"
                log.warning("%s (attempt %d)", last_error, attempts)
                continue

            if detect_leaked_markup(text):
                # Discard entirely. Never speak it, never keep it.
                last_error = "agent leaked tool-call markup"
                log.warning("%s (attempt %d) - discarding and retrying", last_error, attempts)
                continue

            return AgentReply(ok=True, text=text, attempts=attempts)

        return AgentReply(
            ok=False,
            error=last_error,
            leaked=last_error.endswith("markup"),
            attempts=attempts,
        )
