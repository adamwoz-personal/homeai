"""Content policy hooks for art (AVATAR_PLAN §7.7, 6a step 7).

Every art request passes ``check_request`` (the words), ``check_spec`` (the
patched spec) and ``check_image`` (each finished picture) before anything is
shown. ``Allow`` lets it through, ``Rewrite`` swaps in a safer spec, ``Deny``
stops it with a reason that is always spoken, never silent.

Only ``allow-all`` exists today (Adam: hooks now, policies per person later).
An unknown policy name fails closed: everything is denied until it is fixed,
so a typo can never quietly switch a child's account to allow-all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MSG_UNKNOWN_POLICY = "Drawing is switched off for this account until its settings are fixed."


@dataclass(frozen=True)
class Decision:
    verdict: str            # "allow", "rewrite" or "deny"
    reason: str = ""
    spec: dict | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict != "deny"


ALLOW = Decision("allow")


class AllowAll:
    name = "allow-all"

    def check_request(self, text: str, profile: str) -> Decision:
        return ALLOW

    def check_spec(self, spec: dict, profile: str) -> Decision:
        return ALLOW

    def check_image(self, path: Path, profile: str) -> Decision:
        return ALLOW


class DenyAll:
    def __init__(self, name: str) -> None:
        self.name = name

    def _deny(self, *args) -> Decision:
        return Decision("deny", MSG_UNKNOWN_POLICY)

    check_request = check_spec = check_image = _deny


POLICIES = {"allow-all": AllowAll}


def load(name: str):
    factory = POLICIES.get(name)
    return factory() if factory else DenyAll(name)
