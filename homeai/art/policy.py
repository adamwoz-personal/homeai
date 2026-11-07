"""Content policy hooks for art (AVATAR_PLAN §7.7, 6a step 7).

Every art request passes ``check_request`` (the words), ``check_spec`` (the
patched spec) and ``check_image`` (each finished picture) before anything is
shown. ``Allow`` lets it through, ``Rewrite`` swaps in a safer spec, ``Deny``
stops it with a reason that is always spoken, never silent.

Policies: ``allow-all`` (registered adults) and ``kid-safe`` (registered kids
and any voice Jarvis doesn't know; Adam 2026-11-07). An unknown policy name fails closed: everything is denied until it is fixed,
so a typo can never quietly switch a child's account to allow-all.
"""
from __future__ import annotations

import re
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


# Kid-safe: words that stop a request or a spec. Whole words only, so "killer whale" and "bloodhound" pass.
# Weapons, monsters and battles stay allowed (samurai, dragons, knights); sex, nudity, gore, drugs and hate don't.
KID_BLOCKED = re.compile(r"\b(naked|nude|nudity|topless|sexy|sex|sexual|lingerie|underwear|undies|bikini|panties|"
                         r"bra|boobs?|breasts?|butt|nsfw|blood|bloody|gore|gory|guts|decapitat\w*|dismember\w*|"
                         r"corpses?|dead bod(y|ies)|severed|kill|kills|killing|killed|murder\w*|suicide|torture\w*|"
                         r"drugs?|weed|cocaine|beer|alcohol|drunk|cigarettes?|smoking|vape|nazis?|swastika|kkk)\b",
                         re.I)
PEOPLE = re.compile(r"\b(person|people|man|men|woman|women|girl|boy|kid|child|children|lady|guy|samurai|knight|"
                    r"princess|prince|queen|king|warrior|hero|character|he|she|they|family|friends?|teacher|"
                    r"wizard|witch|pirate|ninja|astronaut|mermaid|fairy)\b", re.I)
MSG_KID_DENIED = "Let's draw something else. How about an animal, a place, or a dragon?"
KID_STYLE = "family-friendly"
KID_CLOTHED = "everyone fully clothed"


class KidSafe:
    """For kids and unknown voices: refuses adult words, and steers every picture family-friendly.

    The image check can't look at pictures yet (no classifier on the box), so it
    allows; the request and spec checks plus the positive steering are what keep
    pictures kid-safe today. Positive words, not "avoid nudity": klein reads the
    whole prompt as a description, and naming the thing primes it.
    """

    name = "kid-safe"

    def _text(self, spec: dict) -> str:
        return " ".join(str(v) for k, v in spec.items() if isinstance(v, str))

    def check_request(self, text: str, profile: str) -> Decision:
        return Decision("deny", MSG_KID_DENIED) if KID_BLOCKED.search(text) else ALLOW

    def check_spec(self, spec: dict, profile: str) -> Decision:
        text = self._text(spec)
        if KID_BLOCKED.search(text):
            return Decision("deny", MSG_KID_DENIED)
        safe = dict(spec)
        style = str(safe.get("style") or "")
        if KID_STYLE not in style:
            safe["style"] = f"{style}, {KID_STYLE}" if style else KID_STYLE
        details = str(safe.get("details") or "")
        if PEOPLE.search(text) and KID_CLOTHED not in details:
            safe["details"] = f"{details}, {KID_CLOTHED}" if details else KID_CLOTHED
        return Decision("rewrite", spec=safe) if safe != spec else ALLOW

    def check_image(self, path: Path, profile: str) -> Decision:
        return ALLOW


POLICIES = {"allow-all": AllowAll, "kid-safe": KidSafe}


def load(name: str):
    factory = POLICIES.get(name)
    return factory() if factory else DenyAll(name)
