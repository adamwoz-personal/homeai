# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Hardware discovery, so configuration is not welded to one machine.

Why this module exists
----------------------
This project was developed on one specific desktop, and several values were
originally hard-coded to it -- most visibly ``plughw:2,0``, the analog output
of that box's ALC897 codec. Card *numbers are not stable*: they depend on
which controllers the kernel enumerates, so a machine with no discrete GPU, or
with a USB headset plugged in at boot, will number its cards differently and
the hard-coded value will either fail outright or, worse, play audio silently
into an unplugged HDMI monitor.

Everything here is therefore a *detector with an override*. Detection picks a
sensible default on an unknown machine; an explicit ``HOMEAI_*`` environment
variable always wins, because no heuristic should be able to overrule a human
who knows their own hardware.

Notes for whoever maintains this next (human or AI)
---------------------------------------------------
The ranking below encodes intent, not trivia. For a *voice assistant* we want
the output a person will actually hear from across a room:

* **USB audio** is ranked first. On a purpose-built assistant it is almost
  always a deliberate choice -- a USB speaker, speakerphone, or DAC someone
  plugged in on purpose.
* **Analog line-out / headphone** is next. It is the built-in speaker jack on
  most desktops and laptops.
* **HDMI and S/PDIF are ranked last** and are near-disqualified. HDMI audio
  usually goes to a monitor that is powered off or has no speakers. This is
  the failure that is *hardest to diagnose*, because every layer reports
  success and the room stays silent.

If you are adding a new platform, prefer extending ``_SCORES`` over adding
branches elsewhere. If ALSA is absent entirely (a container, PipeWire-only
system, or macOS) detection returns ``"default"``, which is the one device
name that is always valid when any sound system exists at all.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


# ``aplay -l`` line, e.g.
#   card 2: Generic_1 [HD-Audio Generic], device 0: ALC897 Analog [ALC897 Analog]
_CARD_LINE = re.compile(
    r"^card\s+(?P<card>\d+):\s+(?P<cardid>\S+)\s+\[(?P<cardname>[^]]*)\]"
    r",\s*device\s+(?P<device>\d+):\s+(?P<devname>[^[]*)\[(?P<devdesc>[^]]*)\]"
)

# Higher wins. Ordered most- to least-specific; the first pattern that matches
# a device's combined description decides its score.
_SCORES: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bUSB\b", re.IGNORECASE), 100),
    (re.compile(r"\bheadphone\b", re.IGNORECASE), 80),
    (re.compile(r"\banalog\b", re.IGNORECASE), 70),
    (re.compile(r"\bline\s*out\b", re.IGNORECASE), 60),
    (re.compile(r"\b(?:hdmi|displayport|\bdp\b)\b", re.IGNORECASE), 5),
    (re.compile(r"\b(?:digital|s/?pdif|iec958)\b", re.IGNORECASE), 10),
)

_FALLBACK = "default"


@dataclass(frozen=True)
class AlsaDevice:
    """One ALSA playback or capture endpoint."""

    card: int
    device: int
    card_name: str
    device_name: str

    @property
    def alsa_id(self) -> str:
        """The ``plughw`` identifier.

        ``plughw`` rather than ``hw`` is deliberate and load-bearing. ``hw``
        demands that the application match the hardware's native format
        exactly; the ALC897 on the development box accepts only stereo at
        44.1 kHz or above, while Piper emits mono 22.05 kHz, so ``hw:2,0``
        fails with "Channels count non available". ``plughw`` inserts ALSA's
        conversion plugin and resolves this. Do not "simplify" it to ``hw``.
        """
        return f"plughw:{self.card},{self.device}"

    @property
    def description(self) -> str:
        return f"{self.card_name} {self.device_name}".strip()

    def score(self) -> int:
        for pattern, value in _SCORES:
            if pattern.search(self.description):
                return value
        return 30


def parse_aplay(output: str) -> list[AlsaDevice]:
    """Parse ``aplay -l`` / ``arecord -l`` output into devices.

    Split out from the subprocess call so it can be tested against captured
    output from machines this code will never run on.
    """
    devices: list[AlsaDevice] = []
    for line in output.splitlines():
        match = _CARD_LINE.match(line.strip())
        if not match:
            continue
        devices.append(
            AlsaDevice(
                card=int(match.group("card")),
                device=int(match.group("device")),
                card_name=match.group("cardname").strip(),
                device_name=match.group("devdesc").strip(),
            )
        )
    return devices


def _run_lister(command: str) -> str:
    """Return listing output, or '' if the tool is missing or fails.

    Detection must never raise. A machine without alsa-utils should degrade to
    ``"default"``, not crash the assistant at import time.
    """
    if shutil.which(command) is None:
        return ""
    try:
        proc = subprocess.run(
            [command, "-l"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def rank_playback(devices: list[AlsaDevice]) -> list[AlsaDevice]:
    """Best-first ordering. Ties break on card then device for determinism."""
    return sorted(devices, key=lambda d: (-d.score(), d.card, d.device))


def detect_output_device(listing: str | None = None) -> str:
    """Pick the most plausible speaker output.

    Returns a ``plughw:C,D`` string, or ``"default"`` when nothing can be
    enumerated. Pass ``listing`` to test without hardware.
    """
    text = _run_lister("aplay") if listing is None else listing
    ranked = rank_playback(parse_aplay(text))
    if not ranked:
        return _FALLBACK
    best = ranked[0]
    # Refuse to auto-select HDMI/digital. Guessing wrong here produces a
    # silent assistant with no error anywhere, which is far worse than
    # falling back to whatever the system mixer already considers default.
    if best.score() <= 10:
        return _FALLBACK
    return best.alsa_id


def describe_audio() -> str:
    """Human-readable audio inventory for diagnostics and bug reports."""
    playback = rank_playback(parse_aplay(_run_lister("aplay")))
    capture = parse_aplay(_run_lister("arecord"))

    lines = ["Playback devices (best first):"]
    if not playback:
        lines.append("  none found (is alsa-utils installed?)")
    for dev in playback:
        lines.append(f"  {dev.alsa_id:<16} score={dev.score():<4} {dev.description}")

    lines.append("Capture devices:")
    if not capture:
        lines.append("  none found")
    for dev in capture:
        lines.append(f"  {dev.alsa_id:<16} {dev.description}")

    lines.append(f"Auto-selected output: {detect_output_device()}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_audio())
