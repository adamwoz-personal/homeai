# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""AMD graphics cards' memory from sysfs, in one place (CODE_REVIEW_PLAN item 4).

The mode switcher, the installer's assessment and the art benchmarks each read
``mem_info_vram_total/used`` their own way, in three different units. This
reads bytes; each caller keeps its own unit and rounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DRM = Path("/sys/class/drm")


@dataclass(frozen=True)
class Card:
    name: str           # "card1"
    total: int          # bytes
    used: int           # bytes (0 if the card doesn't say)


def amd_cards(drm_root: Path = DRM) -> list[Card]:
    """Every card that reports its VRAM, in name order. Unreadable cards are skipped."""
    cards = []
    for total_file in sorted(Path(drm_root).glob("card*/device/mem_info_vram_total")):
        device = total_file.parent
        try:
            total = int(total_file.read_text().strip())
            used_file = device / "mem_info_vram_used"
            used = int(used_file.read_text().strip()) if used_file.exists() else 0
        except (OSError, ValueError):
            continue
        cards.append(Card(device.parent.name, total, used))
    return cards


def biggest(drm_root: Path = DRM) -> Card | None:
    """The card with the most VRAM (the 7900 XT here), or None."""
    return max(amd_cards(drm_root), key=lambda c: c.total, default=None)
