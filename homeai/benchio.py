# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Where benchmarks, probes and evals put their results (CODE_REVIEW_PLAN item 5).

Every tool wrote ``~/homeai-bench/<name>-<YYYYmmdd-HHMMSS>`` by hand (19 copies,
two spellings of the time stamp). The folder isn't created here: each tool
already creates it the way it needs (or writes a single file).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path.home() / "homeai-bench"


def bench_path(name: str, suffix: str = "", now: datetime | None = None) -> Path:
    """``~/homeai-bench/<name>-<time><suffix>``, e.g. bench_path("rider") or bench_path("contention", ".json")."""
    return ROOT / f"{name}-{(now or datetime.now()):%Y%m%d-%H%M%S}{suffix}"
