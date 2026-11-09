# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Importing homeai turns onnxruntime's telemetry off before onnxruntime loads.

Why: with home read-only (the system unit's sandbox) onnxruntime wrote its
telemetry session file ":memory:.ses" into the repo (2026-11-09).
"""
from __future__ import annotations

import os
import subprocess
import sys


def test_importing_homeai_disables_onnxruntime_telemetry():
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    out = subprocess.run([sys.executable, "-c", "import homeai, os; print(os.environ.get('ORT_DISABLE_TELEMETRY'))"],
                         env=env, capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "1"
