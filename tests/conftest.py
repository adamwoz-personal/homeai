# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Keep unit tests away from the real house.

This machine has real Home Assistant credentials. Without this, the MCP
tests would list (and could call) the real home tools, and timer ledgers
and quiet windows would be written to the real runtime directory.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMEAI_HA_ENV", str(tmp_path / "no-ha.env"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("HA_TOKEN", raising=False)
    monkeypatch.delenv("HA_URL", raising=False)
    (tmp_path / "run").mkdir(exist_ok=True)
