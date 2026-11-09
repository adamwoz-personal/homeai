# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Home AI voice assistant package."""

import os

# onnxruntime (wake word, speaker ID, embeddings) keeps a telemetry device ID in the
# home cache. In the sandboxed system unit home is read-only, so it wrote ":memory:.ses"
# into the repo instead (2026-11-09). We never wanted the telemetry: off, before any import.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"
