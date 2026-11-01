# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Long-term memory: PostgreSQL + pgvector. See plans/MEMORY_PLAN.md.

Separate from ``homeai.memory`` (the short, in-process conversation window).
Nothing here is imported by the voice loop unless long-term memory is on, and
every failure degrades to "no memories", never to a broken turn.
"""
