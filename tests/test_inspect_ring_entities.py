"""tools/inspect_ring_entities.py: the distinct activities in a history response."""
from __future__ import annotations

from tools import inspect_ring_entities as R


def test_activity_times_are_distinct_timestamps_only():
    raw = [[{"state": "2026-11-15T15:49:39+00:00"}, {"state": "unavailable"},
            {"state": "2026-11-15T15:49:39+00:00"}, {"state": "2026-11-15T16:02:00+00:00"}]]
    assert R.activity_times(raw) == ["2026-11-15T15:49:39+00:00", "2026-11-15T16:02:00+00:00"]
    assert R.activity_times(None) == []
