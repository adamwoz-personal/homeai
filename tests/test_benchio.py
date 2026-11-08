"""homeai/benchio.py: where tools put results, and tools/inspect_tool_imports.py (checks tools without running them)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from homeai import benchio as B


def test_bench_path():
    when = datetime(2026, 11, 8, 14, 5, 9)
    assert B.bench_path("rider", now=when) == Path.home() / "homeai-bench" / "rider-20261108-140509"
    assert B.bench_path("contention", ".json", now=when).name == "contention-20261108-140509.json"


def test_the_import_check_never_runs_a_tool(tmp_path):
    from tools.inspect_tool_imports import try_import
    ran = tmp_path / "ran"
    tool = tmp_path / "tool.py"
    tool.write_text(f"import json\nif __name__ == '__main__':\n    open({str(ran)!r}, 'w').write('x')\n")
    broken = tmp_path / "broken.py"
    broken.write_text("import no_such_module_here\n")
    assert try_import(tool) is None and not ran.exists()
    assert "no_such_module_here" in try_import(broken)
